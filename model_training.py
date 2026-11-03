#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""模型级前向：动作加噪、注意力可见性、专家速度预测和 flow matching 损失。

Model forwards: noisy actions, attention visibility, Expert velocity and flow-matching loss.

原始实现拆分副本，需配合原仓库依赖运行。
Extracted implementation; requires the original repository dependencies.
"""

from ._runtime import F, Tensor, make_training_inputs, torch
from .tensor_utils import make_att_2d_masks, recover_clean_actions

class ModelTrainingMixin:
    # 基础 π0.5 双流训练入口：真实动作与噪声插值得到 x_t，目标速度为 noise-actions；这里没有预测未来条件块，Stage 2 使用专用缓存路径。
    # Base two-stream π0.5 training: interpolate actions and noise into x_t with target velocity noise-actions. This path has no predicted-future block; Stage 2 uses the dedicated cached-context path.
    def forward(
        self,
        images,
        img_masks,
        tokens,
        masks,
        actions,
        noise=None,
        time=None,
        return_intermediates: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Do a full training forward pass and compute the loss."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        # t=0 是真实动作，t=1 是噪声。模型预测 dx_t/dt，推理从噪声端向较小 t 积分以恢复动作。
        # At t=0 the sample is the clean action; at t=1 it is noise. Predict dx_t/dt and integrate toward smaller t during inference.
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        losses = F.mse_loss(u_t, v_t, reduction="none")
        # 额外中间量用于分析去噪行为；predicted_clean_actions 是当前速度对应的一步估计，不替代完整多步采样。
        # Optional intermediates aid analysis; predicted_clean_actions is a one-step estimate from current velocity, not a replacement for iterative sampling.
        if return_intermediates:
            # x_t = t * noise + (1 - t) * action and v = dx/dt, hence
            # the one-step clean-action estimate at t=0 is x_t - t * v.
            predicted_clean_actions = recover_clean_actions(x_t, v_t, time)
            return losses, {
                "x_t": x_t,
                "time": time,
                "predicted_velocity": v_t,
                "predicted_clean_actions": predicted_clean_actions,
            }
        return losses

    # 从真实动作和噪声构造 flow 输入，用未来条件构造额外 token；输出逐元素损失供策略层屏蔽 padding 和 RTC 前缀。
    # Construct flow inputs from actions and noise and condition on future tokens; return elementwise losses for policy-level padding and RTC-prefix masking.
    # 准备动作流训练输入、未来条件和掩码。
    # Expert training entry: prepare flow-matching inputs, future conditions and masks, then call the layerwise Q/K/V path.
    def forward_expert_with_cached_context(
        self,
        context_pad_masks: Tensor,
        context_key_values: list[tuple[Tensor, Tensor]],
        actions: Tensor,
        future_condition: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None,
        prefix_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute flow loss without recomputing the VLM image/language prefix."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        # flow matching 将真实动作与噪声插值；模型学习随时间变化的速度，而不是直接做动作分类。
        # Flow matching interpolates actions and noise and learns a time-dependent velocity, rather than classifying actions.
        # RTC 已知前缀的构造交给原 RTC 工具处理，损失聚合时还会排除这些已知位置。
        # The original RTC utility constructs known-prefix inputs; loss aggregation later excludes those known positions.
        x_t, target_velocity, model_time = make_training_inputs(actions, noise, time, prefix_mask)
        action_embeds, action_pad_masks, action_att_masks, action_adarms_cond = self.embed_suffix(x_t, model_time)

        batch_size = actions.shape[0]
        expected_future_tokens = (
            len(self.config.world_model_image_features)
            * self.config.wam_future_condition_spatial_pool_size**2
        )
        expected_future_shape = (
            batch_size,
            expected_future_tokens,
            self.action_in_proj.out_features,
        )
        # 未来 token 数由相机数量与输出网格决定，宽度必须等于 Expert 输入宽度；提前报错可避免错误布局默默进入注意力。
        # Camera count and output grid determine future-token count; feature width must match the Expert. Validate before an incorrect layout enters attention.
        if future_condition.shape != expected_future_shape:
            raise ValueError(
                "Future condition has an invalid shape; "
                f"expected {expected_future_shape}, got {tuple(future_condition.shape)}."
            )
        if context_pad_masks.ndim != 2 or context_pad_masks.shape[0] != batch_size:
            raise ValueError(
                "Context pad mask must have shape [B,C]; "
                f"got {tuple(context_pad_masks.shape)} for batch size {batch_size}."
            )

        future_pad_masks = torch.ones(
            batch_size,
            expected_future_tokens,
            dtype=torch.bool,
            device=actions.device,
        )
        future_att_masks = torch.zeros_like(future_pad_masks)
        future_att_masks[:, 0] = True
        future_adarms_cond = action_adarms_cond.new_zeros(batch_size, self.action_in_proj.out_features)

        expert_dtype = self.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.q_proj.weight.dtype
        future_condition = future_condition.to(dtype=expert_dtype)
        action_embeds = action_embeds.to(dtype=expert_dtype)

        # 未来看上下文和未来；动作看上下文、未来和动作；屏蔽 padding。
        # Visibility: future sees context+future; action sees context+future+action. Context has no query rows in this pass. Padding stays masked.
        # Construct the same blockwise visibility as the original three-stream
        # forward, then retain only the future/action query rows. Context KV is
        # supplied explicitly and therefore has no query rows in this pass.
        context_att_masks = torch.zeros_like(context_pad_masks)
        pad_masks = torch.cat(
            [context_pad_masks, future_pad_masks, action_pad_masks],
            dim=1,
        )
        att_masks = torch.cat(
            [
                context_att_masks,
                future_att_masks,
                action_att_masks.to(dtype=torch.bool),
            ],
            dim=1,
        )
        # 按 context -> future -> action 排列块标记；同块内可双向关注，后块可读前块，前块不能读后块。
        # Blocks are ordered context -> future -> action. Tokens see their own block bidirectionally and earlier blocks, never later blocks.
        # 因此动作 token 之间不是默认逐步因果自回归；整个动作 chunk 共同参与去噪。
        # Action tokens are therefore not strictly autoregressive by timestep; the chunk participates jointly in denoising.
        full_attention = make_att_2d_masks(pad_masks, att_masks)
        context_length = context_pad_masks.shape[1]
        expert_attention = self._prepare_attention_masks_4d(full_attention[:, context_length:, :])
        full_position_ids = torch.cumsum(pad_masks, dim=1) - 1
        # 位置编号按有效 token 累加，跳过 padding；Expert 位置从上下文之后继续，保持 Q/K 的相对位置一致。
        # Positions advance over valid tokens, skipping padding; Expert positions continue after context to align Q/K positions.
        expert_position_ids = full_position_ids[:, context_length:]

        _, action_output = self.paligemma_with_expert.forward_expert_with_context_kv(
            context_key_values=context_key_values,
            future_embeds=future_condition,
            action_embeds=action_embeds,
            attention_mask=expert_attention,
            position_ids=expert_position_ids,
            future_adarms_cond=future_adarms_cond,
            action_adarms_cond=action_adarms_cond,
        )
        action_output = action_output[:, -self.config.chunk_size :].float()
        # 输出动作速度预测，计算 flow matching 均方误差。
        # Project Expert hidden states to action velocity predictions and compute elementwise flow-matching MSE.
        # 最终线性层才把 Expert 隐藏状态变成每个时间步、每个动作维度的速度预测。
        # This final linear layer converts Expert hidden states into per-timestep, per-action-dimension velocity predictions.
        predicted_velocity = self.action_out_proj(action_output)
        return F.mse_loss(target_velocity, predicted_velocity, reduction="none")


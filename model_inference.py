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

"""推理缓存裁剪、未来条件预填、多步去噪和动作采样。

Inference cache cropping, future-condition prefill, iterative denoising and action sampling.

原始实现拆分副本，需配合原仓库依赖运行。
Extracted implementation; requires the original repository dependencies.
"""

from ._runtime import Any, Tensor, Unpack, prepare_inference_prefix, torch
from .tensor_utils import ActionSelectKwargs, make_att_2d_masks, pad_vector

# 把一次观测的 VLM/未来条件预填与多步动作去噪分离，避免每个去噪步重复执行视觉语言和世界模型。
# Separate observation-level VLM/future-condition prefill from iterative action denoising so vision/language and dynamics are not rerun each step.
class ModelInferenceMixin:
    # 兼容新 Cache 对象和旧逐层元组；仅从序列尾部移除规划 token，保留每层原始上下文 K/V。
    # Support both modern Cache objects and legacy per-layer tuples; remove planner tokens only from the sequence tail, retaining original-context KV.
    @staticmethod
    def _crop_query_tokens_from_cache(past_key_values: Any, query_count: int) -> Any:
        """Remove latent-action query KV while preserving the original PI0.5 KV."""
        if past_key_values is None:
            raise RuntimeError("VLM did not return a KV cache for WAM inference.")
        if hasattr(past_key_values, "crop"):
            sequence_length = past_key_values.get_seq_length()
            if sequence_length < query_count:
                raise RuntimeError(
                    f"Cannot crop {query_count} query tokens from cache length {sequence_length}."
                )
            past_key_values.crop(-query_count)
            return past_key_values

        cropped_layers = []
        for layer in past_key_values:
            if len(layer) < 2:
                raise RuntimeError("Unsupported legacy KV-cache layer format.")
            key, value, *rest = layer
            cropped_layers.append(
                (
                    key[..., :-query_count, :],
                    value[..., :-query_count, :],
                    *rest,
                )
            )
        return tuple(cropped_layers)

    # 未来条件在一个动作块内固定，只经 Expert 编码一次；返回包含原始上下文和未来条件的增广 KV。
    # Future conditions remain fixed within a chunk and pass through the Expert once; return augmented KV containing original context and future tokens.
    def _prefill_future_condition_cache(
        self,
        *,
        prefix_pad_masks: Tensor,
        past_key_values: Any,
        future_condition: Tensor,
    ) -> tuple[Tensor, Any]:
        """Append fixed predicted-future Expert KV once before denoising."""
        batch_size, future_length, _ = future_condition.shape
        expert_dtype = self.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.q_proj.weight.dtype
        future_pad_masks = torch.ones(
            batch_size,
            future_length,
            dtype=torch.bool,
            device=future_condition.device,
        )
        future_att_masks = torch.zeros_like(future_pad_masks)
        # 所有未来 token 属于同一双向块；它们只能读取已有上下文和未来块，动作尚未加入缓存。
        # All future tokens share one bidirectional block; they see existing context and their future block before any actions are added.
        future_att_masks[:, 0] = True

        prefix_attention = prefix_pad_masks[:, None, :].expand(
            batch_size,
            future_length,
            prefix_pad_masks.shape[1],
        )
        future_attention = make_att_2d_masks(
            future_pad_masks,
            future_att_masks,
        )
        attention_mask = self._prepare_attention_masks_4d(
            torch.cat([prefix_attention, future_attention], dim=2)
        ).to(dtype=expert_dtype)
        prefix_offsets = prefix_pad_masks.sum(dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(future_pad_masks, dim=1) - 1
        future_adarms_cond = torch.zeros(
            batch_size,
            self.action_in_proj.out_features,
            dtype=future_condition.dtype,
            device=future_condition.device,
        )
        _, augmented_cache = self.paligemma_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, future_condition.to(dtype=expert_dtype)],
            use_cache=True,
            adarms_cond=[None, future_adarms_cond],
        )
        if augmented_cache is None:
            raise RuntimeError("Action Expert did not return future-condition KV cache.")
        return (
            torch.cat([prefix_pad_masks, future_pad_masks], dim=1),
            augmented_cache,
        )

    # 先取 VLM 缓存再移除内部查询，经世界模型和重采样器生成未来条件，最终返回可重复使用的专家缓存。
    # Obtain VLM cache, remove internal query entries, predict and resample future conditions, then return reusable Expert context.
    @torch.no_grad()
    # 推理准备：VLM 缓存、裁剪查询、预测未来、预填未来条件缓存。
    # Inference path: VLM prefill -> remove planner-query KV -> predict future -> prefill future-condition KV for repeated denoising.
    def _prepare_wam_inference_context(
        self,
        images,
        img_masks,
        tokens,
        masks,
        state,
        world_images=None,
        rtc_prefix=None,
        rtc_prefix_mask=None,
    ) -> tuple[Tensor, Any]:
        """Compute latent action, predicted future state, and reusable Expert KV."""
        (
            _,
            latent_action_tokens,
            wam_prefix_pad_masks,
            _,
            past_key_values,
        ) = self.encode_wam_prefix(
            images,
            img_masks,
            tokens,
            masks,
            use_cache=True,
            rtc_prefix=rtc_prefix, rtc_prefix_mask=rtc_prefix_mask,
            action_state=pad_vector(state, self.config.max_action_dim).float(),
        )
        # 同时移除潜在动作查询及 RTC 槽；避免动作专家直接读取规划查询而绕过预测未来瓶颈。
        # Remove latent-action queries and RTC slots together so the Expert cannot bypass predicted-future conditioning by reading planner queries directly.
        query_count = self.config.wam_latent_action_num_queries + self.config.rtc_training_max_delay
        past_key_values = self._crop_query_tokens_from_cache(
            past_key_values,
            query_count,
        )
        prefix_pad_masks = wam_prefix_pad_masks[:, :-query_count]

        # VLM inputs can include masked camera slots absent from world-model training.
        if world_images is None:
            world_images = self._select_world_model_images(images)
        current_latent = self.encode_world_images(world_images)
        state = pad_vector(state, self.config.max_state_dim).to(
            device=current_latent.device,
            dtype=torch.float32,
        )
        state_token = self.world_state_encoder(state)
        if self.config.rtc_training_max_delay:
            latent_action_tokens, _ = self.constrain_wam_plan(
                latent_action_tokens, pad_vector(state, self.config.max_action_dim).float(),
                rtc_prefix, rtc_prefix_mask)
        predicted_delta = self.world_model(
            current_latent,
            state_token,
            latent_action_tokens.to(dtype=torch.float32),
        )
        predicted_future = current_latent + predicted_delta
        future_condition = self.wam_future_condition_resampler(predicted_future)
        return self._prefill_future_condition_cache(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            future_condition=future_condition,
        )

    # 从 t=1 的高斯噪声向 t=0 积分，输出完整动作块；已知 RTC 前缀在每一步更新前后都重新固定。
    # Integrate Gaussian noise from t=1 toward t=0 to produce an action chunk; clamp known RTC prefixes before and after every update.
    @torch.no_grad()  # see openpi `sample_actions` (slightly adapted)
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        noise=None,
        num_steps=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            # Sample noise with padded dimension as expected by action_in_proj
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )  # Use config max_action_dim for internal processing
            noise = self.sample_noise(actions_shape, device)

        trained_prefix = trained_prefix_mask = None
        rtc_mode = kwargs.get('rtc_mode', 'trained' if self.config.rtc_training_max_delay else 'guided')
        if self.config.rtc_training_max_delay:
            if rtc_mode != 'trained' or self._rtc_enabled():
                raise ValueError('This checkpoint requires trained RTC without gradient guidance')
            inference_delay = kwargs.get('inference_delay')
            trained_prefix, trained_prefix_mask = prepare_inference_prefix(
                noise, kwargs.get('prev_chunk_left_over'), 0 if inference_delay is None else inference_delay,
                self.config.rtc_training_max_delay, kwargs.get('execution_horizon'))
        elif rtc_mode == 'trained':
            raise ValueError('Trained RTC requires a checkpoint trained with a nonzero max delay')

        if self.config.world_model_stage in {"stage2", "inference"}:
            state = kwargs.pop("state", None)
            if state is None:
                raise ValueError("WAVE-0 inference requires observation.state.")
            prefix_pad_masks, past_key_values = self._prepare_wam_inference_context(
                images,
                img_masks,
                tokens,
                masks,
                state,
                world_images=kwargs.pop("world_images", None),
                rtc_prefix=trained_prefix, rtc_prefix_mask=trained_prefix_mask,
            )
        else:
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images,
                img_masks,
                tokens,
                masks,
            )
            prefix_att_2d_masks = make_att_2d_masks(
                prefix_pad_masks,
                prefix_att_masks,
            )
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
            self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (  # noqa: SLF001
                "eager"
            )
            _, past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )

        # 积分方向从噪声 t=1 指向动作 t=0，所以步长为负；这是固定步长的一阶 Euler 更新。
        # Integrate from noise at t=1 toward action at t=0, hence a negative step size; this is fixed-step first-order Euler integration.
        dt = -1.0 / num_steps

        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)
            if trained_prefix_mask is not None:
                x_t = torch.where(trained_prefix_mask[..., None], trained_prefix, x_t)
                # 已知动作前缀的时间设为 0，未知后缀继续去噪，形成 [B,T] 的逐 token 时间条件。
                # Known-prefix actions use time zero while the unknown suffix keeps denoising, producing per-token [B,T] time conditions.
                time_tensor = torch.where(trained_prefix_mask, 0.0, time_tensor[:, None])

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=input_x_t,
                    timestep=current_timestep,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t
            if trained_prefix_mask is not None:
                x_t = torch.where(trained_prefix_mask[..., None], trained_prefix, x_t)

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        return x_t

    # 复用预填上下文预测当前时间的动作速度；返回速度而非完整去噪后的动作，积分更新由采样循环执行。
    # Reuse prefilled context to predict action velocity at the current time; the sampling loop performs the integration update.
    # 推理每步去噪：根据时间形状选择普通缓存或自定义专家路径。
    # Denoising entry: samplewise time uses Gemma cache; per-token time uses the custom Expert path.
    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)
        expert_dtype = self.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.q_proj.weight.dtype
        suffix_embs = suffix_embs.to(dtype=expert_dtype)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks).to(dtype=expert_dtype)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        if timestep.ndim == 2:
            # The installed legacy Gemma forward accepts sample-wise AdaRMS
            # only. Reuse our token-aware Expert path and the prefilled future KV.
            cache = past_key_values.to_legacy_cache() if hasattr(past_key_values, 'to_legacy_cache') else list(past_key_values)
            # 未来条件已在缓存中，不再作为新 token 重算；空流保留共享 Expert 接口的两流约定。
            # Future conditions already live in cache and are not appended again; an empty stream preserves the two-stream Expert interface.
            empty_future = suffix_embs[:, :0]
            _, suffix_out = self.paligemma_with_expert.forward_expert_with_context_kv(
                context_key_values=cache, future_embeds=empty_future,
                action_embeds=suffix_embs, attention_mask=full_att_2d_masks_4d,
                position_ids=position_ids,
                future_adarms_cond=adarms_cond.new_zeros(adarms_cond.shape[0], adarms_cond.shape[-1]),
                action_adarms_cond=adarms_cond)
        else:
            outputs_embeds, _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_2d_masks_4d, position_ids=position_ids,
                past_key_values=past_key_values, inputs_embeds=[None, suffix_embs],
                use_cache=False, adarms_cond=[None, adarms_cond])
            suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


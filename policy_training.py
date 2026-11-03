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

"""第一阶段监督、第二阶段总训练流程、负样本与损失统计。

Stage-1 supervision, Stage-2 training orchestration, negative samples and loss aggregation.

原始实现拆分副本，需配合原仓库依赖运行。
Extracted implementation; requires the original repository dependencies.
"""

from ._runtime import ACTION, F, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE, Tensor, masked_suffix_mean, sample_training_prefix_mask, torch
from .tensor_utils import pad_vector

class PolicyTrainingMixin:
    # 在每个任务组内循环移动样本索引，构造同任务但不同轨迹的负动作；单样本任务组没有可用同伴，会返回无效标记。
    # Rotate indices within each task group to form different-trajectory negatives from the same task; singleton groups have no valid peer and are marked invalid.
    @staticmethod
    def _same_task_action_permutation(tasks, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """Create a cyclic non-identity permutation independently within each task."""
        permutation = torch.arange(batch_size, device=device)
        valid = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if tasks is None:
            if batch_size > 1:
                permutation = torch.roll(permutation, shifts=1)
                valid.fill_(True)
            return permutation, valid

        groups: dict[str, list[int]] = {}
        for index, task in enumerate(tasks):
            groups.setdefault(str(task), []).append(index)
        for indices in groups.values():
            if len(indices) < 2:
                continue
            destination = torch.tensor(indices, dtype=torch.long, device=device)
            source = torch.roll(destination, shifts=1)
            permutation[destination] = source
            valid[destination] = True
        return permutation, valid

    # 优先使用同任务其他样本动作，缺少同伴时对自身轨迹做半时间窗循环移动；额外检查动作确实变化，避免把完全相同轨迹当作负样本。
    # Prefer same-task peer actions, otherwise roll the sample trajectory by half a horizon; verify that values changed so identical trajectories are not treated as negatives.
    @classmethod
    def _make_negative_actions(cls, actions: Tensor, tasks) -> tuple[Tensor, Tensor]:
        """Build task-preserving negatives using peers or a temporal trajectory roll.

        RoboTwin stores many paraphrased instructions, so exact instruction groups can
        be singletons. Those samples use a half-horizon temporal roll of their own GT
        trajectory instead of an unrelated task's action.
        """
        permutation, peer_valid = cls._same_task_action_permutation(tasks, actions.shape[0], actions.device)
        negative_actions = actions[permutation].clone()
        temporal_roll = torch.roll(actions, shifts=max(1, actions.shape[1] // 2), dims=1)
        negative_actions[~peer_valid] = temporal_roll[~peer_valid]
        changed = (negative_actions - actions).abs().flatten(1).amax(dim=1) > 1e-6
        return negative_actions, changed

    # 输入每视觉 token 的值 [B,相机,空间token] 与相机有效位；只对有效相机求均值，并返回该样本是否至少有一个有效目标。
    # Reduce [B,camera,spatial_token] values over valid cameras only and report whether each sample has any valid target.
    @staticmethod
    def _masked_per_sample_mean(values: Tensor, camera_valid_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Reduce [B,C,P] values while excluding padded future-camera targets."""
        mask = camera_valid_mask[:, :, None].expand_as(values)
        denominator = mask.sum(dim=(1, 2))
        per_sample = (values * mask).sum(dim=(1, 2)) / denominator.clamp_min(1)
        return per_sample, denominator > 0

    # 没有有效样本时返回与计算图连接的零，而不是对空集合求均值产生 NaN。
    # When no sample is valid, return a graph-connected zero instead of taking an empty mean and producing NaN.
    @staticmethod
    def _mean_over_valid_samples(values: Tensor, valid: Tensor) -> Tensor:
        if valid.any():
            return values[valid].mean()
        return values.sum() * 0.0

    # 第一阶段只训练动作编码器、状态编码器和潜在动力学：从真实动作预测未来视觉特征，并用错误动作轨迹检验预测是否真正依赖动作。
    # Stage 1 trains action/state encoders and latent dynamics to predict future visual features from ground-truth actions; wrong-action trajectories test action dependence.
    def forward_world_model(
        self,
        batch: dict[str, Tensor],
        reduction: str = "mean",
        compute_diagnostics: bool = True,
    ) -> tuple[Tensor, dict]:
        """Stage-1 GT-action latent world-model training forward."""
        current_images, future_images, future_valid_mask = self._prepare_world_images(batch)
        current_latent, future_latent = self.model.encode_world_image_pairs(current_images, future_images)

        state = pad_vector(batch[OBS_STATE], self.config.max_state_dim).to(
            device=current_latent.device, dtype=torch.float32
        )
        action_state = pad_vector(batch[OBS_STATE], self.config.max_action_dim).to(
            device=current_latent.device, dtype=torch.float32
        )
        horizon = self.config.world_model_future_horizons[0]
        actions = self.prepare_action(batch)[:, : horizon + 1].to(
            device=current_latent.device, dtype=torch.float32
        )

        action_tokens = self.model.world_action_encoder(actions, action_state)
        state_token = self.model.world_state_encoder(state)
        pred_delta = self.model.world_model(current_latent, state_token, action_tokens)
        pred_future = current_latent + pred_delta
        target_delta = future_latent - current_latent

        # 未来余弦距离约束特征方向，变化量 Smooth L1 补充幅值约束；两者都先逐 token 计算，再屏蔽无效未来相机。
        # Future cosine distance constrains direction while delta Smooth L1 constrains magnitude; both are computed per token before invalid future cameras are excluded.
        future_distance_tokens = 1.0 - F.cosine_similarity(pred_future.float(), future_latent.float(), dim=-1)
        future_per_sample, future_sample_valid = self._masked_per_sample_mean(
            future_distance_tokens, future_valid_mask
        )
        delta_tokens = F.smooth_l1_loss(pred_delta.float(), target_delta.float(), reduction="none").mean(
            dim=-1
        )
        delta_per_sample, _ = self._masked_per_sample_mean(delta_tokens, future_valid_mask)

        negative_actions, rank_sample_valid = self._make_negative_actions(
            actions, batch.get("world_model_task", batch.get("task"))
        )
        rank_sample_valid = rank_sample_valid & future_sample_valid
        wrong_per_sample = future_per_sample.detach()
        rank_per_sample = torch.zeros_like(future_per_sample)
        # 启用排序且有有效负样本时才增加第二次世界模型预测；希望真实动作的未来距离至少比错误动作小一个 margin。
        # Run an extra dynamics prediction only when ranking is enabled and negatives are valid; correct actions should beat wrong actions by a margin.
        if self.config.world_model_rank_loss_weight > 0 and rank_sample_valid.any():
            wrong_action_tokens = self.model.world_action_encoder(negative_actions, action_state)
            wrong_delta = self.model.world_model(current_latent, state_token, wrong_action_tokens)
            wrong_future = current_latent + wrong_delta
            wrong_distance_tokens = 1.0 - F.cosine_similarity(
                wrong_future.float(), future_latent.float(), dim=-1
            )
            wrong_per_sample, _ = self._masked_per_sample_mean(wrong_distance_tokens, future_valid_mask)
            rank_per_sample = F.relu(
                self.config.world_model_rank_margin + future_per_sample - wrong_per_sample
            )
            rank_per_sample = rank_per_sample * rank_sample_valid

        total_per_sample = (
            self.config.world_model_future_loss_weight * future_per_sample
            + self.config.world_model_delta_loss_weight * delta_per_sample
            + self.config.world_model_rank_loss_weight * rank_per_sample
        )
        valid_total = future_sample_valid

        # 只在存在真实未来监督的样本上聚合；序列边界的无效未来不能把损失均值稀释。
        # Aggregate only samples with valid future supervision; boundary samples lacking targets must not dilute the mean.
        total_loss = self._mean_over_valid_samples(total_per_sample, valid_total)

        loss_dict: dict[str, float] = {}
        if compute_diagnostics:
            future_loss = self._mean_over_valid_samples(future_per_sample, future_sample_valid)
            delta_loss = self._mean_over_valid_samples(delta_per_sample, future_sample_valid)
            rank_loss = self._mean_over_valid_samples(rank_per_sample, rank_sample_valid)
            wrong_distance = self._mean_over_valid_samples(wrong_per_sample, rank_sample_valid)
            correct_rank_distance = self._mean_over_valid_samples(future_per_sample, rank_sample_valid)
            loss_dict = {
                "loss": total_loss.item(),
                "loss/world_total": total_loss.item(),
                "loss/world_future": future_loss.item(),
                "loss/world_delta": delta_loss.item(),
                "loss/world_rank": rank_loss.item(),
                "metric/correct_future_distance": future_loss.item(),
                "metric/shuffled_future_distance": wrong_distance.item(),
                "metric/action_sensitivity_gap": (wrong_distance - correct_rank_distance).item(),
                "metric/future_valid_fraction": (future_sample_valid.float().mean().item()),
                "metric/rank_valid_fraction": rank_sample_valid.float().mean().item(),
            }

        if compute_diagnostics and self.config.world_model_compute_zero_action_metric:
            with torch.no_grad():
                zero_action_tokens = self.model.world_action_encoder(torch.zeros_like(actions), action_state)
                zero_delta = self.model.world_model(current_latent, state_token, zero_action_tokens)
                zero_future = current_latent + zero_delta
                zero_distance_tokens = 1.0 - F.cosine_similarity(
                    zero_future.float(), future_latent.float(), dim=-1
                )
                zero_per_sample, _ = self._masked_per_sample_mean(zero_distance_tokens, future_valid_mask)
                zero_distance = self._mean_over_valid_samples(zero_per_sample, future_sample_valid)
            loss_dict["metric/zero_action_future_distance"] = zero_distance.item()

        if reduction == "none":
            return total_per_sample, loss_dict
        return total_loss, loss_dict

    # 复用第一阶段的未来与变化量回归目标，返回每样本分数和有效位；这里不含错误动作排序项，避免把饱和排序信号当作主要第二阶段监督。
    # Reuse Stage-1 future/delta regression, returning per-sample scores and validity; omit the ranking term so saturated ranking signals do not drive Stage-2 supervision.
    def _world_score_per_sample(
        self,
        pred_delta: Tensor,
        current_latent: Tensor,
        future_latent: Tensor,
        future_valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return the Stage-1 regression score without its saturated ranking term."""
        pred_future = current_latent + pred_delta
        target_delta = future_latent - current_latent
        future_distance_tokens = 1.0 - F.cosine_similarity(pred_future.float(), future_latent.float(), dim=-1)
        future_per_sample, sample_valid = self._masked_per_sample_mean(
            future_distance_tokens, future_valid_mask
        )
        delta_tokens = F.smooth_l1_loss(pred_delta.float(), target_delta.float(), reduction="none").mean(
            dim=-1
        )
        delta_per_sample, _ = self._masked_per_sample_mean(delta_tokens, future_valid_mask)
        score = (
            self.config.world_model_future_loss_weight * future_per_sample
            + self.config.world_model_delta_loss_weight * delta_per_sample
        )
        return score, future_per_sample, delta_per_sample, sample_valid

    # 逐时间步对齐 VLM 计划与冻结真实动作教师；余弦项约束方向，Smooth L1 同时约束数值尺度，随后在 token 和样本内聚合。
    # Align VLM plans with frozen action-teacher tokens: cosine constrains direction and Smooth L1 constrains value scale, then reduce over tokens and features.
    def _latent_action_alignment_per_sample(
        self,
        predicted_tokens: Tensor,
        teacher_tokens: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Measure how well VLM latent actions match the frozen Stage-1 GT-action encoder.

        Both inputs must contain one token per action from ``t`` through the
        configured future horizon. The teacher input is detached defensively so
        this helper can never update the Stage-1 action encoder.
        """
        if predicted_tokens.shape != teacher_tokens.shape:
            raise ValueError(
                "VLM and teacher latent-action tokens must have identical shapes; "
                f"got {tuple(predicted_tokens.shape)} and {tuple(teacher_tokens.shape)}."
            )
        if predicted_tokens.ndim != 3:
            raise ValueError(
                f"Latent-action tokens must have shape [B,T,D]; got {tuple(predicted_tokens.shape)}."
            )

        teacher_tokens = teacher_tokens.detach()
        predicted_float = predicted_tokens.float()
        teacher_float = teacher_tokens.float()
        cosine_per_sample = (1.0 - F.cosine_similarity(predicted_float, teacher_float, dim=-1)).mean(dim=1)
        huber_per_sample = F.smooth_l1_loss(
            predicted_float,
            teacher_float,
            reduction="none",
        ).mean(dim=(1, 2))
        alignment_per_sample = (
            self.config.wam_alignment_cosine_weight * cosine_per_sample
            + self.config.wam_alignment_huber_weight * huber_per_sample
        )
        return alignment_per_sample, cosine_per_sample, huber_per_sample

    # 先编码视觉语言并取得逐层 KV，再用冻结世界模型预测未来，最后计算动作 flow、规划对齐和未来监督损失。
    # Encode context and retain layer KV, predict the future with frozen dynamics, then combine flow, planning-alignment and future losses.
    # 推荐从这里开始：第二阶段训练总入口。
    # Start reading here: Stage-2 training orchestrates VLM, frozen world model, future conditioning and Expert flow loss.
    def forward_stage2(self, batch: dict[str, Tensor], reduction: str = "mean",
                       optimizer_step: int | None = None) -> tuple[Tensor, dict]:
        """Train pre-expert PI0.5-WAM with flow matching and frozen WAM supervision.

        The WAM branch is:

        ``VLM latent-action queries -> frozen World Model -> future latent loss``.

        The frozen Stage-1 ``world_action_encoder`` supplies a persistent
        GT-action teacher target for the latent-action queries. Neither WAM
        term traverses the Action Expert.
        """
        current_images, future_images, future_valid_mask = self._prepare_world_images(batch)
        if "world_future_valid" in batch:
            future_valid_mask = future_valid_mask & batch["world_future_valid"].to(
                device=future_valid_mask.device, dtype=torch.bool).reshape(-1, 1)
        current_latent, future_latent = self.model.encode_world_image_pairs(current_images, future_images)

        images, img_masks = self._preprocess_images(batch)
        tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        gt_actions = self.prepare_action(batch).to(dtype=torch.float32)

        horizon = self.config.world_model_future_horizons[0]
        state = pad_vector(batch[OBS_STATE], self.config.max_state_dim).to(
            device=current_latent.device, dtype=torch.float32
        )
        action_state = pad_vector(batch[OBS_STATE], self.config.max_action_dim).to(
            device=current_latent.device, dtype=torch.float32
        )
        teacher_actions = gt_actions[:, : horizon + 1].to(
            device=current_latent.device,
            dtype=torch.float32,
        )
        action_is_pad = batch.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=gt_actions.device, dtype=torch.bool)
        prefix_mask = None
        rtc_prefix = None
        # 训练 RTC 先准备连续的已知动作前缀；必须留有至少一个有效自由动作，随后 flow 损失只监督自由后缀。
        # Training RTC first prepares a contiguous committed prefix, retaining at least one valid free action; flow loss later supervises only the free suffix.
        if self.config.rtc_training_max_delay:
            prefix_mask = batch.get("rtc_prefix_mask")
            if prefix_mask is None:
                prefix_mask = sample_training_prefix_mask(gt_actions, self.config.rtc_training_max_delay, action_is_pad)
            else:
                prefix_mask = prefix_mask.to(device=gt_actions.device, dtype=torch.bool)
                if prefix_mask.shape != gt_actions.shape[:2]:
                    raise ValueError('Training RTC mask must match action [B,T]')
                delays = prefix_mask.sum(1)
                expected = torch.arange(gt_actions.shape[1], device=gt_actions.device)[None, :] < delays[:, None]
                if not torch.equal(expected, prefix_mask) or (delays > self.config.rtc_training_max_delay).any():
                    raise ValueError('Training RTC mask must be a contiguous prefix within D')
                if action_is_pad is not None and (prefix_mask & action_is_pad).any():
                    raise ValueError('Training RTC prefix cannot include padding')
                valid_suffix = ~prefix_mask if action_is_pad is None else ~(prefix_mask | action_is_pad)
                if not valid_suffix.any(1).all():
                    raise ValueError('Training RTC requires at least one valid suffix action')
            rtc_prefix = torch.where(prefix_mask[..., None], gt_actions, 0.0).detach()

        # 取得可微 VLM K/V，后面直接复用，不再编码一遍图像语言。
        # Main call chain: obtain differentiable VLM context_key_values, then pass them directly to the Expert.
        # One differentiable VLM prefill produces both the explicit latent
        # action and per-layer original-context KV. The Expert reuses this KV;
        # image/language context is not encoded a second time.
        (
            predicted_action_tokens,
            context_pad_masks,
            context_key_values,
        ) = self.model.encode_wam_prefix_for_training(
            images,
            img_masks,
            tokens,
            masks,
            rtc_prefix=rtc_prefix, rtc_prefix_mask=prefix_mask, action_state=action_state,
        )
        predicted_action_tokens = predicted_action_tokens.to(
            device=current_latent.device,
            dtype=torch.float32,
        )
        physical_plan = None
        # 训练 RTC 先准备连续的已知动作前缀；必须留有至少一个有效自由动作，随后 flow 损失只监督自由后缀。
        # Training RTC first prepares a contiguous committed prefix, retaining at least one valid free action; flow loss later supervises only the free suffix.
        if self.config.rtc_training_max_delay:
            predicted_action_tokens, physical_plan = self.model.constrain_wam_plan(
                predicted_action_tokens, action_state, rtc_prefix, prefix_mask)

        # Training-only teacher and state condition are fully frozen. The GT
        # action never enters the World Model branch itself.
        with torch.no_grad():
            teacher_action_tokens = self.model.world_action_encoder(
                teacher_actions,
                action_state,
            ).detach()
            state_token = self.model.world_state_encoder(state)

        alignment_per_sample, alignment_cosine_per_sample, alignment_huber_per_sample = (
            self._latent_action_alignment_per_sample(
                predicted_action_tokens,
                teacher_action_tokens,
            )
        )

        # 世界模型参数冻结，但保留对输入的梯度以训练 VLM 规划分支。
        # Frozen world-model parameters do not imply no_grad: preserve gradients to latent-action inputs to train the VLM planner.
        # World Model parameters stay frozen, but this call must remain outside
        # no_grad so future/delta losses backpropagate into z_action_vlm.
        # 输入为当前视觉潜变量、状态 token 和 VLM 预测的潜在动作；真实动作只通过教师目标提供监督。
        # Inputs are current visual latents, state tokens and VLM-predicted latent actions; ground-truth actions supervise through teacher targets.
        # 预测的是变化量，因此后续与 current_latent 相加得到未来视觉潜变量，不是直接生成 RGB 图像。
        # The model predicts a delta, added to current_latent to obtain future visual latents rather than RGB images.
        predicted_delta = self.model.world_model(current_latent, state_token, predicted_action_tokens)
        predicted_future = current_latent + predicted_delta
        predicted_score, predicted_future_score, predicted_delta_score, sample_valid = (
            self._world_score_per_sample(
                predicted_delta,
                current_latent,
                future_latent,
                future_valid_mask,
            )
        )

        # Flow supervision sees the predicted future through a dedicated Expert
        # condition block. The default detach keeps Flow gradients out of the
        # frozen WAM/VLM latent-action branch while still training the resampler.
        # 按配置分离预测未来条件梯度；这里没有分离上下文 K/V。
        # By default only the predicted-future conditioning branch is detached here; context KV is not detached. Behavior follows configuration.
        future_condition_source = (
            predicted_future.detach() if self.config.wam_detach_future_condition else predicted_future
        )
        # 重采样器把世界模型的视觉特征映射到 Expert 隐藏宽度和约定 token 数，解决两个模块表示空间不同的问题。
        # The resampler maps world-model visual features to the Expert width and expected token count, bridging their representation spaces.
        # 即使输入未来特征已 detach，重采样器自身的参数仍可由动作 flow loss 更新。
        # Even when future features are detached, the resampler's own parameters can still learn from action flow loss.
        future_condition = self.model.wam_future_condition_resampler(future_condition_source)
        # 将原始上下文 K/V 和预测未来条件交给动作专家。
        # Pass original VLM KV and predicted-future conditions to the Expert, supervised by flow loss.
        flow_losses = self.model.forward_expert_with_cached_context(
            context_pad_masks,
            context_key_values,
            gt_actions,
            future_condition,
            prefix_mask=prefix_mask,
        )
        original_action_dim = self.config.output_features[ACTION].shape[0]
        flow_losses = flow_losses[:, :, :original_action_dim]
        flow_per_sample = masked_suffix_mean(flow_losses, prefix_mask, action_is_pad)
        flow_valid = torch.ones(flow_losses.shape[:2], device=flow_losses.device, dtype=torch.bool)
        if prefix_mask is not None:
            flow_valid &= ~prefix_mask
        if action_is_pad is not None:
            flow_valid &= ~action_is_pad
        flow_per_dim = torch.where(flow_valid[..., None], flow_losses, 0.0).sum((0, 1)) / flow_valid.sum().clamp_min(1)

        teacher_valid = torch.ones_like(sample_valid)
        if action_is_pad is not None:
            teacher_valid = ~action_is_pad[:, :horizon+1].any(1)
        if physical_plan is not None:
            plan_elements = F.smooth_l1_loss(physical_plan[:, :, :original_action_dim],
                                           teacher_actions[:, :, :original_action_dim], reduction='none')
            plan_per_sample = masked_suffix_mean(plan_elements,
                None if prefix_mask is None else prefix_mask[:, :horizon+1],
                None if action_is_pad is None else action_is_pad[:, :horizon+1])
        else:
            plan_per_sample = flow_per_sample * 0.0
        # Match the original Training-RTC mean over all free action elements,
        # rather than giving a 29-step suffix the weight of a 50-step suffix.
        flow_steps = flow_valid.sum(1).float()
        plan_steps = flow_valid[:, :horizon+1].sum(1).float()
        flow_scale = batch.get('flow_loss_scale', flow_steps.numel()*flow_steps/flow_steps.sum().clamp_min(1))
        plan_scale = batch.get('plan_loss_scale', plan_steps.numel()*plan_steps/plan_steps.sum().clamp_min(1))
        flow_for_loss = flow_per_sample * flow_scale
        plan_for_loss = plan_per_sample * plan_scale

        current_step = int(self.model.stage2_step.item()) if optimizer_step is None else int(optimizer_step)
        if current_step < 0:
            raise ValueError('optimizer_step must be nonnegative')
        if optimizer_step is not None:
            self.model.stage2_step.fill_(current_step)
        alignment_weight = self.config.get_wam_alignment_weight(current_step)
        # Synthetic dwell frames supervise the action path and teacher alignment,
        # while only real, continuous visual windows supervise future dynamics.
        valid_count = sample_valid.float().sum().clamp_min(1)
        teacher_count = teacher_valid.float().sum().clamp_min(1)
        # The distributed trainer supplies effective-batch scales computed over
        # all ranks and accumulation micros, matching a true global-batch mean.
        world_scale = batch.get('world_loss_scale', sample_valid.numel()/valid_count)
        teacher_scale = batch.get('teacher_loss_scale', teacher_valid.numel()/teacher_count)
        world_for_loss = predicted_score * sample_valid * world_scale
        alignment_for_loss = alignment_per_sample * teacher_valid * teacher_scale
        wam_per_sample_for_loss = world_for_loss + alignment_weight * alignment_for_loss
        wam_loss = wam_per_sample_for_loss.mean()
        valid_count = sample_valid.float().sum().clamp_min(1)
        # Preserve λ_wam regardless of how many samples near an episode boundary
        # have an invalid future target.
        warmup_steps = self.config.wam_loss_warmup_steps
        if warmup_steps > 0:
            wam_weight_scale = min(1.0, (current_step + 1) / warmup_steps)
        else:
            wam_weight_scale = 1.0
        effective_wam_weight = self.config.wam_loss_weight * wam_weight_scale
        if self.training and optimizer_step is None:
            self.model.stage2_step.add_(1)
        # 总目标把动作生成、世界预测/规划对齐以及可选物理规划监督合并；各支路系数来自配置与 warmup。
        # The total objective combines action generation, world prediction/planning alignment and optional physical-plan supervision, weighted by configuration and warmup.
        # 分布式缩放在前面处理，目的是让不同有效样本数量的批次仍按正确的全局样本或元素均值训练。
        # Earlier distributed scaling accounts for valid-sample counts to preserve the intended global sample or element mean.
        total_per_sample = (flow_for_loss + effective_wam_weight * wam_per_sample_for_loss
                            + self.config.wam_rtc_plan_loss_weight * plan_for_loss)

        flow_loss = flow_for_loss.mean()
        total_loss = total_per_sample.mean()

        def valid_mean(values: Tensor) -> Tensor:
            return (values * sample_valid).sum() / valid_count

        predicted_norm = predicted_action_tokens.float().norm(dim=-1).mean(dim=1)
        teacher_norm = teacher_action_tokens.float().norm(dim=-1).mean(dim=1)
        norm_ratio = predicted_norm / teacher_norm.clamp_min(1e-6)
        latent_cosine_similarity = 1.0 - alignment_cosine_per_sample
        world_loss = valid_mean(predicted_score)
        alignment_loss = self._mean_over_valid_samples(alignment_per_sample, teacher_valid)
        alignment_cosine_loss = self._mean_over_valid_samples(alignment_cosine_per_sample, teacher_valid)
        alignment_huber_loss = self._mean_over_valid_samples(alignment_huber_per_sample, teacher_valid)
        loss_dict = {
            "loss": total_loss.item(),
            "loss/action_flow": flow_loss.item(),
            "loss/rtc_plan": plan_for_loss.mean().item(),
            "metric/rtc_delay_mean": 0.0 if prefix_mask is None else prefix_mask.sum(1).float().mean().item(),
            "metric/wam_teacher_valid_fraction": teacher_valid.float().mean().item(),
            "loss/wam_total": wam_loss.item(),
            "loss/wam_total_weighted": (effective_wam_weight * wam_loss).item(),
            "loss/wam_world": world_loss.item(),
            "loss/wam_future": valid_mean(predicted_future_score).item(),
            "loss/wam_delta": valid_mean(predicted_delta_score).item(),
            "loss/wam_alignment": alignment_loss.item(),
            "loss/wam_alignment_cosine": alignment_cosine_loss.item(),
            "loss/wam_alignment_huber": alignment_huber_loss.item(),
            "loss_per_dim": flow_per_dim.detach().cpu().numpy().tolist(),
            "metric/wam_alignment_weight": alignment_weight,
            "metric/wam_latent_action_cosine": valid_mean(latent_cosine_similarity).item(),
            "metric/wam_latent_action_norm_ratio": valid_mean(norm_ratio).item(),
            "metric/wam_future_condition_norm": future_condition.float().norm(dim=-1).mean().item(),
            "metric/correct_future_distance": valid_mean(predicted_future_score).item(),
            "metric/future_valid_fraction": sample_valid.float().mean().item(),
            "metric/wam_weight_scale": wam_weight_scale,
            "metric/stage2_step": int(self.model.stage2_step.item()),
        }

        # 这条额外世界预测只计算诊断指标，不加入总损失；默认关闭以避免每步增加额外前向开销。
        # This additional world prediction computes diagnostics only, not training loss; it is disabled by default to avoid another forward each step.
        if self.config.wam_compute_shuffled_action_metric:
            with torch.no_grad():
                shuffled_tokens, shuffled_valid = self._make_negative_actions(
                    predicted_action_tokens.detach(),
                    batch.get("world_model_task", batch.get("task")),
                )
                shuffled_valid = shuffled_valid & sample_valid
                shuffled_delta = self.model.world_model(
                    current_latent,
                    state_token,
                    shuffled_tokens,
                )
                shuffled_score, _, _, _ = self._world_score_per_sample(
                    shuffled_delta,
                    current_latent,
                    future_latent,
                    future_valid_mask,
                )
                shuffled_distance = self._mean_over_valid_samples(
                    shuffled_score,
                    shuffled_valid,
                )
                correct_distance = self._mean_over_valid_samples(
                    predicted_score,
                    shuffled_valid,
                )
            loss_dict["metric/wam_shuffled_future_distance"] = shuffled_distance.item()
            loss_dict["metric/wam_action_sensitivity_gap"] = (shuffled_distance - correct_distance).item()
            loss_dict["metric/wam_shuffled_valid_fraction"] = shuffled_valid.float().mean().item()

        if reduction == "none":
            return total_per_sample, loss_dict
        return total_loss, loss_dict


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

"""动作队列、RTC 处理器与策略级动作推理。

Action queues, RTC processor setup and policy-level inference.

原始实现拆分副本，需配合原仓库依赖运行。
Extracted implementation; requires the original repository dependencies.
"""

from ._runtime import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE, RTCProcessor, Tensor, Unpack, deque, torch
from .tensor_utils import ActionSelectKwargs

# 提供环境逐步动作接口和完整动作块接口；世界模型每次生成新块时执行，不在队列取动作时重复执行。
# Expose stepwise and chunk-level action interfaces; run world prediction for each new chunk, not whenever popping a queued action.
class PolicyInferenceMixin:
    # 环境重置时清空动作队列，防止上一回合剩余动作泄漏到新回合。
    # Clear action queues on environment reset so leftover actions from an earlier episode cannot carry over.
    def reset(self):
        """Reset internal state - called when environment resets."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    # RTC 配置即使未开启引导也可用于去噪调试；模型已存在时同步处理器引用。
    # RTC configuration can support denoising diagnostics even without guidance; synchronize the processor reference when the model already exists.
    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Create processor if config provided
        # If RTC is not enabled - we can still track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    # 队列为空才采样一个动作块，只缓存 n_action_steps 步；每次返回 [B,Da]，不支持需要跨块状态的 RTC 引导。
    # Sample only when the queue is empty, cache n_action_steps and return [B,Da] per call; RTC guidance requiring cross-chunk state uses the chunk interface.
    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # Transpose to get shape (n_action_steps, batch_size, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    # 返回 [B,T,Da实际]；世界模型相机按键重新排序，不能依赖策略相机列表中存在相机优先的布局。
    # Return [B,T,actual_Da]; reorder world cameras by key rather than relying on the policy list layout, which puts present cameras first.
    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        if self.config.world_model_stage == "stage1":
            raise RuntimeError(
                "A compact WAVE-0 Stage-1 checkpoint contains no Action Expert and cannot run policy "
                "inference. Load its world modules into the Stage-2 PI0.5 policy first."
            )
        self.eval()

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        if self.config.world_model_stage in {"stage2", "inference"}:
            kwargs["state"] = batch[OBS_STATE]
            # _preprocess_images emits present cameras first, then padded slots.
            # Select by key, not by mask or position, to preserve world-camera order.
            present_keys = [key for key in self.config.image_features if key in batch]
            # 先按存在的相机键建立映射，再按照世界模型训练时的相机顺序取回，避免条件与 camera embedding 错配。
            # Map present cameras by key and retrieve them in world-model training order to avoid mismatching camera embeddings and inputs.
            images_by_key = dict(zip(present_keys, images, strict=False))
            missing_world_keys = [
                key for key in self.config.world_model_image_features if key not in images_by_key
            ]
            if missing_world_keys:
                raise ValueError(f"WAVE-0 inference is missing world-model cameras: {missing_world_keys}")
            kwargs["world_images"] = [images_by_key[key] for key in self.config.world_model_image_features]

        # Sample actions using the model. WAM inference additionally consumes
        # state to predict its future-latent condition once per action chunk.
        actions = self.model.sample_actions(images, img_masks, tokens, masks, **kwargs)

        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions


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

"""图像预处理、动作准备和世界模型监督窗口。

Image preprocessing, action preparation and world-model supervision windows.

原始实现拆分副本，需配合原仓库依赖运行。
Extracted implementation; requires the original repository dependencies.
"""

from ._runtime import ACTION, Tensor, torch, world_model_future_image_key
from .tensor_utils import pad_vector, resize_with_pad_torch

# 把 LeRobot 字典中的多相机观测转为策略与世界模型共享的图像格式，同时保留缺失/未来 padding 的有效位。
# Convert multi-camera LeRobot batches into policy/world-model image inputs while retaining validity for missing cameras and future padding.
class PolicyDataMixin:
    # 单相机输入为 [B,3,H,W] 或 [B,H,W,3]，假定像素已在 [0,1]；转到模型设备并映射到 [-1,1]。
    # Accept one camera as [B,3,H,W] or [B,H,W,3], assuming [0,1] pixels; move to model device and map to [-1,1].
    def _preprocess_image_tensor(self, image: Tensor) -> Tensor:
        """Apply the exact PI0.5 image resize and SigLIP normalization to one camera."""
        if image.ndim != 4:
            raise ValueError(f"Expected one batched image tensor [B,C,H,W] or [B,H,W,C], got {image.shape}.")

        device = next(self.parameters()).device
        image = image.to(device=device, dtype=torch.float32)
        is_channels_first = image.shape[1] == 3
        if is_channels_first:
            image = image.permute(0, 2, 3, 1)
        elif image.shape[-1] != 3:
            raise ValueError(f"Could not identify the RGB channel in image shape {tuple(image.shape)}.")

        if image.shape[1:3] != self.config.image_resolution:
            image = resize_with_pad_torch(image, *self.config.image_resolution)
        image = image * 2.0 - 1.0
        if is_channels_first:
            image = image.permute(0, 3, 1, 2)
        return image

    # 先按配置遍历存在的相机，再添加缺失相机占位；占位像素值为 -1 且 mask=False，因此不能当作真实观测。
    # Emit present configured cameras first, then missing-camera placeholders; placeholders contain -1 pixels and false validity masks.
    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model.

        Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
        PaliGemma expects images in [B, C, H, W] format and normalized to [-1, 1].
        """
        images = []
        img_masks = []

        # Get device from model parameters
        device = next(self.parameters()).device

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = self._preprocess_image_tensor(batch[key])

            images.append(img)
            # Create mask (all ones for real images)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

        # Create image features not present in the batch as fully 0 padded images
        # 占位只保持策略输入槽位数，mask=False 才是缺失语义；不能仅凭像素值判断相机是否有效。
        # Placeholders preserve policy slot count; false masks convey missingness, which cannot be inferred from pixel values alone.
        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(img) * -1  # Padded with -1 for SigLIP
            mask = torch.zeros_like(mask)  # Mask is zero for empty cameras
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    # 只把动作特征维补齐到模型统一宽度，不在这里归一化、裁剪时间长度或处理损失掩码。
    # Pad action features to the model width only; normalization, temporal truncation and loss masking are handled elsewhere.
    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    # 按世界模型相机顺序准备当前/指定 horizon 的未来图像，返回逐相机 [B,C] 有效位以排除序列边界伪目标。
    # Prepare current/future-horizon pairs in world-camera order and return [B,C] validity to exclude padded boundary targets.
    def _prepare_world_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor], Tensor]:
        """Prepare current/future camera pairs and their per-camera validity mask."""
        horizon = self.config.world_model_future_horizons[0]
        current_images: list[Tensor] = []
        future_images: list[Tensor] = []
        future_valid_masks: list[Tensor] = []

        for image_key in self.config.world_model_image_features:
            if image_key not in batch:
                raise ValueError(f"Stage-1 batch is missing current camera {image_key!r}.")
            future_key = world_model_future_image_key(image_key, horizon)
            if future_key not in batch:
                raise ValueError(
                    f"Stage-1 batch is missing {future_key!r}. Ensure WAVE-0 per-camera "
                    "delta timestamps and the future-image processor are enabled."
                )

            current_images.append(self._preprocess_image_tensor(batch[image_key]))
            future_images.append(self._preprocess_image_tensor(batch[future_key]))

            pad_key = f"{image_key}_is_pad"
            if pad_key in batch:
                pad_mask = batch[pad_key].to(device=current_images[-1].device, dtype=torch.bool)
                if pad_mask.ndim != 2 or pad_mask.shape[1] != 2:
                    raise ValueError(f"Expected {pad_key!r} with shape [B,2], got {tuple(pad_mask.shape)}.")
                # 第 0/1 列分别对应当前/未来采样；只用未来有效位控制动态监督，避免末尾重复帧成为真实未来标签。
                # Columns 0/1 refer to current/future samples; future validity controls dynamics supervision so padded terminal frames are not treated as real future labels.
                future_valid_masks.append(~pad_mask[:, 1])
            else:
                future_valid_masks.append(
                    torch.ones(
                        current_images[-1].shape[0],
                        dtype=torch.bool,
                        device=current_images[-1].device,
                    )
                )

        return current_images, future_images, torch.stack(future_valid_masks, dim=1)


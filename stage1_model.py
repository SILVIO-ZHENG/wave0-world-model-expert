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

"""第一阶段世界模型组合与训练模式管理。

Stage-1 world-model composition and training-mode management.

原始实现拆分副本，需配合原仓库依赖运行。
Extracted implementation; requires the original repository dependencies.
"""

from ._runtime import Tensor, WAVE0Config, logging, nn, torch
from .vlm_expert import get_gemma_config
from .world_model import FrozenPaliGemmaVisionBackbone, LatentWorldModel, WorldActionEncoder, WorldStateEncoder, spatial_pool_image_tokens

# 第一阶段只训练动作编码器、状态编码器和世界动力学；无需实例化完整 VLM 或动作专家。
# Stage 1 trains only action/state encoders and latent dynamics, without instantiating the full VLM or Action Expert.
class WAVE0Stage1Model(nn.Module):
    """Minimal Stage-1 model: frozen vision backbone plus trainable world modules."""

    # 根据相机数与池化网格创建动力学网络；视觉特征维度取自基础 PaliGemma 配置。
    # Construct dynamics using the camera count and pooled grid; visual feature width follows the base PaliGemma configuration.
    def __init__(self, config: WAVE0Config):
        super().__init__()
        self.config = config
        paligemma_config = get_gemma_config(config.paligemma_variant)
        num_cameras = len(config.world_model_image_features)
        if num_cameras == 0:
            raise ValueError("WAVE-0 Stage 1 requires at least one visual input feature.")

        self.vision_backbone = FrozenPaliGemmaVisionBackbone(config)
        self.world_action_encoder = WorldActionEncoder(
            max_action_dim=config.max_action_dim,
            width=config.world_model_width,
            depth=config.world_action_encoder_depth,
            num_heads=config.world_model_num_heads,
            max_sequence_length=config.chunk_size,
        )
        self.world_state_encoder = WorldStateEncoder(
            max_state_dim=config.max_state_dim,
            width=config.world_model_width,
        )
        self.world_model = LatentWorldModel(
            image_latent_dim=paligemma_config.width,
            width=config.world_model_width,
            depth=config.world_model_depth,
            num_heads=config.world_model_num_heads,
            mlp_dim=config.world_model_mlp_dim,
            num_cameras=num_cameras,
            spatial_tokens_per_camera=config.world_model_spatial_pool_size**2,
        )

        if config.compile_model:
            # Keep masking, loss assembly, and scalar diagnostics eager. Only
            # compile the fixed-shape neural modules used on every Stage-1 step.
            torch.set_float32_matmul_precision("high")
            self.vision_backbone.forward = torch.compile(
                self.vision_backbone.forward,
                mode=config.compile_mode,
            )
            self.world_action_encoder.forward = torch.compile(
                self.world_action_encoder.forward,
                mode=config.compile_mode,
            )
            self.world_state_encoder.forward = torch.compile(
                self.world_state_encoder.forward,
                mode=config.compile_mode,
            )
            self.world_model.forward = torch.compile(
                self.world_model.forward,
                mode=config.compile_mode,
            )
            logging.info(
                "Enabled module-level torch.compile for WAVE-0 Stage 1: "
                "target vision, action encoder, state encoder, and latent World Model "
                "(mode=%s)",
                config.compile_mode,
            )

    # 对每个相机把当前/未来图像沿批次轴拼接编码，再拆回 [B,C,P,D]；输出明确 detach，目标空间不参与反向更新。
    # Encode current/future images together along the batch axis for each camera, then split into [B,C,P,D]; detach outputs so the target space does not learn.
    @torch.no_grad()
    def encode_world_image_pairs(
        self, current_images: list[Tensor], future_images: list[Tensor]
    ) -> tuple[Tensor, Tensor]:
        if len(current_images) != len(future_images):
            raise ValueError("Current and future image lists must contain the same cameras.")
        self.vision_backbone.eval()
        current_latents = []
        future_latents = []
        for current_image, future_image in zip(current_images, future_images, strict=True):
            batch_size = current_image.shape[0]
            image_tokens = self.vision_backbone(torch.cat([current_image, future_image], dim=0))
            pooled = spatial_pool_image_tokens(
                image_tokens,
                output_size=self.config.world_model_spatial_pool_size,
            ).to(dtype=torch.float32)
            current_latents.append(pooled[:batch_size])
            future_latents.append(pooled[batch_size:])
        return (
            torch.stack(current_latents, dim=1).detach(),
            torch.stack(future_latents, dim=1).detach(),
        )

    # 切换世界模块训练模式时仍强制视觉骨干 eval，避免目标编码受到训练模式影响。
    # Keep the vision backbone in eval even when enabling training for world modules, preserving deterministic targets.
    def train(self, mode: bool = True):
        super().train(mode)
        self.vision_backbone.eval()
        return self


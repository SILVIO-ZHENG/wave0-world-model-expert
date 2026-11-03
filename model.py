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

"""模型组装入口：保留初始化及基础控制；通过 mixin 组合 token、训练和推理方法。

Model assembly: initialization and basic controls, composed with token, training and inference mixins.

原始实现拆分副本，需配合原仓库依赖运行。
Extracted implementation; requires the original repository dependencies.
"""

from ._runtime import ACTION, OPENPI_ATTENTION_MASK_VALUE, RTCProcessor, WAVE0Config, logging, nn, torch
from .model_inference import ModelInferenceMixin
from .model_tokens import ModelTokensMixin
from .model_training import ModelTrainingMixin
from .tensor_utils import sample_beta
from .vlm_expert import PaliGemmaWithExpertModel, get_gemma_config
from .world_model import FrozenPaliGemmaVisionBackbone, LatentWorldModel, SharedFrozenPaliGemmaVisionBackbone, WAMFutureConditionResampler, WorldActionEncoder, WorldStateEncoder

# 组合 token 构造、训练前向和推理 mixin；模块属性仍直接注册在同一 nn.Module 上，保持权重键结构。
# Compose token, training and inference mixins while registering modules on one nn.Module to preserve state-dict naming.
class PI05Pytorch(ModelTokensMixin, ModelTrainingMixin, ModelInferenceMixin, nn.Module):
    """Core PI05 PyTorch model."""

    # 先建 Pi0.5 VLM/Expert，再按阶段接入规划查询、冻结世界模型和未来条件投影；训练冻结边界由策略层设置。
    # Build Pi0.5 VLM/Expert first, then add planner queries, world-model and future-condition modules by stage; the policy applies freeze boundaries.
    def __init__(self, config: WAVE0Config, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor

        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        if config.image_resolution[0] != config.image_resolution[1]:
            raise ValueError(
                f"PaliGemma expects square image resolution, invalid resolution: {config.image_resolution}"
            )

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
            image_size=config.image_resolution[0],
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
        )

        # 动作输入和速度输出之间共享 Expert 隐藏宽度，但两个投影是独立参数，不互为转置。
        # Action input and velocity output share the Expert hidden width, but the two projections have independent parameters.
        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)

        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        # WAVE-0 obtains an explicit latent action from the VLM before calling
        # the World Model or Action Expert. Stage 1 is a compact World Model
        # trainer and deliberately does not instantiate the VLM-side branch.
        # 可学习查询进入 VLM 后读取视觉语言上下文，随后投影成世界模型所需的潜在动作；不是最终机器人的关节动作。
        # Learned queries read VLM context and are projected into world-model latent actions; they are not final robot joint actions.
        self.wam_latent_action_queries = nn.Parameter(
            torch.empty(
                1,
                config.wam_latent_action_num_queries,
                paligemma_config.width,
            )
        )
        nn.init.normal_(self.wam_latent_action_queries, std=0.02)
        self.wam_latent_action_projection = nn.Sequential(
            nn.LayerNorm(paligemma_config.width),
            nn.Linear(paligemma_config.width, config.world_model_width),
        )
        if config.rtc_training_max_delay:
            # A physical-action bridge enforces the committed prefix before the
            # frozen bidirectional action encoder. Never splice teacher latents:
            # those latents may contain information from the unknown GT suffix.
            self.wam_rtc_prefix_projection = nn.Sequential(
                nn.Linear(2 * config.max_action_dim, paligemma_config.width), nn.SiLU(),
                nn.Linear(paligemma_config.width, paligemma_config.width),
            )
            self.wam_rtc_prefix_positions = nn.Parameter(
                torch.randn(1, config.rtc_training_max_delay, paligemma_config.width) * 0.02
            )
            physical_dim = config.output_features[ACTION].shape[0]
            self.wam_rtc_plan_head = nn.Linear(config.world_model_width, physical_dim)

        # World modules supervise Stage-2 training only. ``disabled`` is the
        # deployment/evaluation mode and deliberately constructs none of them.
        # 世界模型分支同时用于第二阶段训练和带世界模型推理；disabled 模式不会构造这些模块。
        # Construct the world branch for both Stage-2 training and world-conditioned inference; disabled mode omits these modules.
        if config.world_model_stage in {"stage2", "inference"}:
            num_cameras = len(config.world_model_image_features)
            if num_cameras == 0:
                raise ValueError("WAVE-0 Stage 2/inference requires at least one visual input feature.")
            pooled_tokens = config.world_model_spatial_pool_size**2
            if config.world_model_stage == "stage2" or config.rtc_training_max_delay:
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
                spatial_tokens_per_camera=pooled_tokens,
            )
            # 共享视觉可节省一份权重，但会固定策略视觉；需要策略视觉全参数微调时应使用独立目标视觉。
            # Sharing vision saves a parameter copy but freezes policy vision; full policy-vision finetuning requires a separate target encoder.
            if config.wam_share_frozen_vision_encoder:
                # The projector is part of PaliGemma's image-feature mapping and
                # must remain fixed together with SigLIP for the Stage-1 latent
                # target space to stay unchanged.
                policy_paligemma = self.paligemma_with_expert.paligemma
                policy_paligemma.vision_tower.requires_grad_(False)
                policy_paligemma.multi_modal_projector.requires_grad_(False)
                policy_paligemma.vision_tower.eval()
                policy_paligemma.multi_modal_projector.eval()
                self.world_vision_backbone = SharedFrozenPaliGemmaVisionBackbone(
                    policy_paligemma.vision_tower,
                    policy_paligemma.multi_modal_projector,
                )
            else:
                # Keep a separate target encoder when policy vision is trainable
                # or intentionally differs from the Stage-1 latent space.
                self.world_vision_backbone = FrozenPaliGemmaVisionBackbone(config)
            self.wam_future_condition_resampler = WAMFutureConditionResampler(
                input_width=paligemma_config.width,
                output_width=action_expert_config.width,
                num_cameras=num_cameras,
                input_spatial_size=config.world_model_spatial_pool_size,
                output_spatial_size=config.wam_future_condition_spatial_pool_size,
            )
            if config.world_model_stage == "stage2":
                self.register_buffer(
                    "stage2_step",
                    torch.zeros((), dtype=torch.long),
                    persistent=True,
                )

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Compile model if requested
        # 编译绑定的 forward 而非替换整个 Module，避免检查点键增加 _orig_mod 前缀；编译不能取代依赖版本检查。
        # Compile bound forwards rather than replacing whole modules to avoid _orig_mod state keys; compilation does not replace dependency-version checks.
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            if config.world_model_stage == "stage2":
                # Stage 2 does not call the generic ``forward`` method. Compile
                # every fixed-shape neural path that is executed during
                # training. Compile bound ``forward`` methods rather than whole
                # nn.Module objects so checkpoint state-dict keys remain
                # unchanged (i.e. no ``_orig_mod.`` prefix).
                self.encode_wam_prefix_for_training = torch.compile(
                    self.encode_wam_prefix_for_training,
                    mode=config.compile_mode,
                )
                self.forward_expert_with_cached_context = torch.compile(
                    self.forward_expert_with_cached_context,
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
                self.world_vision_backbone.forward = torch.compile(
                    self.world_vision_backbone.forward,
                    mode=config.compile_mode,
                )
                self.wam_future_condition_resampler.forward = torch.compile(
                    self.wam_future_condition_resampler.forward,
                    mode=config.compile_mode,
                )
                logging.info(
                    "Enabled torch.compile for all WAVE-0 Stage-2 neural paths: "
                    "VLM prefill, cached-context Expert, target vision, GT-action "
                    "teacher, state encoder, latent World Model, and future resampler"
                )
            else:
                self.forward = torch.compile(self.forward, mode=config.compile_mode)

        msg = """An incorrect transformer version is used, please create an issue on https://github.com/huggingface/lerobot/issues"""

        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    # 同时为语言、视觉和动作专家打开重计算，节省激活显存；不是关闭梯度。
    # Enable activation recomputation for language, vision and Expert paths to save memory without disabling gradients.
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for PI05Pytorch model")

    # 关闭各分支重计算标志，后续前向直接保留正常激活。
    # Disable recomputation flags so subsequent forwards retain ordinary activations.
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for PI05Pytorch model")

    # 判断是否启用了推理时 RTC 引导；它与 rtc_training_max_delay 表示的训练式 RTC 不是同一开关。
    # Check inference-time RTC guidance; this differs from trained RTC selected by rtc_training_max_delay.
    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    # 仅训练且启用检查点时重算函数；推理或关闭该选项时直接调用，避免无意义的重计算。
    # Recompute only during training with checkpointing enabled; otherwise call directly to avoid unnecessary recomputation.
    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    # 将布尔 [B,Q,K] 扩为 [B,1,Q,K] 加性 mask；允许位置为 0，禁止位置为大负值。
    # Convert boolean [B,Q,K] visibility into additive [B,1,Q,K] masks: zero for allowed entries, a large negative value otherwise.
    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    # 创建 float32 标准高斯噪声，形状与填充后的动作块 [B,T,Da] 一致。
    # Create float32 standard Gaussian noise matching the padded action-chunk shape [B,T,Da].
    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    # 按配置的 Beta 分布采样每个样本的去噪时间，再缩放和平移；返回 [B] float32。
    # Sample per-example denoising time from the configured Beta distribution, scale/offset it and return float32 [B].
    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)


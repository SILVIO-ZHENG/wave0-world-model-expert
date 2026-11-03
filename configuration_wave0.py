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

import math
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

DEFAULT_IMAGE_SIZE = 224
WORLD_MODEL_FUTURE_IMAGE_PREFIX = "observation.world_model.future"


# 把真实相机键和离散 horizon 拼成内部未来帧键，使当前观测与监督目标在批次字典中保持不同名称。
# Build an internal future-frame key from the real camera key and discrete horizon so current inputs and supervision remain distinct batch entries.
def world_model_future_image_key(image_key: str, horizon: int) -> str:
    """Return the internal observation key used for a future camera frame."""
    observation_prefix = "observation."
    suffix = image_key.removeprefix(observation_prefix)
    return f"{WORLD_MODEL_FUTURE_IMAGE_PREFIX}.h{horizon}.{suffix}"


# 此配置同时覆盖 Stage 1、Stage 2、世界模型推理和禁用世界模型四种模式；默认值不是特定云端作业的完整配置。
# This configuration covers Stage 1, Stage 2, world-conditioned inference and disabled-world modes; defaults do not encode a complete cloud run.
@PreTrainedConfig.register_subclass("wave0")
@dataclass
class WAVE0Config(PreTrainedConfig):
    # 骨干与精度：VLM 与 Expert 可以使用不同隐藏宽度，但交换 K/V 时要求兼容的层数、注意力头和单头维度。
    # Backbone/precision: VLM and Expert may differ in hidden width, but KV exchange requires compatible layers, attention heads and head dimensions.
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    # 时间布局：chunk_size 是一次生成的动作步数，n_action_steps 是队列实际取用的步数；二者不是图像帧数。
    # Temporal layout: chunk_size is generated action length and n_action_steps is the executed queue length; neither is an image-frame count.
    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 50  # Number of action steps to execute

    # Shorter state and action vectors will be padded to these dimensions
    # 特征宽度：真实机器人的状态/动作可短于这些值，输入补零；计算监督前会裁回真实动作维度。
    # Feature widths: robot state/actions may be narrower and are zero-padded; supervision later slices back to physical action dimensions.
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters: see openpi `PI0Pytorch`
    # 连续时间：t=1 对应噪声，t=0 对应干净动作；采样步数控制 Euler 积分次数，Beta 参数控制训练采样分布。
    # Continuous time: t=1 is noise and t=0 clean action; inference steps set Euler iterations, while Beta parameters control training-time sampling.
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    # 两种 RTC 路径：rtc_config 控制推理引导；rtc_training_max_delay>0 启用训练式已知前缀，当前实现禁止二者同时开启。
    # Two RTC paths: rtc_config controls inference guidance; positive rtc_training_max_delay enables trained known prefixes, and this implementation forbids combining them.
    rtc_config: RTCConfig | None = None
    # Trained RTC is distinct from inference-time gradient guidance.
    rtc_training_max_delay: int = 0
    wam_rtc_plan_loss_weight: float = 1.0
    tokenizer_name: str = 'google/paligemma-3b-pt-224'

    # 图像与语言：相机先按比例缩放补边到这个正方形尺寸；tokenizer_max_length 限制文本上下文，不改变动作块长度。
    # Vision/language: resize and pad cameras to this square resolution; tokenizer_max_length bounds text context, not action length.
    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )  # see openpi `preprocessing_pytorch.py`

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    # 归一化约定：视觉在模型预处理中映射到 [-1,1]；状态和动作采用数据集分位数统计，RTC previous 也必须处在同一归一化空间。
    # Normalization: model preprocessing maps images to [-1,1]; state/actions use dataset quantiles, and RTC previous actions must use the same normalized space.
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for state
            "ACTION": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for action
        }
    )

    # Training settings
    # 内存与编译：检查点通过重算激活节省显存，torch.compile 优化固定形状神经路径；都不决定哪些参数被冻结。
    # Memory/compilation: checkpointing recomputes activations and torch.compile optimizes fixed-shape neural paths; neither determines parameter freezing.
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)

    # Finetuning settings
    # 策略微调边界：freeze_vision_encoder 只固定策略视觉，train_expert_only 固定整个 VLM；世界模型冻结由阶段规则另外控制。
    # Policy finetuning: freeze_vision_encoder freezes policy vision only and train_expert_only freezes the VLM; stage rules separately freeze dynamics.
    freeze_vision_encoder: bool = False  # Freeze only the vision encoder
    train_expert_only: bool = False  # Freeze entire VLM, train only action expert and projections

    # Latent World Model settings. Stage 1 loads only the frozen PI0.5 vision
    # backbone and trains the modules whose names start with ``world_``.
    # 阶段与动态窗口：未来 horizon 是相对于当前帧的离散步偏移；h=31 使用 t 到 t+31 共 32 步动作条件。
    # Stage/dynamics window: the horizon is a discrete offset from the current frame; h=31 uses 32 action conditions from t through t+31.
    world_model_stage: str = "stage1"
    world_model_future_horizons: tuple[int, ...] = (31,)
    # P-large preset aligned with LightWAM. WAVE-0 additionally keeps the full
    # action-token sequence while using its pooled representation for AdaLN-Zero.
    # 世界模型容量与监督：空间池化尺寸 S 对应每相机 S² 个视觉 token；future 与 delta 损失衡量未来状态和变化量，rank 项区分正确与打乱动作。
    # Dynamics capacity/objectives: pool size S yields S² visual tokens per camera; future/delta losses supervise future state and change, while ranking distinguishes correct from shuffled actions.
    world_model_width: int = 1024
    world_model_depth: int = 12
    world_model_num_heads: int = 16
    world_model_mlp_dim: int = 4096
    world_action_encoder_depth: int = 2
    world_model_spatial_pool_size: int = 4
    world_model_future_loss_weight: float = 1.0
    world_model_delta_loss_weight: float = 0.5
    world_model_rank_loss_weight: float = 0.1
    world_model_rank_margin: float = 0.1
    world_model_compute_zero_action_metric: bool = True
    world_model_optimizer_lr: float = 1e-4
    # Stage 2 starts from a complete PI0.5 policy and injects/freeze-loads the
    # three Stage-1 world modules from this compact checkpoint.
    # 两类检查点路径：stage1_checkpoint 是学到的世界模块；base_model 是固定目标视觉空间的来源，不应被恢复训练检查点覆盖。
    # Two checkpoint sources: stage1_checkpoint contains learned world modules; base_model defines fixed target vision and should not be overwritten by a resume checkpoint.
    world_model_stage1_checkpoint_path: str | None = None
    # Permanent reference to the PI0.5 checkpoint that supplies the frozen vision
    # tower. Unlike ``pretrained_path``, this is not replaced by a Stage-1 resume
    # checkpoint path.
    world_model_base_model_path: str | None = None

    # Pre-expert WAM settings. WAVE-0 always follows the
    # VLM -> World Model -> Action Expert path.
    # One query per step covered by the Stage-1 future horizon: h=31 -> 32 tokens.
    # 接入位置：VLM 查询预测潜在动作，再经冻结世界模型得到未来视觉特征；重采样器把它变成 Expert 的未来条件 token。
    # Insertion point: VLM queries predict latent actions, frozen dynamics predicts future visual features, and the resampler produces Expert future tokens.
    wam_latent_action_num_queries: int = 32
    # Resample each camera's 4x4 predicted-future grid to 2x2 before the Expert.
    wam_future_condition_spatial_pool_size: int = 2
    # 默认只阻断 flow loss 经未来条件支路返回规划器，重采样器仍可训练；WAM 监督和原始上下文 KV 路径的梯度不由此开关截断。
    # By default, block flow gradients through predicted-future features into planning while still training the resampler; this does not detach WAM supervision or original-context KV.
    wam_detach_future_condition: bool = True
    # When the policy vision path is frozen, reuse it as the World Model target
    # encoder instead of constructing and checkpointing a duplicate copy.
    wam_share_frozen_vision_encoder: bool = False

    # The frozen Stage-1 action encoder teaches the VLM latent-action branch the
    # input manifold expected by the frozen World Model.
    # 教师对齐：冻结 Stage-1 动作编码器提供目标，余弦项约束方向、Huber 项约束数值；随训练衰减但保留下限。
    # Teacher alignment: the frozen Stage-1 action encoder supplies targets; cosine constrains direction, Huber constrains values, and scheduled weighting retains a floor.
    wam_alignment_cosine_weight: float = 1.0
    wam_alignment_huber_weight: float = 0.5
    wam_alignment_initial_weight: float = 1.0
    wam_alignment_final_weight: float = 0.1
    wam_alignment_decay_start_step: int = 5_000
    wam_alignment_decay_end_step: int = 20_000
    # Weight of the complete pre-expert WAM objective relative to PI0.5 flow
    # matching. This objective contains future/delta prediction and the scheduled
    # latent-action teacher alignment term.
    # 总损失系数：把世界预测与教师对齐合为 WAM 目标，再相对 flow loss 加权；warmup 逐渐提高这条监督的影响。
    # Total-loss weight: combine world prediction and teacher alignment as the WAM objective, weight it relative to flow loss and warm its contribution up gradually.
    wam_loss_weight: float = 0.1
    wam_loss_warmup_steps: int = 1_000
    # Optional diagnostic only. It adds a second frozen World Model pass and is
    # therefore disabled by default for full-scale training.
    wam_compute_shuffled_action_metric: bool = False

    # Optimizer settings: see openpi `AdamW`
    # 优化器预设：只影响外部构造的 AdamW；真正传入优化器的参数已经按 requires_grad 筛选。
    # Optimizer presets configure the external AdamW; parameters passed to it are separately filtered by requires_grad.
    optimizer_lr: float = 2.5e-5  # see openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0
    optimizer_fused: bool = False

    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    # 继承基础设备设置后检查时间范围、模型尺寸、未来查询数量及冻结共享条件；错误组合在创建网络前失败。
    # After base initialization, validate temporal ranges, model sizes, query counts and shared-freezing constraints before network construction.
    def __post_init__(self):
        super().__post_init__()

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )
        if isinstance(self.rtc_training_max_delay, bool) or not isinstance(self.rtc_training_max_delay, int):
            raise ValueError('rtc_training_max_delay must be an integer')
        if not 0 <= self.rtc_training_max_delay < self.chunk_size:
            raise ValueError('rtc_training_max_delay must be smaller than chunk_size')
        if self.rtc_training_max_delay:
            if self.world_model_stage not in {'stage2', 'inference'}:
                raise ValueError('The RTC physical-plan bridge requires Stage 2 or inference')
            if self.rtc_training_max_delay >= self.world_model_future_horizons[0] + 1:
                raise ValueError('RTC prefix must leave a free suffix in the world-model plan')
            if self.rtc_config is not None and self.rtc_config.enabled:
                raise ValueError('Trained RTC cannot be combined with guided RTC')
        if self.wam_rtc_plan_loss_weight < 0:
            raise ValueError('wam_rtc_plan_loss_weight must be nonnegative')

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        if self.world_model_stage not in {"stage1", "stage2", "inference", "disabled"}:
            raise ValueError(
                "world_model_stage must be one of 'stage1', 'stage2', 'inference', "
                "or 'disabled'; "
                f"got {self.world_model_stage!r}."
            )
        if self.world_model_stage == "stage2" and not self.world_model_stage1_checkpoint_path:
            raise ValueError("world_model_stage1_checkpoint_path is required for Stage-2 training.")
        if self.wam_share_frozen_vision_encoder and not self.freeze_vision_encoder:
            raise ValueError("wam_share_frozen_vision_encoder requires freeze_vision_encoder=True.")
        if len(self.world_model_future_horizons) != 1:
            raise ValueError(
                "The first WAVE-0 implementation supports exactly one future horizon; "
                f"got {self.world_model_future_horizons}."
            )
        horizon = self.world_model_future_horizons[0]
        if horizon < 1 or horizon >= self.chunk_size:
            raise ValueError(
                "The future horizon must be in [1, chunk_size - 1]; "
                f"got horizon={horizon}, chunk_size={self.chunk_size}."
            )
        if self.world_model_width % self.world_model_num_heads != 0:
            raise ValueError(
                "world_model_width must be divisible by world_model_num_heads; "
                f"got {self.world_model_width} and {self.world_model_num_heads}."
            )
        if self.world_model_spatial_pool_size <= 0:
            raise ValueError("world_model_spatial_pool_size must be positive.")
        if self.wam_latent_action_num_queries <= 0:
            raise ValueError("wam_latent_action_num_queries must be positive.")
        if self.wam_latent_action_num_queries != horizon + 1:
            raise ValueError(
                "The first pre-expert WAM implementation requires one latent-action query "
                "per action from t through the future horizon; expected "
                f"{horizon + 1}, got {self.wam_latent_action_num_queries}."
            )
        if not (0 < self.wam_future_condition_spatial_pool_size <= self.world_model_spatial_pool_size):
            raise ValueError(
                "wam_future_condition_spatial_pool_size must be in "
                f"[1, {self.world_model_spatial_pool_size}]."
            )
        if self.wam_alignment_cosine_weight < 0:
            raise ValueError("wam_alignment_cosine_weight must be non-negative.")
        if self.wam_alignment_huber_weight < 0:
            raise ValueError("wam_alignment_huber_weight must be non-negative.")
        if self.wam_alignment_initial_weight < self.wam_alignment_final_weight:
            raise ValueError(
                "wam_alignment_initial_weight must be greater than or equal to wam_alignment_final_weight."
            )
        if self.wam_alignment_final_weight < 0:
            raise ValueError("wam_alignment_final_weight must be non-negative.")
        if self.wam_alignment_decay_start_step < 0:
            raise ValueError("wam_alignment_decay_start_step must be non-negative.")
        if self.wam_alignment_decay_end_step <= self.wam_alignment_decay_start_step:
            raise ValueError(
                "wam_alignment_decay_end_step must be greater than wam_alignment_decay_start_step."
            )
        if self.wam_loss_weight < 0:
            raise ValueError("wam_loss_weight must be non-negative.")
        if self.wam_loss_warmup_steps < 0:
            raise ValueError("wam_loss_warmup_steps must be non-negative.")

    # 教师对齐权重先保持初值，在指定步数区间余弦下降，之后保留非零下限；step 应为优化器更新步。
    # Keep the initial teacher-alignment weight, cosine-decay it over the configured update interval, then retain the final floor; step denotes optimizer updates.
    def get_wam_alignment_weight(self, step: int) -> float:
        """Return the persistent latent-action teacher weight at a training step.

        The first implementation keeps weight 1.0 through step 5k, applies a
        smooth cosine decay to 0.1 from 5k to 20k, and then keeps the 0.1 floor
        for the rest of the 100k run.
        """
        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}.")
        if step <= self.wam_alignment_decay_start_step:
            return self.wam_alignment_initial_weight
        if step >= self.wam_alignment_decay_end_step:
            return self.wam_alignment_final_weight

        decay_span = self.wam_alignment_decay_end_step - self.wam_alignment_decay_start_step
        progress = (step - self.wam_alignment_decay_start_step) / decay_span
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (
            self.wam_alignment_final_weight
            + (self.wam_alignment_initial_weight - self.wam_alignment_final_weight) * cosine_factor
        )

    # 补充缺省状态、动作和占位相机的特征声明；实际张量由数据流水线提供，不在此创建训练样本。
    # Supply default state/action and placeholder-camera feature declarations; the data pipeline still provides actual tensors.
    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = OBS_IMAGES + f".empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features[ACTION] = action_feature

    # 过滤 VLM 预训练布局所需的空相机；世界模型只预测真实相机的未来，因为占位相机没有物理未来标签。
    # Filter empty cameras needed for the VLM layout; dynamics predicts real-camera futures because placeholders have no physical future targets.
    @property
    def world_model_image_features(self) -> dict[str, PolicyFeature]:
        """Return only real cameras used as World Model inputs and targets.

        ``empty_cameras`` adds masked visual placeholders so the PI0.5 VLM can
        preserve the camera-slot layout used during pretraining. Those placeholders
        contain no observations or future frames and therefore must not be treated
        as physical cameras by the latent World Model.
        """
        empty_camera_prefix = f"{OBS_IMAGES}.empty_camera_"
        return {
            key: feature
            for key, feature in self.image_features.items()
            if not key.startswith(empty_camera_prefix)
        }

    # 第一阶段使用世界模型学习率，其余阶段使用策略学习率；参数组仍由策略中 requires_grad 的筛选决定。
    # Use the world-model learning rate in Stage 1 and policy learning rate otherwise; policy requires_grad filtering still selects optimized parameters.
    def get_optimizer_preset(self) -> AdamWConfig:
        lr = self.world_model_optimizer_lr if self.world_model_stage == "stage1" else self.optimizer_lr
        return AdamWConfig(
            lr=lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
            fused=self.optimizer_fused,
        )

    # 学习率先 warmup 后余弦衰减，最低学习率不高于本阶段峰值；此处返回设置而非实际执行调度。
    # Warm up then cosine-decay learning rate, keeping the floor no higher than this stage peak; return scheduler settings rather than advancing it.
    def get_scheduler_preset(self):
        peak_lr = self.world_model_optimizer_lr if self.world_model_stage == "stage1" else self.optimizer_lr
        decay_lr = min(self.scheduler_decay_lr, peak_lr)
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=peak_lr,
            decay_lr=decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    # 不对所有观测统一做时间堆叠，避免状态也被加载成当前/未来两帧；相机专用规则由下方属性提供。
    # Do not time-stack every observation, which would also duplicate state; camera-specific sampling is configured below.
    @property
    def observation_delta_indices(self) -> None:
        return None

    # 只在训练阶段为图像请求当前帧和未来 horizon；机器人状态仍只用当前时刻，推理不索取真实未来帧。
    # Only training requests current/future camera frames; robot state remains current-only and inference never requests ground-truth future frames.
    @property
    def observation_delta_indices_by_key(self) -> dict[str, list[int]]:
        """Load current and future frames for cameras without time-stacking robot state.

        The wildcard is resolved by ``datasets.factory.resolve_delta_timestamps``.
        It deliberately applies only to ``observation.images.*`` features.
        """
        if self.world_model_stage not in {"stage1", "stage2"}:
            return {}
        return {
            f"{OBS_IMAGES}.*": [0, *self.world_model_future_horizons],
        }

    # 动作块覆盖离散步 0 到 chunk_size-1；世界模型只使用其中 horizon+1 步，二者长度并非必须相等。
    # The action chunk spans steps 0 through chunk_size-1; dynamics uses only horizon+1 of these steps, so the two lengths need not match.
    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    # 当前监督式训练目标不请求时间展开的奖励序列。
    # The supervised objectives do not request a time-expanded reward sequence.
    @property
    def reward_delta_indices(self) -> None:
        return None

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

"""策略组装入口：保留初始化、训练模式及调度；组合权重、数据、损失和推理方法。

Policy assembly: initialization, mode management and dispatch; compose checkpoint, data, loss and inference methods.

原始实现拆分副本，需配合原仓库依赖运行。
Extracted implementation; requires the original repository dependencies.
"""

from ._runtime import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, Path, PreTrainedPolicy, Tensor, WAVE0Config, logging
from .model import PI05Pytorch
from .policy_checkpoint import PolicyCheckpointMixin
from .policy_data import PolicyDataMixin
from .policy_inference import PolicyInferenceMixin
from .policy_training import PolicyTrainingMixin
from .stage1_model import WAVE0Stage1Model

# 面向 LeRobot 的统一入口：选择训练阶段、控制冻结边界、装载模型并提供损失及动作接口。
# Unified LeRobot-facing entry: choose a stage, enforce freezing, construct models and expose loss/action interfaces.
class WAVE0Policy(PolicyCheckpointMixin, PolicyDataMixin, PolicyTrainingMixin, PolicyInferenceMixin, PreTrainedPolicy):
    """WAVE-0: a PI0.5 policy conditioned on predicted future visual latents."""

    config_class = WAVE0Config

    name = "wave0"

    # 验证输入输出特征后选择紧凑 Stage-1 或完整策略；先设置可训练参数，再供外部构建优化器。
    # Validate features and choose compact Stage 1 or the full policy; configure trainable parameters before external optimizer construction.
    def __init__(
        self,
        config: WAVE0Config,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Initialize the core PI05 model
        self.init_rtc_processor()
        if config.world_model_stage == "stage1":
            self.model = WAVE0Stage1Model(config)
        else:
            self.model = PI05Pytorch(config, rtc_processor=self.rtc_processor)

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing and config.world_model_stage != "stage1":
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)
        self._configure_trainability()

        self.reset()

    # Stage 1 只训三个世界模块；Stage 2 冻结世界模块参数但允许对其输入求导；inference 冻结全部参数。
    # Stage 1 trains three world modules; Stage 2 freezes their parameters while retaining input derivatives; inference freezes all parameters.
    def _configure_trainability(self) -> None:
        """Apply the stage-specific freeze boundary before optimizer construction."""
        if self.config.world_model_stage == "stage1":
            self.model.requires_grad_(False)
            self.model.world_action_encoder.requires_grad_(True)
            self.model.world_state_encoder.requires_grad_(True)
            self.model.world_model.requires_grad_(True)
        elif self.config.world_model_stage == "stage2":
            # The World Model is a differentiable supervisor: parameters stay
            # frozen, while gradients with respect to its action input are kept.
            self.model.world_action_encoder.requires_grad_(False)
            self.model.world_state_encoder.requires_grad_(False)
            # 冻结的是参数更新，不是整个计算图；第二阶段世界损失仍能通过动力学对 VLM 的潜在动作求导。
            # Freeze parameter updates, not the entire computation graph; Stage-2 world losses can still differentiate through dynamics into VLM latent actions.
            self.model.world_model.requires_grad_(False)
            self.model.world_vision_backbone.requires_grad_(False)
        elif self.config.world_model_stage == "inference":
            self.model.requires_grad_(False)

    # 父类会递归切换所有子模块，因此随后把冻结的教师、世界视觉和动力学重新设为 eval。
    # The parent recursively switches submodules, so frozen teacher/world-vision/dynamics modules are then restored to eval.
    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.world_model_stage == "stage1":
            # Frozen PI0.5 is a deterministic target encoder during Stage 1.
            self.model.vision_backbone.eval()
            self.model.world_action_encoder.train(mode)
            self.model.world_state_encoder.train(mode)
            self.model.world_model.train(mode)
        elif self.config.world_model_stage == "stage2":
            self.model.world_action_encoder.eval()
            self.model.world_state_encoder.eval()
            self.model.world_model.eval()
            self.model.world_vision_backbone.eval()
        elif self.config.world_model_stage == "inference":
            self.model.world_state_encoder.eval()
            self.model.world_model.eval()
            self.model.world_vision_backbone.eval()
        return self

    # Stage 1 只保存三个已训练世界模块，并记录恢复视觉空间所需的基础模型路径；其他阶段采用常规策略保存。
    # Save only the three trained world modules in Stage 1, recording the base model needed to restore vision; other stages use regular policy saving.
    def _save_pretrained(self, save_directory: Path) -> None:
        """Save Stage 1 as a compact world-only checkpoint; never duplicate frozen vision."""
        if self.config.world_model_stage != "stage1":
            super()._save_pretrained(save_directory)
            if self.config.wam_share_frozen_vision_encoder:
                from safetensors import safe_open

                model_file = save_directory / "model.safetensors"
                with safe_open(model_file, framework="pt", device="cpu") as checkpoint:
                    duplicated = [key for key in checkpoint.keys() if "world_vision_backbone" in key]
                if duplicated:
                    raise RuntimeError(
                        "Shared vision checkpoint unexpectedly contains duplicate World Model "
                        f"vision tensors: {duplicated[:5]}"
                    )
                logging.info(
                    "Saved shared-vision WAVE-0 checkpoint with one policy-owned vision copy to %s",
                    save_directory,
                )
            return
        if not self.config.world_model_base_model_path:
            raise RuntimeError(
                "world_model_base_model_path is missing; cannot save a restorable Stage-1 checkpoint."
            )

        from safetensors.torch import save_file

        self.config._save_pretrained(save_directory)
        world_state: dict[str, Tensor] = {}
        # 紧凑检查点不重复存储固定视觉；恢复时必须提供与训练目标空间一致的基础视觉权重。
        # The compact checkpoint excludes fixed vision; restoring it requires base visual weights matching the training target space.
        for module_name in ("world_action_encoder", "world_state_encoder", "world_model"):
            module = getattr(self.model, module_name)
            for key, value in module.state_dict().items():
                world_state[f"model.{module_name}.{key}"] = value.detach().cpu().contiguous()
        save_file(
            world_state,
            str(save_directory / "model.safetensors"),
            metadata={
                "format": "pt",
                "wave0_checkpoint_type": "stage1_world_only",
                "base_model_path": str(self.config.world_model_base_model_path),
            },
        )
        logging.info(
            "Saved %d Stage-1 world tensors without the frozen vision backbone to %s",
            len(world_state),
            save_directory,
        )

    # 只返回 requires_grad=True 的参数，避免把冻结教师与世界模型送入优化器；没有可训练参数时明确报错。
    # Return only requires_grad parameters so frozen teachers/dynamics are excluded from optimization; fail clearly when none remain trainable.
    def get_optim_params(self) -> dict:
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError("WAVE-0 has no trainable parameters for the selected stage.")
        return parameters

    # 查询推理式 RTC 引导开关；训练式 RTC 的前缀逻辑由独立的延迟配置驱动。
    # Query inference-time RTC guidance; trained-prefix RTC is driven by separate delay configuration.
    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    # 按阶段分发到世界模型训练或 Stage-2 联合目标；disabled 才走基础 Pi0.5 flow 损失，inference 阶段不允许训练。
    # Dispatch to world-model or Stage-2 objectives; disabled mode uses base Pi0.5 flow loss, while inference mode rejects training.
    def forward(
        self,
        batch: dict[str, Tensor],
        reduction: str = "mean",
        compute_diagnostics: bool = True,
        optimizer_step: int | None = None,
    ) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training.

        Args:
            batch: Training batch containing observations and actions.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        if self.config.world_model_stage == "stage1":
            return self.forward_world_model(
                batch,
                reduction=reduction,
                compute_diagnostics=compute_diagnostics,
            )
        if self.config.world_model_stage == "stage2":
            return self.forward_stage2(batch, reduction=reduction, optimizer_step=optimizer_step)
        if self.config.world_model_stage == "inference":
            raise RuntimeError(
                "world_model_stage='inference' is deployment-only and cannot be trained. "
                "Use world_model_stage='stage2' for WAM training."
            )

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.prepare_action(batch)

        # Compute loss (no separate state needed for PI05)
        losses = self.model.forward(images, img_masks, tokens, masks, actions)

        # Truncate losses to actual action dimensions
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]

        loss_dict = {
            "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
        }

        if reduction == "none":
            # Return per-sample losses (B,) by averaging over time and action dims
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss
            loss = losses.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    # 提供可选 PEFT 的默认正则匹配目标，不会在这里自动启用 LoRA；实际微调策略由外部配置决定。
    # Provide optional PEFT target patterns without enabling LoRA here; external configuration determines the finetuning strategy.
    def _get_default_peft_targets(self) -> dict[str, any]:
        """Return default PEFT target modules for PI0.5 fine-tuning."""
        common_projections = (
            "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        )
        target_modules = rf"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.({common_projections}))"
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }


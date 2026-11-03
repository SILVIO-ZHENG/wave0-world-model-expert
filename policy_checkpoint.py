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

"""检查点解析、权重注入和键名兼容。保存方法因使用 super 保留在策略入口。

Checkpoint resolution, weight injection and key compatibility. Saving stays in the policy class because it uses super.

原始实现拆分副本，需配合原仓库依赖运行。
Extracted implementation; requires the original repository dependencies.
"""

from ._runtime import Path, PreTrainedConfig, T, WAVE0Config, builtins, dataclasses, logging, nn
from .tensor_utils import _is_wam_teacher_state_key, _is_world_model_state_key

# 集中处理模型来源解析、第一阶段注入和历史键名适配，避免训练前向耦合到检查点文件格式。
# Centralize source resolution, Stage-1 injection and legacy-key adaptation so training forwards do not depend on checkpoint layouts.
class PolicyCheckpointMixin:
    # 先接受本地文件或含 model.safetensors 的目录；其他名称交由 Transformers 缓存解析，并遵守 local_files_only。
    # Accept local files or directories containing model.safetensors first; otherwise use Transformers resolution, honoring local_files_only.
    @staticmethod
    def _resolve_safetensors_file(
        pretrained_name_or_path: str | Path,
        *,
        cache_dir: str | Path | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> str:
        model_path = Path(pretrained_name_or_path)
        if model_path.is_file():
            return str(model_path)
        if model_path.is_dir():
            model_file = model_path / "model.safetensors"
            if not model_file.is_file():
                raise FileNotFoundError(f"model.safetensors not found in {model_path}.")
            return str(model_file)

        from transformers.utils import cached_file

        resolved = cached_file(
            str(pretrained_name_or_path),
            "model.safetensors",
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            revision=revision,
            local_files_only=local_files_only,
        )
        if resolved is None:
            raise FileNotFoundError(f"Could not resolve model.safetensors from {pretrained_name_or_path}.")
        return resolved

    # 仅检查 safetensors 键名，不提前加载大张量；用动作教师键识别包含世界模块的检查点。
    # Inspect safetensors keys without loading large tensors; identify world-containing checkpoints through action-encoder keys.
    @staticmethod
    def _checkpoint_contains_world_modules(model_file: str) -> bool:
        from safetensors import safe_open

        with safe_open(model_file, framework="pt", device="cpu") as checkpoint:
            return any(
                key.startswith(("model.world_action_encoder.", "world_action_encoder."))
                for key in checkpoint.keys()
            )

    # 通过视觉塔参数前缀判断是否可从同一检查点恢复固定视觉目标编码器。
    # Detect vision-tower parameter prefixes to determine whether the fixed target encoder can be restored from this checkpoint.
    @staticmethod
    def _checkpoint_contains_vision_backbone(model_file: str) -> bool:
        from safetensors import safe_open

        vision_markers = (
            "paligemma_with_expert.paligemma.model.vision_tower.",
            "model.paligemma_with_expert.paligemma.model.vision_tower.",
        )
        with safe_open(model_file, framework="pt", device="cpu") as checkpoint:
            return any(key.startswith(vision_markers) for key in checkpoint.keys())

    # 只选择视觉塔和多模态投影权重并严格匹配目标键，避免把语言模型权重无意加载进目标编码器。
    # Select only vision-tower/projector tensors with strict key matching, preventing accidental language-weight loading into the target encoder.
    @staticmethod
    def _load_frozen_vision_backbone(backbone: nn.Module, model_file: str) -> None:
        """Selectively read vision tower/projector tensors into a frozen target encoder."""
        from safetensors import safe_open

        source_prefixes = (
            "paligemma_with_expert.paligemma.model.",
            "model.paligemma_with_expert.paligemma.model.",
        )
        expected_keys = set(backbone.state_dict())
        vision_state = {}
        with safe_open(model_file, framework="pt", device="cpu") as checkpoint:
            for source_key in checkpoint.keys():
                target_key = None
                for prefix in source_prefixes:
                    if source_key.startswith(prefix):
                        candidate = source_key.removeprefix(prefix)
                        if candidate.startswith(("vision_tower.", "multi_modal_projector.")):
                            target_key = candidate
                        break
                if target_key in expected_keys:
                    vision_state[target_key] = checkpoint.get_tensor(source_key)

        missing = sorted(expected_keys - set(vision_state))
        unexpected = sorted(set(vision_state) - expected_keys)
        if missing or unexpected:
            raise RuntimeError(
                "Selective PI0.5 vision loading failed. "
                f"Missing keys: {missing[:10]}; unexpected keys: {unexpected[:10]}."
            )
        backbone.load_state_dict(vision_state, strict=True)
        backbone.requires_grad_(False)
        backbone.eval()
        logging.info("Loaded %d frozen PI0.5 vision tensors from %s", len(vision_state), model_file)

    # 第一阶段目标视觉位于 model.vision_backbone；实际权重筛选由公共加载器完成。
    # Stage-1 target vision lives at model.vision_backbone; the shared loader performs tensor selection.
    @classmethod
    def _load_stage1_vision_backbone(cls, model: "WAVE0Policy", model_file: str) -> None:
        """Load the minimal Stage-1 model's frozen target encoder."""
        cls._load_frozen_vision_backbone(model.model.vision_backbone, model_file)

    # 第二阶段独立目标视觉位于 world_vision_backbone，与可训练策略视觉保持分离。
    # Stage-2 target vision lives at world_vision_backbone, separate from trainable policy vision.
    @classmethod
    def _load_stage2_vision_backbone(cls, model: "WAVE0Policy", model_file: str) -> None:
        """Load Stage 2's vision target encoder, separate from trainable policy vision."""
        cls._load_frozen_vision_backbone(model.model.world_vision_backbone, model_file)

    # 分别恢复动作编码器、状态编码器和动力学，逐模块 strict=True；缺少任一模块即停止加载。
    # Restore action encoder, state encoder and dynamics separately with strict=True, failing if any module is missing.
    @staticmethod
    def _load_stage1_world_modules(model: "WAVE0Policy", model_file: str) -> None:
        """Load the three trainable Stage-1 modules from a world-only checkpoint."""
        from safetensors import safe_open

        module_names = ("world_action_encoder", "world_state_encoder", "world_model")
        states: dict[str, dict[str, Tensor]] = {name: {} for name in module_names}
        with safe_open(model_file, framework="pt", device="cpu") as checkpoint:
            for source_key in checkpoint.keys():
                normalized_key = source_key.removeprefix("model.")
                for module_name in module_names:
                    prefix = f"{module_name}."
                    if normalized_key.startswith(prefix):
                        states[module_name][normalized_key.removeprefix(prefix)] = checkpoint.get_tensor(
                            source_key
                        )
                        break

        for module_name in module_names:
            module = getattr(model.model, module_name)
            if not states[module_name]:
                raise RuntimeError(f"Stage-1 checkpoint has no tensors for {module_name}.")
            module.load_state_dict(states[module_name], strict=True)
        logging.info("Loaded Stage-1 world modules from %s", model_file)

    # 按配置加载基础策略和第一阶段世界模块，处理检查点差异及冻结设置；外部模型路径仍由调用者配置。
    # Load base-policy and Stage-1 world modules, reconcile checkpoint differences and configure freezing; callers still supply model paths.
    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Override the from_pretrained method to handle key remapping and display important disclaimer."""
        print(
            "WAVE-0 extends the LeRobot PI0.5 / OpenPI implementation. \n"
            "The PI0.5 backbone retains the original OpenPI structure for compatibility. \n"
            "Original implementation: https://github.com/Physical-Intelligence/openpi"
        )
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        # Use provided config if available, otherwise create default config
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        # Calling WAVE0Policy directly on an official PI0.5 folder yields a
        # PI05Config. Convert the shared fields so Stage 1 can still be created.
        if not isinstance(config, WAVE0Config):
            # v0.4.4 PreTrainedConfig has no to_dict(). Keep nested PolicyFeature
            # objects intact instead of flattening them with dataclasses.asdict.
            source_config = {
                field.name: getattr(config, field.name) for field in dataclasses.fields(config) if field.init
            }
            init_fields = {field.name for field in dataclasses.fields(WAVE0Config) if field.init}
            config = WAVE0Config(**{key: value for key, value in source_config.items() if key in init_fields})

        # 先确认检查点是否包含视觉：world-only 检查点必须另行解析固定视觉来源，不能用新随机视觉代替。
        # Determine whether vision is embedded first: a world-only checkpoint requires a separate fixed-vision source, not newly randomized vision.
        if config.world_model_stage == "stage1":
            checkpoint_file = cls._resolve_safetensors_file(
                pretrained_name_or_path,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                revision=revision,
                local_files_only=local_files_only,
            )
            contains_world = cls._checkpoint_contains_world_modules(checkpoint_file)
            contains_vision = cls._checkpoint_contains_vision_backbone(checkpoint_file)

            if contains_vision:
                vision_checkpoint_file = checkpoint_file
                if not config.world_model_base_model_path:
                    config.world_model_base_model_path = str(pretrained_name_or_path)
            else:
                if not config.world_model_base_model_path:
                    raise RuntimeError(
                        "This is a world-only Stage-1 checkpoint, but its config does not contain "
                        "world_model_base_model_path."
                    )
                vision_checkpoint_file = cls._resolve_safetensors_file(
                    config.world_model_base_model_path,
                    cache_dir=cache_dir,
                    force_download=False,
                    resume_download=resume_download,
                    proxies=proxies,
                    token=token,
                    revision=revision,
                    local_files_only=local_files_only,
                )

            model = cls(config, **kwargs)
            cls._load_stage1_vision_backbone(model, vision_checkpoint_file)
            if contains_world:
                cls._load_stage1_world_modules(model, checkpoint_file)
            model.eval()
            return model

        # 从第一阶段配置恢复视觉基准路径，保证未来监督使用的潜在空间与世界模型训练时一致。
        # Recover the vision-base source from Stage-1 configuration so future supervision uses the latent space learned by the world model.
        if config.world_model_stage == "stage2" and not config.world_model_base_model_path:
            stage1_config = PreTrainedConfig.from_pretrained(
                config.world_model_stage1_checkpoint_path,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            config.world_model_base_model_path = getattr(stage1_config, "world_model_base_model_path", None)
            if not config.world_model_base_model_path:
                raise RuntimeError(
                    "Could not recover world_model_base_model_path from the Stage-1 checkpoint. "
                    "Set it explicitly to the PI0.5 checkpoint used by Stage 1."
                )

        # Initialize model without loading weights
        # Check if dataset_stats were provided in kwargs
        model = cls(config, **kwargs)

        # Now manually load and remap the state dict
        try:
            # Try to load the pytorch_model.bin or model.safetensors file
            print(f"Loading model from: {pretrained_name_or_path}")
            try:
                from transformers.utils import cached_file

                # Try safetensors first
                resolved_file = cached_file(
                    pretrained_name_or_path,
                    "model.safetensors",
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    token=token,
                    revision=revision,
                    local_files_only=local_files_only,
                )
                # 部署按键选择性读取，减少无用教师权重的内存占用；inference 仍保留世界模型及未来条件路径。
                # Deployment reads tensors selectively to avoid unused teacher memory; inference still retains dynamics and future conditioning.
                if config.world_model_stage in {"disabled", "inference"}:
                    # Avoid allocating or even reading training-only World Model
                    # tensors during policy deployment. WAM inference still
                    # keeps the World Model, state/current-image encoders, and
                    # future-condition modules, but skips the GT-action teacher.
                    from safetensors import safe_open

                    with safe_open(resolved_file, framework="pt", device="cpu") as checkpoint:
                        all_keys = list(checkpoint.keys())
                        if config.world_model_stage == "disabled":
                            policy_keys = [key for key in all_keys if not _is_world_model_state_key(key)]
                        else:
                            policy_keys = [key for key in all_keys if not _is_wam_teacher_state_key(
                                key, keep_action_encoder=bool(config.rtc_training_max_delay))]
                        original_state_dict = {key: checkpoint.get_tensor(key) for key in policy_keys}
                    print(
                        "✓ Selectively loaded state dict from model.safetensors "
                        f"(skipped {len(all_keys) - len(policy_keys)} deployment-unused tensors)"
                    )
                else:
                    from safetensors.torch import load_file

                    original_state_dict = load_file(resolved_file)
                    print("✓ Loaded state dict from model.safetensors")
            except Exception as e:
                raise RuntimeError(
                    f"Could not resolve model.safetensors from {pretrained_name_or_path}: {e}"
                ) from e

            # First, fix any key differences # see openpi `model.py, _fix_pytorch_state_dict_keys`
            fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)

            # Then add "model." prefix for all keys that don't already have it
            remapped_state_dict = {}
            remap_count = 0

            for key, value in fixed_state_dict.items():
                if not key.startswith("model."):
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                    remap_count += 1
                    if remap_count <= 10:  # Only print first 10 to avoid spam
                        print(f"Remapped: {key} -> {new_key}")
                else:
                    remapped_state_dict[key] = value

            if remap_count > 0:
                print(f"Remapped {remap_count} state dict keys")

            if config.world_model_stage == "disabled":
                # Defensive filtering for any key spelling introduced by the
                # compatibility remapper.
                remapped_state_dict = {
                    key: value
                    for key, value in remapped_state_dict.items()
                    if not _is_world_model_state_key(key)
                }
            elif config.world_model_stage == "inference":
                remapped_state_dict = {
                    key: value
                    for key, value in remapped_state_dict.items()
                    if not _is_wam_teacher_state_key(key, keep_action_encoder=bool(config.rtc_training_max_delay))
                }
            if config.wam_share_frozen_vision_encoder:
                # Older checkpoints may contain a separate frozen target copy.
                # The shared layout always restores vision from the policy-owned
                # keys, so discard the redundant legacy keys before loading.
                remapped_state_dict = {
                    key: value
                    for key, value in remapped_state_dict.items()
                    if not key.startswith("model.world_vision_backbone.")
                }

            # Load with ``strict=False`` first because an official PI0.5 checkpoint
            # intentionally has no Stage-1 world-model parameters. Strictness is
            # applied below after whitelisting only these new module prefixes.
            missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)

            allowed_missing_prefixes = (
                "model.world_action_encoder.",
                "model.world_state_encoder.",
                "model.world_model.",
                "model.world_vision_backbone.",
                "model.stage2_step",
                "model.wam_latent_action_queries",
                "model.wam_latent_action_projection.",
                "model.wam_future_condition_resampler.",
                "model.wam_rtc_prefix_projection.",
                "model.wam_rtc_prefix_positions",
                "model.wam_rtc_plan_head.",
            )
            invalid_missing_keys = [
                key for key in missing_keys if not key.startswith(allowed_missing_prefixes)
            ]
            if config.world_model_stage == 'inference':
                # Deployment may never initialize a missing world/bridge tensor
                # randomly. In RTC mode the action encoder is a live component.
                invalid_missing_keys = list(missing_keys)
            if strict and (invalid_missing_keys or unexpected_keys):
                raise RuntimeError(
                    "WAVE-0 checkpoint loading found incompatible keys. "
                    f"Missing non-world keys: {invalid_missing_keys[:10]}; "
                    f"unexpected keys: {unexpected_keys[:10]}."
                )

            if missing_keys:
                print(f"Missing keys when loading state dict: {len(missing_keys)} keys")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"  - {key}")
                else:
                    for key in missing_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(missing_keys) - 5} more")

            if unexpected_keys:
                print(f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"  - {key}")
                else:
                    for key in unexpected_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(unexpected_keys) - 5} more")

            if not missing_keys and not unexpected_keys:
                print("All keys loaded successfully!")

        except Exception as e:
            raise RuntimeError(f"Could not load WAVE-0 pretrained weights: {e}") from e

        if config.world_model_stage == "stage2":
            stage1_model_file = cls._resolve_safetensors_file(
                config.world_model_stage1_checkpoint_path,
                cache_dir=cache_dir,
                force_download=False,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                revision=revision,
                local_files_only=local_files_only,
            )
            cls._load_stage1_world_modules(model, stage1_model_file)
            if config.wam_share_frozen_vision_encoder:
                logging.info(
                    "World Model target vision shares the frozen policy vision tower/projector; "
                    "no duplicate target vision weights were loaded."
                )
            else:
                target_vision_file = cls._resolve_safetensors_file(
                    config.world_model_base_model_path,
                    cache_dir=cache_dir,
                    force_download=False,
                    resume_download=resume_download,
                    proxies=proxies,
                    token=token,
                    revision=revision,
                    local_files_only=local_files_only,
                )
                cls._load_stage2_vision_backbone(model, target_vision_file)
            model._configure_trainability()

        return model

    # 处理 AdaRMS、时间 MLP、旧状态投影及共享词嵌入的键名差异；返回新字典，不原地改输入。
    # Adapt AdaRMS, time-MLP, legacy state-projection and tied-embedding keys; return a new dictionary without mutating the input mapping.
    def _fix_pytorch_state_dict_keys(
        self, state_dict, model_config
    ):  # see openpi `BaseModelConfig, _fix_pytorch_state_dict_keys`
        """Fix state dict keys to match current model architecture."""
        import re

        fixed_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Handle layer norm structure changes: .weight -> .dense.weight + .dense.bias
            # For gemma expert layers
            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                key,
            ):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping layer norm key (adaRMS mismatch): {key}")
                    continue

            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping norm key (adaRMS mismatch): {key}")
                    continue

            # Handle MLP naming changes for pi05
            # pi05 model expects time_mlp_*, but checkpoint might have action_time_mlp_*
            if key.startswith("action_time_mlp_in."):
                new_key = key.replace("action_time_mlp_in.", "time_mlp_in.")
            elif key.startswith("action_time_mlp_out."):
                new_key = key.replace("action_time_mlp_out.", "time_mlp_out.")
            # Also handle state_proj which shouldn't exist in pi05
            if key.startswith("state_proj."):
                logging.warning(f"Skipping state_proj key in pi05 mode: {key}")
                continue

            # Handle vision tower embedding layer potential differences
            if "patch_embedding" in key:
                # Some checkpoints might have this, but current model expects different structure
                logging.warning(f"Vision embedding key might need handling: {key}")

            # Safetensors stores tied PaliGemma lm_head/embed_tokens weights only once.
            # Restore the alias expected by strict state-dict loading.
            if key in {
                "model.paligemma_with_expert.paligemma.lm_head.weight",
                "paligemma_with_expert.paligemma.lm_head.weight",
            }:
                fixed_state_dict[
                    "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
                ] = value.clone()

            fixed_state_dict[new_key] = value

        return fixed_state_dict


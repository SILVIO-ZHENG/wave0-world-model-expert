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

"""创建 VLM 和动作专家，按对应层导出和消费上下文 K/V。

Build the VLM and Expert; export and consume matching-layer context KV.

原始实现拆分副本，需配合原仓库依赖运行。
Extracted implementation; requires the original repository dependencies.
"""

from ._runtime import CONFIG_MAPPING, DEFAULT_IMAGE_SIZE, GemmaForCausalLM, Literal, PaliGemmaForConditionalGeneration, Tensor, apply_per_token_norm, nn, torch
from .attention import compute_expert_layer_with_context_kv, compute_layer_complete, compute_vlm_layer_with_context_kv

class GemmaConfig:  # see openpi `gemma.py: Config`
    """Configuration for Gemma model variants."""

    # 保存网络结构尺寸；width 是隐藏宽度，head_dim 是注意力单头宽度，两者不要直接等同。
    # Store structural dimensions: width is hidden width and head_dim is per-attention-head width; they are not interchangeable.
    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


# 两个预设的隐藏宽度不同，但层数、Q 头数、KV 头数和单头宽度一致，这使 VLM 与 Expert 可以按层交换 K/V。
# Presets have different hidden widths but matching depth, Q/KV head counts, and head dimension, enabling layerwise VLM/Expert KV exchange.
def get_gemma_config(variant: str) -> GemmaConfig:  # see openpi `gemma.py: get_config`
    """Returns config for specified gemma variant."""
    if variant == "gemma_300m":
        return GemmaConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    elif variant == "gemma_2b":
        return GemmaConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


class PaliGemmaWithExpertModel(
    nn.Module
):  # see openpi `gemma_pytorch.py: PaliGemmaWithExpertModel` this class is almost a exact copy of PaliGemmaWithExpertModel in openpi
    """PaliGemma model with action expert for PI05."""

    # 按结构配置创建视觉语言模型与动作专家，再设置精度和冻结边界；这里创建网络结构，预训练权重由策略检查点加载流程注入。
    # Create the VLM and action Expert from structural configs, then set precision and freezing. This constructs networks; policy checkpoint loading supplies pretrained weights.
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        image_size: int = DEFAULT_IMAGE_SIZE,
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.image_size = image_size
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        # 专家输入来自连续动作与未来条件投影，不走离散词表嵌入，因此移除未使用的 embed_tokens。
        # Expert inputs are continuous action/future projections rather than vocabulary IDs, so its unused token embedding is removed.
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)
        self._set_requires_grad()

    # bfloat16 模式保留视觉嵌入与归一化等选定参数为 float32，其余使用 bfloat16；并非把所有参数统一压成低精度。
    # In bfloat16 mode, selected vision embeddings and normalization parameters remain float32 while other parameters use bfloat16; precision is intentionally mixed.
    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    # 冻结视觉开关只作用于视觉塔；仅训练专家开关会冻结整个 PaliGemma。世界模型的阶段冻结另由策略层管理。
    # The vision-freeze flag targets the vision tower; expert-only training freezes all PaliGemma parameters. Policy-level code separately manages stage-specific world-module freezing.
    def _set_requires_grad(self):
        if self.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()
            for param in self.paligemma.vision_tower.parameters():
                param.requires_grad = False
        if self.train_expert_only:
            self.paligemma.eval()
            for param in self.paligemma.parameters():
                param.requires_grad = False

    # 父类 train 会递归修改子模块模式，因此随后把冻结部分重新设为 eval，避免冻结参数却意外改变前向行为。
    # Parent train recursively changes module modes; frozen components are reset to eval afterward so their forward behavior remains fixed.
    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()
        if self.train_expert_only:
            self.paligemma.eval()

    # 调用 PaliGemma 的视觉塔和多模态投影，输出能进入语言模型隐藏空间的图像 token，而不是词表 token ID。
    # Run PaliGemma vision and multimodal projection to obtain image-token features in language-model hidden space, not vocabulary token IDs.
    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    # 只做语言 token ID 到词嵌入的查表；缩放和图像语言拼接由上层 embed_prefix 完成。
    # Look up embeddings for language token IDs; embed_prefix handles scaling and concatenation with image features.
    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    # 按 inputs_embeds 中是否有空流选择执行路径：仅 VLM 适合前缀预填，仅 Expert 可消费缓存，两流都有则逐层联合计算。
    # Select a path by absent entries in inputs_embeds: VLM-only for prefix prefill, Expert-only for cached decoding, or joint layerwise evaluation when both streams exist.
    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        # 只编码上下文，缓存中可能仍含内部规划查询；上层推理准备函数负责在交给 Expert 前裁剪。
        # Encode only the prefix; the resulting cache may still include internal planner queries, which inference preparation crops before Expert reuse.
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        # 仅运行专家，以已有前缀 KV 为记忆。是否追加或返回缓存由 use_cache 和对应 Transformers 实现决定。
        # Run the Expert against existing prefix KV; cache updates/returns follow use_cache and the compatible Transformers implementation.
        elif inputs_embeds[0] is None:
            suffix_model_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_model_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = suffix_model_output.past_key_values
        else:
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )

            # final norm
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            # 联合双流训练在这里不构造可复用缓存；需要可微逐层 KV 的第二阶段走 forward_vlm_with_context_kv。
            # This joint training branch does not build reusable KV; Stage 2 uses forward_vlm_with_context_kv for explicit differentiable layer caches.
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values

    # 遍历 VLM 各层，逐层保存上下文 K/V。
    # Collect context KV from every VLM layer in layer order, not only the final layer.
    def forward_vlm_with_context_kv(
        self,
        *,
        prefix_embeds: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
        context_length: int,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        """Run one differentiable VLM prefill and retain per-layer context KV."""
        hidden_states = prefix_embeds
        context_key_values = []
        num_layers = self.paligemma.config.text_config.num_hidden_layers
        use_gradient_checkpointing = (
            hasattr(self.paligemma.language_model, "gradient_checkpointing")
            and self.paligemma.language_model.gradient_checkpointing
            and self.training
        ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

        for layer_idx in range(num_layers):
            if use_gradient_checkpointing:
                hidden_states, context_key, context_value = torch.utils.checkpoint.checkpoint(
                    compute_vlm_layer_with_context_kv,
                    layer_idx,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    context_length,
                    use_reentrant=False,
                    preserve_rng_state=False,
                    paligemma=self.paligemma,
                )
            else:
                hidden_states, context_key, context_value = compute_vlm_layer_with_context_kv(
                    layer_idx,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    context_length,
                    paligemma=self.paligemma,
                )
            # 列表下标就是 Transformer 层号，每项是一对 K/V 张量；这些张量带有训练计算图，不只是推理用的常量缓存。
            # The list index is the Transformer layer number; each entry is a KV pair retaining its training graph, not merely an inference constant.
            context_key_values.append((context_key, context_value))

        hidden_states, _ = self.paligemma.language_model.norm(
            hidden_states,
            cond=None,
        )
        return hidden_states, context_key_values

    # 专家第 l 层读取 VLM 第 l 层 K/V。
    # Consume KV layer by layer: Expert layer l uses VLM layer l KV, avoiding another VLM encoding pass.
    def forward_expert_with_context_kv(
        self,
        *,
        context_key_values: list[tuple[Tensor, Tensor]],
        future_embeds: Tensor,
        action_embeds: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
        future_adarms_cond: Tensor,
        action_adarms_cond: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run only the Expert streams against a single cached VLM prefill."""
        inputs_embeds = [future_embeds, action_embeds]
        adarms_cond = [future_adarms_cond, action_adarms_cond]
        num_layers = self.paligemma.config.text_config.num_hidden_layers
        if len(context_key_values) != num_layers:
            raise ValueError(
                "Expected one context KV pair per transformer layer; "
                f"got {len(context_key_values)} for {num_layers} layers."
            )
        use_gradient_checkpointing = (
            hasattr(self.gemma_expert.model, "gradient_checkpointing")
            and self.gemma_expert.model.gradient_checkpointing
            and self.training
        ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

        # 必须保持 VLM 与 Expert 的层配对及头维兼容；这里不会把多个 VLM 层压成一个公共记忆张量。
        # VLM/Expert layer pairing and head dimensions must remain compatible; this loop does not compress all VLM layers into one shared memory.
        for layer_idx, (context_key, context_value) in enumerate(context_key_values):
            if use_gradient_checkpointing:
                inputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_expert_layer_with_context_kv,
                    layer_idx,
                    inputs_embeds,
                    context_key,
                    context_value,
                    attention_mask,
                    position_ids,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                    paligemma=self.paligemma,
                    gemma_expert=self.gemma_expert,
                )
            else:
                inputs_embeds = compute_expert_layer_with_context_kv(
                    layer_idx,
                    inputs_embeds,
                    context_key,
                    context_value,
                    attention_mask,
                    position_ids,
                    adarms_cond,
                    paligemma=self.paligemma,
                    gemma_expert=self.gemma_expert,
                )

        outputs = []
        for index, hidden_states in enumerate(inputs_embeds):
            output, _ = apply_per_token_norm(self.gemma_expert.model.norm, hidden_states, adarms_cond[index])
            outputs.append(output)
        return outputs[0], outputs[1]


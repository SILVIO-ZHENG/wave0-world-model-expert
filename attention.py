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

"""逐层注意力计算。重点看 compute_expert_layer_with_context_kv 中的 Q 投影和上下文 K/V 拼接。

Layerwise attention kernels. Focus on Expert Q projections and context KV concatenation.

原始实现拆分副本，需配合原仓库依赖运行。
Extracted implementation; requires the original repository dependencies.
"""

from ._runtime import apply_per_token_norm, modeling_gemma, torch

# 基础双流路径：输入依次为 VLM 上下文和动作 Expert 状态。两侧用各自参数投影到兼容的注意力头空间，共享一次带掩码注意力，再按原长度切回各自流。
# Base two-stream path: VLM context and action Expert states use separate projections into compatible attention heads, share one masked attention operation, then split back by original stream lengths.
def compute_layer_complete(
    layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond, paligemma, gemma_expert
):
    models = [paligemma.language_model, gemma_expert.model]
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)
    # Concatenate and process attention
    # dim=2 是 [B,头数,token数,单头宽度] 的 token 轴；不同隐藏宽度由各自投影层转换后才能在这里拼接。
    # dim=2 is the token axis of [B,heads,tokens,head_dim]; separate projections first reconcile differing hidden widths.
    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )
    batch_size = query_states.shape[0]
    scaling = paligemma.language_model.layers[layer_idx].self_attn.scaling
    # Attention computation
    att_output, _ = modeling_gemma.eager_attention_forward(
        paligemma.language_model.layers[layer_idx].self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    # Get head_dim from the current layer, not from the model
    head_dim = paligemma.language_model.layers[layer_idx].self_attn.head_dim
    # 这里保留原实现的 8 个 Query 头假设，对应 get_gemma_config 的预设；若研究其他头数，不能只修改隐藏宽度配置。
    # The retained reshape assumes eight query heads, matching get_gemma_config presets; other head counts require more than a hidden-width change.
    att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)
    # Process layer outputs
    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        # first residual
        out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
        after_first_residual = out_emb.clone()
        out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
        # Convert to bfloat16 if the next layer (mlp) uses bfloat16
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        # second residual
        out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


# 三流参考路径：输入依次为上下文、预测未来条件、带噪动作。未来与动作共享 Expert 权重；与当前 Stage 2 的缓存路径不同，这里还显式计算上下文 Query 和上下文输出。
# Three-stream reference path: context, predicted-future conditions, and noisy actions. Future/action streams share Expert weights; unlike the cached Stage-2 path, this also computes context queries and outputs.
def compute_layer_with_future_condition(
    layer_idx,
    inputs_embeds,
    attention_mask,
    position_ids,
    adarms_cond,
    paligemma,
    gemma_expert,
):
    """Run one shared-attention layer over VLM, future, and action streams.

    The future and action streams share the Action Expert weights, but use
    independent AdaRMS conditions. This retained reference implementation is
    used to verify that the cached-context Expert path is numerically equivalent
    to the original interleaved three-stream layer.
    """
    models = [
        paligemma.language_model,
        gemma_expert.model,
        gemma_expert.model,
    ]
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for index, hidden_states in enumerate(inputs_embeds):
        # index=0 使用 VLM 层；index=1 和 2 使用同一 Expert 层，不会为未来条件额外创建一套专家参数。
        # Index 0 selects a VLM layer; indices 1 and 2 select the same Expert layer rather than a separate future-condition expert.
        layer = models[index].layers[layer_idx]
        hidden_states, gate = layer.input_layernorm(
            hidden_states,
            cond=adarms_cond[index],
        )
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_states.append(layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2))
        key_states.append(layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2))
        value_states.append(layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2))

    # 三条流的 Query 与 Key 顺序相同，mask 决定上下文、未来和动作能否互相读取；一次计算并不意味着完全双向可见。
    # All three streams have the same query/key ordering; the mask controls cross-block visibility, so joint computation does not imply unrestricted bidirectional attention.
    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        unsqueeze_dim=1,
    )
    scaling = paligemma.language_model.layers[layer_idx].self_attn.scaling
    attention_output, _ = modeling_gemma.eager_attention_forward(
        paligemma.language_model.layers[layer_idx].self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    head_dim = paligemma.language_model.layers[layer_idx].self_attn.head_dim
    # 底层 helper 返回各 Query 位置的多头结果；合并头维后保留 Query 顺序，供各流的 o_proj 使用。
    # The helper returns multi-head outputs for query positions; merge head features while retaining query order for each stream's o_proj.
    attention_output = attention_output.reshape(
        query_states.shape[0],
        -1,
        8 * head_dim,
    )

    outputs_embeds = []
    start_position = 0
    for index, hidden_states in enumerate(inputs_embeds):
        # index=0 使用 VLM 层；index=1 和 2 使用同一 Expert 层，不会为未来条件额外创建一套专家参数。
        # Index 0 selects a VLM layer; indices 1 and 2 select the same Expert layer rather than a separate future-condition expert.
        layer = models[index].layers[layer_idx]
        end_position = start_position + hidden_states.shape[1]
        stream_attention = attention_output[:, start_position:end_position]
        if stream_attention.dtype != layer.self_attn.o_proj.weight.dtype:
            stream_attention = stream_attention.to(layer.self_attn.o_proj.weight.dtype)
        # o_proj 把多头特征映射回 Expert 隐藏宽度；它不是最终动作输出层。
        # o_proj maps merged attention heads back to Expert hidden width; it is not the final action-output projection.
        output = layer.self_attn.o_proj(stream_attention)
        output = modeling_gemma._gated_residual(hidden_states, output, gates[index])  # noqa: SLF001
        first_residual = output
        output, gate = layer.post_attention_layernorm(
            output,
            cond=adarms_cond[index],
        )
        if output.dtype != layer.mlp.up_proj.weight.dtype:
            output = output.to(layer.mlp.up_proj.weight.dtype)
        # MLP 逐 token 变换特征，不在这里交换 token 信息；跨 token 信息交换已经在注意力步骤完成。
        # The MLP transforms each token independently; cross-token information exchange occurred in the attention step.
        output = layer.mlp(output)
        output = modeling_gemma._gated_residual(first_residual, output, gate)  # noqa: SLF001
        outputs_embeds.append(output)
        start_position = end_position
    return outputs_embeds


# 先做归一化与 Q/K/V 投影，对 Q/K 应用 RoPE；裁剪原始上下文 KV，同时继续计算 VLM 隐状态。
# Normalize, project Q/K/V and rotate Q/K with RoPE; retain original-context KV while continuing the VLM hidden-state update.
# VLM 每层生成上下文 K/V；不是最终隐状态再投影。
# Step 1: export original image/language KV from the matching VLM layer, not from final hidden states.
def compute_vlm_layer_with_context_kv(
    layer_idx,
    hidden_states,
    attention_mask,
    position_ids,
    context_length,
    paligemma,
):
    """Run one VLM layer and expose the original-context KV before attention.

    ``hidden_states`` contains the original PI0.5 image/language context followed
    by the internal latent-action queries.  The blockwise attention mask prevents
    context tokens from reading the queries, so the first ``context_length`` KV
    entries are numerically identical to a query-free PI0.5 prefix.  Returning
    them from every layer lets the Action Expert reuse this single differentiable
    VLM prefill instead of recomputing the VLM a second time.
    """
    layer = paligemma.language_model.layers[layer_idx]
    residual = hidden_states
    normalized, gate = layer.input_layernorm(hidden_states, cond=None)

    input_shape = normalized.shape[:-1]
    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
    # 这里是 VLM 自身的 Q，不是动作专家的 Q。
    # These queries belong to the VLM, not the Action Expert. Shape: [B,Hq,L,Dh].
    query_states = layer.self_attn.q_proj(normalized).view(hidden_shape).transpose(1, 2)
    key_states = layer.self_attn.k_proj(normalized).view(hidden_shape).transpose(1, 2)
    value_states = layer.self_attn.v_proj(normalized).view(hidden_shape).transpose(1, 2)

    # 这个全零张量只提供位置编码所需的形状、设备和类型，不会替换实际的 Query 特征。
    # This zero tensor supplies shape, device and dtype for rotary embeddings; it does not replace actual query features.
    rotary_input = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    # RoPE 旋转 Q/K，不旋转 V；缓存 K 已旋转。
    # RoPE rotates Q and K, not V. The exported context keys are already rotated.
    cos, sin = paligemma.model.language_model.rotary_emb(rotary_input, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        unsqueeze_dim=1,
    )

    # Keep the compact, non-repeated GQA KV representation
    # [B, num_kv_heads, context_length, head_dim].
    # 裁剪出原始上下文，排除规划槽和潜在动作查询；保留梯度。
    # Keep only original context; exclude RTC planner slots and latent-action queries. No detach: gradients can reach the VLM.
    # 这里保存的是本层注意力之前的投影结果。下一层会重新投影自己的输入，不能把最后一层 KV 复制给所有层。
    # These projections precede this layer's attention. Every later layer projects its own input; final-layer KV cannot substitute for all layers.
    context_key = key_states[:, :, :context_length]
    context_value = value_states[:, :, :context_length]

    attention_output, _ = modeling_gemma.eager_attention_forward(
        layer.self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        layer.self_attn.scaling,
    )
    # 底层 helper 返回各 Query 位置的多头结果；合并头维后保留 Query 顺序，供各流的 o_proj 使用。
    # The helper returns multi-head outputs for query positions; merge head features while retaining query order for each stream's o_proj.
    attention_output = attention_output.reshape(
        query_states.shape[0],
        -1,
        query_states.shape[1] * layer.self_attn.head_dim,
    )
    if attention_output.dtype != layer.self_attn.o_proj.weight.dtype:
        attention_output = attention_output.to(layer.self_attn.o_proj.weight.dtype)
    hidden_states = layer.self_attn.o_proj(attention_output)
    hidden_states = modeling_gemma._gated_residual(residual, hidden_states, gate)  # noqa: SLF001

    residual = hidden_states
    hidden_states, gate = layer.post_attention_layernorm(hidden_states, cond=None)
    if hidden_states.dtype != layer.mlp.up_proj.weight.dtype:
        hidden_states = hidden_states.to(layer.mlp.up_proj.weight.dtype)
    hidden_states = layer.mlp(hidden_states)
    hidden_states = modeling_gemma._gated_residual(residual, hidden_states, gate)  # noqa: SLF001
    return hidden_states, context_key, context_value


# 只更新未来条件和动作两条流；VLM 上下文只提供对应层 K/V。动作 Q 可看上下文、未来和动作，未来 Q 不能看动作。
# Update only future and action streams. VLM context supplies matching-layer KV; action queries see all three blocks, while future queries cannot see actions.
# 核心：动作专家的未来条件流和动作流读取对应层 VLM K/V。
# Core function: future/action Expert streams read matching-layer VLM KV. Both streams share Expert layer weights.
def compute_expert_layer_with_context_kv(
    layer_idx,
    inputs_embeds,
    context_key,
    context_value,
    attention_mask,
    position_ids,
    adarms_cond,
    paligemma,
    gemma_expert,
):
    """Run future/action Expert streams against cached VLM context KV.

    Only the two Expert streams are evaluated here.  Query KV is never included:
    ``context_key`` / ``context_value`` contain exactly the original PI0.5
    image/language prefix from the matching VLM layer.
    """
    # 两项引用同一个 Expert，而不是两套独立参数。两条流长度可以不同，但隐藏宽度必须匹配 Expert。
    # Both entries reference the same Expert, not two parameter sets. Stream lengths may differ; hidden widths must match the Expert.
    # 下文 B 为批量大小，C 为上下文长度，F 为未来 token 数，A 为动作 token 数，Dh 为单头维度。
    # Below, B is batch size, C context length, F future-token count, A action-token count, and Dh head dimension.
    models = [gemma_expert.model, gemma_expert.model]
    query_states = []
    key_states = []
    value_states = []
    gates = []

    # 两条流依次是未来条件和动作；动作 Q 来自第二条流。
    # inputs_embeds is [future_embeds, action_embeds]; index=1 supplies the action queries.
    for index, hidden_states in enumerate(inputs_embeds):
        layer = models[index].layers[layer_idx]
        # 先归一化再做注意力投影；时间条件通过 AdaRMS 调制归一化及残差门控。
        # Normalize before attention projections; time conditioning modulates adaptive normalization and residual gates.
        # 未来条件通常使用零条件，动作流使用采样的去噪时间；RTC 情况下时间还可以逐 token 不同。
        # Future tokens normally use a zero condition; action tokens use denoising time, which may vary per token under RTC.
        normalized, gate = apply_per_token_norm(layer.input_layernorm, hidden_states, adarms_cond[index])
        gates.append(gate)
        input_shape = normalized.shape[:-1]
        # Explicit head counts also support the empty future stream at RTC
        # inference; reshape(..., 0, -1, head_dim) is ambiguous for zero elements.
        # 投影后拆分注意力头；GQA 的 Q 头数与 KV 头数可不同。
        # Split projected features into heads: [B,L,H*Dh] -> [B,H,L,Dh]. GQA permits different Q and KV head counts.
        def project(projection):
            heads = projection.out_features // layer.self_attn.head_dim
            return projection(normalized).view(*input_shape, heads, layer.self_attn.head_dim).transpose(1, 2)
        # 这就是动作专家自身 q_proj 生成 Q 的位置。
        # The requested Q comes from Expert q_proj; action queries occupy the action segment after concatenation.
        query_states.append(project(layer.self_attn.q_proj))
        # 专家也产生自己的 K/V，所以同时存在跨上下文注意力和本地注意力。
        # The Expert also generates local KV, combining cross-context attention with attention over Expert tokens.
        key_states.append(project(layer.self_attn.k_proj))
        value_states.append(project(layer.self_attn.v_proj))

    # 沿 token 维拼接未来 Q 和动作 Q。
    # Concatenate on the token axis: Q=[Q_future,Q_action], not on the head or feature axis.
    query_states = torch.cat(query_states, dim=2)
    local_key_states = torch.cat(key_states, dim=2)
    local_value_states = torch.cat(value_states, dim=2)
    # 这个全零张量只提供位置编码所需的形状、设备和类型，不会替换实际的 Query 特征。
    # This zero tensor supplies shape, device and dtype for rotary embeddings; it does not replace actual query features.
    rotary_input = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = paligemma.model.language_model.rotary_emb(rotary_input, position_ids)
    query_states, local_key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states,
        local_key_states,
        cos,
        sin,
        unsqueeze_dim=1,
    )

    # 上下文 K 已在 VLM 中做过 RoPE，因此这里只旋转本地 Q/K，不能把上下文 K 再旋转一次。
    # Context keys already received RoPE in the VLM; rotate only local Q/K here to avoid rotating context keys twice.
    # 类型转换保留自动求导关系；to(dtype) 与 detach 不同，不会主动切断到 VLM 的梯度。
    # Dtype conversion preserves autograd connectivity; unlike detach, to(dtype) does not intentionally cut gradients into the VLM.
    if context_key.dtype != local_key_states.dtype:
        context_key = context_key.to(local_key_states.dtype)
    if context_value.dtype != local_value_states.dtype:
        context_value = context_value.to(local_value_states.dtype)
    # 关键：拼接 VLM、未来条件、动作三部分 K/V，不重新投影 VLM K/V。
    # Key merge: K=[K_vlm,K_future,K_action], likewise V. VLM KV is not reprojected by the Expert.
    key_states = torch.cat([context_key, local_key_states], dim=2)
    value_states = torch.cat([context_value, local_value_states], dim=2)

    # 此时 Q 为 [B,Hq,F+A,Dh]，K/V 为 [B,Hkv,C+F+A,Dh]。
    # Q now has shape [B,Hq,F+A,Dh], while K/V have shape [B,Hkv,C+F+A,Dh].
    # 兼容的头数和 Dh 让不同隐藏宽度的 VLM 与 Expert 能交换 K/V；拼接不要求两者原始隐藏宽度相等。
    # Compatible head counts and Dh enable KV exchange even when VLM and Expert hidden widths differ.
    # mask 的最后两维分别对应 Query 位置和 Key 位置；上下文没有 Query 行。
    # The final mask axes index query and key positions respectively; context has no query rows.
    # 对动作 Query，一次 softmax 同时在上下文、未来和动作 Key 中分配权重，不是三个结果相加。
    # Each action query uses one softmax over context, future and action keys, rather than summing three independent attention outputs.
    expert_attention = gemma_expert.model.layers[layer_idx].self_attn
    # 注意力入口：softmax(scale × QK转置 + mask) V；具体计算由 Transformers helper 完成。
    # Attention entry: conceptually softmax(scale * Q K^T + mask) V; the imported Transformers helper implements softmax and GQA expansion.
    attention_output, _ = modeling_gemma.eager_attention_forward(
        expert_attention,
        query_states,
        key_states,
        value_states,
        attention_mask,
        expert_attention.scaling,
    )
    # 底层 helper 返回各 Query 位置的多头结果；合并头维后保留 Query 顺序，供各流的 o_proj 使用。
    # The helper returns multi-head outputs for query positions; merge head features while retaining query order for each stream's o_proj.
    attention_output = attention_output.reshape(
        query_states.shape[0],
        -1,
        query_states.shape[1] * expert_attention.head_dim,
    )

    outputs = []
    start_position = 0
    # 按未来与动作切分输出，再执行输出投影、残差和 MLP。
    # Split outputs back into future/action streams, then apply o_proj, residuals, normalization and MLP.
    for index, hidden_states in enumerate(inputs_embeds):
        layer = models[index].layers[layer_idx]
        end_position = start_position + hidden_states.shape[1]
        stream_attention = attention_output[:, start_position:end_position]
        if stream_attention.dtype != layer.self_attn.o_proj.weight.dtype:
            stream_attention = stream_attention.to(layer.self_attn.o_proj.weight.dtype)
        # o_proj 把多头特征映射回 Expert 隐藏宽度；它不是最终动作输出层。
        # o_proj maps merged attention heads back to Expert hidden width; it is not the final action-output projection.
        output = layer.self_attn.o_proj(stream_attention)
        output = modeling_gemma._gated_residual(hidden_states, output, gates[index])  # noqa: SLF001

        residual = output
        output, gate = apply_per_token_norm(layer.post_attention_layernorm, output, adarms_cond[index])
        if output.dtype != layer.mlp.up_proj.weight.dtype:
            output = output.to(layer.mlp.up_proj.weight.dtype)
        # MLP 逐 token 变换特征，不在这里交换 token 信息；跨 token 信息交换已经在注意力步骤完成。
        # The MLP transforms each token independently; cross-token information exchange occurred in the attention step.
        output = layer.mlp(output)
        output = modeling_gemma._gated_residual(residual, output, gate)  # noqa: SLF001
        outputs.append(output)
        start_position = end_position
    return outputs


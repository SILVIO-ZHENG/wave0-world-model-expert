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

"""构造视觉语言上下文、规划查询和带噪动作 token，输出可复用的逐层 K/V。

Build context, planner and noisy-action tokens; expose reusable per-layer KV.

原始实现拆分副本，需配合原仓库依赖运行。
Extracted implementation; requires the original repository dependencies.
"""

from ._runtime import Any, F, Tensor, clamp_plan_prefix, math, torch
from .tensor_utils import create_sinusoidal_pos_embedding, make_att_2d_masks, pad_vector
from .world_model import spatial_pool_image_tokens

class ModelTokensMixin:
    # 按相机名称匹配世界模型需要的物理相机，避免把策略中的空相机槽或错误相机顺序交给世界模型。
    # Select physical cameras by configured names so policy empty slots or differing camera order do not reach world dynamics.
    def _select_world_model_images(self, images: list[Tensor]) -> list[Tensor]:
        """Select physical cameras from inputs ordered like config.image_features.

        Direct model callers supply the full VLM camera layout, including empty
        slots. The policy entry point instead passes explicitly named world images.
        """
        image_keys = list(self.config.image_features)
        if len(images) != len(image_keys):
            raise ValueError(
                f"Preprocessed image count {len(images)} != image_features {len(image_keys)}."
            )
        world_keys = list(self.config.world_model_image_features)
        if world_keys == image_keys:
            return images
        images_by_key = dict(zip(image_keys, images, strict=True))
        return [images_by_key[key] for key in world_keys]

    # 世界视觉特征是固定的监督空间：用冻结编码器提取每相机特征，按二维网格池化后返回 [B,相机数,空间token数,视觉宽度]。
    # World visual features form a fixed target space: frozen per-camera encoding and spatial pooling return [B,cameras,spatial_tokens,visual_width].
    @torch.no_grad()
    def encode_world_images(self, images: list[Tensor]) -> Tensor:
        """Extract frozen, spatially pooled PaliGemma image tokens for each camera."""
        target_backbone = getattr(self, "world_vision_backbone", None)
        if target_backbone is not None:
            target_backbone.eval()
        else:
            self.paligemma_with_expert.paligemma.vision_tower.eval()
            self.paligemma_with_expert.paligemma.multi_modal_projector.eval()
        pooled_latents = []
        for image in images:
            image_tokens = (
                target_backbone(image)
                if target_backbone is not None
                else self.paligemma_with_expert.embed_image(image)
            )
            pooled = spatial_pool_image_tokens(
                image_tokens,
                output_size=self.config.world_model_spatial_pool_size,
            )
            pooled_latents.append(pooled.to(dtype=torch.float32))
        return torch.stack(pooled_latents, dim=1).detach()

    # 同一相机的当前帧与未来帧沿 batch 轴合并，一次调用视觉骨干后再切开，减少重复调用；未来帧仅用于训练监督。
    # Concatenate current/future frames of each camera along the batch axis, encode them together, then split the batch. Future frames serve training supervision only.
    @torch.no_grad()
    def encode_world_image_pairs(
        self, current_images: list[Tensor], future_images: list[Tensor]
    ) -> tuple[Tensor, Tensor]:
        """Encode current/future frames together to improve frozen vision throughput."""
        if len(current_images) != len(future_images):
            raise ValueError("Current and future image lists must contain the same cameras.")
        target_backbone = getattr(self, "world_vision_backbone", None)
        if target_backbone is not None:
            target_backbone.eval()
        else:
            self.paligemma_with_expert.paligemma.vision_tower.eval()
            self.paligemma_with_expert.paligemma.multi_modal_projector.eval()
        current_latents = []
        future_latents = []
        for current_image, future_image in zip(current_images, future_images, strict=True):
            batch_size = current_image.shape[0]
            # 合并后形状为 [2B,通道,高,宽]，不是把两帧沿通道轴拼成新的六通道图像。
            # The merged shape is [2B,channels,height,width], not a six-channel image formed by channel concatenation.
            image_batch = torch.cat([current_image, future_image], dim=0)
            image_tokens = (
                target_backbone(image_batch)
                if target_backbone is not None
                else self.paligemma_with_expert.embed_image(image_batch)
            )
            pooled = spatial_pool_image_tokens(
                image_tokens,
                output_size=self.config.world_model_spatial_pool_size,
            ).to(dtype=torch.float32)
            # 视觉骨干不会混合 batch 中不同帧；按原 batch 边界还原当前和未来特征，再沿相机轴 stack。
            # Vision processes frames independently across the batch; split at the original batch boundary, then stack along cameras.
            current_latents.append(pooled[:batch_size])
            future_latents.append(pooled[batch_size:])
        return (
            torch.stack(current_latents, dim=1).detach(),
            torch.stack(future_latents, dim=1).detach(),
        )

    # 构建原始上下文：每台相机的视觉 token 后接语言 token。返回嵌入、有效位和注意力分块标记，三者的 token 顺序必须一致。
    # Build the original context by concatenating camera tokens followed by language tokens. Return embeddings, validity masks and block markers in matching token order.
    def embed_prefix(
        self, images, img_masks, tokens, masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed the original PI0.5 image/language context.

        This method deliberately contains no WAM latent-action queries. The
        Action Expert continues to receive the original PI0.5 context KV; the
        queries are internal VLM readout tokens created by ``embed_wam_prefix``.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            # 一台相机的有效位扩展到它的每个视觉 token；缺失相机仍可占槽，但其 Key/Query 位置会被有效位屏蔽。
            # Expand camera validity over all its visual tokens; absent-camera slots remain in layout but are masked out.
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            lang_emb_dim = lang_emb.shape[-1]
            # 按基础 Gemma 嵌入约定缩放语言特征，保持与预训练输入尺度一致，不是注意力中的 1/sqrt(Dh)。
            # Scale language embeddings according to the Gemma convention; this differs from attention score scaling by 1/sqrt(Dh).
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)

        num_lang_embs = lang_emb.shape[1]
        # 图像与语言都使用同一个初始块标记，因而有效上下文 token 彼此可见；后续规划/动作块通过新块标记与它们分离。
        # Images and language share the initial block marker, allowing mutual context attention; later planner/action blocks introduce new markers.
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    # VLM 输入为原始上下文、可选 RTC 规划槽、潜在动作查询。
    # VLM input: original context + optional RTC planner slots + learned latent-action queries. The latter two are excluded from Expert context KV.
    def embed_wam_prefix(self, images, img_masks, tokens, masks, *, rtc_prefix=None,
                         rtc_prefix_mask=None, action_state=None) -> tuple[Tensor, Tensor, Tensor]:
        """Append an internal latent-action query block to PI0.5 context.

        The attention-block markers are ``0`` for image/language context and
        ``[1, 0, ..., 0]`` for the query block. Context tokens therefore cannot
        read the learned queries, while every query can read the full context
        and all other queries. Query KV is removed before conditioning the
        Action Expert; only the predicted future-state condition is added.
        """
        context_embs, context_pad_masks, context_att_masks = self.embed_prefix(
            images, img_masks, tokens, masks
        )
        if self.config.rtc_training_max_delay:
            slots = self.config.rtc_training_max_delay
            batch = context_embs.shape[0]
            # 没有已知前缀也保留固定 RTC 槽数，但全部标成无效，以保持 token 布局和训练/推理结构一致。
            # Retain a fixed number of RTC slots even with no committed prefix, marking them invalid to preserve layout and graph structure.
            if rtc_prefix is None:
                known = context_embs.new_zeros(batch, slots, self.config.max_action_dim)
                known_mask = torch.zeros(batch, slots, dtype=torch.bool, device=context_embs.device)
            else:
                if rtc_prefix_mask is None or action_state is None:
                    raise ValueError('RTC planner prefix requires a mask and normalized action state')
                known_mask = rtc_prefix_mask[:, :slots].to(device=context_embs.device, dtype=torch.bool)
                known = torch.where(known_mask[..., None], rtc_prefix[:, :slots], 0.0).detach()
            if action_state is None:
                action_state = known.new_zeros(batch, self.config.max_action_dim)
            # 规划前缀同时包含绝对动作和相对当前状态的偏移；未知位置清零，避免把占位值当成已知动作。
            # Planner-prefix features include absolute actions and offsets from current state; unknown positions are zeroed rather than treated as committed commands.
            relative = torch.where(known_mask[..., None], known-action_state[:, None, :], 0.0)
            prefix_tokens = self.wam_rtc_prefix_projection(torch.cat([known, relative], dim=-1).float())
            prefix_tokens = prefix_tokens + self.wam_rtc_prefix_positions
            prefix_att = torch.zeros_like(known_mask)
            prefix_att[:, 0] = True
            context_embs = torch.cat([context_embs, prefix_tokens.to(context_embs.dtype)], dim=1)
            context_pad_masks = torch.cat([context_pad_masks, known_mask], dim=1)
            context_att_masks = torch.cat([context_att_masks, prefix_att], dim=1)
        bsize = context_embs.shape[0]
        latent_action_queries = self.wam_latent_action_queries.expand(bsize, -1, -1)
        latent_action_queries = latent_action_queries.to(
            device=context_embs.device,
            dtype=context_embs.dtype,
        )
        query_pad_masks = torch.ones(
            bsize,
            self.config.wam_latent_action_num_queries,
            dtype=torch.bool,
            device=context_pad_masks.device,
        )
        query_att_masks = torch.zeros_like(query_pad_masks)
        query_att_masks[:, 0] = True

        return (
            torch.cat([context_embs, latent_action_queries], dim=1),
            torch.cat([context_pad_masks, query_pad_masks], dim=1),
            torch.cat([context_att_masks, query_att_masks], dim=1),
        )

    # 普通 VLM 前缀入口同时返回最终隐藏状态、投影后的潜在动作和可选推理 KV 缓存；训练阶段的显式可微上下文 KV 由另一个专用入口生成。
    # The ordinary VLM-prefix path returns final states, projected latent actions and optional inference KV; the dedicated training entry produces explicit differentiable context KV.
    def encode_wam_prefix(
        self,
        images,
        img_masks,
        tokens,
        masks,
        *,
        use_cache: bool,
        rtc_prefix=None,
        rtc_prefix_mask=None,
        action_state=None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Any]:
        """Run the VLM prefix and return its explicit latent-action tokens.

        Returns:
            ``prefix_output``:
                Final VLM hidden states for context and latent-action queries.
            ``latent_action_tokens``:
                Projected final query states with shape
                ``[B, wam_latent_action_num_queries, world_model_width]``.
            ``prefix_pad_masks`` / ``prefix_att_masks``:
                The 1-D masks needed by the following Action Expert pass.
            ``past_key_values``:
                Optional per-layer VLM KV cache used by inference.
        """
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_wam_prefix(
            images, img_masks, tokens, masks, rtc_prefix=rtc_prefix,
            rtc_prefix_mask=rtc_prefix_mask, action_state=action_state,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (  # noqa: SLF001
            "eager"
        )
        (prefix_output, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=use_cache,
        )
        if prefix_output is None:
            raise RuntimeError("WAVE-0 VLM prefix forward returned no hidden states.")

        query_count = self.config.wam_latent_action_num_queries
        # 取最后的可学习规划查询输出，用投影层映射到世界模型动作表示空间；这与 Expert q_proj 生成注意力 Q 是两件事。
        # Take learned planner-query outputs and project them into the world-model action space; this differs from generating attention Q with Expert q_proj.
        query_output = prefix_output[:, -query_count:].to(dtype=torch.float32)
        latent_action_tokens = self.wam_latent_action_projection(query_output)
        expected_shape = (
            prefix_output.shape[0],
            query_count,
            self.config.world_model_width,
        )
        if latent_action_tokens.shape != expected_shape:
            raise RuntimeError(
                "WAVE-0 latent-action projection returned an invalid shape: "
                f"expected {expected_shape}, got {tuple(latent_action_tokens.shape)}."
            )

        return (
            prefix_output,
            latent_action_tokens,
            prefix_pad_masks,
            prefix_att_masks,
            past_key_values,
        )

    # 返回潜在动作 token、原始上下文有效位和逐层 KV；规划查询和 RTC 规划槽不会进入返回的上下文 KV。
    # Return latent actions, original-context validity and per-layer KV; exclude planner queries and RTC planner slots from context KV.
    # 训练时一次可微 VLM 编码同时返回潜在动作与逐层上下文 K/V。
    # Stage-2 VLM entry: one differentiable prefill returns latent actions and per-layer original-context KV.
    def encode_wam_prefix_for_training(
        self,
        images,
        img_masks,
        tokens,
        masks,
        *, rtc_prefix=None, rtc_prefix_mask=None, action_state=None,
    ) -> tuple[Tensor, Tensor, list[tuple[Tensor, Tensor]]]:
        """Run the Stage-2 VLM exactly once and expose differentiable context KV.

        The returned KV contains only the original PI0.5 image/language context.
        Latent-action query KV is intentionally excluded so the Action Expert can
        consume query information only through the predicted-future bottleneck.
        """
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_wam_prefix(
            images,
            img_masks,
            tokens,
            masks,
            rtc_prefix=rtc_prefix, rtc_prefix_mask=rtc_prefix_mask, action_state=action_state,
        )
        query_count = self.config.wam_latent_action_num_queries
        # 扣除规划槽和查询长度，避免专家直接读取规划查询。
        # Subtract latent queries and RTC slots so the Expert cannot directly read planner queries and bypass the predicted-future branch.
        # RTC 槽位虽然参与 VLM 规划，但不属于原始图像语言上下文。扣除槽位数量，而不是只扣除其中有效位置。
        # RTC slots participate in VLM planning but are not original image/language context. Subtract all allocated slots, not only valid ones.
        context_length = prefix_embs.shape[1] - query_count - self.config.rtc_training_max_delay
        if context_length <= 0:
            raise ValueError(
                "WAM training prefix must contain original PI0.5 context before "
                f"the {query_count} latent-action queries."
            )

        prefix_attention = self._prepare_attention_masks_4d(
            make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        )
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        vlm_dtype = self.paligemma_with_expert.paligemma.language_model.layers[
            0
        ].self_attn.q_proj.weight.dtype
        prefix_output, context_key_values = self.paligemma_with_expert.forward_vlm_with_context_kv(
            prefix_embeds=prefix_embs.to(dtype=vlm_dtype),
            attention_mask=prefix_attention,
            position_ids=prefix_position_ids,
            context_length=context_length,
        )

        # 取最后的可学习规划查询输出，用投影层映射到世界模型动作表示空间；这与 Expert q_proj 生成注意力 Q 是两件事。
        # Take learned planner-query outputs and project them into the world-model action space; this differs from generating attention Q with Expert q_proj.
        query_output = prefix_output[:, -query_count:].to(dtype=torch.float32)
        latent_action_tokens = self.wam_latent_action_projection(query_output)
        expected_shape = (
            prefix_output.shape[0],
            query_count,
            self.config.world_model_width,
        )
        if latent_action_tokens.shape != expected_shape:
            raise RuntimeError(
                "WAVE-0 training latent-action projection returned an invalid shape: "
                f"expected {expected_shape}, got {tuple(latent_action_tokens.shape)}."
            )
        return (
            latent_action_tokens,
            prefix_pad_masks[:, :context_length],
            context_key_values,
        )

    # RTC 先把潜在计划解码成物理动作，用已提交前缀硬覆盖，再经冻结动作编码器回到世界模型的动作空间；冻结编码器仍对自由计划部分保留梯度。
    # RTC decodes a physical plan, hard-clamps committed prefixes, and re-encodes through the frozen action encoder; gradients to the free plan remain available.
    def constrain_wam_plan(self, latent_tokens, action_state, rtc_prefix=None, rtc_prefix_mask=None):
        """Use the same hard physical prefix for WAM planning and action flow."""
        physical_plan = self.wam_rtc_plan_head(latent_tokens.float())
        physical_plan = pad_vector(physical_plan, self.config.max_action_dim)
        physical_plan = clamp_plan_prefix(physical_plan, rtc_prefix, rtc_prefix_mask)
        # Frozen weights, differentiable inputs. This call is also used at d=0
        # and during inference, so every trained-RTC request follows one graph.
        action_tokens = self.world_action_encoder(physical_plan, action_state)
        return action_tokens, physical_plan

    # 动作值经线性层变为 token；时间编码进入自适应归一化条件，不直接拼成另一组视觉 token。
    # Project action values into tokens; time embeddings condition adaptive normalization rather than becoming visual tokens.
    # 将带噪动作投影成专家 token，时间作为 AdaRMS 条件。
    # Project noisy actions into Expert tokens; encode time through an MLP for AdaRMS conditioning.
    def embed_suffix(self, noisy_actions, timestep):
        """Embed noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep.reshape(-1),
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        # 样本级时间为 [B]，RTC 可使用 [B,A] 的逐动作时间；先展平生成正余弦，再还原时间轴。
        # Samplewise time has shape [B]; RTC may use per-action [B,A] time. Flatten for sinusoidal encoding, then restore the action axis.
        if timestep.ndim == 2:
            time_emb = time_emb.reshape(*timestep.shape, -1)
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        action_time_emb = action_emb
        adarms_cond = time_emb

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        # 整个动作 chunk 属于一个新块：不让前面的上下文读取动作，但允许同一动作块内部双向交互。
        # The action chunk forms one new block: earlier context cannot read actions, while tokens within the action block interact bidirectionally.
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond


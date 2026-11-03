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

"""冻结视觉骨干、动作/状态编码、潜在世界模型与未来条件重采样器。

Vision backbones, action/state encoders, latent dynamics and future-condition resampling.

原始实现拆分副本，需配合原仓库依赖运行。
Extracted implementation; requires the original repository dependencies.
"""

from ._runtime import CONFIG_MAPPING, F, Literal, PaliGemmaMultiModalProjector, SiglipVisionModel, Tensor, WAVE0Config, math, nn, torch
from .vlm_expert import get_gemma_config

# 将 [B,N,D] 恢复为方形 patch 网格并做二维平均池化，再返回 [B,output_size²,D]，保留邻近空间关系。
# Restore [B,N,D] to a square patch grid, average-pool spatially and return [B,output_size²,D], preserving spatial neighborhoods.
def spatial_pool_image_tokens(image_tokens: Tensor, output_size: int) -> Tensor:
    """Pool square SigLIP patch tokens while preserving their 2-D spatial layout."""
    if image_tokens.ndim != 3:
        raise ValueError(f"Expected image tokens with shape [B,N,D], got {tuple(image_tokens.shape)}.")
    batch_size, num_tokens, width = image_tokens.shape
    spatial_size = math.isqrt(num_tokens)
    if spatial_size * spatial_size != num_tokens:
        raise ValueError(f"Expected a square patch grid, but got {num_tokens} image tokens.")
    if output_size > spatial_size:
        raise ValueError(
            f"Cannot pool a {spatial_size}x{spatial_size} grid to the larger size {output_size}."
        )

    features = image_tokens.reshape(batch_size, spatial_size, spatial_size, width)
    features = features.permute(0, 3, 1, 2)
    features = F.adaptive_avg_pool2d(features, (output_size, output_size))
    return features.flatten(2).transpose(1, 2).contiguous()


# 只实例化 SigLIP 与多模态投影，不加载语言模型或动作专家；输出固定的视觉潜在目标空间。
# Instantiate only SigLIP and the multimodal projector, without a language model or Expert, to define a fixed visual target space.
class FrozenPaliGemmaVisionBackbone(nn.Module):
    """The only pretrained PI0.5 component required by Stage 1.

    This is equivalent to ``PaliGemmaModel.get_image_features`` but deliberately
    does not instantiate the PaliGemma language model, LM head, or action expert.
    """

    def __init__(self, config: WAVE0Config):
        super().__init__()
        vlm_config = get_gemma_config(config.paligemma_variant)
        paligemma_config = CONFIG_MAPPING["paligemma"]()
        paligemma_config.text_config.hidden_size = vlm_config.width
        paligemma_config.vision_config.image_size = config.image_resolution[0]
        paligemma_config.vision_config.intermediate_size = 4304
        paligemma_config.vision_config.projection_dim = vlm_config.width
        paligemma_config.vision_config.projector_hidden_act = "gelu_fast"
        paligemma_config.vision_config.torch_dtype = "float32"

        self.vision_tower = SiglipVisionModel(paligemma_config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(paligemma_config)
        self._set_precision(config.dtype)
        self.requires_grad_(False)
        self.eval()

    def _set_precision(self, precision: Literal["bfloat16", "float32"]) -> None:
        if precision == "float32":
            self.to(dtype=torch.float32)
            return
        if precision != "bfloat16":
            raise ValueError(f"Unsupported vision precision: {precision}")

        self.to(dtype=torch.bfloat16)
        keep_float32 = (
            "vision_model.embeddings.patch_embedding",
            "vision_model.embeddings.position_embedding",
        )
        for name, parameter in self.named_parameters():
            if any(selector in name for selector in keep_float32):
                parameter.data = parameter.data.to(dtype=torch.float32)

    def train(self, mode: bool = True):
        # The Stage-1 latent target space must remain fixed and deterministic.
        return super().train(False)

    def forward(self, image: Tensor) -> Tensor:
        vision_output = self.vision_tower(image)
        return self.multi_modal_projector(vision_output.last_hidden_state)


# 以非注册引用复用策略视觉模块，避免同一参数在 state_dict 中保存两份；共享模式要求策略视觉也冻结。
# Reuse policy vision through unregistered references to avoid duplicate state_dict tensors; shared mode also requires frozen policy vision.
class SharedFrozenPaliGemmaVisionBackbone(nn.Module):
    """Unregistered view of the frozen policy vision path.

    The referenced modules remain owned by ``paligemma_with_expert``. Keeping
    the references unregistered makes the World Model consume the exact same
    tensors without adding a second set of state-dict keys to checkpoints.
    """

    def __init__(self, vision_tower: nn.Module, multi_modal_projector: nn.Module):
        super().__init__()
        object.__setattr__(self, "_vision_tower_ref", vision_tower)
        object.__setattr__(self, "_multi_modal_projector_ref", multi_modal_projector)
        self.train(False)

    def train(self, mode: bool = True):
        super().train(False)
        self._vision_tower_ref.eval()
        self._multi_modal_projector_ref.eval()
        return self

    def forward(self, image: Tensor) -> Tensor:
        vision_output = self._vision_tower_ref(image)
        return self._multi_modal_projector_ref(vision_output.last_hidden_state)


# 将归一化真实动作序列 [B,T,Da] 编码为 [B,T,W]；同时输入绝对动作与相对当前关节状态的偏移。
# Encode normalized ground-truth actions [B,T,Da] into [B,T,W], using both absolute actions and offsets from current joint state.
class WorldActionEncoder(nn.Module):
    """Encode a normalized GT joint-action sequence into latent action tokens."""

    def __init__(
        self,
        max_action_dim: int,
        width: int,
        depth: int,
        num_heads: int,
        max_sequence_length: int,
    ):
        super().__init__()
        hidden_width = min(width, 256)
        self.max_action_dim = max_action_dim
        self.input_projection = nn.Sequential(
            nn.Linear(2 * max_action_dim, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, width),
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, max_sequence_length, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=num_heads,
            dim_feedforward=4 * width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=depth,
            norm=nn.LayerNorm(width),
            enable_nested_tensor=False,
        )
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, actions: Tensor, state: Tensor) -> Tensor:
        if actions.ndim != 3 or actions.shape[-1] != self.max_action_dim:
            raise ValueError(
                f"Expected padded actions [B,T,{self.max_action_dim}], got {tuple(actions.shape)}."
            )
        if state.ndim != 2 or state.shape[-1] != self.max_action_dim:
            raise ValueError(f"Expected padded state [B,{self.max_action_dim}], got {tuple(state.shape)}.")
        if actions.shape[1] > self.position_embedding.shape[1]:
            raise ValueError(
                f"Action sequence length {actions.shape[1]} exceeds configured maximum "
                f"{self.position_embedding.shape[1]}."
            )
        # 绝对关节目标和相对位移一起编码，向网络同时提供目标姿态与运动幅度；位置嵌入保留动作时间顺序。
        # Encode absolute joint targets and relative displacements together to expose target pose and motion magnitude; position embeddings preserve action order.
        relative_actions = actions - state[:, None, :]
        tokens = self.input_projection(torch.cat([actions, relative_actions], dim=-1))
        tokens = tokens + self.position_embedding[:, : actions.shape[1]]
        return self.encoder(tokens)


# 将归一化机器人状态 [B,Ds] 映射为一个 [B,1,W] token，与视觉和动作共同条件化动力学。
# Map normalized robot state [B,Ds] into one [B,1,W] token to condition dynamics alongside vision and action.
class WorldStateEncoder(nn.Module):
    """Encode the normalized, padded robot state into one world-model token."""

    def __init__(self, max_state_dim: int, width: int):
        super().__init__()
        hidden_width = min(width, 256)
        self.max_state_dim = max_state_dim
        self.projection = nn.Sequential(
            nn.Linear(max_state_dim, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, width),
        )

    def forward(self, state: Tensor) -> Tensor:
        if state.ndim != 2 or state.shape[-1] != self.max_state_dim:
            raise ValueError(f"Expected padded state [B,{self.max_state_dim}], got {tuple(state.shape)}.")
        return self.projection(state).unsqueeze(1)


# 对每个样本应用通道级 scale/shift，借助 None 维广播到全部序列位置。
# Apply per-sample channel scale/shift, broadcasting across sequence positions through the inserted axis.
def _adaln_modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Apply sample-wise adaptive LayerNorm shift and scale."""
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


# 使用双向自注意力混合当前图像、状态、动作及未来查询；动作平均表示通过 AdaLN 调制整层。
# Mix current-image, state, action and future-query tokens with bidirectional attention; pooled actions modulate the block through AdaLN.
class AdaLNWorldModelBlock(nn.Module):
    """Bidirectional transformer block with action-conditioned AdaLN-Zero."""

    def __init__(self, width: int, num_heads: int, mlp_dim: int):
        super().__init__()
        if width % num_heads != 0:
            raise ValueError(f"width={width} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        self.head_dim = width // num_heads

        self.attention_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.qkv_projection = nn.Linear(width, 3 * width)
        self.attention_output_projection = nn.Linear(width, width)
        self.ffn_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(width, mlp_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, width),
        )

        # Per block: attention shift/scale/gate + FFN shift/scale/gate.
        # Zero initialization makes the new block an identity at initialization;
        # action conditioning is then learned progressively and stably.
        # 六组输出分别为注意力和 FFN 的 shift、scale、gate；零初始化使新增残差分支最初关闭。
        # Six outputs provide shift, scale and gate for attention and FFN; zero initialization initially closes the new residual branches.
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)

    def _attention(self, x: Tensor) -> Tensor:
        batch_size, sequence_length, width = x.shape
        # Q/K/V 一次投影后拆成 [B,H,L,Dh]；未传因果 mask，因此这个世界模型块允许序列内双向注意力。
        # Project Q/K/V jointly and reshape to [B,H,L,Dh]; no causal mask is passed, so this world-model block uses bidirectional attention.
        qkv = self.qkv_projection(x).reshape(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)
        output = F.scaled_dot_product_attention(query, key, value)
        output = output.transpose(1, 2).reshape(batch_size, sequence_length, width)
        return self.attention_output_projection(output)

    def forward(self, x: Tensor, action_condition: Tensor) -> Tensor:
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = self.adaln(
            action_condition
        ).chunk(6, dim=-1)
        attention_input = _adaln_modulate(self.attention_norm(x), shift_attn, scale_attn)
        x = x + gate_attn[:, None, :] * self._attention(attention_input)
        ffn_input = _adaln_modulate(self.ffn_norm(x), shift_ffn, scale_ffn)
        return x + gate_ffn[:, None, :] * self.ffn(ffn_input)


# 从当前视觉潜变量、状态和潜在动作预测未来视觉变化量；它不是生成 RGB 图像的解码器。
# Predict a future visual-latent delta from current latents, state and latent actions; this is not an RGB-image decoder.
class LatentWorldModel(nn.Module):
    """Predict a future visual-latent delta from current latents, state, and actions."""

    def __init__(
        self,
        image_latent_dim: int,
        width: int,
        depth: int,
        num_heads: int,
        mlp_dim: int,
        num_cameras: int,
        spatial_tokens_per_camera: int,
    ):
        super().__init__()
        self.num_cameras = num_cameras
        self.spatial_tokens_per_camera = spatial_tokens_per_camera
        self.visual_projection = nn.Linear(image_latent_dim, width)
        self.future_output_projection = nn.Linear(width, image_latent_dim)

        self.future_queries = nn.Parameter(torch.zeros(1, num_cameras, spatial_tokens_per_camera, width))
        self.camera_embedding = nn.Parameter(torch.zeros(1, num_cameras, 1, width))
        self.horizon_embedding = nn.Parameter(torch.zeros(1, 1, width))
        self.current_type_embedding = nn.Parameter(torch.zeros(1, 1, width))
        self.state_type_embedding = nn.Parameter(torch.zeros(1, 1, width))
        self.action_type_embedding = nn.Parameter(torch.zeros(1, 1, width))
        self.query_type_embedding = nn.Parameter(torch.zeros(1, 1, width))

        self.action_condition_norm = nn.LayerNorm(width)
        self.blocks = nn.ModuleList(
            AdaLNWorldModelBlock(width=width, num_heads=num_heads, mlp_dim=mlp_dim) for _ in range(depth)
        )
        self.output_norm = nn.LayerNorm(width)

        for parameter in (
            self.future_queries,
            self.camera_embedding,
            self.horizon_embedding,
            self.current_type_embedding,
            self.state_type_embedding,
            self.action_type_embedding,
            self.query_type_embedding,
        ):
            nn.init.normal_(parameter, std=0.02)

    def forward(self, current_latent: Tensor, state_token: Tensor, action_tokens: Tensor) -> Tensor:
        expected_shape = (self.num_cameras, self.spatial_tokens_per_camera)
        if current_latent.ndim != 4 or tuple(current_latent.shape[1:3]) != expected_shape:
            raise ValueError(
                "Expected current latent [B,C,P,D] with "
                f"(C,P)={expected_shape}, got {tuple(current_latent.shape)}."
            )
        batch_size = current_latent.shape[0]

        # 输入 [B,C,P,D视觉] 投影到世界模型宽度；C 为相机数、P 为单相机池化 patch 数。
        # Project [B,C,P,Dvision] to world-model width; C is camera count and P is the pooled patch count per camera.
        current_tokens = self.visual_projection(current_latent)
        current_tokens = current_tokens + self.camera_embedding + self.current_type_embedding[:, None]
        current_tokens = current_tokens.flatten(1, 2)

        queries = self.future_queries + self.camera_embedding
        queries = queries.flatten(1, 2).expand(batch_size, -1, -1)
        queries = queries + self.horizon_embedding + self.query_type_embedding

        # 序列顺序为当前视觉、状态、动作、未来查询；只有末尾查询对应的输出被映射回视觉变化量。
        # Concatenate current vision, state, actions and future queries; only trailing query outputs are projected back to visual deltas.
        sequence = torch.cat(
            [
                current_tokens,
                state_token + self.state_type_embedding,
                action_tokens + self.action_type_embedding,
                queries,
            ],
            dim=1,
        )
        # 动作 token 既作为序列输入，也取均值作为逐样本 AdaLN 条件；这不是替代动作序列信息。
        # Actions enter the sequence and also provide a pooled per-sample AdaLN condition; pooling does not replace the action sequence.
        action_condition = self.action_condition_norm(action_tokens.mean(dim=1))
        output = sequence
        for block in self.blocks:
            output = block(output, action_condition)
        query_output = output[:, -queries.shape[1] :]
        query_output = self.output_norm(query_output)
        pred_delta = self.future_output_projection(query_output)
        return pred_delta.reshape(
            batch_size,
            self.num_cameras,
            self.spatial_tokens_per_camera,
            -1,
        )


# 世界模型与 Expert 的接口：降低空间 token 数并映射到专家宽度，输出 [B,C*Sout²,Dexpert]。
# Bridge world-model features to the Expert by pooling spatial tokens and projecting width, yielding [B,C*Sout²,Dexpert].
class WAMFutureConditionResampler(nn.Module):
    """Convert predicted World Model latents into Action Expert condition tokens."""

    def __init__(
        self,
        *,
        input_width: int,
        output_width: int,
        num_cameras: int,
        input_spatial_size: int,
        output_spatial_size: int,
    ) -> None:
        super().__init__()
        if output_spatial_size > input_spatial_size:
            raise ValueError("Future-condition output grid cannot be larger than its input grid.")
        self.num_cameras = num_cameras
        self.input_spatial_size = input_spatial_size
        self.output_spatial_size = output_spatial_size
        self.input_norm = nn.LayerNorm(input_width)
        self.projection = nn.Linear(input_width, output_width)
        self.camera_embedding = nn.Parameter(
            torch.zeros(1, num_cameras, output_spatial_size**2, output_width)
        )
        self.type_embedding = nn.Parameter(torch.zeros(1, 1, output_width))
        nn.init.normal_(self.camera_embedding, std=0.02)
        nn.init.normal_(self.type_embedding, std=0.02)

    @property
    def num_output_tokens(self) -> int:
        return self.num_cameras * self.output_spatial_size**2

    def forward(self, predicted_future: Tensor) -> Tensor:
        expected_tokens = self.input_spatial_size**2
        if predicted_future.ndim != 4:
            raise ValueError(
                f"Predicted future latent must have shape [B,C,P,D]; got {tuple(predicted_future.shape)}."
            )
        batch_size, num_cameras, num_tokens, input_width = predicted_future.shape
        if num_cameras != self.num_cameras or num_tokens != expected_tokens:
            raise ValueError(
                "Predicted future latent has an incompatible camera/grid shape; "
                f"expected C={self.num_cameras}, P={expected_tokens}, got "
                f"C={num_cameras}, P={num_tokens}."
            )

        # 先把相机并入批次，每个相机单独进行空间池化，避免跨相机平均混合。
        # Fold cameras into the batch axis and pool each camera spatially, avoiding averaging across cameras.
        flattened = predicted_future.reshape(
            batch_size * num_cameras,
            num_tokens,
            input_width,
        )
        pooled = spatial_pool_image_tokens(
            flattened,
            output_size=self.output_spatial_size,
        )
        pooled = pooled.reshape(
            batch_size,
            num_cameras,
            self.output_spatial_size**2,
            input_width,
        )
        # 池化后归一化再投影到 Expert 宽度；相机和类型嵌入告诉专家 token 的来源。
        # Normalize pooled features and project to Expert width; camera/type embeddings identify token origin to the Expert.
        tokens = self.projection(self.input_norm(pooled))
        tokens = tokens + self.camera_embedding
        tokens = tokens.flatten(1, 2)
        return tokens + self.type_embedding


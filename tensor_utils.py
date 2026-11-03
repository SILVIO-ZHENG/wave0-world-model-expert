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

"""张量、时间编码、掩码、图像缩放和权重键辅助函数。

Tensor, time embedding, attention masks, image resizing and checkpoint-key utilities.

原始实现拆分副本，需配合原仓库依赖运行。
Extracted implementation; requires the original repository dependencies.
"""

from ._runtime import F, Tensor, TypedDict, math, torch

# 动作采样的可选输入：RTC 延迟、上一动作块剩余部分、机器人状态和世界模型相机输入。键均可省略，具体模式会检查必需项。
# Optional sampling inputs include RTC delay, leftover actions, robot state and world-model images. Mode-specific checks enforce required keys.
class ActionSelectKwargs(TypedDict, total=False):
    rtc_mode: str
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None
    state: Tensor | None
    world_images: list[Tensor] | None


# 剥离模型包装前缀后判断权重是否属于世界模型扩展，供禁用世界模型的部署加载过滤使用。
# Strip model-wrapper prefixes and identify world-extension tensors so deployment without the world model can exclude them.
def _is_world_model_state_key(key: str) -> bool:
    """Return whether a checkpoint tensor belongs to the WAM extension."""
    parts = key.split(".")
    while parts and parts[0] in {"model", "_orig_mod"}:
        parts.pop(0)
    if not parts:
        return False
    return parts[0] in {
        "world_action_encoder",
        "world_state_encoder",
        "world_model",
        "world_vision_backbone",
        "stage2_step",
        "wam_latent_action_queries",
        "wam_latent_action_projection",
        "wam_future_condition_resampler",
        "wam_rtc_prefix_projection",
        "wam_rtc_prefix_positions",
        "wam_rtc_plan_head",
    }


# 识别只在训练中使用的教师和步数；RTC 物理规划仍需要动作编码器时可以保留它。
# Identify training-only teacher and step tensors; retain the action encoder when RTC physical planning still needs it.
def _is_wam_teacher_state_key(key: str, keep_action_encoder: bool = False) -> bool:
    """Return whether a tensor is needed only for Stage-2 training."""
    parts = key.split(".")
    while parts and parts[0] in {"model", "_orig_mod"}:
        parts.pop(0)
    if not parts:
        return False
    return parts[0] == "stage2_step" or (parts[0] == "world_action_encoder" and not keep_action_encoder)


# 为指定设备选择辅助计算精度；这里显式将 CPU bfloat16 和 MPS float64 降为 float32。
# Choose a device-compatible helper dtype; this implementation maps CPU bfloat16 and MPS float64 to float32.
def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


# 输入 time 为 [B]，输出 [B,dimension]；不同周期的正弦和余弦让连续去噪时间可供神经网络使用。
# Map time [B] to [B,dimension]; sine/cosine features at different periods encode continuous denoising time.
def create_sinusoidal_pos_embedding(  # see openpi `create_sinusoidal_pos_embedding` (exact copy)
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


# 每个样本独立采样一个 Beta 随机数，输出 [B]；随后由配置缩放到训练时间区间。
# Draw one Beta variate per sample, returning [B]; configuration later scales these into the training-time interval.
def sample_beta(alpha, beta, bsize, device):  # see openpi `sample_beta` (exact copy)
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


# 注意力掩码：每个 Query 只看本块及之前的有效 Key。
# Attention mask: each query sees valid keys in its own or earlier blocks.
# pad_masks 表示位置是否有效，att_masks 表示新块起点；输出 [B,N,N]，最后两维是 Query、Key。
# pad_masks marks valid positions and att_masks marks block starts; output [B,N,N] indexes query then key positions.
def make_att_2d_masks(pad_masks, att_masks):  # see openpi `make_att_2d_masks` (exact copy)
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    # 块编号累加后，Key 块号小于等于 Query 块号才可见；外积有效位同时屏蔽 Query 和 Key padding。
    # Cumulative block IDs permit keys whose block ID is no greater than the query block ID; validity also masks padded queries and keys.
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


# 只补最后一个特征维度，保留批次和时间轴；已达到目标宽度时直接返回，不会裁剪更宽的输入。
# Pad only the final feature axis, preserving batch/time axes; inputs already at or above target width are returned unchanged.
def pad_vector(vector, new_dim):
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


# 由 x_t=t*noise+(1-t)*action 和速度预测估算 t=0 动作，返回与 x_t 同形状的 [B,T,D] 张量。
# Estimate the t=0 action from x_t=t*noise+(1-t)*action and predicted velocity, preserving the [B,T,D] shape.
def recover_clean_actions(x_t: Tensor, predicted_velocity: Tensor, time: Tensor) -> Tensor:
    """Recover the t=0 action estimate from PI0.5's flow-matching parameterization."""
    if time.ndim != 1 or time.shape[0] != x_t.shape[0]:
        raise ValueError(f"Expected time shape ({x_t.shape[0]},), got {tuple(time.shape)}.")
    if predicted_velocity.shape != x_t.shape:
        raise ValueError(
            "x_t and predicted_velocity must have identical shapes; "
            f"got {tuple(x_t.shape)} and {tuple(predicted_velocity.shape)}."
        )
    return x_t - time[:, None, None] * predicted_velocity


# 按原比例缩放再居中补边，避免把物体拉伸；支持通道优先或通道末尾，内部统一使用 NCHW。
# Resize proportionally and center-pad to avoid geometric distortion; accept channels-first/last and use NCHW internally.
def resize_with_pad_torch(  # see openpi `resize_with_pad_torch` (exact copy)
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input
    """
    # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

    batch_size, channels, cur_height, cur_width = images.shape

    # Calculate resize ratio
    # 取较大的缩放比例以确保图像完整落在目标框内，剩余区域在后面补边。
    # Use the larger scale ratio so the entire image fits inside the target frame; pad the remaining area later.
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize
    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    # Handle dtype-specific clipping
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad
    constant_value = 0 if images.dtype == torch.uint8 else -1.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # left, right, top, bottom
        mode="constant",
        value=constant_value,
    )

    # Convert back to original format if needed
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

    return padded_images


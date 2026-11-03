"""Training-time action conditioning for PI05 (arXiv:2512.05964).

PI05 uses noise at t=1 and clean actions at t=0, the reverse of the paper's
example. These helpers do not add parameters or change ordinary PI05 training.
"""

import torch
from torch import Tensor


# 输入 actions 为 [B,T,A]，逐样本均匀采样已知前缀长度 d，返回 [B,T] 布尔 mask；max_delay=0 返回 None。
# Given actions [B,T,A], sample a known-prefix length d uniformly per example and return boolean [B,T]; max_delay=0 returns None.
def sample_training_prefix_mask(
    actions: Tensor, max_delay: int, action_is_pad: Tensor | None = None
) -> Tensor | None:
    """Sample each prefix uniformly, retaining at least one valid target step.

Episode-tail padding limits the sampled delay. A prefix never includes padding;
an unusual mask with internal gaps conservatively uses its first valid run.
"""
    if max_delay == 0:
        return None
    batch_size, horizon = actions.shape[:2]
    if max_delay < 0 or max_delay >= horizon:
        raise ValueError(f"rtc_training_max_delay must be in [0, {horizon}), got {max_delay}")
    # 每条样本可选 d 属于 [0,limit]；遇到 episode 尾部 padding 时减小上限，必须留下至少一个有效后缀目标。
    # Each example samples d in [0,limit]; episode-tail padding reduces the limit so at least one valid suffix target remains.
    limits = torch.full((batch_size,), max_delay, dtype=torch.long, device=actions.device)
    if action_is_pad is not None:
        if action_is_pad.shape != actions.shape[:2]:
            raise ValueError("action_is_pad must match the action batch and horizon")
        padding = action_is_pad.to(device=actions.device, dtype=torch.bool)
        if padding.all(dim=1).any():
            raise ValueError("Training-time RTC requires at least one valid action per sample")
        positions = torch.arange(horizon, device=actions.device).expand(batch_size, -1)
        first_pad = torch.where(padding, positions, horizon).amin(dim=1)
        limits = torch.minimum(limits, (first_pad - 1).clamp_min(0))
    # 用 [0,1) 随机数乘 limit+1 再取整，得到包含 0 和 limit 的离散均匀长度；不是逐 token 独立丢弃。
    # Scale [0,1) draws by limit+1 and truncate to obtain a discrete uniform length including 0 and limit; this is not independent token dropout.
    delays = (torch.rand(batch_size, device=actions.device) * (limits + 1)).to(torch.long)
    return torch.arange(horizon, device=actions.device)[None, :] < delays[:, None]


# 输入动作/噪声 [B,T,A] 和时间 [B]；无 RTC 时返回时间 [B]，有前缀时返回逐 token 时间 [B,T]。
# Inputs are actions/noise [B,T,A] and time [B]; return time [B] without RTC, or per-token time [B,T] when a prefix exists.
def make_training_inputs(
    actions: Tensor, noise: Tensor, time: Tensor, prefix_mask: Tensor | None
) -> tuple[Tensor, Tensor, Tensor]:
    """Keep the known prefix clean and identify it with per-action t=0."""
    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    # 所有位置先计算同一种 flow 速度目标，已知前缀不改 target；后续聚合损失时会用 prefix_mask 排除它们。
    # Compute the same flow velocity target at all positions, including known prefixes; later loss reduction excludes prefix positions using prefix_mask.
    target = noise - actions
    if prefix_mask is None:
        return x_t, target, time
    if prefix_mask.shape != actions.shape[:2]:
        raise ValueError("RTC prefix mask must match the action batch and horizon")
    prefix_mask = prefix_mask.to(device=actions.device, dtype=torch.bool)
    x_t = torch.where(prefix_mask[:, :, None], actions, x_t)
    # 前缀保持干净动作并以 t=0 标记，后缀保留样本采样时间；时间因此从 [B] 扩展为 [B,T]。
    # Keep clean prefix actions at t=0 and sampled time on the suffix, expanding time from [B] to [B,T].
    model_time = torch.where(prefix_mask, 0.0, time[:, None])
    return x_t, target, model_time


# 把上次动作块中已承诺执行的前 d 步写入填充宽度的 [B,T,A] 前缀，并返回 [B,T] mask；d=0 不约束动作。
# Copy the first d committed actions of the previous chunk into padded [B,T,A] prefix storage and return [B,T] masks; d=0 leaves actions unconstrained.
def prepare_inference_prefix(
    noise: Tensor,
    previous: Tensor | None,
    inference_delay: int,
    training_max_delay: int,
    execution_horizon: int | None = None,
) -> tuple[Tensor | None, Tensor | None]:
    """Validate an explicit trained-RTC request and pad its normalized prefix.

`previous` starts at the observation's action time, before the delay elapses.
Its values must already be in the policy's normalized action space.
"""
    if training_max_delay <= 0:
        raise ValueError("rtc_mode='trained' requires rtc_training_max_delay > 0 in the checkpoint")
    if isinstance(inference_delay, bool) or not isinstance(inference_delay, int):
        raise ValueError("inference_delay must be an integer number of controller steps")
    horizon = noise.shape[1]
    if not 0 <= inference_delay <= training_max_delay or inference_delay >= horizon:
        raise ValueError(
            f"inference_delay={inference_delay} must be within the trained range "
            f"[0, {training_max_delay}] and smaller than chunk_size={horizon}"
        )
    if execution_horizon is not None:
        if isinstance(execution_horizon, bool) or not isinstance(execution_horizon, int):
            raise ValueError("execution_horizon must be an integer number of controller steps")
        if not training_max_delay <= execution_horizon <= horizon - training_max_delay:
            raise ValueError(
                "Trained RTC requires rtc_training_max_delay <= execution_horizon <= "
                "chunk_size - rtc_training_max_delay"
            )
    # inference_delay 是控制器离散步数，不是毫秒；positive delay 要求提供从当前观测对应动作时刻开始的 previous。
    # inference_delay counts discrete controller steps, not milliseconds; positive delay requires previous actions beginning at the current observation action time.
    if inference_delay == 0:
        return None, None
    if previous is None:
        raise ValueError("A positive trained RTC delay requires prev_chunk_left_over")
    previous = previous.to(device=noise.device, dtype=noise.dtype)
    if previous.ndim == 2:
        previous = previous.unsqueeze(0)
    if previous.ndim != 3:
        raise ValueError("prev_chunk_left_over must have shape (B, T, A) or (T, A)")
    if previous.shape[0] == 1 and noise.shape[0] > 1:
        previous = previous.expand(noise.shape[0], -1, -1)
    if previous.shape[0] != noise.shape[0]:
        raise ValueError("RTC prefix batch size does not match the policy batch")
    if previous.shape[1] < inference_delay:
        raise ValueError("RTC prefix is shorter than inference_delay")
    if not 0 < previous.shape[2] <= noise.shape[2]:
        raise ValueError("RTC prefix action dimension exceeds the model action dimension or is empty")
    if not torch.isfinite(previous[:, :inference_delay]).all():
        raise ValueError("RTC prefix contains NaN or Inf")
    # 仅复制已承诺的前 d 步，其他位置填零且 mask=False；未承诺的上一块后缀不会成为本次硬约束。
    # Copy only the d committed steps; other positions are zero with false masks, so the previous uncommitted suffix does not become a hard constraint.
    prefix = torch.zeros_like(noise)
    prefix[:, :inference_delay, : previous.shape[2]] = previous[:, :inference_delay]
    mask = torch.arange(horizon, device=noise.device)[None, :] < inference_delay
    return prefix, mask.expand(noise.shape[0], -1)


# 先屏蔽已知前缀和 padding，再按每个样本的有效时间步乘真实动作维度求均值，输出 [B]。
# Exclude known prefixes and padding, then average over each example valid steps times physical action dimensions, returning [B].
def masked_suffix_mean(losses, prefix_mask=None, action_is_pad=None):
    """Per-example mean over real action dimensions and free, nonpadded steps.

    Callers slice losses to the physical action dimension before this reduction.
    An empty target contributes zero, with a connected zero gradient.
    """
    if losses.ndim != 3:
        raise ValueError('Expected [B,T,A] losses')
    valid = torch.ones(losses.shape[:2], dtype=torch.bool, device=losses.device)
    for mask in [prefix_mask, action_is_pad]:
        if mask is not None:
            if mask.shape != valid.shape:
                raise ValueError('Action masks must match [B,T]')
            valid &= ~mask.to(device=losses.device, dtype=torch.bool)
    # 使用 where 而非 losses*mask 可避免被屏蔽位置的 NaN 经 0*NaN 污染总和；空有效集返回连接计算图的零。
    # Use where rather than losses*mask to prevent masked NaNs contaminating the sum through 0*NaN; empty valid sets return graph-connected zero.
    values = torch.where(valid[..., None], losses, 0.0).sum((1, 2))
    denominator = valid.sum(1).clamp_min(1) * losses.shape[-1]
    return values / denominator


# 只在 mask=True 的已承诺位置替换预测计划，其他位置保持模型输出；已知前缀 detach，未知后缀继续传梯度。
# Replace the predicted plan only at committed mask positions; detach known-prefix values while preserving gradients through the predicted free suffix.
def clamp_plan_prefix(plan, prefix=None, mask=None):
    """Hard constrain the physical plan without reading an uncommitted suffix."""
    if (prefix is None) != (mask is None):
        raise ValueError('Prefix and mask must be provided together')
    if prefix is None:
        return plan
    if plan.ndim != 3 or prefix.ndim != 3 or mask.ndim != 2:
        raise ValueError('Expected plan/prefix [B,T,A] and mask [B,T]')
    if prefix.shape[:2] != mask.shape or prefix.shape[0] != plan.shape[0] or prefix.shape[2] != plan.shape[2] or prefix.shape[1] < plan.shape[1]:
        raise ValueError('Prefix shape is incompatible with the physical plan')
    mask = mask.to(device=plan.device, dtype=torch.bool)[:, :plan.shape[1]]
    known = prefix.to(device=plan.device, dtype=plan.dtype)[:, :plan.shape[1]]
    if not torch.isfinite(known[mask]).all():
        raise ValueError('Committed prefix contains NaN or Inf')
    # 已知动作视为外部常量，不反向优化旧动作；只训练新规划尚未承诺的后缀，避免未知 GT 信息泄漏。
    # Treat committed actions as external constants, never optimizing past commands; train only the new uncommitted plan suffix to prevent unknown-ground-truth leakage.
    return torch.where(mask[..., None], known.detach(), plan)


# 兼容旧 Gemma 只接受逐样本 AdaRMS 条件的接口；展平归一化的 B/T 轴，不改变注意力序列组织。
# Adapt legacy sample-conditioned Gemma AdaRMS by flattening B/T only for normalization, never changing attention sequence structure.
def apply_per_token_norm(norm, hidden, cond=None):
    """Adapt sample-conditioned legacy Gemma AdaRMS to per-action flow time.

    Flatten only the token-wise normalization, never attention. No dependency
    patch or new parameters are required; scalar-time behavior is unchanged.
    """
    if cond is None or cond.ndim == 2:
        return norm(hidden, cond=cond)
    if cond.ndim != 3 or hidden.ndim != 3 or cond.shape[:2] != hidden.shape[:2]:
        raise ValueError('Per-token AdaRMS condition must match hidden [B,T,D]')
    batch, length, width = hidden.shape
    # 把每个 token 临时当作单独样本，使它得到自己的 AdaRMS 时间条件；归一化后恢复 [B,T,D]，注意力仍在原始序列上计算。
    # Temporarily treat each token as a separate sample for its own AdaRMS time condition, then restore [B,T,D]; attention still operates on the original sequence.
    output, gate = norm(hidden.reshape(batch*length, 1, width),
                        cond=cond.reshape(batch*length, cond.shape[-1]))
    output = output.reshape(batch, length, width)
    return output, None if gate is None else gate.reshape(batch, length, width)

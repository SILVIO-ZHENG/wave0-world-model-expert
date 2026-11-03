# 源码阅读指南 / Source reading guide

## 先建立整体路线 / Start with the overall route

先读 [README](../README.md) 的架构图，再读 [ARCHITECTURE](ARCHITECTURE.md)。牢记两条路径：当前上下文 KV 直接到 Expert；潜在动作经过世界模型形成未来条件再到 Expert。

Read the architecture diagram in the [README](../README.md), then [ARCHITECTURE](ARCHITECTURE.md). Keep both paths in mind: direct current-context KV to the Expert, and latent actions through the world model into future conditioning.

## 路线一：只看 Expert Q 和 VLM KV / Route 1: Expert Q and VLM KV

1. [`attention.py`](../attention.py) → `compute_expert_layer_with_context_kv`。
   看 `q_proj` 从 Expert 未来/动作状态生成 Q，`torch.cat(..., dim=2)` 拼接 token，再看 `eager_attention_forward` 的输入。
   Inspect Expert Q projections, token-axis concatenation, and arguments to `eager_attention_forward`.
2. 同文件 → `compute_vlm_layer_with_context_kv`。
   看 VLM 每层的归一化、K/V 投影、RoPE 和 `:context_length` 裁剪；注意没有从最终隐状态重新生成全部层 KV。
   Trace VLM layer normalization, KV projections, RoPE and context slicing; all-layer KV is not reconstructed from final hidden states.
3. [`vlm_expert.py`](../vlm_expert.py) → `forward_vlm_with_context_kv`、`forward_expert_with_context_kv`。
   确认 `context_key_values[l]` 与 Expert 第 `l` 层一一配对。
   Confirm that `context_key_values[l]` pairs with Expert layer `l`.
4. [`model_training.py`](../model_training.py) → `forward_expert_with_cached_context`。
   检查 context/future/action 三块掩码，以及为什么 Expert 前向只保留 future/action Query 行。
   Inspect the three attention blocks and why only future/action query rows remain in the Expert pass.
5. [`tensor_utils.py`](../tensor_utils.py) → `make_att_2d_masks`。
   累计块编号决定可见性；`pad_masks` 另行排除无效位置。
   Cumulative block IDs define visibility; `pad_masks` separately excludes invalid positions.

读完应该能解释：动作 Q 的来源、VLM K/V 的来源、拼接维度、对应层关系，以及为什么动作 Q 还能读取未来和动作自身 K/V。

You should then be able to explain query/key/value origins, concatenation axes, layer matching, and why action queries also read future/action KV.

## 路线二：看世界模型加在哪里 / Route 2: World-model integration

1. [`policy_training.py`](../policy_training.py) → `forward_stage2`。
   顺着 `encode_wam_prefix_for_training`、`world_model`、`wam_future_condition_resampler`、`forward_expert_with_cached_context` 四个调用看数据流。
   Follow those four calls to trace the complete data flow.
2. [`model_tokens.py`](../model_tokens.py) → `embed_wam_prefix`、`encode_wam_prefix_for_training`。
   看规划查询的追加、投影，以及规划 KV 为什么不直接交给 Expert。
   Trace appended planner queries, their projection, and why planner KV is excluded from Expert context.
3. [`world_model.py`](../world_model.py) → `LatentWorldModel.forward`。
   看当前视觉、状态、动作和未来查询如何组成序列，以及 `predicted_delta` 如何恢复相机与空间维度。
   Inspect the current/state/action/future-query sequence and the camera/spatial layout of `predicted_delta`.
4. 同文件 → `WAMFutureConditionResampler.forward`。
   看预测未来怎样池化并映射到 Expert 宽度。
   Inspect pooling and projection into Expert width.
5. 回到 `forward_stage2`，区分真实动作教师、VLM 预测潜在动作和最终 Expert 动作三种表示。
   Return to `forward_stage2` and distinguish teacher-action encodings, VLM latent plans, and final Expert actions.

## 路线三：看两个阶段和梯度 / Route 3: Training stages and gradients

1. [`policy.py`](../policy.py) → `_configure_trainability`、`train`、`forward`。
   区分参数冻结、模块 train/eval 模式和阶段调度，它们解决不同问题。
   Separate parameter freezing, train/eval mode and stage dispatch; they serve different purposes.
2. [`stage1_model.py`](../stage1_model.py) → `WAVE0Stage1Model`。
   第一阶段只构造世界模型训练需要的组件，冻结视觉骨干。
   Stage 1 builds only the components required for world-model training and freezes vision.
3. [`policy_training.py`](../policy_training.py) → `forward_world_model`。
   看真实动作如何进入世界模型，以及未来、变化量、错误动作排序损失如何组成。
   Trace ground-truth action conditioning and future/delta/wrong-action ranking losses.
4. 同文件 → `forward_stage2`、`_latent_action_alignment_per_sample`。
   找到教师的 `no_grad()`、世界预测保留输入梯度的位置和可配置的未来条件 `detach()`。
   Locate teacher `no_grad()`, differentiable world-model inputs, and configurable future-condition detachment.
5. [`model_training.py`](../model_training.py) → `forward_expert_with_cached_context`。
   看 flow matching 目标以及为何返回逐元素损失，再由策略层处理 padding 和 RTC 前缀。
   Inspect flow targets and elementwise losses before policy-level padding/RTC reduction.

## 路线四：看推理如何复用缓存 / Route 4: Inference cache reuse

1. [`policy_inference.py`](../policy_inference.py) → `predict_action_chunk`、`select_action`。
   从策略入口观察图像、状态、RTC 参数和动作队列。
   Start from policy inputs, RTC arguments and the action queue.
2. [`model_inference.py`](../model_inference.py) → `_prepare_wam_inference_context`。
   看 VLM 缓存、裁剪规划区域、世界预测和未来条件预填。
   Trace VLM prefill, planner cropping, world prediction and future prefill.
3. 同文件 → `sample_actions`、`denoise_step`。
   当前动作估计随去噪时间更新；上下文和未来条件缓存被复用。
   Action estimates evolve over denoising time while context and future caches are reused.

## 每个文件适合回答的问题 / Questions answered by each module

| 文件 / File | 阅读问题 / Question |
|---|---|
| `attention.py` | Q/K/V 在哪个位置产生、拼接和读取？ / Where are Q/K/V produced, merged and read? |
| `vlm_expert.py` | 两个网络如何逐层配合？ / How do VLM and Expert cooperate layer by layer? |
| `world_model.py` | 潜在动力学怎样预测未来？ / How do latent dynamics predict the future? |
| `stage1_model.py` | 第一阶段需要哪些组件？ / Which components does Stage 1 instantiate? |
| `model.py` | 模型初始化时有哪些子模块？ / Which submodules are initialized? |
| `model_tokens.py` | 图像、语言、计划和动作怎样变成 token？ / How are images, text, plans and actions tokenized? |
| `model_training.py` | 动作生成训练的目标和 mask 是什么？ / What are the action-training targets and masks? |
| `model_inference.py` | 如何缓存固定条件并多步去噪？ / How are fixed conditions cached for denoising? |
| `policy.py` | 训练阶段、冻结和调用入口怎样管理？ / How are stages, freezing and entry points managed? |
| `policy_training.py` | 全部损失在哪里汇合？ / Where are losses combined? |
| `policy_data.py` | 未来图像与动作监督怎样准备？ / How are future images and action supervision prepared? |
| `policy_checkpoint.py` | 基础策略和第一阶段权重怎样合并？ / How are base and Stage-1 checkpoints combined? |
| `policy_inference.py` | 动作 chunk 怎样进入执行队列？ / How do action chunks enter the execution queue? |
| `tensor_utils.py` | 通用形状处理与掩码规则是什么？ / What are the shared shape and masking rules? |
| `configuration_wave0.py` | 哪些模型尺寸、阶段与损失开关可以配置？ / Which model dimensions, stages and loss switches are configurable? |
| `rtc_training.py` | 已知动作前缀怎样约束训练和规划？ / How do committed action prefixes constrain training and planning? |
| `_runtime.py` | 哪些能力仍依赖原项目及修改版依赖？ / Which capabilities require the original project and modified dependencies? |
| `__init__.py` | 对外导出哪些类？ / Which classes are exported? |

## 容易混淆的地方 / Common confusions

- **规划 query token 与注意力 Q 不同。** 前者是可学习输入位置，后者是当前层线性投影产生的矩阵。
  **Planner query tokens differ from attention Q.** The former are learned input positions; the latter is a layerwise projected tensor.
- **冻结参数与关闭自动求导不同。** 冻结世界模型仍可让损失对其输入求导。
  **Frozen parameters differ from disabled autograd.** A frozen world model may still transmit gradients to inputs.
- **世界模型未来查询与 VLM 规划查询不同。** 它们属于不同模块，分别用于读取未来视觉输出和提出潜在动作计划。
  **World-model future queries differ from VLM planner queries.** They belong to different modules and retrieve future visual outputs versus proposing latent action plans.
- **未来条件不是直接替换 VLM KV。** Expert 同时消费当前上下文 KV 和未来条件 token。
  **Future conditions do not replace VLM KV.** The Expert consumes both context KV and future tokens.
- **完整 forward 不等于第二阶段训练入口。** Stage 2 的策略入口是 `forward_stage2`，动作专家路径是 `forward_expert_with_cached_context`。
  **The generic model forward is not the Stage-2 training entry.** Stage 2 uses `forward_stage2` and `forward_expert_with_cached_context`.
- **静态语法树一致不代表完整运行验证。** 依赖、MRO、权重兼容与数值行为还需要对应环境检查。
  **Static AST equivalence is not complete execution validation.** Dependencies, MRO, checkpoint compatibility and numerical behavior still require environment-specific checks.

## 阅读后的最小自检 / Minimal understanding checklist

能从 `forward_stage2` 一直追到实际 attention helper，并回答：当前图像经过几条编码路径、潜在动作由谁预测、冻结世界模型的输入是什么、未来 token 如何形成、Expert 每层读取哪一层 KV，以及哪些梯度默认被 detach。

Follow `forward_stage2` to the attention helper and explain the visual encoding paths, latent-action producer, frozen dynamics inputs, future-token construction, matching-layer KV, and default detachment boundaries.

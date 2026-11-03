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

"""外部依赖集中导入。 / Centralized third-party and original-package imports."""
import builtins
import dataclasses
import logging
import math
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict

# 统一外部依赖只解决导入位置，不会延迟加载整个 PyTorch 或创建无依赖运行环境。
# Centralizing imports does not lazily load all of PyTorch or make this package dependency-free.
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from typing_extensions import Unpack

from lerobot.utils.import_utils import _transformers_available

# 类型检查或已安装 Transformers 时导入模型实现；缺失依赖时占位为 None，不代表模型可执行。
# Import model implementations for typing or when Transformers is available; None placeholders do not make models executable without dependencies.
if TYPE_CHECKING or _transformers_available:
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.gemma import modeling_gemma
    from transformers.models.gemma.modeling_gemma import GemmaForCausalLM
    from transformers.models.paligemma.modeling_paligemma import (
        PaliGemmaForConditionalGeneration,
        PaliGemmaMultiModalProjector,
    )
    from transformers.models.siglip.modeling_siglip import SiglipVisionModel
else:
    CONFIG_MAPPING = None
    modeling_gemma = None
    GemmaForCausalLM = None
    PaliGemmaForConditionalGeneration = None
    PaliGemmaMultiModalProjector = None
    SiglipVisionModel = None

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
# 配置和训练式 RTC 工具随当前包提供；LeRobot 的基础策略、推理 RTC 与常量仍是外部依赖。
# Configuration and trained-RTC helpers ship with this package; base policies, inference RTC and constants remain LeRobot dependencies.
from .configuration_wave0 import (
    DEFAULT_IMAGE_SIZE,
    WAVE0Config,
    world_model_future_image_key,
)
from .rtc_training import (
    apply_per_token_norm,
    clamp_plan_prefix,
    make_training_inputs,
    masked_suffix_mean,
    prepare_inference_prefix,
    sample_training_prefix_mask,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    OPENPI_ATTENTION_MASK_VALUE,
)

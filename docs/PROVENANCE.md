# 来源与发布范围 / Provenance and publication scope

本项目从本地 WAVE-0 衍生实现拆分，保留源文件中的版权声明和上游 Apache-2.0 许可证。

This project is extracted from a local derivative of WAVE-0. Original copyright notices and the upstream Apache-2.0 license are retained.

| 项目 / Item | 记录 / Record |
|---|---|
| 上游 / Upstream | https://github.com/AutoLab-SAI-SJTU/WAVE-0 |
| 原始工作区的 Git 基线 / Original checkout Git baseline | `eaeb49294313140b6924782d56e6a5457ce10e13` |
| 实际拆分源文件 / Actual extraction source | `src/lerobot/policies/wave0/modeling_wave0.py` in the local derivative |
| 拆分源文件 SHA-256 / Extraction-source SHA-256 | `8dcc68acddc13a5214dba3cc2ecd1e4dde5cb040daa80a459b3c5e10a3bb6a83` |
| 特定 Transformers 依赖 / Specific Transformers dependency | `huggingface/transformers@dcddb970176382c0fcf4521b0c0e6fc15894dfe0` |

Git 基线不是完整本地实现的版本标识：原工作区有未提交的 WAM/RTC 等变更。因此，不能仅按此上游提交重建全部本地改动；原始建模文件以其 SHA-256 作为精确来源证据。

The Git baseline does not identify the full local implementation: the working tree contained uncommitted WAM/RTC changes. That upstream commit alone cannot reproduce all local modifications; the modeling-file SHA-256 identifies the precise extraction source.

## 此仓库做了哪些修改？ / Changes made here

- 按职责拆分函数和类，以显式相对导入和 mixin 组合模块。
  Split functions and classes by responsibility using explicit relative imports and mixin composition.
- 添加详细中文及英文说明，保留原模型方法的执行逻辑。
  Add detailed Chinese and English explanations while retaining original model-method logic.
- 随附对应的 `configuration_wave0.py` 与 `rtc_training.py`，将 `_runtime.py` 中这两处依赖改为包内相对导入。
  Include the matching configuration and RTC helpers; change those two `_runtime.py` imports to package-relative imports.
- 提供包配置、架构图、源码阅读路线和纯标准库结构验证工具。
  Provide packaging metadata, architecture diagrams, source-reading routes and standard-library structural checks.

此处没有更换注意力算法、重新训练模型或引入新的性能结果。源码拆分不会自动替换原训练项目的入口。

This work does not replace the attention algorithm, retrain models or introduce new performance results. Extracting modules does not automatically change the original training entry points.

## 发布范围 / Published material

只发布模块源码、随附配置/RTC 工具、许可证及说明文档。没有上传模型权重、训练数据、账号凭据或个人机器路径。

Publication contains module source, the included configuration/RTC helpers, license and documentation. It excludes model weights, datasets, account credentials and personal-machine paths.

## 分支 / Branches

- `main`：完整集成版本。 / Complete integrated version.
- `docs/architecture`：架构和数据流的后续讲解。 / Architecture and data-flow documentation work.
- `study/attention`：注意力、KV 与源码阅读的后续学习。 / Attention, KV and source-reading work.

三个分支从同一初始发布快照创建，便于后续分别改进；这不是三套不同的算法。

All three branches start at the same initial publication snapshot for future focused work; they do not represent three different algorithms.

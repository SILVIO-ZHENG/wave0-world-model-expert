# 验证范围 / Validation scope

本仓库提供模块化源码和解释，验证重点是拆分是否保留原实现，以及本地模块连接是否完整。验证工具仅使用 Python 标准库，不加载 PyTorch、Transformers 或模型权重。

This repository provides modular source and explanations. Validation focuses on preserving the extracted implementation and connecting local modules correctly. The verifier uses only Python's standard library and does not load PyTorch, Transformers or model weights.

## 本地结构检查 / Local structural checks

在仓库根目录执行：

Run from the repository root:

```powershell
python tools/verify_structure.py
```

工具执行以下检查：

The tool performs these checks:

- 解析并在内存中编译 Python 源码，发现语法与编译错误；不生成 `__pycache__`。
  Parse and compile Python source in memory to catch syntax and compilation errors, without creating `__pycache__`.
- 使用符号表检查模块全局名字，包括函数内部引用；识别导入、赋值、函数/类定义以及 Python 内置名字。
  Use symbol tables to check module-global references, accounting for imports, assignments, definitions and Python-provided names.
- 检查根目录模块的本地相对导入是否存在，并检查循环引用。
  Verify that relative imports among root-level modules exist and contain no cycles.

这些检查不执行实际模型。全局名字静态检查不能证明所有运行时路径都有效；外部包的接口、类型、张量形状和设备兼容性仍需要实际环境验证。

These checks do not execute the model. Static name checks do not prove every runtime path is valid; external interfaces, types, tensor shapes and device compatibility still require the real environment.

## 可选的原实现对比 / Optional original-source comparison

如果本地保留了拆分前的原文件，可以通过参数指定其位置。工具不依赖某台电脑的固定目录。

If the original monolithic source is available locally, supply its location explicitly. The tool does not depend on a machine-specific directory.

```powershell
python tools/verify_structure.py --original path/to/modeling_wave0.py
```

对比包含原始顶层函数、未拆分类，以及迁移到 mixin 的 `PI05Pytorch`、`WAVE0Policy` 方法。针对本次来源版本，共对比 84 个定义/方法。新增的配置和 RTC 支持文件不属于原大文件，单独纳入语法、全局名字和导入检查。

Comparison covers original top-level functions, intact classes, and `PI05Pytorch`/`WAVE0Policy` methods moved into mixins. For the source version used in this extraction, the comparison covers 84 declarations/methods. Added configuration and RTC support files are not part of the original monolith and receive the syntax, global-name and import checks separately.

AST 一致表示保留了原函数体与装饰器的语法结构，注释和格式差异不会影响结果。类组装方式因 mixin 拆分而不同，因此两个大类按方法对比；这不能单独证明 GPU 数值等价、序列化兼容或完整训练结果。

AST equality means the original function bodies and decorators retain their syntactic structure; comments and formatting do not affect the result. Because mixins change class assembly, the two large classes are compared method by method. This alone does not prove GPU numerical equivalence, serialization compatibility or identical full-training results.

## 尚未完成的运行验证 / Runtime validation still outstanding

本地可用的 CPU PyTorch 环境缺少完整 LeRobot/Transformers 训练依赖，其 PyTorch 版本也超出来源项目声明的版本范围。它不能代表实际训练环境。

The available local CPU PyTorch environment lacks the complete LeRobot/Transformers training dependencies, and its PyTorch version falls outside the source project's declared range. It is not a substitute for the training environment.

以下结果没有在本次源码整理中得到验证：

The following have not been verified during this source extraction:

- 在匹配的完整依赖环境中导入、实例化和加载真实检查点。
  Importing, instantiating and loading real checkpoints with the complete matching dependency environment.
- GPU 前向、反向及训练/推理数值等价。
  GPU forward/backward execution and numerical equivalence of training or inference.
- 云端端到端训练、机器人实机性能和论文指标复现。
  End-to-end cloud training, real-robot performance and reproduction of paper metrics.

需要运行时验证时，应使用来源项目的匹配依赖、合法可用的模型权重与数据，优先验证单批前向/反向和关键缓存路径，再决定是否扩大训练检查。

For runtime validation, use matching source-project dependencies and available model weights/data. Start with a single-batch forward/backward pass and the relevant cache paths before expanding training checks.


## 本次发布的附加检查 / Additional publication checks

- 包构建通过，wheel 包包含全部 18 个根 Python 模块，压缩包完整性检查通过。
  Wheel building passed; all 18 root Python modules are present and archive integrity passed.
- 独立加载 RTC helper 的三项 CPU 行为检查通过：已知前缀保持 clean 且时间为零、loss 排除前缀与 padding 且反向梯度正确、物理规划硬约束保留已承诺值并阻断相应梯度。
  Three isolated RTC CPU checks passed: clean zero-time prefixes, correct prefix/padding-masked loss and gradients, and committed-prefix clamping with blocked gradients.
- CPU 检查使用 torch 2.14.0+cpu，超出原始 torch<2.11 依赖范围，不能当作完整训练环境兼容性证明。
  CPU checks used torch 2.14.0+cpu, outside the original torch<2.11 range; they do not establish compatibility of the complete training environment.

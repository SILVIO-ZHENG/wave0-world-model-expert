#!/usr/bin/env python
"""检查拆分结构，不加载模型依赖或写入字节码。

Check the modular source without loading model dependencies or writing bytecode.
"""

import argparse
import ast
import builtins
from pathlib import Path
import symtable
import sys


def parse_sources(root):
    """解析并编译源文件；compile 只在内存中运行，不生成 __pycache__。

    Parse and compile sources in memory, without producing __pycache__ files.
    """
    trees = {}
    for path in sorted(root.rglob("*.py")):
        if any(part in {".git", "__pycache__", ".venv", "build", "dist"} for part in path.relative_to(root).parts):
            continue
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(path))
        compile(tree, str(path), "exec", dont_inherit=True)
        trees[path] = (source, tree)
    if not trees:
        raise ValueError("没有找到 Python 文件。 / No Python source files found.")
    return trees


def check_global_names(trees):
    """用 Python 符号表检查缺失的模块级名字，包含函数内的全局引用。

    Use Python symbol tables to detect missing module globals, including reads
    inside functions. Imported and locally assigned names are recognized.
    """
    # 这些名字由 Python 模块加载器提供，不需要源码自行声明。
    # Python's module loader provides these names without source assignments.
    implicit = set(dir(builtins)) | {
        "__file__", "__name__", "__package__", "__spec__", "__loader__",
        "__cached__", "__builtins__", "__doc__", "__annotations__",
    }
    for path, (source, _) in trees.items():
        table = symtable.symtable(source, str(path), "exec")
        bound = {
            symbol.get_name() for symbol in table.get_symbols()
            if symbol.is_imported() or symbol.is_assigned() or symbol.is_namespace()
        }
        referenced = set()

        def visit(scope):
            for symbol in scope.get_symbols():
                if symbol.is_global() and symbol.is_referenced():
                    referenced.add(symbol.get_name())
            for child in scope.get_children():
                visit(child)

        visit(table)
        missing = referenced - bound - implicit
        if missing:
            raise ValueError(f"{path.name}: 未定义的全局名字 / unresolved globals: {sorted(missing)}")


def check_import_graph(root, trees):
    """检查根目录源码模块的相对导入，不导入 PyTorch 或 LeRobot。

    Check relative imports among root-level source modules without importing
    PyTorch or LeRobot. External-package dependencies are outside this check.
    """
    modules = {path.stem: tree for path, (_, tree) in trees.items() if path.parent == root}
    graph = {name: set() for name in modules}
    for name, tree in modules.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level != 1:
                continue
            targets = [node.module.split(".")[0]] if node.module else [alias.name for alias in node.names]
            for target in targets:
                if target not in modules:
                    raise ValueError(f"{name}: 缺少相对导入模块 / missing relative module: {target}")
                graph[name].add(target)

    visited = set()

    def visit(name, stack):
        if name in stack:
            raise ValueError("循环导入 / import cycle: " + " -> ".join(stack + [name]))
        if name in visited:
            return
        for dependency in sorted(graph[name]):
            visit(dependency, stack + [name])
        visited.add(name)

    for name in graph:
        visit(name, [])
    return graph


def compare_original(original, root, trees):
    """逐个比较原函数、完整类和拆分后类方法的 AST。

    Compare original functions, intact classes and methods moved into mixins.
    Added configuration/RTC helpers are not part of the original monolith.
    """
    original_tree = ast.parse(original.read_text(encoding="utf-8-sig"))
    declarations = {}
    for path, (_, tree) in trees.items():
        if path.parent != root:
            continue
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in declarations:
                    raise ValueError(f"重复的顶层定义 / duplicate declaration: {node.name}")
                declarations[node.name] = node

    def equal(before, after, label):
        if ast.dump(before, include_attributes=False) != ast.dump(after, include_attributes=False):
            raise ValueError(f"实现 AST 不一致 / implementation AST differs: {label}")

    def methods_with_mixins(cls):
        methods = {}
        # 类自身优先；随后按声明顺序读取 mixin。这里只检查实际迁出的直接父 mixin。
        # Prefer the class itself, followed by directly declared extraction mixins.
        parents = [cls] + [
            declarations[base.id] for base in cls.bases
            if isinstance(base, ast.Name) and base.id.endswith("Mixin") and base.id in declarations
        ]
        for parent in parents:
            for node in parent.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name in methods:
                        raise ValueError(f"重复的类方法 / duplicate class method: {cls.name}.{node.name}")
                    methods[node.name] = node
        return methods

    checked = 0
    for node in original_tree.body:
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in declarations:
            raise ValueError(f"缺少原始定义 / missing original declaration: {node.name}")
        target = declarations[node.name]
        if node.name not in {"PI05Pytorch", "WAVE0Policy"}:
            equal(node, target, node.name)
            checked += 1
            continue
        available = methods_with_mixins(target)
        for method in node.body:
            if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if method.name not in available:
                    raise ValueError(f"缺少原始方法 / missing original method: {node.name}.{method.name}")
                equal(method, available[method.name], f"{node.name}.{method.name}")
                checked += 1
    return checked


def main():
    # Windows 重定向输出时也使用 UTF-8，避免中文检查结果触发编码异常。
    # Keep redirected Windows output in UTF-8 so diagnostics remain readable.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="项目根目录 / repository root")
    parser.add_argument("--original", type=Path,
                        help="可选的原始 modeling_wave0.py / optional original monolithic source")
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        trees = parse_sources(root)
        check_global_names(trees)
        graph = check_import_graph(root, trees)
        print(f"通过：{len(trees)} 个 Python 文件解析/编译，无缺失全局名字，无本地循环导入。")
        print(f"Passed: {len(trees)} Python sources parsed/compiled; no unresolved globals or local import cycles.")
        print(f"本地模块数 / local modules: {len(graph)}")
        if args.original:
            checked = compare_original(args.original, root, trees)
            print(f"原实现 AST 对比通过 / original implementation AST matches: {checked} declarations/methods")
        else:
            print("未指定 --original，跳过原文件对比。 / No --original supplied; source comparison skipped.")
        print("这里只验证静态结构；没有验证依赖安装、GPU 数值结果或完整训练。")
        print("Static structure only: dependency installation, GPU numerics and full training are not validated.")
    except (OSError, SyntaxError, ValueError) as error:
        print(f"检查失败 / verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

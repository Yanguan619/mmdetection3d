"""spconv_gemm 的 AscendC 后端封装 —— vendored torch 版中的存根。

本目录是从 unum_ops.spconv 迁移的纯 torch 参考实现，刻意不依赖
unum_ops 包。原 AscendC Cube 加速内核（torch.ops.unum.spconv_gemm）
存在输出 stride 缺陷且实测慢于 aclnn torch 2D GEMM 路径，故这里
不携带。``available()`` 恒为 False，``conv.py::_gather`` 的
``_USE_ASCENDC`` 分支永远走 torch 回退（与 unum_ops 默认行为一致）。
"""
from __future__ import annotations


def available() -> bool:
    """本 vendored 版本始终不提供 AscendC 扩展（torch-only）。"""
    return False


def spconv_gemm(feats, weight, bias):
    """AscendC 内核不可用：调用即报错（不应触发，见模块 docstring）。"""
    raise RuntimeError(
        "spconv_ascendc.spconv_gemm 不可用：vendored torch 版不含 AscendC 内核，"
        "请走 conv.py 的 torch 2D GEMM 路径（默认）。")


__all__ = ["available", "spconv_gemm"]

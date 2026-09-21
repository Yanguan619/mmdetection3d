"""NPU-specific runtime patches for BEVFusion on Ascend 310P.

PatchMerging fix: mmdet's Swin PatchMerging uses nn.Unfold(k=2, s=2) which
computes WRONG results on NPU in processes that imported mmcv.ops/mmdet3d
before the first unfold call (first-4-columns data corruption, cos~0.98).
The reshape/permute replacement below is bit-exact with nn.Unfold(k=2,s=2)
on CPU and uses only memory ops, avoiding the buggy NPU unfold kernel.

Apply patch_swin_patchmerging() and patch_get_geometry_cpu() BEFORE building /
running the model (after mmdet3d imports is fine).
"""

import os

import numpy as np
import torch

_ORIG_GET_GEOMETRY = None  # patch_get_geometry_cpu() 捕获的原始实现


def patch_spconv_torch():
    """用 vendored 纯 torch spconv 替换官方 spconv-cu114（mmdet3d 加载之后调用）。

    将 mmdet3d.registry.MODELS 中注册的稀疏卷积模块覆盖为
    ``projects/BEVFusion/bevfusion/ops/spconv`` 的纯 torch 版本（迁移自
    unum_ops.spconv），并把 mmdet3d 各模块 import 的 SparseSequential
    替换为其原生版本（forward 已兼容官方 SparseModule）。
    """
    import importlib

    from projects.BEVFusion.bevfusion.ops import spconv as torch_spconv

    from mmdet3d.registry import MODELS

    MODELS._register_module(torch_spconv.conv.SubMConv3d, "SubMConv3d", force=True)
    MODELS._register_module(torch_spconv.conv.SparseConv3d, "SparseConv3d", force=True)
    MODELS._register_module(
        torch_spconv.conv.SparseInverseConv3d, "SparseInverseConv3d", force=True
    )
    MODELS._register_module(torch_spconv.conv.SubMConv2d, "SubMConv2d", force=True)
    MODELS._register_module(torch_spconv.conv.SparseConv2d, "SparseConv2d", force=True)
    MODELS._register_module(
        torch_spconv.conv.SparseInverseConv2d, "SparseInverseConv2d", force=True
    )

    _torch_seq = torch_spconv.sparse_modules.SparseSequential
    for _mod_name in [
        "mmdet3d.models.layers.sparse_block",
        "mmdet3d.models.middle_encoders.sparse_encoder",
    ]:
        _mod = importlib.import_module(_mod_name)
        _mod.SparseSequential = _torch_seq
    print("[npu_patches] spconv -> vendored torch (MODELS 注册 + SparseSequential 替换)")
    patch_spconv_load5d()


def patch_spconv_load5d():
    """load_state_dict 时把官方 BEVFusion ckpt 的 5D 稀疏卷积权重转成 4D。

    官方 spconv2 checkpoint 的卷积权重布局为 KRSC ``(out, D, H, W, in)``，
    vendored torch spconv 参数为标准 PyTorch ``(out, in, D, H, W)``。若不加
    处理，``load_state_dict(strict=False)`` 会因形状不匹配静默丢弃这些 key
    （sparse encoder 权重全丢，结果失真）。

    钩子装在 vendored ``SparseConvolution`` 基类的 ``_load_from_state_dict``
    上：传入权重为 5D 且与自身参数形状不符时自动 ``permute(0,4,1,2,3)``。
    这样 ``mmdet3d.apis.init_model`` / ``mmengine load_checkpoint`` 可直接
    加载官方 ckpt，无需手工预处理 state_dict。
    """
    from projects.BEVFusion.bevfusion.ops.spconv import conv as _spconv_conv

    _orig = _spconv_conv.SparseConvolution._load_from_state_dict

    def _load5d(self, state_dict, prefix, local_metadata, strict,
                missing_keys, unexpected_keys, error_msgs):
        w = state_dict.get(prefix + "weight")
        if w is not None and w.dim() == 5 and self.weight.shape != w.shape:
            state_dict[prefix + "weight"] = w.permute(0, 4, 1, 2, 3).contiguous()
        return _orig(self, state_dict, prefix, local_metadata, strict,
                     missing_keys, unexpected_keys, error_msgs)

    _spconv_conv.SparseConvolution._load_from_state_dict = _load5d
    print("[npu_patches] spconv load5d: 5D KRSC ckpt 权重自动 permute(0,4,1,2,3)")


def _patched_patchmerging_forward(self, x, input_size):
    """Reshape-based Unfold(k=2, s=2); bit-exact, NPU-safe."""
    H, W = input_size
    B, L, C = x.shape
    assert L == H * W, "input feature has wrong size"
    x = x.view(B, H, W, C).permute([0, 3, 1, 2])
    # crop odd edges (Unfold drops them)
    H2, W2 = H - (H % 2), W - (W % 2)
    if H2 != H or W2 != W:
        x = x[:, :, :H2, :W2]
    x = x.view(B, C, H2 // 2, 2, W2 // 2, 2).permute(0, 1, 3, 5, 2, 4)
    x = x.reshape(B, C * 4, -1).transpose(1, 2)  # (B, L/4, 4C)
    if self.norm is not None:
        x = self.norm(x)
    x = self.reduction(x)
    return x, (H // 2, W // 2)


def patch_swin_patchmerging():
    from mmdet.models.backbones.swin import PatchMerging

    PatchMerging.forward = _patched_patchmerging_forward


# ─── Patch 2: get_geometry on CPU ──────────────────────────────────────────────
# NPU fp32 bmm internally computes at reduced precision (rel ~4e-4, fp16-like
# Cube inputs).  The view-transform geometry chain (frustum -> undo post-rot ->
# perspective decode xy*z -> cam2lidar) amplifies that to absolute coordinate
# errors up to ~30 m, flipping the frustum kept-mask and corrupting bev_pool
# (img_bev cos 0.9856 vs CPU).  CPU fp32 error is ~3e-5, so we run the small
# matrix chain on CPU and ship the result back to the NPU (~24 MB, tens of ms).


def patch_get_geometry_cpu():
    """Run BaseDepthTransform.get_geometry on CPU (fp32) for NPU devices."""
    from projects.BEVFusion.bevfusion.depth_lss import BaseDepthTransform

    if getattr(BaseDepthTransform, "_npu_geometry_patched", False):
        return

    _orig_get_geometry = BaseDepthTransform.get_geometry

    global _ORIG_GET_GEOMETRY
    _ORIG_GET_GEOMETRY = _orig_get_geometry

    def _cpu_get_geometry(self, rots, trans, intrins, post_rots, post_trans, **kwargs):
        dev = trans.device
        if dev.type != "npu":
            return _orig_get_geometry(self, rots, trans, intrins, post_rots, post_trans, **kwargs)

        kwargs_cpu = {k: (v.to("cpu") if torch.is_tensor(v) else v) for k, v in kwargs.items()}
        with torch.no_grad():
            pts = _compute_geometry_cpu(
                self,
                rots.detach().to("cpu"),
                trans.detach().to("cpu"),
                intrins.detach().to("cpu"),
                post_rots.detach().to("cpu"),
                post_trans.detach().to("cpu"),
                **kwargs_cpu,
            )
        pts = pts.to(dev)
        _dump = os.environ.get("NPU_GEOM_DUMP")
        if _dump:
            torch.save({"pts": pts.cpu()}, _dump)
        return pts

    BaseDepthTransform.get_geometry = _cpu_get_geometry
    BaseDepthTransform._npu_geometry_patched = True
    print("[npu_patches] BaseDepthTransform.get_geometry -> CPU fp32 (NPU only)")




def _compute_geometry_cpu(vt, rots, trans, intrins, post_rots, post_trans, **kwargs):
    """在 CPU 张量上运行原始 get_geometry（带 frustum CPU 换入/换出）。

    输入切片先 .contiguous()：即时路径的 D2H 会把非稠密切片转成连续副本，
    而本函数的调用方可能传 CPU 非连续切片——统一布局消除两类路径的
    最后一位舍入差异（实测 1.5e-5m，虽在管线噪声下，仍求一致）。
    """
    rots, trans = rots.contiguous(), trans.contiguous()
    intrins = intrins.contiguous()
    post_rots, post_trans = post_rots.contiguous(), post_trans.contiguous()
    if getattr(vt, "_frustum_cpu_t", None) is None:
        vt._frustum_cpu_t = vt.frustum.detach().to("cpu").clone()
    frustum_backup = vt._parameters.get("frustum")
    vt._parameters["frustum"] = torch.nn.Parameter(vt._frustum_cpu_t, requires_grad=False)
    try:
        return _ORIG_GET_GEOMETRY(vt, rots, trans, intrins, post_rots, post_trans, **kwargs)
    finally:
        vt._parameters["frustum"] = frustum_backup

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


def patch_spconv_unum_ops():
    """用 unum_ops spconv 替换官方 spconv-cu114（mmdet3d 加载之后调用）。

    将 mmdet3d.registry.MODELS 中注册的稀疏卷积模块覆盖为 unum_ops 纯 torch
    版本，并把 mmdet3d 各模块 import 的 SparseSequential 替换为 unum_ops 原生
    版本（其 forward 已兼容官方 SparseModule）。
    """
    import importlib

    import unum_ops.spconv

    from mmdet3d.registry import MODELS

    MODELS._register_module(unum_ops.spconv.conv.SubMConv3d, "SubMConv3d", force=True)
    MODELS._register_module(unum_ops.spconv.conv.SparseConv3d, "SparseConv3d", force=True)
    MODELS._register_module(
        unum_ops.spconv.conv.SparseInverseConv3d, "SparseInverseConv3d", force=True
    )
    MODELS._register_module(unum_ops.spconv.conv.SubMConv2d, "SubMConv2d", force=True)
    MODELS._register_module(unum_ops.spconv.conv.SparseConv2d, "SparseConv2d", force=True)
    MODELS._register_module(
        unum_ops.spconv.conv.SparseInverseConv2d, "SparseInverseConv2d", force=True
    )

    _unum_seq = unum_ops.spconv.sparse_modules.SparseSequential
    for _mod_name in [
        "mmdet3d.models.layers.sparse_block",
        "mmdet3d.models.middle_encoders.sparse_encoder",
    ]:
        _mod = importlib.import_module(_mod_name)
        _mod.SparseSequential = _unum_seq
    print("[npu_patches] spconv -> unum_ops (MODELS 注册 + SparseSequential 替换)")


def patch_mha_with_flash_attention():
    """Replace nn.MultiheadAttention.forward with PromptFlashAttention.

    TransFusion decoder 的 self/cross attention 走 npu_prompt_flash_attention
    (fp16 内核)。历史实测(2026-09-09, Fast-PT 栈)有精度损失(8 dets/0.416 vs
    13/0.503)——本开关用于在当前栈上复测。
    """
    import torch.nn.functional as F
    import torch_npu
    from torch import nn

    def mha_fwd(self, query, key=None, value=None, **kw):
        _Sq, B, E = query.shape
        H = self.num_heads
        D = E // H
        scale = 1.0 / (D**0.5)
        if key is None:
            key = query
        if value is None:
            value = key
        mha_fwd.calls += 1
        qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        q = qkv[..., :E]
        if self._qkv_same_embed_dim:
            k2 = F.linear(key, self.in_proj_weight[E : 2 * E], self.in_proj_bias[E : 2 * E])
            v2 = F.linear(
                value, self.in_proj_weight[2 * E : 3 * E], self.in_proj_bias[2 * E : 3 * E]
            )
        else:
            k2 = F.linear(key, self.k_proj_weight, self.k_proj_bias)
            v2 = F.linear(value, self.v_proj_weight, self.v_proj_bias)
        k, v = k2, v2

        def to_bnsd(t):
            return t.transpose(0, 1).reshape(B, -1, H, D).permute(0, 2, 1, 3).contiguous()

        q, k, v = to_bnsd(q).half(), to_bnsd(k).half(), to_bnsd(v).half()
        out = torch_npu.npu_prompt_flash_attention(
            q, k, v, num_heads=H, scale_value=scale, input_layout="BNSD"
        )
        out = out.float().permute(0, 2, 1, 3).reshape(B, -1, E)
        out = F.linear(out, self.out_proj.weight, self.out_proj.bias)
        return (out.transpose(0, 1), None)

    mha_fwd.calls = 0
    nn.MultiheadAttention.forward = mha_fwd
    print("[NPU PATCH] MultiheadAttention -> npu_prompt_flash_attention (fp16)", flush=True)


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

        # 命中 extract_img_feat 预计算 stash（Swin 执行期间已在 CPU 算好，
        # 此处只做 H2D；D2H 同步开销已被 Swin 异步执行吸收）
        g = getattr(self, "_precomputed_geom_cpu", None)
        _from_stash = g is not None
        if g is not None:
            self._precomputed_geom_cpu = None
            if g.dim() == 6 and tuple(g.shape[:2]) == tuple(rots.shape[:2]):
                pts = g.to(dev)
                _dump = os.environ.get("NPU_GEOM_DUMP")
                if _dump:
                    torch.save({"pts": pts.cpu(), "from_stash": _from_stash}, _dump)
                return pts
            # 形状不匹配（不应发生）→ 落回下面的即时路径

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
            torch.save({"pts": pts.cpu(), "from_stash": _from_stash}, _dump)
        return pts

    BaseDepthTransform.get_geometry = _cpu_get_geometry
    BaseDepthTransform._npu_geometry_patched = True
    print("[npu_patches] BaseDepthTransform.get_geometry -> CPU fp32 (NPU only)")


_GEOM_INPUT_STASH = {}  # extract_feat 入口留存的 numpy 标定矩阵（每帧覆盖）
_GEOM_STASH_KEYS = ("lidar2img", "cam2img", "cam2lidar", "img_aug_matrix", "lidar_aug_matrix")


def patch_get_geometry_overlap():
    """CPU 几何计算省掉 D2H 排干（在 patch_get_geometry_cpu 之后调用）。

    背景：仅靠 patch_get_geometry_cpu 时，矩阵 .cpu() 的 D2H 在主流上排在
    Swin 之后，必须等 Swin（以及 loop 模式下上一帧）排干才开始 CPU 计算，
    期间设备空转——ms 级 A/B 实测 CPU 补丁净成本 ~36-40ms/帧（741ms vs 672ms）。
    v1（矩阵提前 D2H 到 extract_img_feat 入口）实测更慢：提前的 D2H 仍要等
    上一帧排干，且主机在 Swin 入队前就被阻塞，设备空转入队期（~850-890ms）。

    v2（本实现）：矩阵的诞生地在 extract_feat——由 metas 的 numpy 数组经
    imgs.new_tensor 构建。hook extract_feat 在构建前留存 numpy 副本（微秒级，
    零 D2H）；extract_img_feat 用 CPU 副本预计算几何（省掉 get_geometry 的
    D2H 排干等待）；get_geometry 命中 stash 只做 H2D。
    """
    from projects.BEVFusion.bevfusion.bevfusion import BEVFusion
    from projects.BEVFusion.bevfusion.depth_lss import BaseDepthTransform

    if getattr(BEVFusion, "_npu_geom_overlap_patched", False):
        return

    # hook 1: extract_feat 入口留存 metas 的 numpy 矩阵副本
    # （矩阵原本就是从这里 imgs.new_tensor 上 NPU 的，numpy 副本零成本）
    _orig_extract_feat = BEVFusion.extract_feat

    def _extract_feat_stash(self, batch_inputs_dict, batch_input_metas, **kwargs):
        try:
            imgs = (batch_inputs_dict or {}).get("imgs", None)
            if torch.is_tensor(imgs) and imgs.device.type == "npu" and batch_input_metas:
                m = {k: [] for k in _GEOM_STASH_KEYS}
                for meta in batch_input_metas:
                    if not isinstance(meta, dict):
                        break
                    m["lidar2img"].append(np.asarray(meta["lidar2img"]))
                    m["cam2img"].append(np.asarray(meta["cam2img"]))
                    m["cam2lidar"].append(np.asarray(meta["cam2lidar"]))
                    m["img_aug_matrix"].append(np.asarray(meta.get("img_aug_matrix", np.eye(4))))
                    m["lidar_aug_matrix"].append(
                        np.asarray(meta.get("lidar_aug_matrix", np.eye(4)))
                    )
                else:
                    _GEOM_INPUT_STASH["np"] = {k: np.stack(v) for k, v in m.items()}
        except Exception:
            _GEOM_INPUT_STASH.pop("np", None)  # 留存失败 → 回退即时路径
        return _orig_extract_feat(self, batch_inputs_dict, batch_input_metas, **kwargs)

    BEVFusion.extract_feat = _extract_feat_stash

    # hook 2: extract_img_feat —— 先入队 Swin/FPN，再在设备执行期间算几何
    _orig_extract = BEVFusion.extract_img_feat

    def _extract_img_feat_early(
        self,
        x,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ):
        vt = getattr(self, "view_transform", None)
        mats = None
        stash = _GEOM_INPUT_STASH.get("np")
        if (
            isinstance(vt, BaseDepthTransform)
            and stash is not None
            and torch.is_tensor(camera_intrinsics)
            and camera_intrinsics.device.type == "npu"
        ):
            dt = camera_intrinsics.dtype
            t = {k: torch.from_numpy(v).to(dt) for k, v in stash.items()}
            ref = dict(
                camera_intrinsics=camera_intrinsics,
                camera2lidar=camera2lidar,
                img_aug_matrix=img_aug_matrix,
                lidar_aug_matrix=lidar_aug_matrix,
            )
            key_map = dict(
                camera_intrinsics="cam2img",
                camera2lidar="cam2lidar",
                img_aug_matrix="img_aug_matrix",
                lidar_aug_matrix="lidar_aug_matrix",
            )
            if all(t[key_map[k]].shape == ref[k].shape for k in key_map):
                mats = {k: t[key_map[k]] for k in key_map}
        # (a) 原路径入队 backbone + neck
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W).contiguous()
        x = self.img_backbone(x)
        x = self.img_neck(x)
        if not isinstance(x, torch.Tensor):
            x = x[0]
        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)
        # (b) 用 numpy 副本 CPU 预计算几何（省 D2H 排干）
        #     （副本在矩阵上 NPU 前已留存，主机不阻塞主流）
        if mats is not None:
            try:
                with torch.no_grad():
                    vt._precomputed_geom_cpu = _compute_geometry_cpu(
                        vt,
                        mats["camera2lidar"][..., :3, :3],
                        mats["camera2lidar"][..., :3, 3],
                        mats["camera_intrinsics"][..., :3, :3],
                        mats["img_aug_matrix"][..., :3, :3],
                        mats["img_aug_matrix"][..., :3, 3],
                        extra_rots=mats["lidar_aug_matrix"][..., :3, :3],
                        extra_trans=mats["lidar_aug_matrix"][..., :3, 3],
                    )
            except Exception:
                vt._precomputed_geom_cpu = None  # 预计算失败 → 走即时路径
        # (c) view_transform（get_geometry 命中 stash，只做 H2D）
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            x = self.view_transform(
                x,
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas,
            )
        return x

    BEVFusion.extract_img_feat = _extract_img_feat_early
    BEVFusion._npu_geom_overlap_patched = True
    print("[npu_patches] get_geometry CPU 预计算: extract_feat 留存 numpy 矩阵副本（省 D2H 排干）")


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

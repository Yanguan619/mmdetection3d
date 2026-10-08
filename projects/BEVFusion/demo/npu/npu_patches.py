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


# ─── Patch 1b: ShiftWindowMSA shift 路径优化（2026-09-30）──────────────────────
# 实测（npu:7，stage0 形状 6x70x182x96）：
#   torch.roll          2.31ms   F.pad 0.67ms   mask 构建链 1.48ms
# 每个 shift 块每帧：pad+roll（2 次全帧拷贝）+ reverse roll+contiguous（又 2 次）
# + mask 重算；6 个 shift 块合计 ~27ms。
# 两项优化均 bit-exact（数值相同，仅拷贝路径不同）：
#   1) attn_mask 是 (H_pad, W_pad, window, shift) 的纯形状函数 -> 全局缓存，
#      省去每帧重算（1.48ms x 6）。
#   2) pad(右/下)+roll(-s) 代数等价于 4 切片装配（主块+列/行/角卷绕），
#      前向省一次全帧拷贝；反向 roll(+s)+slice+contiguous（2 次拷贝）同样
#      等价于 4 切片装配（1 次拷贝）。成立条件 pad_b>=s 且 pad_r>=s（本模型
#      四个 stage 的 pad 为 6/6,3/3,5/5,6/6，shift=3，均满足）；不满足时
#      严格回退原始路径（不静默出错）。

_SWIN_ATTN_MASK_CACHE = {}


def _swin_cached_attn_mask(self, H_pad, W_pad, device):
    """mmdet ShiftWindowMSA.forward 220-240 行的等价实现 + 全局缓存。"""
    key = (str(device), H_pad, W_pad, self.window_size, self.shift_size)
    mask = _SWIN_ATTN_MASK_CACHE.get(key)
    if mask is not None:
        return mask
    img_mask = torch.zeros((1, H_pad, W_pad, 1))
    h_slices = (slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None))
    w_slices = (slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    ws = self.window_size
    mw = img_mask.view(1, H_pad // ws, ws, W_pad // ws, ws, 1).permute(
        0, 1, 3, 2, 4, 5).reshape(-1, ws * ws)
    attn_mask = mw.unsqueeze(1) - mw.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)
                                      ).masked_fill(attn_mask == 0, float(0.0))
    attn_mask = attn_mask.to(device)
    _SWIN_ATTN_MASK_CACHE[key] = attn_mask
    return attn_mask


def _swin_fused_pad_roll(x, H, W, H_pad, W_pad, s):
    """F.pad(右/下) + roll(-s,-s) 的单次装配等价（要求 pad_b>=s 且 pad_r>=s）。"""
    B, _, _, C = x.shape
    out = x.new_zeros((B, H_pad, W_pad, C))
    out[:, 0:H - s, 0:W - s] = x[:, s:H, s:W]
    out[:, 0:H - s, W_pad - s:W_pad] = x[:, s:H, 0:s]
    out[:, H_pad - s:H_pad, 0:W - s] = x[:, 0:s, s:W]
    out[:, H_pad - s:H_pad, W_pad - s:W_pad] = x[:, 0:s, 0:s]
    return out


def _swin_fused_roll_unpad(sh, H, W, H_pad, W_pad, s):
    """roll(+s,+s) + [:, :H, :W] + contiguous() 的单次装配等价。"""
    B, _, _, C = sh.shape
    out = sh.new_empty((B, H, W, C))
    out[:, s:H, s:W] = sh[:, 0:H - s, 0:W - s]
    out[:, s:H, 0:s] = sh[:, 0:H - s, W_pad - s:W_pad]
    out[:, 0:s, s:W] = sh[:, H_pad - s:H_pad, 0:W - s]
    out[:, 0:s, 0:s] = sh[:, H_pad - s:H_pad, W_pad - s:W_pad]
    return out


def _patched_shiftwinmsa_forward(self, query, hw_shape):
    import torch.nn.functional as F

    from mmdet.models.backbones.swin import _roll_onnx_compat

    B, L, C = query.shape
    H, W = hw_shape
    assert L == H * W, 'input feature has wrong size'
    query = query.view(B, H, W, C)

    pad_r = (self.window_size - W % self.window_size) % self.window_size
    pad_b = (self.window_size - H % self.window_size) % self.window_size
    H_pad, W_pad = H + pad_b, W + pad_r
    s = self.shift_size
    # 融合路径成立条件：shift 块 + 两个方向 pad 都 >= shift + 图像大于 shift
    fused = s > 0 and pad_b >= s and pad_r >= s and H > s and W > s

    if fused:
        shifted_query = _swin_fused_pad_roll(query, H, W, H_pad, W_pad, s)
        attn_mask = _swin_cached_attn_mask(self, H_pad, W_pad, query.device)
    else:
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]
        if s > 0:
            shifted_query = _roll_onnx_compat(
                query, shifts=(-s, -s), dims=(1, 2))
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                                                  attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None

    query_windows = self.window_partition(shifted_query)
    query_windows = query_windows.view(-1, self.window_size**2, C)
    attn_windows = self.w_msa(query_windows, mask=attn_mask)
    attn_windows = attn_windows.view(-1, self.window_size,
                                     self.window_size, C)
    shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)

    if fused:
        x = _swin_fused_roll_unpad(shifted_x, H, W, H_pad, W_pad, s)
    else:
        if s > 0:
            x = _roll_onnx_compat(
                shifted_x, shifts=(s, s), dims=(1, 2))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

    x = x.view(B, H * W, C)
    x = self.drop(x)
    return x


def patch_swin_shift_optimizations():
    """ShiftWindowMSA：mask 缓存 + pad/roll 融合（bit-exact，见模块注释）。

    env 开关 NPU_PATCH_SWIN_SHIFT=0 可关闭（A/B 对照用）。
    """
    if os.environ.get("NPU_PATCH_SWIN_SHIFT", "1") == "0":
        print("[npu_patches] ShiftWindowMSA 优化已通过 NPU_PATCH_SWIN_SHIFT=0 关闭")
        return
    from mmdet.models.backbones.swin import ShiftWindowMSA

    ShiftWindowMSA.forward = _patched_shiftwinmsa_forward
    print("[npu_patches] ShiftWindowMSA: mask 缓存 + pad/roll 融合 (bit-exact)")



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


def _stash_geom_inputs(batch_inputs_dict, batch_input_metas):
    """extract_feat 入口留存 metas 的 numpy 矩阵副本（每帧覆盖，失败清 stash）。

    numpy 互操作段，标记为 dynamo 禁区（torchair 图捕获时在此干净断图、
    eager 执行本段；无 dynamo 时该标记零开销）。见 patch_get_geometry_overlap。
    """
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


_stash_geom_inputs = torch._dynamo.disable(_stash_geom_inputs)


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
    # （矩阵原本就是从这里 imgs.new_tensor 上 NPU 的，numpy 副本零成本；
    #  stash 段为 dynamo 禁区，torchair 图捕获时在此断图）
    _orig_extract_feat = BEVFusion.extract_feat

    def _extract_feat_stash(self, batch_inputs_dict, batch_input_metas, **kwargs):
        _stash_geom_inputs(batch_inputs_dict, batch_input_metas)
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


# ─── Patch 4: LiDAR 投影 trash 列向量化 ───────────────────────────────────────
# 原始 BaseDepthTransform.forward 的 b×6 相机循环是 NPU 反模式组合：
#   每相机一次 .any() 同步（排干队列）+ 布尔索引 gather（NonZero AICPU
#   11.4ms + Index 6.4ms，op_summary 实测）+ advanced-index scatter，
#   外加 depth 的 CPU zeros + 4.3MB H2D。段实测 44.7ms（独立探针）。
# 改法：depth 直接在设备上分配 (B, C, 1, H, W+1)，无效点统一路由到 (0, W)
# trash 格（NaN/越界坐标经 mask 比较自然落 trash），逐相机全量 index_put_
# ——保持每相机内点序，重复单元 last-wins 与原始逐位一致（bit-exact 验证
# 过）；末尾 narrow[..., :W] 丢弃 trash 列。数学链逐语句不动。
# 段实测 44.7→24.8ms（-20ms）；env NPU_PATCH_LIDAR_PROJ=0 关闭。


def patch_lidar_proj_trash():
    """LiDAR 投影（点云→稀疏深度图）trash 列向量化，bit-exact，-20ms/帧。

    env 开关 NPU_PATCH_LIDAR_PROJ=0 可关闭（A/B 对照用）；非 NPU 设备
    运行时自动回退原始实现。
    """
    if os.environ.get("NPU_PATCH_LIDAR_PROJ", "1") == "0":
        print("[npu_patches] LiDAR 投影 trash 优化已通过 NPU_PATCH_LIDAR_PROJ=0 关闭")
        return
    from projects.BEVFusion.bevfusion.depth_lss import BaseDepthTransform

    if getattr(BaseDepthTransform, "_npu_lidar_proj_patched", False):
        return

    _orig_forward = BaseDepthTransform.forward

    def _trash_forward(
        self, img, points, lidar2image, cam_intrinsic, camera2lidar,
        img_aug_matrix, lidar_aug_matrix, metas, **kwargs,
    ):
        if not torch.is_tensor(points[0]) or points[0].device.type != "npu":
            return _orig_forward(
                self, img, points, lidar2image, cam_intrinsic, camera2lidar,
                img_aug_matrix, lidar_aug_matrix, metas, **kwargs,
            )
        intrins = cam_intrinsic[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        batch_size = len(points)
        H_img, W_img = self.image_size
        depth = torch.zeros(
            batch_size, img.shape[1], 1, H_img, W_img + 1,
            device=points[0].device,
        )
        for b in range(batch_size):
            cur_coords = points[b][:, :3]
            cur_img_aug_matrix = img_aug_matrix[b]
            cur_lidar_aug_matrix = lidar_aug_matrix[b]
            cur_lidar2image = lidar2image[b]

            # inverse aug
            cur_coords -= cur_lidar_aug_matrix[:3, 3]
            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                cur_coords.transpose(1, 0))
            # lidar2image
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
            # get 2d coords
            dist = cur_coords[:, 2, :]
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]
            # imgaug
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)
            # normalize coords for grid sample: [..., [1, 0]] -> (y, x)
            cur_coords = cur_coords[..., [1, 0]]

            y = cur_coords[..., 0]
            x = cur_coords[..., 1]
            valid = (y >= 0) & (y < H_img) & (x >= 0) & (x < W_img)
            # 无效点 → (0, W) trash 格（单格重复写无影响，narrow 后丢弃）
            yi = torch.where(valid, y, torch.zeros_like(y)).long()
            xi = torch.where(valid, x, torch.full_like(x, float(W_img))).long()
            flat = yi * (W_img + 1) + xi
            dd = torch.where(valid, dist, torch.zeros_like(dist))
            # 逐相机全量 scatter：保持相机内点序 → 重复单元 last-wins 与原始一致
            for c in range(cur_coords.shape[0]):
                depth[b, c].view(-1)[flat[c]] = dd[c]

        depth = depth[..., :W_img].contiguous()

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]
        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        x = self.get_cam_feats(img, depth)
        x = self.bev_pool(geom, x)
        return x

    BaseDepthTransform.forward = _trash_forward
    BaseDepthTransform._npu_lidar_proj_patched = True
    print("[npu_patches] LiDAR 投影 trash 列向量化 (bit-exact, 段 -20ms)")


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

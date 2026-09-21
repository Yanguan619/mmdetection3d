import os
import torch

try:
    from . import bev_pool_ext
    _HAS_CUDA_EXT = True
except (ImportError, ModuleNotFoundError):
    _HAS_CUDA_EXT = False

try:
    import importlib.util as _ilu
    _HAS_MX_DRIVING_EXT = _ilu.find_spec('mx_driving') is not None
except (ImportError, ModuleNotFoundError, ValueError):
    _HAS_MX_DRIVING_EXT = False


class QuickCumsum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        # NPU bug workaround: torch_npu 上 `kept[:-1] = ...` 就地切片赋值会多写
        # 末位元素（kept[N-1] 被覆盖为垃圾值），导致丢失最后一个 segment（一个
        # BEV cell 清零）。用 cat 构造规避（[1:] 起始的切片赋值不受影响）。
        kept = torch.cat((
            ranks[1:] != ranks[:-1],
            torch.ones(1, device=x.device, dtype=torch.bool)))
        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))
        ctx.save_for_backward(kept)
        ctx.mark_non_differentiable(geom_feats)
        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        (kept, ) = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1
        val = gradx[back]
        return val, None, None


class QuickCumsumCuda(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, geom_feats, ranks, B, D, H, W):
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[1:] = ranks[1:] != ranks[:-1]
        interval_starts = torch.where(kept)[0].int()
        # NPU bug workaround: `interval_lengths[:-1] = ...` 同款切片赋值问题，用 cat 规避
        interval_lengths = torch.cat((
            interval_starts[1:] - interval_starts[:-1],
            (x.shape[0] - interval_starts[-1:]).to(interval_starts.dtype)))
        geom_feats = geom_feats.int()
        out = bev_pool_ext.bev_pool_forward(
            x, geom_feats, interval_lengths, interval_starts, B, D, H, W)
        ctx.save_for_backward(interval_starts, interval_lengths, geom_feats)
        ctx.saved_shapes = B, D, H, W
        return out

    @staticmethod
    def backward(ctx, out_grad):
        interval_starts, interval_lengths, geom_feats = ctx.saved_tensors
        B, D, H, W = ctx.saved_shapes
        out_grad = out_grad.contiguous()
        x_grad = bev_pool_ext.bev_pool_backward(
            out_grad, geom_feats, interval_lengths, interval_starts, B, D, H, W)
        return x_grad, None, None, None, None, None, None


def _sanitize_dense_scatter(feats, coords, B, D, H, W):
    """QuickCumsum / scatter / CUDA 后端的坐标清理。

    这些后端用坐标直接索引输出（负/越界索引会回绕或越界）。depth_lss 的
    bev_pool 现在传入"哨兵坐标"（越界点映射到 (H, W, D) 网格外哨兵格），
    本函数把它们识别为越界并做标准清理：
      1) feats 越界行乘 0（贡献恰好 0.0，bit-exact）
      2) 坐标 clamp 回 [0, dim-1]
    对已清理的输入幂等。坐标列约定 (x, y, z, b)：x<H、y<W、z<D。
    """
    kept = ((coords[:, 0] >= 0) & (coords[:, 0] < H)
            & (coords[:, 1] >= 0) & (coords[:, 1] < W)
            & (coords[:, 2] >= 0) & (coords[:, 2] < D))
    feats = feats * kept.unsqueeze(1).to(feats.dtype)
    coords = torch.cat(
        [coords[:, :3].clamp(
            min=coords.new_tensor([0, 0, 0]),
            max=coords.new_tensor([H - 1, W - 1, D - 1])),
         coords[:, 3:]], dim=1)
    return feats, coords


def _bev_pool_pytorch(feats, coords, B, D, H, W):
    return _bev_pool_pytorch_fallback(feats, coords, B, D, H, W)


def _bev_pool_pytorch_fallback(feats, coords, B, D, H, W):
    assert feats.shape[0] == coords.shape[0]
    feats, coords = _sanitize_dense_scatter(feats, coords, B, D, H, W)
    ranks = (
        coords[:, 0] * (W * D * B) + coords[:, 1] * (D * B) +
        coords[:, 2] * B + coords[:, 3])
    indices = ranks.float().argsort()
    feats = torch.index_select(feats, 0, indices)
    coords = torch.index_select(coords, 0, indices)
    ranks = torch.index_select(ranks, 0, indices)
    x, geom_feats = QuickCumsum.apply(feats, coords, ranks)
    geom_feats = geom_feats.long()
    out = feats.new_zeros(B, D, H, W, feats.shape[-1])
    out[geom_feats[:, 3], geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x
    out = out.permute(0, 4, 1, 2, 3).contiguous()
    return out


def _bev_pool_cuda(feats, coords, B, D, H, W):
    assert feats.shape[0] == coords.shape[0]
    feats, coords = _sanitize_dense_scatter(feats, coords, B, D, H, W)
    ranks = (
        coords[:, 0] * (W * D * B) + coords[:, 1] * (D * B) +
        coords[:, 2] * B + coords[:, 3])
    indices = ranks.argsort()
    feats, coords, ranks = feats[indices], coords[indices], ranks[indices]
    x = QuickCumsumCuda.apply(feats, coords, ranks, B, D, H, W)
    x = x.permute(0, 4, 1, 2, 3).contiguous()
    return x


def bev_pool_torch(feats, coords, B, D, H, W):
    """Pure PyTorch implementation of the BEV pillar pooling operator.

    Equivalent to the CUDA ``bev_pool``. It sums the features of all points
    that fall into the same ``(batch, x, y, z)`` cell and scatters them into a
    dense ``[B, C, D, H, W]`` tensor.

    The implementation relies only on standard PyTorch ops so that it can run
    on CPU / NPU and be exported to ONNX.

    Args:
        feats (Tensor): Flattened features of shape ``[N, C]``.
        coords (Tensor): Coordinates of shape ``[N, 4]``, with columns
            ``(x, y, z, batch)``.
        B (int): Batch size.
        D (int): Size of the ``z`` (depth) dimension.
        H (int): Size of the ``x`` dimension.
        W (int): Size of the ``y`` dimension.

    Returns:
        Tensor: BEV features of shape ``[B, C, D, H, W]``.
    """
    assert feats.shape[0] == coords.shape[0]
    feats, coords = _sanitize_dense_scatter(feats, coords, B, D, H, W)

    N, C = feats.shape
    coords = coords.long()
    # flatten the (batch, z, x, y) cell into a single index
    flat = (coords[:, 3] * (D * H * W) + coords[:, 2] * (H * W) +
            coords[:, 0] * W + coords[:, 1])
    index = flat.unsqueeze(1).expand(-1, C)

    out = torch.zeros(
        B * D * H * W, C, dtype=feats.dtype, device=feats.device)
    out = torch.scatter_add(out, 0, index, feats)
    out = out.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
    return out


def _bev_pool_ascendc(feats, coords, B, D, H, W):
    # 2026-09-18 起 unum_ops AscendC kernel 的 CoordOffset/CoordInGrid 已改为
    # 与官方 bev_pool_cuda.cu 一致的约定（coords=[x,y,z,b] 原生输入，x→H、
    # y→W，输出 out[b,z,x,y]），分发器不再交换 x/y 列。
    # 历史注记：旧 kernel 把 coord[0] 当 W 轴（by*gridW+bx），此处曾用 cat
    # 切片交换（2.9ms；fancy 索引版 14ms 禁用）补偿。
    return _bev_pool_ascend(feats, coords, B, D, H, W).out


def _bev_pool_mxdriving(feats, coords, B, D, H, W):
    """DrivingSDK (mx_driving) bev_pool_v3 后端。

    DrivingSDK 官方实现，无 depth 模式直接收 (N,4) 原始坐标 [x,y,z,b]，kernel
    用 SetAtomicAdd 原子累加，输出 (B,C,D,H,W)。坐标列序与官方约定一致
    （b*DHW + z*HW + x*W + y → out[b,z,x,y]），无需 rank 预处理。C 需 %8==0。
    无越界检查：越界点（哨兵格）经 _sanitize_dense_scatter 乘 0 + clamp。
    """
    assert feats.shape[0] == coords.shape[0]
    feats, coords = _sanitize_dense_scatter(feats, coords, B, D, H, W)
    C = feats.shape[1]
    assert C % 8 == 0, f"mx_driving bev_pool_v3 requires C%8==0, got {C}"
    coords = coords.int()
    from mx_driving.ops.bev_pool_v3 import bev_pool_v3
    return bev_pool_v3(None, feats, None, None, coords, (B, D, H, W, C))


def bev_pool(feats, coords, B, D, H, W):
    if _HAS_CUDA_EXT:
        return _bev_pool_cuda(feats, coords, B, D, H, W)
    backend = os.environ.get('BEVFUSION_BEV_POOL_BACKEND', 'auto')
    if backend == 'scatter':
        return bev_pool_torch(feats, coords, B, D, H, W)
    if backend in ('torch', 'quickcumsum'):
        return _bev_pool_pytorch(feats, coords, B, D, H, W)
    if backend in ('mx_driving', 'mxdriving', 'driving'):
        if not _HAS_MX_DRIVING_EXT:
            raise RuntimeError(
                "BEVFUSION_BEV_POOL_BACKEND=mx_driving 但 mx_driving 未安装。"
                "请先 `pip install mx-driving`（cp311/aarch64 wheel 含 "
                "ascend310p 预编译 bev_pool_v3 内核）。")
        return _bev_pool_mxdriving(feats, coords, B, D, H, W)
    # 【mxdriving 分支】auto 只用 DrivingSDK 官方 bev_pool_v3（原子累加，
    # 全链 ~34.4ms，max_abs 1.95e-5）；不经过 unum_ops 原子加 kernel。
    # 与 main 的差异仅在 auto 缺省路由；显式 'torch'/'scatter' env 路径不变。
    # mx_driving 的 kernel 无 OOB 检查，_bev_pool_mxdriving 内部已做 sanitize。
    if not _HAS_MX_DRIVING_EXT:
        raise RuntimeError(
            "mxdriving 分支要求 mx_driving（DrivingSDK）已安装。"
            "请先 `pip install mx-driving`（cp311/aarch64 wheel 含 "
            "ascend310p 预编译 bev_pool_v3 内核），或切回 main 使用 "
            "unum_ops/QuickCumsum 回退。")
    return _bev_pool_mxdriving(feats, coords, B, D, H, W)

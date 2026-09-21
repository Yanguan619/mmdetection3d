import hashlib
import itertools
import os

import numpy as np
import torch
from torch import nn

from . import spconv_ascendc as ascendc
from .sparse_modules import SparseConvTensor, SparseModule

# AscendC 内核默认关闭（标量实现慢于 torch 2D GEMM），显式开启用环境变量
_USE_ASCENDC = os.environ.get("UNUM_SPCONV_USE_ASCENDC", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# 邻居查找稠密网格上限（int64 条目数）：超过后回退到 sort+searchsorted。
# 8M 条目 = 64MB，覆盖 BEVFusion 常用空间 (200,200,16)=640K / SECOND (100,100,8)=80K。
# 注意：BEVFusion 的 SECOND 空间是 (1440,1440,41)=85M 条目，超过 8M 会回退到
# sort+searchsorted 路径，而该路径在 NPU 上数值严重失真（SubMConv 输出 cos≈0.72）。
# 因此上限需覆盖 BEVFusion 85M grid（256M 条目 = 2GB）。NPU 上 grid 路径
# （index_put_ + gather）与 CPU 完全一致（cos=1.0，rel=0.03%）。
_GRID_LOOKUP_MAX_ENTRIES = 256 * 1024 * 1024


class SparseConvolution(SparseModule):
    """spconv SparseConvolution 的 torch-native 实现（纯 PyTorch，CPU/GPU/NPU 通用）

    提供通用的稀疏 N 维卷积（2D/3D），包括：
      - SubMConv（子流形卷积）：输出坐标 == 输入坐标，空间形状不变
      - SparseConv（步长下采样卷积）：输出坐标 = floor(输入坐标 / stride)
      - SparseInverseConv（转置卷积 / 上采样）：与对应的 SparseConv 配对使用

    优化说明（相对早期逐体素 Python 循环版本）：
      - 邻居查找：dict 逐点查找 → 坐标编码为 int64 key + torch.searchsorted 批量查找
      - 特征聚合：逐体素小矩阵乘 → 按核偏移循环的批量 addmm（torch.addmm_）
      - 邻居表缓存：按 (坐标指纹, 核/步长/padding/空间形状, 设备) 缓存，
        网络内坐标不变且设备不变时只构建一次；NPU 命中后无设备往返。

    设备策略（方案 A）：
      - 邻居表构建全程留在 indices 所在设备（CPU/NPU），避免 host-device 同步
      - _fingerprint 仅用 shape + 标量统计构造缓存 key（NPU 标量 .item() 同步开销小）
      - _neighbor_cached 按 (fp, device) 分别缓存，避免跨设备复用错误
      - _gather 中 neighbor_idx 已与 features 同设备，无需再 .to(device)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        bias: bool = True,
        indice_key: str | None = None,
        ndim: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ndim = ndim
        self.stride = self._triple(stride, ndim)
        self.padding = self._triple(padding, ndim)
        self.dilation = self._triple(dilation, ndim)
        self.kernel_size = self._triple(kernel_size, ndim)
        self.indice_key = indice_key

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        # 惰性缓存
        self._offsets_cache = None
        self._offsets_ndim = None
        self._nb_cache = {}

    def _triple(self, x, ndim=None) -> tuple[int, ...]:
        if ndim is None:
            ndim = self.ndim
        if isinstance(x, (tuple, list)):
            return tuple(int(v) for v in x)
        return (int(x),) * ndim

    def _kernel_offsets(self):
        """返回所有卷积核偏移量，形状 (K^ndim, ndim)，顺序为 itertools.product 字典序"""
        offsets = []
        for o in itertools.product(*[range(k) for k in self.kernel_size]):
            offsets.append(o)
        return offsets

    @property
    def _flat_slice_idx(self):
        """与 _kernel_offsets 顺序一致的平坦权重索引（按 np.ravel_multi_index 规则）"""
        offsets = self._kernel_offsets()
        idx = []
        for o in offsets:
            idx.append(int(np.ravel_multi_index(o, tuple(self.kernel_size))))
        return idx

    @property
    def _offsets_t(self) -> torch.Tensor:
        """核偏移量张量 (K, ndim)，K = prod(kernel_size)。

        注意：返回 CPU 张量，调用方需自行 .to(device)。
        仅取每个偏移的前 ndim 维，与逐体素实现的邻居坐标计算保持一致
        （2D 卷积的 kernel_size 仍按 ndim=3 初始化，此处截断保留原语义）。
        """
        if self._offsets_cache is None or self._offsets_ndim != self.ndim:
            prod = self._kernel_offsets()
            offs = [p[: self.ndim] for p in prod]
            self._offsets_cache = torch.tensor(offs, dtype=torch.int64)
            self._offsets_ndim = self.ndim
        return self._offsets_cache

    def _offsets_on(self, device) -> torch.Tensor:
        """返回指定设备上的核偏移量 (K, ndim)"""
        return self._offsets_t.to(device)

    @property
    def _wT(self) -> torch.Tensor:
        """权重重排 (K, C_in, C_out)，k 与 _offsets_t 第 k 行对应"""
        K = self._offsets_t.shape[0]
        w = self.weight.reshape(self.out_channels, self.in_channels, K)
        return w.permute(2, 1, 0).contiguous()

    @property
    def _wT2d(self) -> torch.Tensor:
        """权重重排为 2D (K*C_in, C_out)，供 2D GEMM 聚合使用（带缓存）。

        _wT 为 (K, C_in, C_out) 连续张量，按行展平 (k, i) → (K*C_in) 后
        第 k*C_in+i 行正是 wT[k, i, :]，与 all_feats.reshape(N, K*C_in)
        的列顺序完全对应。reshape 是零拷贝 view。
        推理期权重不变，按 (weight._version, data_ptr) 缓存，避免每次
        forward 的 permute+contiguous 拷贝。_version 覆盖就地更新
        （load_state_dict 等），data_ptr 覆盖 .data 整体替换（cast_to_fp16）。
        """
        key = (self.weight._version, self.weight.data_ptr())
        cache = self.__dict__.get("_wT2d_cache")
        if cache is None or cache[0] != key:
            K = self._offsets_t.shape[0]
            w = self.weight.reshape(self.out_channels, self.in_channels, K)
            wT = w.permute(2, 1, 0).contiguous()  # (K, C_in, C_out)
            self._wT2d_cache = (
                key,
                wT.reshape(K * self.in_channels, self.out_channels),
            )
        return self._wT2d_cache[1]

    # ------------------------------------------------------------------
    # 邻居表构建（向量化，无 Python 逐点循环，全程留在输入设备）
    # ------------------------------------------------------------------

    def _encode(self, indices: torch.Tensor, spatial_shape) -> torch.Tensor:
        """将坐标 (N, 1+ndim) [batch, spatial...] 编码为单一 int64 key。

        spatial_shape 按 [x, y, z] 顺序给出，坐标列按 [x, y, z] 顺序，
        因此列乘数需从最后一列（z）向第一列（x）累乘空间尺寸。
        """
        ndim = self.ndim
        sp = [int(spatial_shape[i]) for i in range(ndim)]
        strides = [1] * (ndim + 1)
        acc = 1
        for j in range(ndim, 0, -1):
            strides[j] = acc
            acc *= sp[j - 1]
        strides[0] = acc
        # 构造 strides 张量，放在 indices 同设备
        strides_t = torch.tensor(strides, dtype=torch.int64, device=indices.device)
        inds = indices.to(torch.int64)
        keys = inds[:, 0] * strides_t[0]
        for d in range(ndim):
            keys = keys + inds[:, d + 1] * strides_t[d + 1]
        return keys

    def _lookup_grid(
        self,
        in_keys: torch.Tensor,
        cand_keys: torch.Tensor,
        cand_valid: torch.Tensor,
        max_key: int,
    ) -> torch.Tensor:
        """稠密网格 O(1) 查找：in_keys 与 cand_keys 已在同一 key 空间。

        max_key: 网格分配大小下界（>= batch_size*sp_total）。
        NPU 上 index_put_ + gather 远快于 sort + searchsorted（~5x）。
        空间过大（> _GRID_LOOKUP_MAX_ENTRIES）时返回 None 由调用方回退。

        网格用 int32：key 空间 <= 256M 条目 < 2^31，行数 N_in < 2^31 均不会
        溢出；相比 int64 减半内存流量（BEVFusion 85M 网格：680MB → 340MB），
        末尾统一 .long() 回 int64 供 index_select 使用。
        """
        N_in = in_keys.numel()
        max_key = max(max_key, int(in_keys.max().item()) + 1)
        grid_size = max_key
        if grid_size <= _GRID_LOOKUP_MAX_ENTRIES:
            device = in_keys.device
            grid = torch.full((grid_size,), -1, dtype=torch.int32, device=device)
            ar = torch.arange(N_in, dtype=torch.int32, device=device)
            grid.index_put_([in_keys], ar)
            nb = grid[cand_keys.reshape(-1)].reshape(cand_keys.shape).long()
            return torch.where(cand_valid, nb, torch.full_like(nb, -1))
        return None

    def _lookup(
        self,
        in_indices: torch.Tensor,
        spatial_shape: tuple[int, ...],
        cand_keys: torch.Tensor,
        cand_valid: torch.Tensor,
    ) -> torch.Tensor:
        """批量查找候选坐标对应的输入行索引；未命中置 -1。
        全程在 cand_keys 所在设备完成。

        快速路径（稠密网格 O(1) 查找）：
          构建 spatial_shape 大小的稠密索引表，将坐标 key 直映射到行索引。
          NPU 上 index_put_ + gather 远快于 sort + searchsorted（~5x）。

        回退路径（sort + searchsorted）：
          空间过大（> _GRID_LOOKUP_MAX_ENTRIES）时使用，避免大张量内存申请。
        """
        in_keys = self._encode(in_indices, spatial_shape)
        N_in = in_keys.numel()
        if N_in == 0:
            return torch.full_like(cand_keys, -1)

        ndim = self.ndim
        sp_total = 1
        for i in range(ndim):
            sp_total *= int(spatial_shape[i])
        batch_size = int(in_indices[:, 0].max().item()) + 1
        grid_size = batch_size * sp_total

        if grid_size <= _GRID_LOOKUP_MAX_ENTRIES:
            nb = self._lookup_grid(in_keys, cand_keys, cand_valid, grid_size)
            if nb is not None:
                return nb

        # 回退：sort + searchsorted
        in_keys_sorted, order = torch.sort(in_keys.double())
        if torch.jit.is_tracing():
            # ONNX 不支持 searchsorted。用二分查找 O(N*K*logN) 内存友好。
            lo = torch.zeros_like(cand_keys, dtype=torch.long)
            hi = torch.full_like(cand_keys, N_in, dtype=torch.long)
            cd = cand_keys.double()
            for _ in range(22):  # 2^22 = 4M, 覆盖 N 到 4M
                mid = (lo + hi) // 2
                mid_val = in_keys_sorted[mid.clamp(0, N_in - 1)]
                is_lo = mid_val <= cd
                lo = torch.where(is_lo, mid, lo)
                hi = torch.where(is_lo, hi, mid)
            pos = lo
        else:
            pos = torch.searchsorted(in_keys_sorted, cand_keys.double())
        pos = pos.clamp(0, N_in - 1)
        match = (in_keys_sorted[pos] == cand_keys.double()) & cand_valid
        return torch.where(match, order[pos], torch.full_like(pos, -1))

    def _build_neighbor_idx(
        self,
        in_indices: torch.Tensor,
        out_coords: torch.Tensor,
        spatial_shape,
        kernel,
        padding,
        stride,
        dilation=None,
    ) -> torch.Tensor:
        """普通（SubM / 步长）卷积邻居表。

        对每个输出坐标 q 与每个核偏移 k：候选输入坐标 = q*stride + k*dilation - padding，
        SubM 语义下（stride=1, padding=0, dilation=1）退化为 q + k。
        返回 (N_out, K) 邻居行索引，-1 表示邻居不存在。
        全程在 out_coords 所在设备完成（不再强制 .cpu()）。

        快速路径（SubM / in_indices == out_coords 同数据）：
          候选 key = in_key + offs_key - padding_key，
          避免 (N*K, ndim) 候选坐标构造 + 二次编码（~6ms @ 5 万体素）。
        """
        ndim = self.ndim
        device = out_coords.device
        offs = self._offsets_on(device)
        K = offs.shape[0]
        N_out = out_coords.shape[0]
        sp_col = torch.tensor(
            [int(spatial_shape[i]) for i in range(ndim)],
            dtype=torch.int64,
            device=device,
        )
        pad = torch.tensor(
            [int(p) for p in padding[:ndim]], dtype=torch.int64, device=device
        )
        st = torch.tensor(
            [int(s) for s in stride[:ndim]], dtype=torch.int64, device=device
        )
        dil = (
            torch.tensor(
                [int(d) for d in dilation[:ndim]], dtype=torch.int64, device=device
            )
            if dilation is not None
            else torch.ones(ndim, dtype=torch.int64, device=device)
        )
        if N_out == 0:
            return torch.empty(0, K, dtype=torch.int64, device=device)

        oc = out_coords
        cand_sp = (
            oc[:, None, 1:] * st + offs[None] * dil - pad
        )  # (N_out, K, ndim) 列序 [x,y,z]
        # SubM 快速路径：in_indices 与 out_coords 同一数据（stride=1 下采样不变）
        if in_indices.data_ptr() == out_coords.data_ptr() and all(
            int(s) == 1 for s in stride[:ndim]
        ):
            in_keys = self._encode(in_indices, spatial_shape)  # (N,)
            # 偏移量在 key 空间：offs*strides[1:] - padding*strides[1:]
            sp_strides = [1] * (ndim + 1)
            acc = 1
            for j in range(ndim, 0, -1):
                sp_strides[j] = acc
                acc *= int(spatial_shape[j - 1])
            sp_strides[0] = acc
            sp_total = acc
            offs_key = (
                offs
                * dil
                * torch.tensor(sp_strides[1:], dtype=torch.int64, device=device)
            ).sum(-1)
            pad_key = (
                pad * torch.tensor(sp_strides[1:], dtype=torch.int64, device=device)
            ).sum(-1)
            cand_keys = in_keys[:, None] + (offs_key - pad_key)[None, :]  # (N, K)

            # 边界：核窗口内的邻居落在 [0, sp) 内才算有效
            in_range = (cand_sp >= 0) & (cand_sp < sp_col)
            valid = in_range.all(-1)
            cand_keys = torch.where(valid, cand_keys, torch.zeros_like(cand_keys))
            sp_total = sp_strides[0]
            batch_size = int(in_indices[:, 0].max().item()) + 1
            nb = self._lookup_grid(in_keys, cand_keys, valid, batch_size * sp_total)
            if nb is not None:
                return nb
            # 网格过大时回退到 sort + searchsorted
            return self._lookup(in_indices, spatial_shape, cand_keys, valid)

        in_range = (cand_sp >= 0) & (cand_sp < sp_col)
        valid = in_range.all(-1)
        batch = oc[:, None, 0:1].expand(N_out, K, 1).to(torch.int64)
        cand = torch.cat([batch, cand_sp], dim=-1).reshape(-1, ndim + 1)
        cand_keys = self._encode(cand, spatial_shape).reshape(N_out, K)
        cand_keys = torch.where(valid, cand_keys, torch.zeros_like(cand_keys))
        return self._lookup(in_indices, spatial_shape, cand_keys, valid)

    def _build_inverse_neighbor_idx(
        self,
        out_coords: torch.Tensor,
        in_indices: torch.Tensor,
        spatial_shape,
        kernel,
        padding,
        stride,
        dilation=None,
    ) -> torch.Tensor:
        """转置卷积邻居表。

        对每个细坐标 c 与核偏移 k：粗坐标 q = (c + padding - k*dilation) / stride（仅当整除且落在范围内）。
        返回 (N_out, K) 邻居行索引，-1 表示无贡献。
        全程在 out_coords 所在设备完成（不再强制 .cpu()）。
        """
        ndim = self.ndim
        device = out_coords.device
        offs = self._offsets_on(device)
        K = offs.shape[0]
        N_out = out_coords.shape[0]
        sp_col = torch.tensor(
            [int(spatial_shape[i]) for i in range(ndim)],
            dtype=torch.int64,
            device=device,
        )
        pad = torch.tensor(
            [int(p) for p in padding[:ndim]], dtype=torch.int64, device=device
        )
        st = torch.tensor(
            [int(s) for s in stride[:ndim]], dtype=torch.int64, device=device
        )
        dil = (
            torch.tensor(
                [int(d) for d in dilation[:ndim]], dtype=torch.int64, device=device
            )
            if dilation is not None
            else torch.ones(ndim, dtype=torch.int64, device=device)
        )
        if N_out == 0:
            return torch.empty(0, K, dtype=torch.int64, device=device)

        oc = out_coords
        num = oc[:, None, 1:] + pad - offs[None] * dil  # c + padding - k*dilation
        q = num // st
        div = (num % st) == 0
        in_range = (q >= 0) & (q < sp_col)
        valid = (div & in_range).all(-1)
        batch = oc[:, None, 0:1].expand(N_out, K, 1).to(torch.int64)
        cand = torch.cat([batch, q], dim=-1).reshape(-1, ndim + 1)
        cand_keys = self._encode(cand, spatial_shape).reshape(N_out, K)
        cand_keys = torch.where(valid, cand_keys, torch.zeros_like(cand_keys))
        return self._lookup(in_indices, spatial_shape, cand_keys, valid)

    # ------------------------------------------------------------------
    # 特征聚合（批量 GEMM）
    # ------------------------------------------------------------------

    def _gather(
        self,
        features: torch.Tensor,
        neighbor_idx: torch.Tensor,
    ) -> torch.Tensor:
        """out[n] = bias + sum_k wT[k] @ features[neighbor_idx[n,k]]（邻居缺失贡献为 0）

        neighbor_idx 必须与 features 同设备（由上层 _neighbor_cached 保证）。

        设备策略：
          - NPU 上且 AscendC 扩展可加载时，特征聚合（gather 后 GEMM）卸载到
            自定义 AscendC 内核 ``unum.spconv_gemm``（含 autograd 反向）。
          - 其余情况回退到 2D GEMM（原 einsum 在 NPU 上慢 6x，改用 (N, K*C_in) @ (K*C_in, C_out)）。
        """
        N, K = neighbor_idx.shape
        C_in = features.shape[1]
        C_out = self.out_channels
        if N == 0 or features.shape[0] == 0:
            return features.new_zeros(0, C_out)

        # 1. 零行技巧：features 头部拼一行零，无效邻居(-1)+1 → 0 号零行。
        #    等价于旧版 "gather 后乘 valid 掩码"（bit-exact），但省掉 (N*K, C_in)
        #    全量逐元素乘（310P 实测每层 ~4ms，21 层/帧 ~22ms）；cat 仅 ~0.25ms。
        nb_flat = neighbor_idx.reshape(-1)  # (N*K,)
        feats_pad = torch.cat(
            [features.new_zeros(1, C_in), features]
        )  # (N+1, C_in)，第 0 行为零
        all_feats = torch.index_select(
            feats_pad, 0, nb_flat + 1
        )  # -1→0（零行），i→i+1
        all_feats = all_feats.reshape(N, K, C_in)  # (N, K, C_in)

        # 2. AscendC 加速路径（仅 NPU + 扩展可加载 + 显式启用时）
        # Cube 内核：fp16 (N, K*C_in) @ fp16 (K*C_in, C_out) → fp32 (N, C_out)。
        # 默认走 torch 2D GEMM；设置环境变量 UNUM_SPCONV_USE_ASCENDC=1 启用内核。
        if features.device.type == "npu" and _USE_ASCENDC and ascendc.available():
            feats_l = all_feats.reshape(N, K * C_in).to(torch.float16)
            w_l = self._wT2d.to(torch.float16)
            if self.bias is not None:
                b_l = self.bias.to(torch.float32)
            else:
                b_l = torch.zeros(C_out, dtype=torch.float32, device=features.device)
            return ascendc.spconv_gemm(feats_l, w_l, b_l)

        # 3. 回退路径：2D GEMM（NPU 上 3D einsum 慢 6x，展平为 (N, K*C_in) @ (K*C_in, C_out)）
        out = all_feats.reshape(N, K * C_in) @ self._wT2d  # (N, C_out)

        if self.bias is not None:
            out.add_(self.bias)
        return out

    # ------------------------------------------------------------------
    # 邻居表缓存
    # ------------------------------------------------------------------

    def _fingerprint(self, *parts) -> str:
        """构造缓存 key。仅用 shape + 标量统计（sum/min），避免 .cpu().numpy() 同步开销。

        注意：相比原 md5(numpy bytes) 版本，冲突概率略升，但用于缓存 key 可接受
        （误命中只会导致用错邻居表，forward 数值会立刻错误，测试能发现）。

        ONNX 导出期间（torch.jit.is_tracing()）跳过数值统计，只用 shape/类型做 hash——
        否则 .tolist() 会在 ONNX 中展开成 Loop+Sequence 子图，ATC 不支持。
        每个调用返回一个唯一 nonce，确保 cache miss（每层独立 build）。
        """
        if torch.jit.is_tracing():
            # 导出期间：每个 _fingerprint 调用返回唯一标识，确保 _neighbor_cached
            # 必然 miss、每层独立 build 邻居表
            if not hasattr(self, "_trace_fp_counter"):
                self._trace_fp_counter = 0
            self._trace_fp_counter += 1
            return f"trace_fp_{id(self)}_{self._trace_fp_counter}"
        h = hashlib.md5()
        for p in parts:
            if torch.is_tensor(p):
                # 仅取 shape + 数值摘要，避免 .cpu().numpy().tobytes() 的 host 同步
                h.update(str(tuple(p.shape)).encode())
                if p.numel() > 0:
                    # 三统计合并为一次 .tolist() 同步（原来 3 次 .item()）
                    f = p.detach().to(torch.float32)
                    stats = torch.stack((f.sum(), f.min(), f.max())).tolist()
                    h.update(repr(stats).encode())
            else:
                h.update(repr(p).encode())
        return h.hexdigest()

    def _neighbor_cached(self, fp: str, build, device=None):
        """按 (fp, device) 缓存邻居表。

        - device=None 表示与 build 返回值同设备（向后兼容）
        - 命中后无设备往返：缓存的就是目标设备上的 nb
        """
        key = (fp, str(device) if device is not None else "cpu")
        nb = self._nb_cache.get(key)
        if nb is None:
            nb = build()
            self._nb_cache[key] = nb
            if len(self._nb_cache) > 8:
                self._nb_cache.pop(next(iter(self._nb_cache)))
        return nb

    def _downsample_coords(self, indices: torch.Tensor):
        """计算 SparseConv（步长卷积）的输出坐标，与官方 spconv 语义一致。

        官方 spconv 为 grid-based：输入点 c 被输出坐标 o 覆盖当且仅当
          o*stride - padding <= c <= o*stride - padding + kernel - 1
        即 o ∈ [ceil((c + padding - kernel + 1) / stride),
                floor((c + padding) / stride)]
        对所有输入点取并集（含边界过滤 + unique 由调用方完成）。

        向量化实现（无 Python 逐点循环——NPU 上逐点 .item() 同步在 1.7 万点
        时实测 ~9s）：每点的候选数 n_d = hi_d - lo_d + 1，用 repeat_interleave
        展开行、商余分解得到每个维度的偏移。行集合与 itertools.product 版一致
        （顺序不同，调用方会 unique）。
        """
        ndim = self.ndim
        inds = indices.long()
        N = inds.shape[0]
        if N == 0:
            return indices.clone()

        device = indices.device
        los, ns = [], []
        for d in range(ndim):
            c = inds[:, d + 1]
            lo = torch.ceil(
                (c + self.padding[d] - self.kernel_size[d] + 1).float() / self.stride[d]
            ).long()
            hi = torch.floor((c + self.padding[d]).float() / self.stride[d]).long()
            n = (hi - lo + 1).clamp_min(0)
            los.append(lo)
            ns.append(n)

        totals = ns[0]
        for n in ns[1:]:
            totals = totals * n
        if int(totals.sum().item()) == 0:
            return indices[:0].clone()

        keep = totals > 0
        keep_idx = torch.nonzero(keep).flatten()
        los = [torch.index_select(lo, 0, keep_idx) for lo in los]
        ns = [torch.index_select(n, 0, keep_idx) for n in ns]
        batch_kept = torch.index_select(inds, 0, keep_idx)[:, 0]
        Nk = keep_idx.size(0)
        T = torch.index_select(totals, 0, keep_idx)

        # 展开行：第 r 行重复 T[r] 次
        base = torch.repeat_interleave(torch.arange(Nk, device=device), T) # 在 ONNX 导出时会生成 Loop 算子
        starts = torch.zeros(Nk, dtype=torch.long, device=device)
        starts[1:] = torch.cumsum(T, dim=0)[:-1]
        rem = torch.arange(base.numel(), device=device) - starts[base]

        # 按 itertools.product 字典序分解 rem（外层维度变化最慢）：
        # rem = c0*(n1*n2) + c1*n2 + c2，从最内维（d 大）逐层取余
        cds = [None] * ndim
        r = rem
        for d in range(ndim - 1, -1, -1):
            if d > 0:
                div = ns[d][base]
                cds[d] = r % div
                r = r // div
            else:
                cds[d] = r
        cols = [batch_kept[base]] + [los[d][base] + cds[d] for d in range(ndim)]
        return torch.stack(cols, dim=1)

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        raise NotImplementedError


class SubMConv3d(SparseConvolution):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        indice_key=None,
        ndim=3,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            indice_key=indice_key,
            ndim=ndim,
        )

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        features, indices = x.features, x.indices
        ndim = self.ndim
        spatial = tuple(x.spatial_shape[:ndim])
        device = features.device

        # 帧内跨层共享（官方 spconv 按 indice_key 共享 indice pairs 的等价物）：
        # 邻居表只依赖 (coords, 核参数, spatial)。同帧内同参数层作用于相同
        # coords 时直接复用，免去 fingerprint 的 .item() 同步与重复构建
        # （~0.12s/层 @ 1.7 万体素）。缓存挂在 tensor 上随前向传播、每帧新建。
        # coords 以 data_ptr 入 key：同 key 层在 encoder/decoder 侧可能对应
        # 不同 coords（inverse 上采样后），此时必须分开构建。
        cache = getattr(x, "_layer_nb_cache", None)
        lparams = (
            ("subm",)
            + tuple(self.kernel_size)
            + tuple(self.padding)
            + tuple(self.stride)
            + tuple(self.dilation)
            + spatial
            + (indices.shape[0], indices.data_ptr(), str(device))
        )
        nb = cache.get(lparams) if cache is not None else None

        if nb is None:

            def build():
                # 不再 .cpu()，全程留在 features 设备
                in_c = indices.long().to(device).contiguous()
                return self._build_neighbor_idx(
                    in_c,
                    in_c,
                    spatial,
                    self.kernel_size,
                    self.padding,
                    self.stride,
                    self.dilation,
                )

            fp = self._fingerprint(
                indices, ("subm", self.kernel_size, self.padding, self.stride, spatial)
            )
            nb = self._neighbor_cached(fp, build, device)
            if cache is not None:
                cache[lparams] = nb
        out_feat = self._gather(features, nb)
        out = SparseConvTensor(
            out_feat, indices, x.spatial_shape, x.batch_size, grid=x.grid
        )
        out.indice_dict = dict(x.indice_dict)
        out._layer_nb_cache = getattr(x, "_layer_nb_cache", {})
        if self.indice_key is not None:
            out.indice_dict[self.indice_key] = {
                "in_coords": indices.clone(),
                "stride": tuple(self.stride),
                "padding": tuple(self.padding),
                "dilation": tuple(self.dilation),
                "kernel": tuple(self.kernel_size),
            }
        return out


class SubMConv2d(SubMConv3d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        indice_key=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            indice_key=indice_key,
            ndim=2,
        )


class SparseConv3d(SparseConvolution):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        indice_key=None,
        ndim=3,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            indice_key=indice_key,
            ndim=ndim,
        )

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        features, indices = x.features, x.indices
        in_shape = tuple(x.spatial_shape[: self.ndim])
        out_shape = tuple(
            (s + 2 * self.padding[d] - (self.kernel_size[d] - 1) - 1) // self.stride[d]
            + 1
            for d, s in enumerate(in_shape)
        )
        device = features.device

        def build():
            # 不再 .cpu()，全程留在 features 设备
            in_c = indices.long().to(device).contiguous()
            out_i = self._downsample_coords(in_c)
            keep = torch.ones(out_i.shape[0], dtype=torch.bool, device=device)
            for d in range(self.ndim):
                keep = keep & (out_i[:, d + 1] >= 0) & (out_i[:, d + 1] < out_shape[d])
            keep_idx = torch.nonzero(keep).flatten()
            out_i = torch.index_select(out_i, 0, keep_idx)
            # dedup via float64 key sort (AiCore) instead of torch.unique(dim=0)
            # which dispatches to UniqueWithCountsExt2 (AiCPU ~12ms per layer).
            if out_i.size(0) > 1:
                keys = self._encode(out_i, out_shape).double()
                order = torch.topk(keys, k=keys.size(0), largest=False).indices
                sk = keys[order]
                is_first = torch.cat(
                    [torch.ones(1, dtype=torch.bool, device=device), sk[1:] != sk[:-1]]
                )
                first_idx = torch.nonzero(is_first).flatten()
                sorted_out = torch.index_select(out_i, 0, order)
                out_i = torch.index_select(sorted_out, 0, first_idx)
            nb = self._build_neighbor_idx(
                in_c,
                out_i,
                in_shape,
                self.kernel_size,
                self.padding,
                self.stride,
                self.dilation,
            )
            return nb, out_i

        fp = self._fingerprint(
            indices,
            (
                "sparse",
                self.kernel_size,
                self.padding,
                self.stride,
                in_shape,
                out_shape,
            ),
        )
        nb, out_indices = self._neighbor_cached(fp, build, device)
        out_feat = self._gather(features, nb)

        spatial_shape = list(out_shape) + list(x.spatial_shape[self.ndim :])
        out = SparseConvTensor(
            out_feat, out_indices, spatial_shape, x.batch_size, grid=x.grid
        )
        out.indice_dict = dict(x.indice_dict)
        out._layer_nb_cache = getattr(x, "_layer_nb_cache", {})
        if self.indice_key is not None:
            out.indice_dict[self.indice_key] = {
                "in_coords": indices.clone(),
                "in_shape": list(in_shape),
                "stride": tuple(self.stride),
                "padding": tuple(self.padding),
                "dilation": tuple(self.dilation),
                "kernel": tuple(self.kernel_size),
            }
        return out


class SparseConv2d(SparseConv3d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        indice_key=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            indice_key=indice_key,
            ndim=2,
        )


class SparseInverseConv3d(SparseConvolution):
    """转置稀疏卷积（上采样）

    与配对的下采样 SparseConv3d 使用相同的 indice_key。
    输出坐标 = 下采样前的输入坐标（由输入 tensor 携带的 indice_dict 提供）。
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, indice_key=None, bias=True, ndim=3
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            bias=bias,
            indice_key=indice_key,
            ndim=ndim,
        )

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        key = self.indice_key
        # 读取配对下采样卷积记录的 fine coords 与参数
        if key is not None and key in x.indice_dict:
            info = x.indice_dict[key]
            out_indices = info["in_coords"]
            stride = tuple(info["stride"])
            padding = tuple(info["padding"])
            dilation = tuple(info.get("dilation", (1,) * self.ndim))
            kernel = tuple(info["kernel"])
        else:
            raise ValueError(
                f"SparseInverseConv3d requires matching SparseConv3d with indice_key='{key}' "
                f"recorded in the input tensor's indice_dict."
            )

        features = x.features
        in_shape = tuple(x.spatial_shape[: self.ndim])
        device = features.device

        # 帧内跨层共享（同 key 的 inverse 层作用相同 fine/coarse coords 对）
        cache = getattr(x, "_layer_nb_cache", None)
        lparams = (
            ("inv",)
            + kernel
            + padding
            + stride
            + dilation
            + in_shape
            + (
                out_indices.shape[0],
                out_indices.data_ptr(),
                x.indices.shape[0],
                x.indices.data_ptr(),
                str(device),
            )
        )
        nb = cache.get(lparams) if cache is not None else None

        if nb is None:

            def build():
                # 不再 .cpu()，全程留在 features 设备
                fine_c = out_indices.long().to(device).contiguous()
                coarse_c = x.indices.long().to(device).contiguous()
                return self._build_inverse_neighbor_idx(
                    fine_c,
                    coarse_c,
                    in_shape,
                    kernel,
                    padding,
                    stride,
                    dilation,
                )

            fp = self._fingerprint(
                out_indices,
                x.indices,
                ("inverse", kernel, padding, stride, dilation, in_shape),
            )
            nb = self._neighbor_cached(fp, build, device)
            if cache is not None:
                cache[lparams] = nb
        out = self._gather(features, nb)

        spatial_shape = list(
            info.get("in_shape", tuple(s * st for s, st in zip(in_shape, stride)))
        )
        out_t = SparseConvTensor(
            out, out_indices, spatial_shape, x.batch_size, grid=x.grid
        )
        out_t._layer_nb_cache = getattr(x, "_layer_nb_cache", {})
        return out_t


class SparseInverseConv2d(SparseInverseConv3d):
    def __init__(
        self, in_channels, out_channels, kernel_size, indice_key=None, bias=True
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            indice_key=indice_key,
            bias=bias,
            ndim=2,
        )
        self.ndim = 2

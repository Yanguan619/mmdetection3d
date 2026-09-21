from __future__ import annotations

from typing import Any

import torch
from torch import nn


class SparseConvTensor:
    """spconv SparseConvTensor 的 torch-native 实现（纯 PyTorch，CPU/GPU/NPU 通用）

    封装稀疏体素数据：特征 + 坐标 + 空间形状
    """

    def __init__(
        self,
        features: torch.Tensor,
        indices: torch.Tensor,
        spatial_shape,
        batch_size: int,
        grid: torch.Tensor | None = None,
    ):
        """
        Args:
            features: (N, C) 体素特征
            indices:  (N, 4) 坐标 (batch_idx, x, y, z)
            spatial_shape: (D, H, W) 空间大小（3D），(H, W) 空间大小（2D），
                           OpenPCDet 传 [D, H, W, 1, 0, 0] 这类 6 维时取前 ndim 维
            batch_size: batch 大小
            grid: 可选的密集网格索引（用于加速查找）
        """
        self.features = features
        self.indices = indices
        self.spatial_shape = tuple(spatial_shape)
        self.batch_size = batch_size
        self.grid = grid
        # 记录是否有索引映射（用于 replace_feature / SparseInverseConv）
        self._indice_dict: dict[str, Any] = {}
        # 帧级邻居表缓存：同一帧前向传播中，同参数卷积层作用于相同 coords 时
        # 共享邻居表（官方 spconv 按 indice_key 共享 indice pairs 的等价物）。
        # 每次构造新 tensor 即新缓存——无跨帧陈旧数据风险；由各卷积层 forward
        # 与 replace_feature 传播。
        self._layer_nb_cache: dict = {}

    @property
    def indice_dict(self):
        return self._indice_dict

    @indice_dict.setter
    def indice_dict(self, val):
        self._indice_dict = val

    @property
    def ndim(self) -> int:
        """空间维度数（2D -> 2, 3D -> 3）"""
        return self.indices.shape[1] - 1

    @property
    def spatial_size(self):
        return self.spatial_shape[: self.ndim]

    @property
    def shape(self):
        """返回 (N, C)"""
        return self.features.shape

    def replace_feature(self, features: torch.Tensor):
        """用新特征替换当前特征，保持其他属性不变"""
        out = SparseConvTensor(
            features, self.indices, self.spatial_shape, self.batch_size, self.grid
        )
        out.indice_dict = self.indice_dict
        out._layer_nb_cache = self._layer_nb_cache
        out._nb_merge_v2 = getattr(self, "_nb_merge_v2", None)
        return out

    def dense(self, channels_first: bool = True) -> torch.Tensor:
        spatial = self.spatial_shape[: self.ndim]
        _N, C = self.features.shape
        B = int(self.batch_size) if isinstance(self.batch_size, torch.Tensor) else self.batch_size

        if channels_first:
            dense = torch.zeros(
                (B, C, *spatial), dtype=self.features.dtype, device=self.features.device
            )
        else:
            dense = torch.zeros(
                (B, *spatial, C), dtype=self.features.dtype, device=self.features.device
            )

        indices = self.indices.long()
        valid = (indices[:, 0] >= 0) & (indices[:, 0] < B)
        for d in range(self.ndim):
            valid &= (indices[:, d + 1] >= 0) & (indices[:, d + 1] < spatial[d])
        if not valid.any():
            return dense

        valid_idx = indices[valid]
        valid_feat = self.features[valid]

        if channels_first:
            b = valid_idx[:, 0]
            x = valid_idx[:, 1]
            y = valid_idx[:, 2]
            if self.ndim == 3:
                z = valid_idx[:, 3]
                # dense is (B, C, D, H, W); flat index of (b, c, x, y, z):
                # b*C*D*H*W + c*D*H*W + x*H*W + y*W + z
                base = (
                    b * C * spatial[0] * spatial[1] * spatial[2]
                    + x * spatial[1] * spatial[2]
                    + y * spatial[2]
                    + z
                )
                c_offset = (
                    torch.arange(C, device=valid_feat.device)
                    * spatial[0]
                    * spatial[1]
                    * spatial[2]
                )
            else:
                # dense is (B, C, H, W); flat index of (b, c, x, y):
                # b*C*H*W + c*H*W + x*W + y
                base = b * C * spatial[0] * spatial[1] + x * spatial[1] + y
                c_offset = (
                    torch.arange(C, device=valid_feat.device) * spatial[0] * spatial[1]
                )
            flat_idx = base.unsqueeze(1) + c_offset.unsqueeze(0)
            dense.view(-1).scatter_add_(0, flat_idx.reshape(-1), valid_feat.reshape(-1))
        else:
            if self.ndim == 3:
                x, y, z = valid_idx[:, 1], valid_idx[:, 2], valid_idx[:, 3]
                flat = (
                    valid_idx[:, 0] * spatial[0] * spatial[1] * spatial[2]
                    + x * spatial[1] * spatial[2]
                    + y * spatial[2]
                    + z
                )
            else:
                x, y = valid_idx[:, 1], valid_idx[:, 2]
                flat = valid_idx[:, 0] * spatial[0] * spatial[1] + x * spatial[1] + y
            flat = flat.unsqueeze(1).expand(-1, C)
            dense.view(-1, C).scatter_add_(0, flat, valid_feat)

        return dense

    @staticmethod
    def from_dense(
        dense: torch.Tensor, channels_first: bool = True
    ) -> "SparseConvTensor":
        """从密集张量构建稀疏张量"""
        if channels_first:
            # (B, C, D, H, W) -> 提取非零位置
            B, C, D, H, W = dense.shape
            mask = dense.abs().sum(dim=1) > 0  # (B, D, H, W)
        else:
            # (B, D, H, W, C)
            B, D, H, W, C = dense.shape
            mask = dense.abs().sum(dim=-1) > 0  # (B, D, H, W)

        indices_list = []
        features_list = []
        for b in range(B):
            nonzero = mask[b].nonzero()  # (N, 3)
            for idx in nonzero:
                x, y, z = idx[0].item(), idx[1].item(), idx[2].item()
                indices_list.append([b, x, y, z])
                if channels_first:
                    features_list.append(dense[b, :, x, y, z])
                else:
                    features_list.append(dense[b, x, y, z, :])

        if features_list:
            indices = torch.tensor(indices_list, dtype=torch.int32, device=dense.device)
            features = torch.stack(features_list).float()
        else:
            indices = torch.empty(0, 4, dtype=torch.int32, device=dense.device)
            features = torch.empty(
                0, C if channels_first else dense.shape[-1], device=dense.device
            )

        return SparseConvTensor(features, indices, (D, H, W), B)

    def to(self, device) -> "SparseConvTensor":
        """设备迁移，返回新对象。新对象拥有独立的特征/坐标副本，修改不影响原对象。

        注意：官方 spconv 的 SparseConvTensor.to() 同样返回新对象。
        """
        out = SparseConvTensor(
            self.features.to(device).clone(),
            self.indices.to(device).clone(),
            self.spatial_shape,
            self.batch_size,
            self.grid,
        )
        out.indice_dict = self.indice_dict
        return out

    def cpu(self) -> "SparseConvTensor":
        return self.to("cpu")

    def cuda(self) -> "SparseConvTensor":
        return self.to("cuda")

    def __repr__(self):
        return (
            f"SparseConvTensor(features={self.features.shape}, "
            f"indices={self.indices.shape}, "
            f"spatial_shape={self.spatial_shape}, "
            f"batch_size={self.batch_size})"
        )


class SparseModule(nn.Module):
    """spconv SparseModule 的 torch-native 实现（纯 PyTorch，CPU/GPU/NPU 通用）

    所有稀疏模块的基类，接受 SparseConvTensor 作为输入，输出 SparseConvTensor
    """

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        raise NotImplementedError


# 运行时检测官方 spconv / mmcv.ops 是否可用，若可用则把对应的 SparseModule
# 加入稀疏模块类型列表（mmdet3d 的 SparseBasicBlock 继承自官方或 mmcv 的
# SparseModule，具体取决于 is_spconv2_available 的值）。
_SPARSE_MODULE_TYPES: tuple = (SparseModule,)
try:
    import spconv.pytorch as _sp_pt

    _SPARSE_MODULE_TYPES = (SparseModule, _sp_pt.SparseModule)
except ImportError:
    try:
        from mmcv.ops import SparseModule as _mmcv_sp

        _SPARSE_MODULE_TYPES = (SparseModule, _mmcv_sp)
    except (ImportError, AttributeError):
        pass


class SparseSequential(SparseModule):
    """spconv SparseSequential 的 torch-native 实现（纯 PyTorch，CPU/GPU/NPU 通用）

    顺序容器，依次执行各模块，数据流为 SparseConvTensor
    """

    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], dict):
            # 字典方式构造: SparseSequential({'conv1': conv, 'relu1': relu})
            for name, module in args[0].items():
                self.add_module(name, module)
        elif len(args) == 1 and isinstance(args[0], (list, tuple)):
            # 列表方式构造
            for idx, module in enumerate(args[0]):
                self.add_module(str(idx), module)
        else:
            # 位置参数构造: SparseSequential(conv, relu, conv2)
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, x):
        """依次通过所有模块

        - 稀疏模块（SparseModule）直接以 SparseConvTensor 为输入输出；
        - 普通 nn.Module（BatchNorm/ReLU 等）只作用在 .features 上，再 replace_feature。
        """
        for module in self:
            if isinstance(x, SparseConvTensor) and not isinstance(
                module, _SPARSE_MODULE_TYPES
            ):
                x = x.replace_feature(module(x.features))
            else:
                x = module(x)
        return x

    def __len__(self):
        return len(self._modules)

    def __getitem__(self, idx) -> SparseModule | "SparseSequential":
        """支持索引访问"""
        if isinstance(idx, slice):
            modules = list(self._modules.values())[idx]
            return SparseSequential(*modules)
        else:
            return list(self._modules.values())[idx]

    def __iter__(self):
        return iter(self._modules.values())


# ============================================================
# 适配层：让 SparseConv3dCPU / SubMConv3dCPU 接受 SparseConvTensor
# ============================================================


class SparseConv3dAdapter(SparseModule):
    """将 SparseConv3d 适配为 SparseModule 接口"""

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()
        from .conv import SparseConv3d as SparseConv3dCPU

        self.conv = SparseConv3dCPU(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        # SparseConv3d.forward 接受 SparseConvTensor，返回 SparseConvTensor
        return self.conv(x)


class SubMConv3dAdapter(SparseModule):
    """将 SubMConv3d 适配为 SparseModule 接口"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        super().__init__()
        from .conv import SubMConv3d as SubMConv3dCPU

        self.conv = SubMConv3dCPU(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        # SubMConv3d.forward 接受 SparseConvTensor，返回 SparseConvTensor
        return self.conv(x)


# ============================================================
# 常用激活/归一化层的稀疏适配
# ============================================================


class SparseReLU(SparseModule):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        return x.replace_feature(self.relu(x.features))


class SparseBatchNorm1d(SparseModule):
    """对稀疏特征做 BatchNorm（等价于对 N 个体素做 BN）"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        return x.replace_feature(self.bn(x.features))


class SparseLinear(SparseModule):
    """对每个体素独立做线性变换"""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        return x.replace_feature(self.linear(x.features))

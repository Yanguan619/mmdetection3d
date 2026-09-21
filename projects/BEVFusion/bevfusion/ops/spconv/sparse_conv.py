import torch
from torch import nn

from .conv import (
    SparseConv2d,
    SparseConv3d,
    SparseConvolution,
    SparseInverseConv2d,
    SparseInverseConv3d,
    SubMConv2d,
    SubMConv3d,
)

# 兼容旧名
SparseConv3dCPU = SparseConv3d
SubMConv3dCPU = SubMConv3d

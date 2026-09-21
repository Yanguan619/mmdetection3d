from . import conv as conv
from . import spconv_ascendc
from . import utils as utils
from .conv import SparseConv2d, SparseInverseConv2d, SparseInverseConv3d, SubMConv2d

# 兼容旧版 sparse_conv.py 中直接引用 SparseConv3dCPU / SubMConv3dCPU 的写法
from .conv import SparseConv3d as SparseConv3dCPU
from .conv import SubMConv3d as SubMConv3dCPU
from .sparse_conv import SparseConv3d, SubMConv3d
from .sparse_modules import SparseConvTensor, SparseModule, SparseSequential

__version__ = "0.0.1"
__all__ = [
    "SparseConv2d",
    "SparseConv3d",
    "SparseConv3dCPU",
    "SparseConvTensor",
    "SparseInverseConv2d",
    "SparseInverseConv3d",
    "SparseModule",
    "SparseSequential",
    "SubMConv2d",
    "SubMConv3d",
    "SubMConv3dCPU",
    "conv",
    "spconv_ascendc",
    "utils",
]

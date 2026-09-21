import os

from .bev_pool import bev_pool, bev_pool_torch
from .bev_pool.bev_pool import _bev_pool_pytorch
from .voxel import DynamicScatter, Voxelization, dynamic_scatter, voxelization

# Select the bev_pool backend via the `BEVFUSION_BEV_POOL_BACKEND` env var:
#   - 'auto' (default):  由 bev_pool() 内部分发——CUDA ext 可用走 CUDA，否则
#                        优先 unum_ops 原子加 AscendC（2026-09-20 重写：免排序
#                        scatter，1.84M 点 31.4ms / max_abs 4.8e-7），其次
#                        mx_driving（DrivingSDK 官方 bev_pool_v3，34ms），
#                        （~2.4x 快于 QuickCumsum，精度同为 fp32 重排级），
#                        不可加载时回退 QuickCumsum
#   - 'mx_driving':      强制 mx_driving（DrivingSDK）bev_pool_v3 后端
#   - 'torch':           强制 QuickCumsum 纯 PyTorch（CPU / NPU 均可用）
#   - 'scatter':         scatter_add 纯 PyTorch（bit-exact 但 NPU 上 AICPU 慢 ~6x）
#   - 'cuda':            使用编译的 CUDA 扩展
# NPU 性能注意（性能.md §3）：scatter_add 版把 index 展开成 (N,C) int64 走
# AICPU 逐点累加（BEVFusion 相机分支 ~1.84M 点实测 1.32s/帧）；QuickCumsum 版
# 全链路 ~0.40s，unum_ops 原子加版 31.4ms（max_abs 4.8e-7 vs mx_driving
# 1.95e-5，fp32 原子重排噪声级），mx_driving 版 34.4ms。
_backend = os.environ.get('BEVFUSION_BEV_POOL_BACKEND', 'auto')

if _backend == 'scatter':
    bev_pool = bev_pool_torch
elif _backend in ('torch', 'quickcumsum'):
    bev_pool = _bev_pool_pytorch
elif _backend == 'cuda':
    try:
        from .bev_pool import bev_pool_ext  # noqa: F401

        if bev_pool_ext is None:
            bev_pool = _bev_pool_pytorch
    except ImportError:
        bev_pool = _bev_pool_pytorch
# 'auto': 保留顶部导入的 bev_pool()，由其内部分发（CUDA -> AscendC -> QuickCumsum）

__all__ = [
    'bev_pool', 'bev_pool_torch', 'Voxelization', 'voxelization',
    'dynamic_scatter', 'DynamicScatter'
]

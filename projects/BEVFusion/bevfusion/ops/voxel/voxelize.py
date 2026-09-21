import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair

try:
    from .voxel_layer import dynamic_voxelize, hard_voxelize
    _HAS_EXT = True
except (ImportError, ModuleNotFoundError):
    _HAS_EXT = False

try:
    from unum_ops.voxelization import voxelization as _voxelization_ascend
    _HAS_ASCEND_EXT = True
except (ImportError, ModuleNotFoundError):
    _HAS_ASCEND_EXT = False


def _dynamic_voxelize_pytorch(points, voxel_size, coors_range, ndim=3):
    device = points.device
    voxel_size = torch.tensor(voxel_size, dtype=torch.float, device=device)
    coors_range = torch.tensor(coors_range, dtype=torch.float, device=device)
    grid_size = ((coors_range[ndim:] - coors_range[:ndim]) / voxel_size).long()

    coors = torch.floor((points[:, :ndim] - coors_range[:ndim]) / voxel_size).long()
    out_of_range = (coors < 0) | (coors >= grid_size.unsqueeze(0))
    out_of_range = out_of_range.any(dim=1)
    coors[out_of_range] = -1
    return coors


def _hard_voxelize_ascendc(points, voxel_size, coors_range, max_points=35,
                           max_voxels=20000, ndim=3):
    """AscendC voxelization. Requires (N, 4) points, pads extra features with 0."""
    device = points.device
    orig_features = points.size(1)
    points_4 = points[:, :4].contiguous()
    out = _voxelization_ascend(
        points_4,
        voxel_size=list(voxel_size),
        pcr=list(coors_range),
        max_num_points=int(max_points),
        max_voxels=int(max_voxels),
    )
    voxels, coords, num_points, num_voxels = out
    if orig_features > 4:
        pad = torch.zeros(voxels.shape[0], voxels.shape[1],
                          orig_features - 4,
                          dtype=voxels.dtype, device=device)
        voxels = torch.cat([voxels, pad], dim=-1)
    return voxels, coords, num_points


def _hard_voxelize_pytorch(points, voxel_size, coors_range, max_points=35, max_voxels=20000, ndim=3):
    device = points.device
    voxel_size_t = torch.tensor(voxel_size, dtype=torch.float, device=device)
    coors_range_t = torch.tensor(coors_range, dtype=torch.float, device=device)
    grid_size = ((coors_range_t[ndim:] - coors_range_t[:ndim]) / voxel_size_t).long()

    coors = torch.floor((points[:, :ndim] - coors_range_t[:ndim]) / voxel_size_t).long()
    out_of_range = (coors < 0) | (coors >= grid_size.unsqueeze(0))
    out_of_range = out_of_range.any(dim=1)

    valid = ~out_of_range
    valid_idx = torch.nonzero(valid).flatten()
    valid_coors = torch.index_select(coors, 0, valid_idx)
    valid_points = torch.index_select(points, 0, valid_idx)
    num_valid = valid_points.size(0)

    N = points.size(1)
    if num_valid == 0:
        return (points.new_zeros(0, max_points, N),
                points.new_zeros(0, ndim, dtype=torch.int),
                points.new_zeros(0, dtype=torch.int))

    linear_idx = (
        valid_coors[:, 0] * grid_size[1] * grid_size[2] +
        valid_coors[:, 1] * grid_size[2] +
        valid_coors[:, 2])

    # int64 argsort（AICPU，~44ms）——NPU 上无更快的精确整数排序；
    # float64 虽走 AICore 但排序不稳定（tie 打乱），已实测弃用（2026-09-17）
    sort_order = torch.argsort(linear_idx, stable=True)
    sorted_points = valid_points[sort_order]
    sorted_coors = valid_coors[sort_order]
    sorted_lin = linear_idx[sort_order]

    # 边界法替代 unique_consecutive（走 AICPU）：
    # diffs = 每个位置与前一项比较 → AI_CORE 向量化
    # inverse = cumsum(is_first) - 1 → 跳变点累加得 voxel 编号
    # unique_lin = sorted_lin[is_first] → 每个 voxel 的首 key
    # counts = scatter_add(1, inverse) → 每个 voxel 的点数
    diffs = sorted_lin[1:] != sorted_lin[:-1]  # (N-1,) bool
    is_first = torch.cat([torch.ones(1, device=points.device, dtype=torch.bool),
                         diffs])
    inverse = torch.cumsum(is_first.to(torch.int64), 0) - 1

    unique_lin = sorted_lin[is_first]
    num_voxels = unique_lin.size(0)
    counts = torch.zeros(num_voxels, dtype=torch.int64, device=points.device)
    counts.scatter_add_(0, inverse, torch.ones_like(inverse))

    if max_voxels != -1 and num_voxels > max_voxels:
        keep_mask = inverse < max_voxels
        keep_idx = torch.nonzero(keep_mask).flatten()
        sorted_points = torch.index_select(sorted_points, 0, keep_idx)
        sorted_coors = torch.index_select(sorted_coors, 0, keep_idx)
        sorted_lin = torch.index_select(sorted_lin, 0, keep_idx)
        num_voxels = max_voxels
        # 截断后重算边界
        diffs = sorted_lin[1:] != sorted_lin[:-1]
        is_first = torch.cat([torch.ones(1, device=points.device, dtype=torch.bool),
                             diffs])
        inverse = torch.cumsum(is_first.to(torch.int64), 0) - 1
        unique_lin = sorted_lin[is_first]
        counts = torch.zeros(num_voxels, dtype=torch.int64, device=points.device)
        counts.scatter_add_(0, inverse, torch.ones_like(inverse))

    # compute position within each voxel
    cum_counts = torch.cumsum(counts, 0)
    voxel_starts = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=device), cum_counts[:-1]])
    # repeat_interleave 构建 (N,) 展开后起止位置 → 43ms 瓶颈；
    # 用 index_select gather 等价：每个点的起始位置 = voxel_starts[inverse[k]]
    total_pts = sorted_points.size(0)
    point_pos = torch.arange(total_pts, device=device)
    point_pos = point_pos - torch.index_select(voxel_starts, 0, inverse[:total_pts])

    if max_points != -1:
        keep = point_pos < max_points
        keep_idx = torch.nonzero(keep).flatten()
        sorted_points = torch.index_select(sorted_points, 0, keep_idx)
        sorted_coors = torch.index_select(sorted_coors, 0, keep_idx)
        inverse = torch.index_select(inverse, 0, keep_idx)
        point_pos = torch.index_select(point_pos, 0, keep_idx)
        counts = counts.clamp(max=max_points)

    # build output
    voxels = torch.zeros(num_voxels, max_points, N, device=device, dtype=points.dtype)
    num_points_per_voxel = torch.zeros(num_voxels, device=device, dtype=torch.int)

    if sorted_points.size(0) > 0:
        voxels[inverse, point_pos] = sorted_points
        num_points_per_voxel = counts.to(torch.int)

    # recover coords from linear index
    Gx, Gy, Gz = grid_size[0].item(), grid_size[1].item(), grid_size[2].item()
    out_coors = torch.zeros(num_voxels, ndim, device=device, dtype=torch.int)
    out_coors[:, 0] = unique_lin // (Gy * Gz)
    out_coors[:, 1] = (unique_lin % (Gy * Gz)) // Gz
    out_coors[:, 2] = unique_lin % Gz

    return voxels, out_coors, num_points_per_voxel


class _Voxelization(Function):

    @staticmethod
    def forward(ctx,
                points,
                voxel_size,
                coors_range,
                max_points=35,
                max_voxels=20000,
                deterministic=True):
        if _HAS_EXT:
            return _voxelization_ext_forward(points, voxel_size, coors_range, max_points, max_voxels, deterministic)
        elif max_points == -1 or max_voxels == -1:
            return _dynamic_voxelize_pytorch(points, voxel_size, coors_range, 3)
        else:
            return _hard_voxelize_pytorch(points, voxel_size, coors_range, max_points, max_voxels, 3)


def _voxelization_ext_forward(points, voxel_size, coors_range, max_points, max_voxels, deterministic):
    if max_points == -1 or max_voxels == -1:
        coors = points.new_zeros(size=(points.size(0), 3), dtype=torch.int)
        dynamic_voxelize(points, coors, voxel_size, coors_range, 3)
        return coors
    else:
        voxels = points.new_zeros(
            size=(max_voxels, max_points, points.size(1)))
        coors = points.new_zeros(size=(max_voxels, 3), dtype=torch.int)
        num_points_per_voxel = points.new_zeros(
            size=(max_voxels, ), dtype=torch.int)
        voxel_num = hard_voxelize(
            points,
            voxels,
            coors,
            num_points_per_voxel,
            voxel_size,
            coors_range,
            max_points,
            max_voxels,
            3,
            deterministic,
        )
        voxels_out = voxels[:voxel_num]
        coors_out = coors[:voxel_num]
        num_points_per_voxel_out = num_points_per_voxel[:voxel_num]
        return voxels_out, coors_out, num_points_per_voxel_out


voxelization = _Voxelization.apply


class Voxelization(nn.Module):

    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 max_num_points,
                 max_voxels=20000,
                 deterministic=True):
        super(Voxelization, self).__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        if isinstance(max_voxels, tuple):
            self.max_voxels = max_voxels
        else:
            self.max_voxels = _pair(max_voxels)
        self.deterministic = deterministic

        point_cloud_range = torch.tensor(
            point_cloud_range, dtype=torch.float32)
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        grid_size = (point_cloud_range[3:] -
                     point_cloud_range[:3]) / voxel_size
        grid_size = torch.round(grid_size).long()
        input_feat_shape = grid_size[:2]
        self.grid_size = grid_size
        self.pcd_shape = [*input_feat_shape, 1]

    def forward(self, input):
        if self.training:
            max_voxels = self.max_voxels[0]
        else:
            max_voxels = self.max_voxels[1]

        return voxelization(
            input,
            self.voxel_size,
            self.point_cloud_range,
            self.max_num_points,
            max_voxels,
            self.deterministic,
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'voxel_size=' + str(self.voxel_size)
        tmpstr += ', point_cloud_range=' + str(self.point_cloud_range)
        tmpstr += ', max_num_points=' + str(self.max_num_points)
        tmpstr += ', max_voxels=' + str(self.max_voxels)
        tmpstr += ', deterministic=' + str(self.deterministic)
        tmpstr += ')'
        return tmpstr
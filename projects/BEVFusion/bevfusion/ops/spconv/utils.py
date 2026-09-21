import numpy as np


class VoxelGeneratorV2:
    """spconv.utils.VoxelGeneratorV2 的 torch-native/numpy 实现（不依赖编译扩展，CPU/GPU/NPU 通用）

    与 spconv 1.x 的 VoxelGeneratorV2 行为对齐：
      - 输入：points (N, 3 + C)，前 3 列为 [x, y, z]
      - 输出（generate 返回 dict）：
          voxels:             (num_voxels, max_num_points, 3 + C)
          coordinates:        (num_voxels, 3)  顺序 [z, y, x]
          num_points_per_voxel: (num_voxels,)
    """

    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        self.voxel_size = np.asarray(voxel_size, dtype=np.float32)
        self.point_cloud_range = np.asarray(point_cloud_range, dtype=np.float32)
        self.max_num_points = int(max_num_points)
        self.max_voxels = int(max_voxels)
        if self.point_cloud_range.shape[0] != 6:
            raise ValueError(
                f"point_cloud_range must have 6 elements (x,y,z,x_max,y_max,z_max), "
                f"got {self.point_cloud_range.shape[0]}")
        if np.any(self.voxel_size <= 0):
            raise ValueError(f"voxel_size must be positive, got {voxel_size}")
        if self.max_num_points <= 0 or self.max_voxels <= 0:
            raise ValueError(
                f"max_num_points and max_voxels must be positive, "
                f"got {max_num_points}, {max_voxels}")
        span = self.point_cloud_range[3:] - self.point_cloud_range[:3]
        self._grid_size = np.ceil(span / self.voxel_size).astype(np.int32)

    def generate(self, points):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(
                f"points must be 2D with >=3 columns [x,y,z,...], got shape {points.shape}")
        feat_dim = points.shape[1]

        coords = np.floor((points[:, :3] - self.point_cloud_range[:3]) / self.voxel_size).astype(np.int32)

        valid = np.all(coords >= 0, axis=1) & np.all(coords < self._grid_size, axis=1)
        points = points[valid]
        coords = coords[valid]

        # 空输入处理
        if len(points) == 0:
            return {
                'voxels': np.zeros((0, self.max_num_points, feat_dim), dtype=np.float32),
                'coordinates': np.zeros((0, 3), dtype=np.int32),
                'num_points_per_voxel': np.zeros((0,), dtype=np.int32),
            }

        # spconv 1.x 使用 z 优先的字典序排序（lexsort 以最后一个 key 为主，故 z 放最后）
        sort_idx = np.lexsort((coords[:, 0], coords[:, 1], coords[:, 2]))
        points = points[sort_idx]
        coords = coords[sort_idx]

        # 分组（排序后同组连续），向量化填充 voxels
        uniq_coords, inverse, counts = np.unique(coords, axis=0, return_inverse=True, return_counts=True)
        num_voxels = min(len(counts), self.max_voxels)
        if num_voxels == 0:
            return {
                'voxels': np.zeros((0, self.max_num_points, points.shape[1]), dtype=np.float32),
                'coordinates': np.zeros((0, 3), dtype=np.int32),
                'num_points_per_voxel': np.zeros((0,), dtype=np.int32),
            }

        boundaries = np.concatenate([[0], np.cumsum(counts[:num_voxels])])
        keep = inverse < num_voxels
        kept_idx = np.arange(len(points))[keep]
        group_start = np.repeat(boundaries[:-1], counts[:num_voxels])
        pos = kept_idx - group_start  # 组内序号（0-based）

        voxels = np.zeros((num_voxels, self.max_num_points, points.shape[1]), dtype=np.float32)
        mask = pos < self.max_num_points
        voxels[inverse[keep][mask], pos[mask]] = points[keep][mask]
        num_points_per_voxel = np.minimum(counts[:num_voxels], self.max_num_points).astype(np.int32)
        coords_out = uniq_coords[:num_voxels][:, [2, 1, 0]]

        return {
            'voxels': voxels,
            'coordinates': coords_out,
            'num_points_per_voxel': num_points_per_voxel,
        }


class VoxelGenerator(VoxelGeneratorV2):
    pass


class Point2VoxelCPU3d(VoxelGeneratorV2):
    """cumm Point2VoxelCPU3d 接口的简化适配（仅保留 CPU 路径）"""

    def point_to_voxel(self, points):
        raise RuntimeError('Point2VoxelCPU3d requires cumm; use VoxelGeneratorV2 instead')
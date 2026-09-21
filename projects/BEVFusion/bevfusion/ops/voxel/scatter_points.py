import torch
from torch import nn
from torch.autograd import Function

try:
    from .voxel_layer import (dynamic_point_to_voxel_backward,
                              dynamic_point_to_voxel_forward)
    _HAS_EXT = True
except (ImportError, ModuleNotFoundError):
    _HAS_EXT = False


def _dynamic_point_to_voxel_forward_pytorch(feats, coors, reduce_type):
    num_input = feats.size(0)
    num_feats = feats.size(1)

    if num_input == 0:
        return (feats.clone().detach(), coors.clone().detach(),
                coors.new_empty((0, ), dtype=torch.int32),
                coors.new_empty((0, ), dtype=torch.int32))

    coors_clean = coors.masked_fill(coors.lt(0).any(-1, True), -1)
    out_coors, inverse, reduce_count = torch.unique(
        coors_clean, dim=0, return_inverse=True, return_counts=True)

    if out_coors.shape[0] > 0 and out_coors[0, 0] < 0:
        out_coors = out_coors[1:]
        reduce_count = reduce_count[1:]
        inverse = inverse - 1

    coors_map = inverse.to(torch.int32)
    reduce_count = reduce_count.to(torch.int32)

    reduced_feats = torch.zeros((out_coors.size(0), num_feats),
                                dtype=feats.dtype,
                                device=feats.device)
    if coors_map.numel() > 0:
        if reduce_type == 'max':
            reduced_feats = reduced_feats.scatter_reduce(
                0,
                coors_map.unsqueeze(1).expand(-1, num_feats),
                feats,
                reduce='amax',
                include_self=True)
        elif reduce_type == 'sum':
            reduced_feats = reduced_feats.scatter_add(
                0, coors_map.unsqueeze(1).expand(-1, num_feats), feats)
        elif reduce_type == 'mean':
            reduced_feats = reduced_feats.scatter_add(
                0, coors_map.unsqueeze(1).expand(-1, num_feats), feats)
            reduced_feats /= reduce_count.unsqueeze(-1).to(feats.dtype)
        else:
            raise NotImplementedError(f'reduce_type {reduce_type} not '
                                      f'supported')

    return reduced_feats, out_coors, coors_map, reduce_count


def _dynamic_point_to_voxel_backward_pytorch(grad_feats, grad_reduced_feats,
                                             feats, reduced_feats, coors_map,
                                             reduce_count, reduce_type):
    grad_feats.fill_(0)
    num_input = feats.size(0)
    num_reduced = reduced_feats.size(0)
    num_feats = feats.size(1)

    if num_input == 0 or num_reduced == 0:
        return

    coors_map = coors_map.to(torch.long)
    if reduce_type in ('mean', 'sum'):
        if reduce_type == 'mean':
            grad_reduced_feats = grad_reduced_feats / \
                reduce_count.unsqueeze(-1).to(grad_reduced_feats.dtype)
        grad_feats = grad_feats.scatter_add(
            0, coors_map.unsqueeze(1).expand(-1, num_feats),
            grad_reduced_feats[coors_map])
    elif reduce_type == 'max':
        max_vals = reduced_feats[coors_map]
        is_max = (feats == max_vals)
        grad_scatter = grad_reduced_feats[coors_map] * is_max.to(
            grad_reduced_feats.dtype)
        grad_feats = grad_feats.scatter_add(
            0, coors_map.unsqueeze(1).expand(-1, num_feats), grad_scatter)
    else:
        raise NotImplementedError(f'reduce_type {reduce_type} not supported')


class _dynamic_scatter(Function):

    @staticmethod
    def forward(ctx, feats, coors, reduce_type='max'):
        if _HAS_EXT:
            results = dynamic_point_to_voxel_forward(feats, coors, reduce_type)
        else:
            results = _dynamic_point_to_voxel_forward_pytorch(
                feats, coors, reduce_type)
        (voxel_feats, voxel_coors, point2voxel_map,
         voxel_points_count) = results
        ctx.reduce_type = reduce_type
        ctx.save_for_backward(feats, voxel_feats, point2voxel_map,
                              voxel_points_count)
        ctx.mark_non_differentiable(voxel_coors)
        return voxel_feats, voxel_coors

    @staticmethod
    def backward(ctx, grad_voxel_feats, grad_voxel_coors=None):
        (feats, voxel_feats, point2voxel_map,
         voxel_points_count) = ctx.saved_tensors
        grad_feats = torch.zeros_like(feats)
        if _HAS_EXT:
            dynamic_point_to_voxel_backward(
                grad_feats,
                grad_voxel_feats.contiguous(),
                feats,
                voxel_feats,
                point2voxel_map,
                voxel_points_count,
                ctx.reduce_type,
            )
        else:
            _dynamic_point_to_voxel_backward_pytorch(
                grad_feats, grad_voxel_feats.contiguous(), feats, voxel_feats,
                point2voxel_map, voxel_points_count, ctx.reduce_type)
        return grad_feats, None, None


dynamic_scatter = _dynamic_scatter.apply


class DynamicScatter(nn.Module):

    def __init__(self, voxel_size, point_cloud_range, average_points: bool):
        super(DynamicScatter, self).__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.average_points = average_points

    def forward_single(self, points, coors):
        reduce = 'mean' if self.average_points else 'max'
        return dynamic_scatter(points.contiguous(), coors.contiguous(), reduce)

    def forward(self, points, coors):
        if coors.size(-1) == 3:
            return self.forward_single(points, coors)
        else:
            batch_size = coors[-1, 0] + 1
            voxels, voxel_coors = [], []
            for i in range(batch_size):
                inds = torch.where(coors[:, 0] == i)
                voxel, voxel_coor = self.forward_single(
                    points[inds], coors[inds][:, 1:])
                coor_pad = nn.functional.pad(
                    voxel_coor, (1, 0), mode='constant', value=i)
                voxel_coors.append(coor_pad)
                voxels.append(voxel)
            features = torch.cat(voxels, dim=0)
            feature_coors = torch.cat(voxel_coors, dim=0)

            return features, feature_coors

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'voxel_size=' + str(self.voxel_size)
        tmpstr += ', point_cloud_range=' + str(self.point_cloud_range)
        tmpstr += ', average_points=' + str(self.average_points)
        tmpstr += ')'
        return tmpstr
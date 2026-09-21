#!/usr/bin/env python
"""NPU-adapted BEVFusion full-dataset evaluation.

Usage:
  python NPU/npu_test.py \
      projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
      projects/BEVFusion/bevfusion_official.pth \
      --work-dir /tmp/npu_test
"""

import os
import time
from argparse import ArgumentParser
from os import path as osp

_t0 = time.time()

# --- Patch mmcv.ops BEFORE any mmdet3d import ---
import torch
torch.npu.set_compile_mode(jit_compile=False)
import torch_npu  # noqa

# Device selection: NPU_TEST_DEVICE env var must match --device (set_device
# before model.to() to avoid the 507013 d2d error when they disagree).
_DEV = os.environ.get('NPU_TEST_DEVICE', '3')
torch.npu.set_device(int(_DEV.split(':')[-1]) if ':' in _DEV else int(_DEV))
print(f'[TIME] torch_npu init: {time.time()-_t0:.1f}s (device {_DEV})', flush=True)

from npu_patches import (patch_get_geometry_cpu, patch_spconv_torch,
                         patch_swin_patchmerging)
patch_spconv_torch()  # MODELS 注册 + SparseSequential 替换 + 5D 权重加载钩子
patch_swin_patchmerging()  # Swin PatchMerging nn.Unfold kernel bug 修复（bit-exact）
patch_get_geometry_cpu()  # get_geometry 走 CPU（NPU fp32 bmm 实为 fp16 Cube）


def main():
    from mmengine.config import Config, DictAction
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--work-dir', default='/tmp/npu_test', help='Work directory')
    parser.add_argument('--device', default='npu:3', help='Device')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='Override config options')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Limit test to N frames for quick validation')
    parser.add_argument('--start-idx', type=int, default=0,
                        help='Start frame index (for chunked resume)')
    parser.add_argument('--format-only', action='store_true',
                        help='Only format results to nuscenes json (no eval)')
    parser.add_argument('--jsonfile-prefix', default=None,
                        help='Prefix path for the formatted results json')
    args = parser.parse_args()

    # FIX: mmengine BaseDataset.get_data_info overwrites sample_idx with the
    # local index (0-based within the filtered subset). For chunked evaluation
    # this must be the ORIGINAL index into the full pkl data_list, otherwise
    # the metric resolves the wrong frame token (data_infos[0] instead of
    # data_infos[start_idx]).
    from mmengine.dataset.base_dataset import BaseDataset as _BDS
    _orig_get_data_info = _BDS.get_data_info
    def _patched_get_data_info(self, idx):
        data_info = _orig_get_data_info(self, idx)
        if self._indices is not None and idx >= 0 and idx < len(self._indices):
            data_info['sample_idx'] = self._indices[idx]
        return data_info
    _BDS.get_data_info = _patched_get_data_info

    from mmengine.runner import Runner
    from mmengine.registry import init_default_scope

    # Load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    # Strip pretrained Swin (we load full model checkpoint)
    if cfg.model.get('img_backbone', {}).get('init_cfg', {}).get('type') == 'Pretrained':
        cfg.model.img_backbone.init_cfg = None

    # Slice dataset via indices (works with mmengine BaseDataset)
    # Must happen BEFORE any format_only config below, which may trigger
    # lazy build of the dataset before indices is applied.
    if args.max_frames is not None or args.start_idx > 0:
        start = args.start_idx
        end = start + args.max_frames
        cfg.test_dataloader.dataset.indices = list(range(start, end))

    # Format-only: save predictions to json, no evaluation
    if args.format_only:
        if args.jsonfile_prefix is None:
            args.jsonfile_prefix = osp.join(args.work_dir, 'chunk_results')
        cfg.test_evaluator = dict(
            type='NuScenesMetric',
            data_root=cfg.test_dataloader.dataset.data_root,
            ann_file=osp.join(cfg.test_dataloader.dataset.data_root,
                              cfg.test_dataloader.dataset.ann_file),
            metric='bbox',
            backend_args=None,
            format_only=True,
            jsonfile_prefix=args.jsonfile_prefix)

    # Don't let Runner load checkpoint (we handle it manually)
    cfg.load_from = None
    cfg.work_dir = args.work_dir
    cfg.launcher = 'none'

    # Build runner (model is built from cfg.model dict)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    _tb = time.time()
    runner = Runner.from_cfg(cfg)
    print(f'[TIME] runner init: {time.time()-_tb:.1f}s', flush=True)

    # Load checkpoint（5D KRSC 稀疏卷积权重由 patch_spconv_load5d 钩子自动转 4D）
    raw_ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    raw_sd = raw_ckpt['state_dict'] if 'state_dict' in raw_ckpt else raw_ckpt
    missing, unexpected = runner.model.load_state_dict(raw_sd, strict=False)
    print(f'[TIME] load weights: {time.time()-_tb:.1f}s', flush=True)
    if missing:
        print(f'WARNING: {len(missing)} missing keys', flush=True)
        for k in missing[:10]:
            print(f'  missing: {k}', flush=True)
    if unexpected:
        print(f'WARNING: {len(unexpected)} unexpected keys', flush=True)

    # Move to device
    runner.model.to(args.device)
    runner.model.eval()
    print(f'[TIME] to device: {time.time()-_tb:.1f}s', flush=True)

    # Run test
    _tt = time.time()
    runner.test()
    print(f'[TIME] test complete: {time.time()-_tt:.1f}s', flush=True)


if __name__ == '__main__':
    main()
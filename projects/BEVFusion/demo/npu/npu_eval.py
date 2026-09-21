#!/usr/bin/env python
"""Standalone NuScenes evaluation on a merged results_nusc.json.

Usage:
    python projects/BEVFusion/demo/npu_eval.py \
        /tmp/npu_test_chunks/results_nusc_merged.json \
        --data-root /mnt/datasets/nuscenes \
        --version v1.0-trainval \
        --out-dir /tmp/npu_test_chunks/eval
"""

import argparse
import json
import os
import os.path as osp
import sys

from nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_json', help='Merged results_nusc.json')
    parser.add_argument('--data-root', default='/mnt/datasets/nuscenes',
                        help='nuScenes dataset root')
    parser.add_argument('--version', default='v1.0-trainval',
                        choices=['v1.0-trainval', 'v1.0-mini'])
    parser.add_argument('--eval-set', default=None,
                        help='Evaluation split (default: val for '
                        'v1.0-trainval, mini_val for v1.0-mini)')
    parser.add_argument('--out-dir', default='/tmp/npu_eval',
                        help='Output directory for metrics')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    args = parser.parse_args()

    if not osp.exists(args.result_json):
        print(f'ERROR: result file not found: {args.result_json}')
        sys.exit(1)

    # Determine eval set from version
    eval_set_map = {
        'v1.0-mini': 'mini_val',
        'v1.0-trainval': 'val',
    }
    eval_set = args.eval_set or eval_set_map[args.version]

    # Verify results file
    with open(args.result_json) as f:
        data = json.load(f)
    print(f'Loaded {len(data.get("results", {}))} samples from results json')
    if 'results' not in data:
        print('ERROR: results json missing "results" key')
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    # Create NuScenes instance
    print(f'Loading NuScenes ({args.version}, {args.data_root})...')
    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=False)
    print(f'Loaded {len(nusc.sample)} samples, {len(nusc.scene)} scenes')

    # Create detection config (same as mmdet3d's NuScenesMetric)
    eval_version = 'detection_cvpr_2019'
    cfg = config_factory(eval_version)
    print(f'Using eval config: {eval_version}')

    # Run evaluation
    print(f'Evaluating on set: {eval_set}')
    nusc_eval = NuScenesEval(
        nusc,
        config=cfg,
        result_path=args.result_json,
        eval_set=eval_set,
        output_dir=args.out_dir,
        verbose=args.verbose,
    )
    nusc_eval.main(render_curves=False)

    # Print summary
    metrics_path = osp.join(args.out_dir, 'metrics_summary.json')
    if osp.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        print('\n===== Metrics Summary =====')
        for k in ['NDS', 'mAP', 'mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE']:
            if k in metrics:
                print(f'  {k}: {metrics[k]:.4f}')
        print('===========================')
    else:
        # Try to find metrics in the output dir
        for fname in os.listdir(args.out_dir):
            if 'metrics' in fname and fname.endswith('.json'):
                with open(osp.join(args.out_dir, fname)) as f:
                    metrics = json.load(f)
                print(f'\nMetrics from {fname}:')
                for k in ['NDS', 'mAP', 'mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE']:
                    if k in metrics:
                        print(f'  {k}: {metrics[k]:.4f}')
                break

    print(f'\nDetailed results in: {args.out_dir}')


if __name__ == '__main__':
    main()
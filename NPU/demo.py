"""BEVFusion with vendored torch spconv (bevfusion.ops.spconv) replacement.

Runs the same inference as cuda_demo.py but replaces official spconv-cu114
CUDA kernels with the vendored pure-PyTorch spconv implementation, and in CPU
mode also replaces the CUDA bev_pool / voxelization custom ops with torch
-native implementations. Compares results against the original CUDA path.

Model building / data loading / inference reuse mmdet3d.apis:
  - init_model(config, ckpt, device='cpu') + model.to(device)
      (checkpoint 5D KRSC 权重由 npu_patches.patch_spconv_load5d 自动转 4D)
  - inference_multi_modality_detector(..., cam_type='all') 六相机数据构建 + 推理

Usage:
  python demo.py \
    <pcd_path> <img_dir> <ann_path> <config> <checkpoint> \
    [--device cuda:0] [--out-json /tmp/cuda_unumops_results.json]
"""

import json
import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import torch
from mmdet3d.apis import inference_multi_modality_detector, init_model
from mmengine.config import Config
from mmengine.dataset import pseudo_collate

# ============================================================
# NPU 补丁（spconv vendored torch 替换 + Swin PatchMerging 位精确替代 +
# get_geometry CPU fp32 + 5D 权重自动转换），全部集中到 npu_patches.py
# ============================================================
_npu_patch = Path(__file__).resolve() / "npu_patches.py"
sys.path.insert(0, str(_npu_patch.parent))

from npu_patches import (
    patch_get_geometry_cpu,
    patch_spconv_torch,
    patch_swin_patchmerging,
)

patch_spconv_torch()  # 用 vendored torch spconv 替换官方 spconv-cu114（mmdet3d 加载后）
patch_swin_patchmerging()
patch_get_geometry_cpu()  # NPU_GEOMETRY_ON_NPU=1: 几何回 NPU（精度降级）


def post_process(result, model, args):
    result = result[0]
    pred_instances = result.pred_instances_3d
    scores = pred_instances.scores_3d
    labels = pred_instances.labels_3d
    bboxes = pred_instances.bboxes_3d
    classes = model.dataset_meta.get("classes", [])
    classes_list = list(classes)

    print(f"scores range: {scores.min().item():.6f} - {scores.max().item():.6f}", flush=True)
    keep = scores > args.score_thr
    print("=" * 70, flush=True)
    print(f"Total detections (score > {args.score_thr}): {keep.sum().item()}", flush=True)
    box_tensor = bboxes.tensor if hasattr(bboxes, "tensor") else bboxes
    detections = []
    for i in keep.nonzero().flatten().tolist():
        label = (
            classes_list[int(labels[i])]
            if int(labels[i]) < len(classes_list)
            else str(int(labels[i]))
        )
        box = box_tensor[i]
        print(
            f"  [{i}] {label}: score={scores[i].item():.3f} "
            f"box=({box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}) "
            f"size=({box[3]:.2f}, {box[4]:.2f}, {box[5]:.2f}) "
            f"yaw={box[6]:.2f}",
            flush=True,
        )
        detections.append(
            {
                "index": i,
                "label": label,
                "score": round(scores[i].item(), 6),
                "box": [round(box[j].item(), 4) for j in range(7)],
            }
        )
    print("=" * 70, flush=True)
    from collections import Counter

    cnt = Counter(int(labels[i]) for i in keep.nonzero().flatten().tolist())
    print("\nPer-class detections:")
    for cls_idx, n in sorted(cnt.items()):
        name = classes_list[cls_idx] if cls_idx < len(classes_list) else str(cls_idx)
        print(f"  {name:22s}: {n}")
    print()

    if args.out_json:
        out = {
            "model_config": args.config,
            "checkpoint": args.checkpoint,
            "score_thr": args.score_thr,
            "device": args.device,
            "scores_range": [round(float(scores.min()), 6), round(float(scores.max()), 6)],
            "total_detections": int(keep.sum().item()),
            "detections": detections,
        }
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Detections saved to {args.out_json}", flush=True)
    return detections


def main(args):
    # 设置 NPU 当前设备（防止中间 tensor 落到默认 device 0）
    if args.device.startswith("npu"):
        _dev_id = int(args.device.split(":")[1])
        torch.npu.set_device(_dev_id)
        print(f"[NPU] set_device(npu:{_dev_id})", flush=True)

    config = Config.fromfile(args.config)
    if config.model.get("img_backbone", {}).get("init_cfg", {}).get("type") == "Pretrained":
        config.model.img_backbone.init_cfg = None

    # init_model(device='cpu')：跳过其内部 torch.cuda.set_device 硬编码；
    # checkpoint 5D KRSC 权重由 patch_spconv_load5d 在 load_state_dict 时自动
    # permute(0,4,1,2,3)，无需手工预处理。加载后自行搬到目标设备。
    model = init_model(config, args.checkpoint, device="cpu")
    model.to(args.device)

    # cam_type='all'：六相机数据构建 + 首次推理（复用 mmdet3d.apis）
    result, data = inference_multi_modality_detector(
        model, args.pcd, args.img_dir, args.ann, cam_type="all"
    )
    collate_data = pseudo_collate([data])

    # Warmup
    print(f"[NPU] warmup x{args.warmup}: start", flush=True)
    _t4 = time.time()
    with torch.no_grad():
        for _ in range(max(1, args.warmup)):
            _ = model.test_step(collate_data)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elif args.device.startswith("npu"):
        torch.npu.synchronize()
    print(f"[NPU] warmup: {time.time() - _t4:.1f}s", flush=True)

    # Timed inference
    _times = []
    for _round in range(args.loop):
        _t5 = time.time()
        with torch.no_grad():
            result = model.test_step(collate_data)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        elif args.device.startswith("npu"):
            torch.npu.synchronize()
        _elapsed = time.time() - _t5
        _times.append(_elapsed)
        if args.loop > 1:
            print(
                f"[NPU] inference round {_round + 1}/{args.loop}: {_elapsed * 1000:.0f}ms",
                flush=True,
            )
    if args.loop == 1:
        print(f"[NPU] inference: {_times[0] * 1000:.0f}ms", flush=True)
    else:
        print(
            f"[NPU] inference: {_times[0]:.1f}s (rounds: {'/'.join(f'{t:.1f}' for t in _times)})",
            flush=True,
        )

    post_process(result, model, args)


if __name__ == "__main__":
    parser = ArgumentParser(description="vendored torch spconv BEVFusion demo (NPU)")
    parser.add_argument("pcd", help="Point cloud file (.pcd.bin)")
    parser.add_argument("img_dir", help="Image directory (containing 6 camera images)")
    parser.add_argument("ann", help="Annotation .pkl file (with data_list)")
    parser.add_argument("config", help="Config .py file")
    parser.add_argument("checkpoint", help="Checkpoint .pth file")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-thr", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--loop",
        type=int,
        default=1,
        help="Timed inference rounds (each round runs test_step once, "
        "prints per-round time; default 1)",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="/tmp/cuda_unumops_results.json",
        help="Path to save detection JSON",
    )
    args = parser.parse_args()
    main(args)

"""BEVFusion with unum_ops spconv (torch-native) replacement.

Runs the same inference as cuda_demo.py but replaces official spconv-cu114
CUDA kernels with unum_ops's pure-PyTorch spconv implementation, and in CPU
mode also replaces the CUDA bev_pool / voxelization custom ops with torch
-native implementations. Compares results against the original CUDA path.

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
from mmengine.config import Config
from mmengine.dataset import pseudo_collate

from mmdet3d.apis import inference_multi_modality_detector, init_model

# ============================================================
# NPU 补丁（spconv unum_ops 替换 + Swin PatchMerging 位精确替代 +
# get_geometry CPU fp32），全部集中到 npu_patches.py
# ============================================================
_npu_patch = Path(__file__).resolve() / "npu_patches.py"
sys.path.insert(0, str(_npu_patch.parent))

from npu_patches import (
    patch_get_geometry_cpu,
    patch_get_geometry_overlap,
    patch_mha_with_flash_attention,
    patch_spconv_unum_ops,
    patch_swin_patchmerging,
)

patch_spconv_unum_ops()  # 用 unum_ops spconv 替换官方 spconv-cu114（mmdet3d 加载后）
patch_swin_patchmerging()
patch_get_geometry_cpu()  # NPU_GEOMETRY_ON_NPU=1: 几何回 NPU（精度降级）
patch_get_geometry_overlap()  # numpy 副本 CPU 预计算，省掉 D2H 排干（无重叠）
patch_mha_with_flash_attention()


def build_model(config, checkpoint, device, compile_mode="off"):
    # 复用 mmdet3d.apis.init_model 的模型构建部分（convert_SyncBN + MODELS.build +
    # cfg 注入 + eval）。不传 checkpoint / device='cpu' 以跳过 init_model 内部两条
    # 不适用的路径：
    #   - load_checkpoint 不做 spconv2 KRSC (out,D,H,W,in) → PyTorch (out,in,D,H,W) 重排
    #   - device != 'cpu' 时调 torch.cuda.set_device，NPU 上不可用
    # 权重重排加载与 to(device) 在下面手动完成。
    model = init_model(config, checkpoint=None, device="cpu")

    # Load checkpoint
    raw_ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    raw_sd = raw_ckpt.get("state_dict", raw_ckpt)

    model_state = model.state_dict()
    filtered_sd = {}
    for k, v in raw_sd.items():
        if k not in model_state:
            continue
        # Checkpoint weights are spconv2 KRSC (out,D,H,W,in).
        # unum_ops uses standard PyTorch layout (out,in,D,H,W).
        # Convert: permute(0,4,1,2,3): (out,D,H,W,in) → (out,in,D,H,W)
        if len(v.shape) == 5 and len(model_state[k].shape) == 5:
            filtered_sd[k] = v.permute(0, 4, 1, 2, 3).contiguous()
        elif v.shape == model_state[k].shape:
            filtered_sd[k] = v
    missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
    if missing:
        print(f"WARNING: {len(missing)} missing keys", flush=True)
    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected keys", flush=True)

    model.to(device)
    model.eval()

    if device.startswith("npu") and compile_mode != "off":
        import torch_npu
        import torchair

        compiler_config = torchair.CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=compiler_config)
        model = torch.compile(model, backend=npu_backend, fullgraph=(compile_mode == "full"))
        print(f"[NPU] torchair applied (mode={compile_mode})")

    # dataset_meta: init_model(checkpoint=None) 跳过了 checkpoint 分支，这里补上
    if "dataset_meta" in raw_ckpt.get("meta", {}):
        model.dataset_meta = raw_ckpt["meta"]["dataset_meta"]
    elif "CLASSES" in raw_ckpt.get("meta", {}):
        model.dataset_meta = {"classes": raw_ckpt["meta"]["CLASSES"]}
    else:
        model.dataset_meta = {"classes": config.class_names}
    return model


def build_data(model, pcd_path, img_dir, ann_path):
    # 复用 mmdet3d.apis.inference_multi_modality_detector 的数据构建通路
    # （cam_type='all' 加载全部相机），与 multi_modality_demo 同一条路径。
    # 该函数内部会顺带跑一次 test_step（充当一次 warmup），返回的 data[0] 是
    # pipeline 输出的单样本 dict，collate 后用于后续 warmup / 计时循环。
    _, data = inference_multi_modality_detector(model, pcd_path, img_dir, ann_path, cam_type="all")
    return pseudo_collate([data])


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
        print(f"[unum_ops] set_device(npu:{_dev_id})", flush=True)

    config = Config.fromfile(args.config)
    if config.model.get("img_backbone", {}).get("init_cfg", {}).get("type") == "Pretrained":
        config.model.img_backbone.init_cfg = None

    # 非 CUDA 设备：repo 自带的 torch 回退实现已经可用且向量化——
    #   - voxelization: _hard_voxelize_pytorch（argsort + unique_consecutive，
    #     无 Python 逐点循环；unum_ops.voxelization_torch 旧版为逐点循环，
    #     NPU 上 .item() 同步实测 13.8s/帧，已废弃）
    #   - bev_pool: _bev_pool_pytorch（QuickCumsum，kept 构造已修复 NPU
    #     切片赋值 bug；比 scatter_add 版快 ~6.7x）
    # spconv 仍由文件头部的 unum_ops 注册替换。无需在此 monkeypatch。

    model = build_model(
        config, args.checkpoint, args.device, compile_mode=getattr(args, "compile", "off")
    )

    collate_data = build_data(model, args.pcd, args.img_dir, args.ann)

    # Warmup
    print(f"[unum_ops] warmup x{args.warmup}: start", flush=True)
    _t4 = time.time()
    with torch.no_grad():
        for _ in range(max(1, args.warmup)):
            _ = model.test_step(collate_data)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elif args.device.startswith("npu"):
        torch.npu.synchronize()
    print(f"[unum_ops] warmup: {time.time() - _t4:.1f}s", flush=True)

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
                f"[unum_ops] inference round {_round + 1}/{args.loop}: {_elapsed * 1000:.0f}ms",
                flush=True,
            )
    if args.loop == 1:
        print(f"[unum_ops] inference: {_times[0] * 1000:.0f}ms", flush=True)
    else:
        print(
            f"[unum_ops] inference: {_times[0]:.1f}s (rounds: {'/'.join(f'{t:.1f}' for t in _times)})",
            flush=True,
        )

    post_process(result, model, args)


if __name__ == "__main__":
    parser = ArgumentParser(description="unum_ops spconv BEVFusion demo")
    parser.add_argument("pcd", help="Point cloud file (.pcd.bin)")
    parser.add_argument("img_dir", help="Image directory (containing 6 camera images)")
    parser.add_argument("ann", help="Annotation .pkl file (with data_list)")
    parser.add_argument("config", help="Config .py file")
    parser.add_argument("checkpoint", help="Checkpoint .pth file")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--compile",
        choices=["off", "full", "default"],
        default="off",
        help="NPU torchair compile mode: off (eager) / full (fullgraph=True) / "
        "default (max-autotune, fullgraph=False)",
    )
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

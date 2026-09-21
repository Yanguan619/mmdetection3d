# BEVFusion NPU Demo（Ascend 310P）

BEVFusion 在昇腾 310P 上的全 PyTorch 推理 demo 与全量评测工具链。

- 性能结论：**0.54s/帧**（`[TIME] inference` = `model.test_step` 纯前向，不含数据/权重加载；2026-09-21 npu:0 实测；npu:7 共享设备噪声 ~0.70s，A/B 须同卡同时段）
- 精度结论：**全量 6019 帧 val 集 mAP 68.74 / NDS 71.49**，与官方 GPU 基线一致（100.2% / 100.1%）
- 详细分析见 [`性能.md`](性能.md)（profiling + 优化轨迹）与 [`精度.md`](精度.md)（验证记录 + 补丁清单）

## 环境

- NPU: Ascend 310P（`npu:7`；device 0 被系统占用会 OOM，勿用）
- torch_npu 2.7.1 / CANN 9.0.0，`unum_ops`（纯 torch 稀疏卷积 + AscendC bev_pool）
- 设备切换：demo 用 `--device npu:7`；`npu_test.py` 用 `NPU_TEST_DEVICE` 环境变量

## demo.py：单帧推理 + 计时

```bash
python projects/BEVFusion/demo/npu/demo.py \
  demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin \
  demo/data/nuscenes/ \
  demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl \
  projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
  projects/BEVFusion/bevfusion_official.pth \
  --device npu:7 --score-thr 0.2 --loop 3
```

位置参数：`pcd`(点云.bin) `img_dir`(6 相机图像目录) `ann`(标注.pkl) `config`(配置.py) `checkpoint`(.pth)

| 参数 | 默认 | 说明 |
|------|------|------|
| `--device` | `cuda:0` | `npu:7`（310P 必须） |
| `--score-thr` | `0.0` | 检测分数阈值 |
| `--warmup` | `2` | 预热轮数（吸收 JIT 编译 + 数据加载） |
| `--loop` | `1` | 计时轮数（每轮跑一次 `test_step`，打印逐轮耗时） |
| `--out-json` | `/tmp/cuda_unumops_results.json` | 检测结果 JSON 输出 |
| `--compile` | `off` | torchair 编译模式 `off/full/default`（310P 上 eager 即可，无收益） |

输出示例：

```
[unum_ops] warmup x2: start
[unum_ops] warmup: 2.1s
[unum_ops] inference: 615ms
[unum_ops] Detections (score > 0.2): 48 classes
[unum_ops] time/frame: 615ms (loop x3)
```

> 说明：demo 内部复用 `mmdet3d.apis.init_model`（构建）+ `inference_multi_modality_detector`（数据通路，`cam_type='all'` 加载全部相机）。NPU 专属补丁（spconv 替换 / Swin PatchMerging / get_geometry CPU 链）在文件头部的 `npu_patches.py` 中注入。

## bev_pool 后端切换（环境变量）

`demo.py` 无 `--bev-pool-backend` 参数，用环境变量 `BEVFUSION_BEV_POOL_BACKEND` 控制：

```bash
BEVFUSION_BEV_POOL_BACKEND=auto        python ...   # 默认：CUDA→unum_ops 原子加→mx_driving→QuickCumsum
BEVFUSION_BEV_POOL_BACKEND=torch       python ...   # QuickCumsum（精度参照）
BEVFUSION_BEV_POOL_BACKEND=scatter     python ...   # scatter_add（bit-exact，AICPU 慢）
BEVFUSION_BEV_POOL_BACKEND=mx_driving  python ...   # DrivingSDK bev_pool_v3（对照）
```

## 全量评测（6019 val 帧）

### 单进程 / 分块

```bash
# 单进程（慢，~3.5h）
python projects/BEVFusion/demo/npu/npu_test.py \
  projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
  projects/BEVFusion/bevfusion_official.pth \
  --device npu:3 --format-only --jsonfile-prefix /tmp/npu_test/results_nusc

# 分块驱动（3 chunk 顺序 + 合并 + 评测；必须 setsid 脱离 bash timeout）
setsid bash projects/BEVFusion/demo/npu/run_npu_test_chunks.sh 3 6019 \
  >/tmp/npu_test_chunks/driver.log 2>&1
# 监控
grep -E "=== Chunk|exit:|Merged|FAILED" /tmp/npu_test_chunks/driver.log
```

`npu_test.py` 关键参数：`--start-idx N`（起始帧，用于断点续跑）、`--max-frames N`（快速验证）、`--format-only`（只出结果 JSON 不评测）、`--jsonfile-prefix`（结果前缀）。设备经 `NPU_TEST_DEVICE` 环境变量设置（如 `NPU_TEST_DEVICE=3`），必须与 `--device` 一致。

### 评测指标

```bash
python projects/BEVFusion/demo/npu/npu_eval.py \
  /tmp/npu_test_chunks/results_nusc_merged.json \
  --data-root /mnt/datasets/nuscenes --version v1.0-trainval \
  --eval-set val --out-dir /tmp/npu_eval
```

- 全量结果（2026-09-20，原子加 bev_pool）：mAP 68.74 / NDS 71.49（详见 `精度.md` §1.0）
- 8 卡并行（每卡一 chunk，753 帧）~21 分钟；并发坑：满载时偶发 507015 内核故障，需预算一轮重试

## 目录结构

| 文件 | 用途 |
|------|------|
| `demo.py` | 单帧推理 + 计时（本 README 主入口） |
| `npu_test.py` | 全量评测 Runner（分块/format-only） |
| `run_npu_test_chunks.sh` | 分块驱动 + 合并脚本 |
| `npu_eval.py` | NuScenesEval（NDS/mAP/ATE/ASE/AOE/AVE/AAE） |
| `npu_patches.py` | NPU 精度/性能补丁（spconv 替换、Swin、get_geometry） |
| `性能.md` | 性能分析详情（profiling、优化轨迹 57s→0.54s、bev_pool A/B） |
| `精度.md` | 精度验证详情（全量/单帧/补丁清单/复测方法） |

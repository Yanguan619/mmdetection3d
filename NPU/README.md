# BEVFusion 昇腾 NPU（Ascend 310P）推理适配

>  [BEVFusion 多模态 3D 检测](https://github.com/open-mmlab/mmdetection3d/tree/main/projects/BEVFusion)（nuScenes lidar-cam 双模态）移植到华为昇腾 NPU

## 1. 概述

| 项目 | 内容 |
|---|---|
| 模型 | BEVFusion |
| 任务 | nuScenes 3D 目标检测（10 类） |
| 平台 | Ascend 310P，torch_npu |
| 单帧性能 | **0.75s / 帧**（`test_step` 纯前向口径），48 dets / max score 0.8195 |
| 全量精度 | **mAP 68.74 / NDS 71.49**（官方 GPU 68.6 / 71.4，100% 保留率） |

## 2. 已验证环境

## 3. 基础容器镜像环境

| 组件 | 版本 | 说明 |
|---|---|---|
| 设备 | Ascend 300IDUO | - |
| CANN | 9.0.0 | 含 AscendC 算子编译工具链 |
| torch / torch_npu | 2.7.1 / 2.7.1 | 版本需配套 |

启动容器:

```bash
#!/bin/bash
IMAGE=swr.cn-south-1.myhuaweicloud.com/ascendhub/torch-npu:2.7.1.post4-310p-ubuntu22.04-py3.11
docker run -itd \
    --name npu-bevfusion \
    -w /workspace \
    --privileged \
    --ipc=host \
    --net=host \
    --shm-size=5g \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /usr/bin/hostname:/usr/bin/hostname \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /var/log/npu/:/usr/slog \
    -v /etc/hccn.conf:/etc/hccn.conf \
    -v /etc/localtime:/etc/localtime \
    -v /etc/hosts:/etc/hosts \
    -v /data:/data \
    -v /home:/home \
    -v /mnt:/mnt \
    $IMAGE bash

docker exec -it npu-bevfusion bash
```

## 4. 代码与依赖准备

### 4.1 源码仓库（全部 editable 安装，编辑即生效）

```bash
git clone https://gitcode.com/ascend/ModelZoo-PyTorch.git
cd ModelZoo-PyTorch/ACL_PyTorch/built-in/embodied_ai/BevFusion/

git clone https://github.com/open-mmlab/mmdetection3d.git --depth 1
mkdir -p mmdetection3d/npu && cp -r ./*.* mmdetection3d/npu
```

| 仓库 | 路径 | 说明 |
|---|---|---|
| mmdetection3d | `/workspace/mmdetection3d` | 本项目（`pip install -e .`）；spconv 已 vendored 入仓 |
| mmcv | `/workspace/mmcv` | **必须源码编译**（含 `_ext` aarch64 .so），`pip install -e .` |
| mmengine | `/workspace/mmengine` | `pip install -e .` |



### 4.2 关键版本锁定

```bash
pip install torch==2.7.1 torch-npu==2.7.1 numpy==1.26.4 nuscenes-devkit
# mmcv 2.2.0 由源码构建（mmdet3d 版本门已放宽 <2.3.0）
pip install -e /workspace/mmcv
pip install -e /workspace/mmengine
pip install -e /workspace/mmdetection3d
# bev_pool 后端：mx_driving（DrivingSDK bev_pool_v3，cp311 linux aarch64 wheel）
pip install /tmp/opencode/mxdriving_wheel_test/mx_driving-26.1.0.post2-cp311-cp311-manylinux2014_aarch64.whl
```

### 4.3 依赖清单（本容器实测已就绪）

torch 2.7.1（CPU wheel）· torch_npu 2.7.1 · torchair（随 torch_npu）· mmengine 0.11.0rc3 ·
mmcv 2.2.0 · mmdet3d 1.4.0 · mx_driving 26.1.0.post2 · numpy 1.26.4 · scipy 1.17.1 ·
nuscenes-devkit

### 4.4 数据集与权重

**数据集（nuScenes Full v1.0，评测用）**
- 下载：https://www.nuscenes.org/download（需注册，下载 Full dataset (v1.0) 所有 zip）
- symlink 到仓库根 `data/nuscenes`（结构：`samples/` `sweeps/` `maps/` `v1.0-trainval/`），然后生成 infos：
  ```bash
  pip install nuscenes-devkit
  python tools/create_data.py nuscenes --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes --extra-tag nuscenes
  ```
  → 产出 `nuscenes_infos_val.pkl` / `nuscenes_infos_train.pkl` / `nuscenes_dbinfos_train.pkl`
- 单帧 demo 样例已含在 `demo/data/nuscenes/`（pcd + 6 相机图 + ann.pkl），无需下载

**权重（官方 BEVFusion lidar-cam，71.4 NDS / 68.6 mAP）**
- 下载：`https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth`
- 放置：`projects/BEVFusion/bevfusion_official.pth`（demo 用）；chunk 评测脚本默认读 `NPU/models/bevfusion_official.pth`
- 训练才需额外 Swin 预训练：`https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/swint-nuimages-pretrained.pth`

## 5. 目录结构

| 文件 | 作用 |
|---|---|
| `demo.py` | **主入口**：全 PyTorch 单帧推理 + 计时（`--loop N` 毫秒级） |
| `demo_profiler.py` | profiling 变体：torch_npu.profiler 采集算子统计 |
| `npu_test.py` | 全量数据集评测 Runner（单进程 / 分块，format-only 出 JSON） |
| `run_npu_test_chunks.sh` | 3/8 块顺序评测驱动 + 结果合并（**必须 `setsid` 脱离 timeout**） |
| `npu_eval.py` | 对合并的 results_nusc.json 跑标准 NuScenesEval（NDS/mAP/ATE/…） |
| `npu_patches.py` | NPU 精度/性能补丁（spconv 替换、Swin unfold、geometry CPU） |
| `assets/` | 评测日志与结果 JSON（`npu_full_eval_*.log`、`*_metrics.json`） |

## 6. 单帧推理 demo

### 6.1 准备

- nuScenes 单帧样例数据已含在 `demo/data/nuscenes/`（pcd + 6 相机图 + ann.pkl）；
- 官方权重放入 `projects/BEVFusion/bevfusion_official.pth`；
- 首次运行会自动触发各算子的 JIT 编译（warmup 吸收，约几秒，每进程一次）。

### 6.2 运行

```bash
# 在仓库根目录（mmdetection3d/）执行
python NPU/demo.py \
  demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin \
  demo/data/nuscenes/ \
  demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl \
  projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
  projects/BEVFusion/bevfusion_official.pth \
  --device npu:0 --score-thr 0.2 --loop 3 --out-json /tmp/demo_out.json
```

### 6.3 参数说明

| 参数 | 默认 | 说明 |
|---|---|---|
| `--device` | `cuda:0` | NPU 用 `npu:7`（设备 0 被占用） |
| `--score-thr` | `0.0` | 检测输出阈值（demo 验证常用 `0.2`） |
| `--warmup` | `2` | warmup 轮数（吸收 JIT 编译） |
| `--loop` | `1` | 计时的推理轮数，>1 逐轮打印毫秒级耗时 |
| `--out-json` | `/tmp/cuda_unumops_results.json` | 检测结果落盘 |

bev_pool 后端由环境变量 `BEVFUSION_BEV_POOL_BACKEND` 选择（`auto` 默认 / `mx_driving` /
`torch` / `scatter`），须在导入前导出。

## 7. 全量数据集评测（6019 帧 val）

### 7.1 单进程（分块格式）

```bash
# 单块快速验证（如 100 帧）
python NPU/npu_test.py \
  projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
  projects/BEVFusion/bevfusion_official.pth \
  --device npu:7 --work-dir /tmp/npu_test \
  --start-idx 0 --max-frames 100 --format-only --jsonfile-prefix /tmp/npu_test/chunk
```

### 7.2 分块驱动 + 合并（推荐）

```bash
# 需要先把 checkpoint 放到 NPU/models/bevfusion_official.pth
setsid bash NPU/run_npu_test_chunks.sh 3 \
  >/tmp/npu_test_chunks/driver.log 2>&1
# 监控：grep -E "=== Chunk|exit:|Merged|FAILED" /tmp/npu_test_chunks/driver.log
```

### 7.3 结果评测

```bash
python NPU/npu_eval.py \
  /tmp/npu_test_chunks/results_nusc_merged.json \
  --data-root data/nuscenes --version v1.0-trainval \
  --out-dir /tmp/npu_test_chunks/eval
```

### 7.4 8 卡并行（最快，约 21-44 分钟）

每卡一个 chunk（6019/8 ≈ 753 帧），format_only 分块出 JSON 后合并 + eval。
8 进程满载时偶发 507015 ViewCopy 内核故障（同物理芯片逻辑对冲突），需预算一轮重试
（chunk 脚本驱动重试，`NPU_TEST_DEVICE` 必须与 `--device` 一致，否则 507013 d2d 错误）。

## 8. Profiling

```bash
# torch_npu.profiler 采集（输出到 ./result_profiler_*/）
python NPU/demo_profiler.py \
  <pcd> <img_dir> <ann> <config> <ckpt> --device npu:7 --loop 4
```


## 9. 单帧 demo 验证（`demo.py`，npu:7，score-thr 0.2）

- **48 detections，max score 0.8195**


### 9.1 性能结果

单帧推理（npu:7，score-thr 0.2）

| 指标 | 值 |
|---|---|
| 检测数 / max score | **48 dets / 0.8195** |
| E2E 推理 | **0.75s / 帧**（`test_step` 纯前向，mx_driving bev_pool 后端，4 轮 734-766ms） |

### 9.2 主要耗时分布（2026-09-18 串行探针，E2E ~0.63s/帧）

| 阶段 | 耗时 (ms) | 说明 |
|---|---|---|
| img_backbone_neck | ~158 | Swin-B Backbone + FPN Neck |
| sparse_encoder | ~170 | SubMConv3D / SparseConv3D 体素特征提取 |
| view_transform | ~183 | dtransform 5 + depthnet 7 + get_geometry 58(CPU) + bev_pool 105 + downsample 8 |
| 体素化 | ~45 | stable argsort（AICPU） |
| detector | ~48 | fusion + backbone + neck + head（含 top-k 后处理 ~8ms CPU） |
| **合计** | **≈670** | **E2E 实测 0.63s** |

## 10. 精度结果

### 10.1 全量 val 集（6019 帧，2026-09-20，8 卡并行 ~21 分钟）

| 指标 | 值 |
|---|---|
| **mAP** | **68.74** |
| **NDS** | **71.49** |
| mATE / mASE / mAOE | 0.2764 / 0.2540 / 0.2970 |
| mAVE / mAAE | 0.2743 / 0.1858 |

逐类 AP：car 0.894 / pedestrian 0.882 / traffic_cone 0.797 / bus 0.766 /
motorcycle 0.765 / barrier 0.718 / truck 0.641 / bicycle 0.629 / trailer 0.486 /
construction_vehicle 0.295。

评测日志：`assets/npu_full_eval_20260920_atomic.log`；
指标 JSON：`assets/npu_full_eval_20260920_atomic_metrics.json`。

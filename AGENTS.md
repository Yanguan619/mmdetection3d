# NPU (Ascend 310P) 适配记录

## BEVFusion Demo 在 NPU 上运行

### 设备环境
- NPU: Ascend 310P (8 个逻辑设备), 43GB 显存
- torch_npu 2.7.1, CANN 9.0.0
- 关键依赖: `unum_ops`（纯 torch 稀疏卷积、AscendC bev_pool/voxelization 算子）

> **2026-09-21 迁移注记**：mmdetection3d 已**不再依赖 unum_ops 包**。spconv 纯 torch 实现
> 已逐字节迁移到 `projects/BEVFusion/bevfusion/ops/spconv/`（`npu_patches.py` 的
> `patch_spconv_torch()`，原 `patch_spconv_unum_ops`）；bev_pool 用 mx_driving 或
> QuickCumsum 回退。下文 §1/§2/§14 中的 `unum_ops/spconv/...` 路径均指该迁移后的
> vendored 版本（内容逐字节一致）。

### 历史性能（npu_demo.py 旧路径；当前入口为 NPU/demo.py，0.66s/帧，见 §12/§14/§15）
- 推理时间: ~8.3s/frame (0.12 FPS)
- 检测结果: 15 detections (score > 0.2), max score 0.501（当时代码未含精度补丁，仅作历史记录）
- 主要瓶颈: view_transform (3.5s), img_backbone (1.5s)
- **spconv 优化后实测**: SubMConv3d (N=50000, C=64) 聚合 19.9ms, 邻居表构建 16.4ms, 每帧总 < 40ms

### 关键修改

#### 1. 稀疏卷积 (`unum_ops/spconv/conv.py`)
- **坐标顺序修复**: `_encode` 和 `sp_col` 按 `[x, y, z]` 顺序处理，**不要**反转 spatial_shape
  - `_encode`: strides 从 `sp[0]*sp[1]*sp[2], sp[0]*sp[1], sp[0], 1` 改为 `sp[0]*sp[1]*sp[2], sp[1]*sp[2], sp[2], 1`
  - `sp_col`: 去掉 `range(ndim-1, -1, -1)` 反转，直接用 `range(ndim)`
  - 原因: BEVFusion 的坐标是 `(batch, x, y, z)`，spatial_shape 是 `(x_dim, y_dim, z_dim)`
- **特征聚合用 2D GEMM 而非 einsum**（NPU 上 3D einsum 慢 2.4~6x）:
  - 原 `torch.einsum('nki,kio->no', all_feats, wT)`（SubM N=50000 C=64 约 48ms）
  - 改为 `all_feats.reshape(N, K*C_in) @ self._wT2d`（约 7ms matmul），`_wT2d` 是 `_wT.reshape(K*C_in, C_out)` 零拷贝 view
- **邻居查找用稠密网格 O(1) 而非 sort+searchsorted**:
  - `_lookup`: `grid.index_put_([in_keys], ar)` 建网格 + `grid[cand_keys]` 一次 gather（~5x），
    空间 > `_GRID_LOOKUP_MAX_ENTRIES`(8M 条目) 自动回退 sort+searchsorted
  - `_build_neighbor_idx`: SubM（in_indices==out_coords 且 stride=1）用 key-offset 快速路径
    `cand_key = in_key + offs_key - padding_key`，省去 (N*K, ndim) 候选坐标构造 + 二次编码（34ms → 16.4ms）
- 基准数据生成注意：`make_tensor` 的 spatial_shape 必须与坐标生成一致为 `(x_dim, y_dim, z_dim)`，
  若按 `(D,H,W)` 生成坐标会导致非立方空间 key 碰撞（旧 `benchmark/bench_spconv.py` 有此隐患）

#### 2. dense() 方法 (`unum_ops/spconv/sparse_modules.py`)
- 用 `scatter_add_` 替代 Python 逐体素循环（50K 体素从数秒降到 135ms）
- 关键: flat index 计算要匹配 C-major 布局 `(B, C, D, H, W)`

#### 3. bev_pool (`projects/BEVFusion/bevfusion/ops/`)
- **当前默认（auto）：CUDA ext → unum_ops 原子加 AscendC → mx_driving → QuickCumsum 四级分发**
  （`bev_pool()` 内部实现；**2026-09-20 unum_ops kernel 重写为免排序原子加 scatter**：
  host 预乘元素偏移 ranks（官方约定 `((bD+z)*H+x)*W+y` 再 ×C），kernel 逐点
  SetAtomicAdd + 80-float DataCopy 直写、OOB 零写流量跳过、tile 间 PipeBarrier 串行纪律；
  1.84M 点实测全链 **31.4ms**（ranks 2.5 + op 19.7 + permute 6.4）vs mx_driving 34.4ms、
  旧 segment-sum 133.7ms、QuickCumsum 399ms；对 CPU 参考精度 max_abs 4.8e-7，优于 mx 的
  1.95e-5；demo E2E 48 dets / 0.819399 逐位一致）
  - **NPU 大 int32 坑**：≥2^28 的 int32 过 `torch.where`/比较算子会被内部 FP32 量化
    （268435455→268435456），B>1 高 rank 区曾因此丢写；wrapper 的 OOB 修正用小值域判定
    （坐标逐维比较 + b→B 小值 where）规避，乘加大 int32 域精确（321M 实测逐位对）
  - 首版整块原子提升（tile 级 SetAtomicAdd 提升 + ping-pong 事件）在 310P 上间歇性挂死
    （后续 kernel 被 MTE 卡死、报 Transpose 背锅），最终版用 mx 同款逐点切换 + 旧 kernel 的
    PipeBarrier 纪律，15/15 稳定
  - mx_driving（DrivingSDK bev_pool_v3 wheel）保留为 auto 第二优先级/显式对照：
    `BEVFUSION_BEV_POOL_BACKEND=mx_driving`；其 kernel 无 OOB 检查，需 host sanitize
  - `BEVFUSION_BEV_POOL_BACKEND=torch|scatter` 显式选 QuickCumsum / scatter_add 版
  - **kernel 状态勘误（2026-09-18）**：9/6 的三个提交（70b5600 coords/starts/lengths UB 批量装载、
    dac1c36 UB 缓冲改 tilePoints、50a2322 约定改动）**从未重编安装过**——线上一直跑的是 9/4
    3c7c9bd 时代的旧二进制；当天重编后首次执行即 507015 MTE burst 崩溃（int32 coords 批装
    `DataCopy(rowOff*4)` 每 16B/行，rowOff 奇数 → 非 32B 倍数 → MTE 突发长度非法）。修复：
    kernel 回退 3c7c9bd + 仅叠加官方约定改动（OutputOffset/InBounds 两处）。教训：改 AscendC
    kernel 后必须立即重编验证，未构建的提交≠已生效；MTE DataCopy GM→UB 长度必须 32B 对齐
    （不对齐用 DataCopyPad）
- scatter_add 版对 CPU fp32 是 bit-exact 的，但走 AICPU 逐点累加（~1.84M 点实测
  1.32s/帧），仅作参照；QuickCumsum 版 fp32 求和重排级误差（对检测结果影响 <0.001）
- **torch_npu `t[:-1] = src` 就地切片赋值 bug**（CANN 25.5.2）：会多写末位元素，
  QuickCumsum 的 `kept` 曾因此丢最后一个 segment（一个 BEV cell 清零）；修复用
  `torch.cat` 构造（`[1:] =` 起始切片安全），见 bev_pool.py 内注释

#### 4. voxelization (`projects/BEVFusion/bevfusion/ops/voxel/voxelize.py`)
- 纯 torch 向量化实现（`torch.argsort + torch.unique_consecutive`）
- AscendC 的 `unum_ops.voxelization.voxelization` 在 34 万点以上卡死，不可用
- argsort 用 int64（float32 会因精度 >2^24 丢失精度）

#### 5. 7D matmul (`projects/BEVFusion/bevfusion/depth_lss.py`)
- NPU 不支持 >6D 张量的 `matmul`，改为 `torch.bmm` + reshape
- `get_geometry` 和 `BaseDepthTransform.forward` 中的 `(B,N,1,1,1,3,3).matmul(points.unsqueeze(-1))` 改为 `bmm`

#### 6. `autocast` 设备类型 (`bevfusion.py`)
- `torch.autocast(device_type='cuda', ...)` → `device_type=x.device.type`（动态适配 npu）

#### 7. `torch.cuda.set_device()` (`mmdet3d/apis/inference.py`)
- 改为 NPU 感知：`device.startswith('npu')` 时用 `torch_npu.npu.set_device()`

#### 8. 权重格式转换
- OpenMMLab 官方 checkpoint 的 5D 稀疏卷积权重: `(C_out, kD, kH, kW, C_in)` → `permute(0, 4, 1, 2, 3)` → `(C_out, C_in, kD, kH, kW)`
- 2D/1D 权重已是标准 PyTorch 格式，无需转换

### 运行命令

#### 全 PyTorch NPU（当前入口 NPU/demo.py；旧 NPU/npu_demo.py 已于 2026-09-18 删除）
```bash
python NPU/demo.py \
  demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin \
  demo/data/nuscenes/ \
  demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl \
  projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
  projects/BEVFusion/bevfusion_official.pth \
  --device npu:7 --score-thr 0.2 --loop 3
```

#### 混合 OM 推理（历史方案，2026-09-21 起从仓库移除）
> 以下 OM/ONNX 方案已整体删除：`om_legacy/`、`npu_om_test.py`、`export_full_onnx.py`、
> `run_npu_om_test_chunks.sh`、`tools/export_onnx.py`、`OM_DESIGN.md`、`ONNX_OM_EXPORT.md`。
> 当前唯一方案是全 PyTorch 路径。本小节保留作为历史记录（ATC 精度约束见下文）。
```bash
python NPU/om_legacy/demo_om.py \
  demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin \
  demo/data/nuscenes/ \
  demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl \
  projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
  projects/BEVFusion/bevfusion_official.pth \
  --score-thr 0.2 --warmup 1
```

### 混合 OM 推理结果（2026-09-09）
- **推理时间: 1.38s**（全 PyTorch 2.1s，1.52x 加速）
- **检测结果: 13 detections (score > 0.2), max score 0.503**（与全 PyTorch 基线 13-14 dets 一致）
- OM 加载: 5.8s，warmup: 2.0s，推理: 1.38s

### view_transform.om 精度修复（关键发现）

参见 `ascend-onnx-atc-pipeline` skill 的 `references/PRECISION_DEBUGGING.md`。

三个叠加的 ATC 精度问题:

| 问题 | 算子 | 表现 | 修复 |
|------|------|------|------|
| **FP16 溢出** | MatMul (geometry) | 坐标值 ±1e7~1e8 → FP16 溢出到 ±65504 | `--precision_mode=force_fp32` |
| **NonZero 垃圾输出** | NonZero | 动态输出多出行未初始化内存数据 | 算法级避免布尔索引 |
| **ScatterElements add 失效** | ScatterElements | 重复索引覆盖而非累加 | 排序+去重+唯一索引 scatter |

**bev_pool 修复算法**（`export_viewtransform_onnx.py` 中 `ViewTransformONNX` 的新版 `bev_pool_v2`）：
1. 按 flat index 排序
2. CumSum 累加
3. 相邻比较定位桶边界（is_last）
4. 段和 = cum[last] - cum[prev_last]
5. 唯一索引 scatter（overwrite 等价 add）

### 已知问题
- （2026-09-21 起 mmdetection3d 不再包含 unum_ops.spconv AscendC Cube 内核——spconv 已
  迁移为 `projects/BEVFusion/bevfusion/ops/spconv/` 纯 torch 实现，`spconv_ascendc.available()` 恒 False）
- torch 路径（默认）已用 aclnn 优化 GEMM（Cube 加速）：聚合 19.9ms、邻居表 16.4ms，每帧 spconv 总耗时 < 40ms
- NPU 7 必须用 `npu:7`（设备 0 被系统占用导致 OOM，37454/44278 MB）
- `jit_compile=False` 对精度无影响，但可避免 NPU 内存申请失败
- 多卡/多进程场景需注意 NPU 内存碎片
### 9. Mixed OM + Fast-PT detector（2026-09-09）

将 detector head 从 OM 改为 Fast-PT（纯 PyTorch detector），核心优化：
- **`patch_for_fp16`** 消除 autocast 的 Cast 操作（~12.5ms/iter）：Conv/Linear/BN forward 自动转换输入 dtype 到权重 dtype
- **Flash Attention（2026-09-21 已删除）**：历史"PromptFA 精度损失不可接受"是旧 patch 的投影 bug（`kv.chunk(2)` 把 k 序列劈半、v 错误投影），非 fp16 精度问题
  - 修复后实测曾验证：48 dets / max 0.819478，与基线逐位一致（score diff ≤3.6e-4，fp16 噪声级），
    但 decoder attention 占比小，E2E 无可见加速（~2% 噪声内），`FusedInferAttentionScore` 310P 不支持
  - 2026-09-21：`patch_mha_with_flash_attention` 与 `--flash-attn` 已从 `npu_patches.py` / demo /
    demo_profiler 删除（无收益），本段保留作历史记录

#### 修改的文件
- `projects/BEVFusion/demo/npu_fast_detector.py` (新增, 46 行，含 patch_for_fp16)
- `NPU/om_legacy/demo_fast_det.py` (Fast-PT detector 参考；2026-09-18 移入 om_legacy/)

#### 性能对比

| 方案 | 时间 | 检测数 | max score |
|---|---|---|---|
| 全 PyTorch（baseline） | 2090ms | 14 | ~0.48 |
| 混合 OM（detector.om） | 1280ms | 13 | 0.503 |
| 混合 OM + Fast-PT detector | **1230ms** | **13** | **0.503** |

#### 注意事项
- `img_bev_np` 从 OM 是 1D 拉平 (2592000,)，需 reshape 到 (1, 80, 180, 180) 再送入 Fast-PT detector
- `torch.npu.set_device(7)`（device 0 被系统占用 OOM）
- `patch_for_fp16` 自动把输入 cast 到权重 dtype，避免 ONNX 内 `.float()` 引入 fp32
- 注意：`cast_to_fp16(model)` 转换全部权重到 fp16 也会引入精度损失（8 dets），建议只保留 `patch_for_fp16` 保持 fp32 权重

### 10. auto_optimizer 新增 Knowledge: KnowledgeScatterNdToPad（2026-09-09）

`/workspace/msit/msit/components/debug/surgeon/auto_optimizer/pattern/knowledges/knowledge_scatternd_to_pad.py`

**识别模式**：ScatterND 满足
- `data` 是 zeros initializer
- `indices` 是 5D 常量 (batch, classes, ny, nx, 4)，覆盖规则矩形
- 矩形覆盖率 = updates 的前两维

**替换为**：单个 Pad 节点，padding 位置从 indices 的 ymin/xmin 偏移反推

**应用效果**：detector.om inference **130ms → 44ms**（3x 加速）

**调用方式**：
```bash
python /root/.agents/skills/ascend-onnx-atc-pipeline/scripts/optimize_onnx.py \
  --model detector.onnx \
  --save-dir output \
  --autoopt-knowledges KnowledgeScatterNdToPad
```

### 11. ascend-onnx-atc-pipeline skill 的 ONNX 精度对比工具（2026-09-09）

当 OM 输出与 PyTorch reference 不一致时，使用 `compare_precision.py` 进行逐张量对比：
```bash
python /root/.agents/skills/ascend-onnx-atc-pipeline/scripts/compare_precision.py \
  --model /tmp/opt/detector/detector_onnxslim_auto_optimizer.onnx \
  --inputs /tmp/test_inputs \
  --output-dir /tmp/precision_check
```

通过 `(cosine, max_abs_diff, max_relative_diff)` 三个指标判定精度。


### 12. 完整数据集精度测试

#### 最新结果（2026-09-20，原子加 bev_pool kernel，6019 val 帧，8 卡并行）

| 指标 | NPU | 官方 GPU | 保留率 |
|------|-----|---------|--------|
| **mAP** | **68.74** | 68.6 | **100.2%** |
| **NDS** | **71.49** | 71.4 | **100.1%** |

- 2026-09-20 重写的免排序原子加 bev_pool kernel 的全量回归：与 9/17 QuickCumsum
  （68.71/71.44）、9/18 旧 AscendC（68.67/71.49）及 GPU 基线完全一致（评测噪声内）
- 逐类 AP：car 0.894 / ped 0.882 / cone 0.797 / cv 0.295（与 9/17 一致）
- mATE 0.276 / mASE 0.254 / mAOE 0.297 / mAVE 0.274 / mAAE 0.186
- 评测日志: `NPU/assets/npu_full_eval_20260920_atomic.log`；
  合并结果: `/tmp/npu_test_atomic/results_nusc_merged.json`（358MB，重启会丢，
  指标已存 assets/npu_full_eval_20260920_atomic_metrics.json）
- 8 卡并行 ~21 分钟（1.66s/帧/卡，8 进程满载），8/8 chunk 一次通过无重试

#### 历史结果（2026-09-17，6019 val 帧，8 卡并行，当日算子修复后）

| 指标 | NPU | 官方 GPU | 保留率 |
|------|-----|---------|--------|
| **mAP** | **68.71** | 68.6 | **100.2%** |
| **NDS** | **71.44** | 71.4 | **100.1%** |

- 与官方 GPU 基线完全一致（差异在评测噪声内）——NPU 移植精度闭环
- 评测日志: `NPU/npu_full_eval_20260917.log`；
  合并结果 JSON: `/tmp/npu_test_8way/results_nusc_merged.json`（358MB，重启会丢）
- 8 卡并行（每卡一个 chunk，753 帧）总耗时 ~65 分钟（单进程串行需 ~3.5h）；
  **并发坑**: 8 进程同时满载时 2/8 chunk 死于 507015 ViewCopy 内核故障（同物理芯片
  逻辑对间歇性冲突），等芯片空闲后补跑即可——8 卡方案要预算一轮重试
- mATE 0.277 / mASE 0.254 / mAOE 0.302 / mAVE 0.272 / mAAE 0.186；
  car 0.895 / ped 0.882 / cone 0.797 最好，construction_vehicle 0.290 最差（与 GPU 同款难点）
- 早期（2026-09-17 前）的评测数字已被代码勘误作废（见 §13），不再保留

#### NPU 文档索引（NPU/）
- `README.md`：唯一文档，含使用指南 + **性能结果（§9）+ 精度结果（§10）**
  （`性能.md`/`精度.md` 已于 2026-09-21 并入 README 并删除）
- （ONNX/OM 工具链与 `OM_DESIGN.md`/`ONNX_OM_EXPORT.md` 已于 2026-09-21 整体删除）

#### 关键工具
- `npu_test.py`：单进程 Runner，支持 `--start-idx/--max-frames/--format-only/--jsonfile-prefix`（分块评估）
- `run_npu_test_chunks.sh`：3 chunk 顺序驱动 + 合并（**必须用 `setsid` 脱离 bash timeout**）
- `npu_eval.py`：对合并结果跑标准 NuScenesEval（NDS/mAP/ATE/ASE/AOE/AVE/AAE）

#### 分块评估关键修复
- **mmengine `BaseDataset.get_data_info` 会把 `sample_idx` 覆盖为本地子集索引**（即使 `_indices` 存在），导致所有 chunk 输出 data_infos[0] 的 token
- 修复：monkey-patch `get_data_info`，`_indices` 存在时恢复 `data_info['sample_idx'] = self._indices[idx]`
- 3 chunk 各 ~2007 帧，~95 分钟/chunk，全部 exit 0；合并 6019 samples 唯一

### 13. 精度排查关键结论（2026-09-16）

#### 勘误：早期错误结果的成因（数字已废弃，仅保留诊断）
- OM 混合评估 mAP 23.9 是**类映射错误**：`npu_om_test.py` 曾用 `metric.dataset_meta =
  dataset.METAINFO`（类级默认顺序）而 Runner 用实例 metainfo（配置规范顺序
  `car,truck,cv,bus,trailer,barrier,motorcycle,bicycle,ped,cone`）。修复为
  `dataset.metainfo` 后 hybrid-OM 与全 PT 帧级一致——OM 路径本身精度健康
- 全 PT 评估 mAP 47.11 是**过时代码**：transfusion_head 未提交的 .half()、spconv
  sort+searchsorted 破损路径（pts_middle_encoder cos 0.95）、无 unfold/geometry patch
- 修复后全量结果见 §12（mAP 68.71 / NDS 71.44，GPU parity）；分析检测时**必须用 JSON 的
  detection_name**（名字安全），不要手写 label index→名字映射

#### 两个真实 NPU bug（张量级验证，npu_patches.py）
1. **Swin PatchMerging nn.Unfold kernel 错误**：主进程 import mmcv.ops 后首次 unfold 调用选错
   kernel（前 4 列数据错位，stage2 cos 0.9887）。修复：reshape/permute 等价替换（bit-exact）。
   `import mmcv.utils.device_type` 是触发链，kernel 选择是一次性 lazy 的
2. **NPU fp32 bmm 实为 fp16 Cube**（rel ~4e-4 vs CPU 5e-8）：get_geometry 坐标链放大到 ~30m
   误差，翻转 frustum mask（kept 点数差 7365），img_bev cos 0.9856。修复：get_geometry 移到
   CPU 计算（~24MB 传输），img_bev cos→0.9997
- 两 patch 对最终检测影响小（top3 差 ~0.002），但张量级严格更优
- `npu_patches.py` 提供 `patch_swin_patchmerging()` / `patch_get_geometry_cpu()`，
  npu_test.py 已接入（npu_om_test.py 已随 ONNX/OM 方案删除）。历史性能补丁
  `patch_get_geometry_overlap()`（extract_feat numpy 副本预计算，省 ~37ms/帧 D2H 排干）
  已于 2026-09-21 因复杂度/收益比差删除——整份 extract_feat 复制 + 双份几何实现的维护成本

#### 排查工具链教训
- NPU-vs-CPU 逐张量 bisect（cos/L1）可靠；检测级 per-class 统计易犯索引映射错
- "同进程同模型，dataloader batch vs 手动 batch"对照可分离数据通路与进程状态
- `runner.model.to('npu:X')` 前必须 `torch.npu.set_device(X)`（否则模型先落在坏 device 0，
  d2d copy 报 507013 DMA error）

### 14. NPU demo 算子性能优化（2026-09-17）——57s → 1.1s

完整分析见 `NPU/README.md` §9（原 `性能.md` 已于 2026-09-21 并入 README）。

#### 性能轨迹
| 版本 | inference | 说明 |
|---|---|---|
| NPU 初版 | 57-61s | Python 循环算子（voxelization_torch 逐点 .item()、_downsample_coords 逐点） |
| 算子向量化 | 2.2s | 见下 |
| + bev_pool 分发修复 | **1.1s** | 48 dets / max 0.8193 保持不变 |
| + AscendC bev_pool（auto 默认） | 0.9-1.2s | demo 级 48 dets 一致 |
| + O3 热点消除（NonZero/Unique/Index） | 0.7s | 2026-09-18 |
| + get_geometry numpy 副本预计算 | **0.66s（653-660ms）** | 2026-09-18 下午；几何逐位不变，纯 CPU 补丁的 D2H 排干阻塞净成本 ~37ms 被消除（0.1s 精度计时曾误判为 0） |
| + bev_pool 包装层快赢（cat 交换/int32 ranks/sort 一次性） | **0.63s（631-639ms）** | 2026-09-18 傍晚；包装 78→48ms（fancy 交换 14→3、int32 ranks 9→5、torch.sort 一次性 33→18、int32 coords gather 7→6）；48 dets / 0.819399 噪声内。坑：int32 直接 sort 落 AICPU 慢 20x（367ms），排序必须 float32。剩余：feats 640MB gather 15ms（kernel 侧索引可消）、sort 18ms（atomic kernel 可消） |
| + 移除 x/y 交换（kernel 改官方约定） | **0.62s（613-618ms）** | 2026-09-18 晚；unum_ops AscendC kernel 回退至 3c7c9bd+官方约定（coord[0]→H），分发器直用原生 [x,y,z,b]；48 dets / max 0.819478 逐位一致 |

#### 关键修复
1. **`unum_ops.voxelization_torch` 向量化**（13.8s → 0.37s）：stable argsort(int64) +
   unique_consecutive + repeat_interleave；voxel 集合/内容与循环版完全一致（仅行序变为 key 升序）
2. **`spconv/conv.py _downsample_coords` 向量化**（8.9s → <50ms）：repeat_interleave + 商余分解
3. **QuickCumsum kept bug**（torch_npu `t[:-1] = src` 多写末位元素，见 §3）：每次 bev_pool
   丢最后一个 segment；`torch.cat` 规避
4. **bev_pool 分发**：`ops/__init__.py` auto 模式 ext 缺失时曾错发 `bev_pool_torch`
   （scatter_add，AICPU 1.32s/帧），改发 `_bev_pool_pytorch`（QuickCumsum，-1.1s/帧）
5. **SubM 邻居表帧内跨层共享**（`SparseConvTensor._layer_nb_cache`，随 replace_feature 传播）：
   同 indice_key + 同参数的层复用一张表，eval 流式场景省 ~16 层 × 0.12s/帧
6. **spconv 杂项**：`_wT2d` 按 weight 版本缓存（免每 forward 重排）、fingerprint 单次同步
   （.tolist 一次取 sum/min/max）、`_lookup_grid` int32 网格（680MB → 340MB）

#### Profiling 教训（310P + CANN 25.5.2）
- msprof op_statistic 的 `Task Type` 列区分 AI_CORE/AI_CPU：**AICPU 是性能红旗**，
  本例 ScatterElements(ScatterAdd) 58.65% 占比直接定位 bev_pool
- 长条 host 空洞（~285ms × N）= `AscendCL@aclopCompileAndExecute` 遗留算子 JIT 编译，
  每进程一次（warmup 吸收）；推理期不应出现
- `task_time.csv` 字段带内嵌引号和 \t，需 strip("'").strip() 清洗
- 计算内核（Swin 152ms / spconv 70ms / BEV conv 43ms）远小于数据搬运/布局转换——
  310P 优化主战场是 gather/scatter/sort/小算子风暴，不是 GEMM/Conv
- torchair（default/full）对当前全 PT 路径无收益（device-bound，1.1s 三模式同速同结果）

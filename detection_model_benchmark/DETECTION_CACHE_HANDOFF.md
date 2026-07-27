# 长视频检测缓存后端交接手册

> 更新日期：2026-07-27
> 适用范围：为 MOT/分类计数实验生成统一逐帧 `Nx6` 检测缓存，以及把冻结前端替换为其他检测模型。
> 当前参考实现：RT-DETR-R18。
> 权威研究约束：`research_docs/MOT算法计数性能对比实验协议.md` v2.1。
> 数值格式实现：`benchmark_common/numeric_cache.py`。

## 1. 这份文档放在这里的原因

本文件属于检测基准工作区的运行契约，而不是某一天的实验日志。以后无论换成 YOLO、DEIM、Faster R-CNN 还是其他检测器，都应先读本文件，再为该检测器实现或审核独立后端。

各类文档的分工如下：

- 本文件：如何实现、验证、生成、校验和移交一套检测缓存；
- `research_docs/MOT算法计数性能对比实验协议.md`：研究设计和冻结参数；
- `research_docs/当前进展与下一步.md`：当前做到哪一步；
- `research_docs/变更日志.md`：已经发生的变更与验证证据；
- 各模型 `RUNBOOK.md`、`PROJECT_INFO.md`：模型环境、官方源码和权重来源。

## 2. 核心原则

8 个 tracker 必须读取同一份逐帧数字缓存。检测器只推理一次，tracker 是后续比较中的唯一变量。

一套可用于正式实验的检测缓存必须同时满足：

1. 模型、checkpoint、实际加载的 state dict、源码 commit 和推理参数均已冻结；
2. 预处理、颜色空间、数据类型、batch、后处理和坐标恢复复现该模型的冻结评测路径；
3. 对已有冻结预测做逐图、逐框、逐字段数值一致性验证，不能只比较 mAP；
4. 按 `video_data/manifests/segments.csv` 的 `video_id + segment_index` 显式读取；
5. 以实际解码成功帧数为准，`CAP_PROP_FRAME_COUNT` 只作诊断；
6. 帧号从 1 开始连续，空检测帧写空数组，不能跳帧；
7. 每个缓存都有完整 footer；中断时不能留下可被误认成正式结果的文件；
8. 完整记录视频、checkpoint、配置和 numeric payload SHA-256；
9. 生成后由独立集合校验器重新遍历 25 个缓存；
10. 任何缓存格式或字段精度变化都必须升级版本并使旧缓存失效。

## 3. 通用输入输出契约

### 3.1 后端输入

建议每个检测器后端提供批量接口：

```text
输入：按视频解码顺序排列的 RGB 图像，以及冻结的最低保存置信度
输出：与输入图像一一对应的检测列表
```

不要让 tracker、ROI 或计数逻辑进入检测后端。

### 3.2 后端输出

每帧统一输出：

```text
Nx6 [x1, y1, x2, y2, confidence, class_id]
```

约束：

- 坐标是原始完整画面的 `xyxy`；
- 当前正式视频画面为 2304×1296；
- `class_id=0` 是 `Bactrocera dorsalis`；
- `class_id=1` 是 `Bactrocera correcta`；
- 两个类别都保留，不在缓存阶段做物种过滤；
- 当前共享保存下限为 `confidence >= 0.10`；
- 输出顺序应保持官方 postprocessor 的确定顺序。

### 3.3 数值缓存 v2

`NumericCacheWriter` 负责以下不变量：

| 字段 | 写入精度 |
|---|---:|
| `x1,y1,x2,y2` | 2 位小数 |
| `confidence` | 4 位小数 |
| `class_id` | 整数 |

舍入方式是“四舍五入远离零”，不是 Python 的银行家进位。非有限数、非整数 class ID、列数错误和不连续帧号都会直接报错。

缓存结构为：

```text
header
frame 1
frame 2
...
frame N
footer(complete=true)
```

`cache_id` 校验 header；数字 payload 的 SHA-256 另存于片段 manifest，并用于生成集合指纹。

### 3.4 每个检测器必须记录的 metadata

至少记录：

- 检测器名称和架构；
- 官方源码路径、origin 和 commit；
- checkpoint 路径及 SHA-256；
- 实际加载的权重节点，例如 `ema.module` 或 `model`；
- 推理配置文件及 SHA-256；
- 输入尺寸和 resize/letterbox 规则；
- RGB/BGR、数值范围、mean/std；
- FP32/FP16/AMP；
- batch size；
- 官方后处理器和 top-k；
- NMS 是否启用；若启用则记录 IoU，若 NMS-free 则记录 `not_applicable`；
- 最低保存置信度；
- 类别映射；
- 原始尺寸分量顺序和坐标恢复方式；
- 能唯一代表上述配置的 detector config hash。

## 4. 当前 RT-DETR-R18 参考实现

### 4.1 代码入口

| 文件 | 作用 |
|---|---|
| `rtdetr_r18_backend.py` | 加载冻结源码、配置和 EMA；执行精确 FP32 批量推理 |
| `validate_rtdetr_val_consistency.py` | 对冻结 val 结果做全量逐框一致性验证 |
| `cache_rtdetr_detections.py` | 严格按 manifest 解码视频并写缓存 |
| `verify_rtdetr_detection_caches.py` | 校验 25 个缓存并生成合并清单与集合报告 |

### 4.2 已冻结的模型事实

- checkpoint：`results/fruitfly_detection_v1/benchmark_640_e100_s42/runs/10_rt_detr_r18/checkpoints/best.pth`
- checkpoint SHA-256：`558b7fd2c7b6488649d64cada5d6c860841301d00b2afec7907f29eeef44cc6f`
- 官方源码 commit：`1c8ac3f7ba84f14bd5651ab7b1b70d69a5f55f47`
- checkpoint 同时含 `model` 和 `ema`；
- 正式后端强制加载 `ema.module`，共 533 个 tensor；
- EMA updates=50,900，warmups=191；
- 推理为 FP32，AMP 关闭；
- val 预处理为直接拉伸 `Resize([640,640]) -> ToImageTensor -> ConvertDtype`；
- 不做 mean/std normalization；
- `orig_target_sizes` 顺序为 `(width,height)`；
- postprocessor 为 NMS-free top-300；
- batch size 固定为 16；
- `--iou` 不存在，传入时 argparse 必须拒绝；
- header 中记录 `nms=false`、`nms_iou=not_applicable`。

这些值不是“推荐默认值”，而是当前 RT-DETR-R18 冻结结果的一部分。换模型时必须重新核验，不能照抄。

### 4.3 首轮验收证据

冻结 val 一致性：

- 1,530 张图片；
- 430,357 行冻结预测；
- 430,357 行完全一致；
- mismatched image=0；
- bbox 最大绝对误差=0；
- confidence 最大绝对误差=0；
- 冻结预测 SHA-256：`25d0b96e3b22554a8fde18b73f39f8b4422236c55e1ff206d1b4301b6d9ddffe`。

正式视频缓存：

| 项目 | 结果 |
|---|---:|
| 逻辑视频 | 6 |
| 物理片段 | 25 |
| 实际解码帧 | 10,007,059 |
| 检测总数 | 8,817,155 |
| class 0 检测 | 5,521,598 |
| class 1 检测 | 3,295,557 |
| 空检测帧 | 6,269,427 |
| 缓存字节 | 550,443,689 |
| 平均字节/帧 | 55.01 |
| footer | 25/25 完整 |

集合 numeric payload SHA-256：

```text
83d60d1e0ce66d77f948c4b3072426bba66faccb55fe224ec87ef68a8b64bd18
```

结果文件：

```text
project_results/detection_cache/rtdetr_r18_official_v1_validation/val_consistency.json
project_results/detection_cache/rtdetr_r18_official_v1/detection_cache_manifest.csv
project_results/detection_cache/rtdetr_r18_official_v1/verification_report.json
```

`project_results/`、checkpoint、外部源码和正式缓存均被 Git 忽略，必须通过项目存储或备份单独保留；Git 提交只保存代码、配置和本文档。

## 5. 标准执行流程

### 阶段 A：建立冻结运行单

运行前写清：

```text
detector_id
environment
source_commit
checkpoint_path
checkpoint_sha256
state_dict_key
preprocess
precision
batch_size
postprocess
confidence
class_mapping
output_root
reference_prediction_path
reference_prediction_sha256
```

输出目录必须包含检测器和版本，例如：

```text
project_results/detection_cache/<detector_id>_<cache_version_name>/
```

更换模型、checkpoint、预处理、batch、后处理、conf、类别映射或数值格式时必须换目录，不能覆盖既有缓存。

### 阶段 B：核验 checkpoint 与官方评测路径

必须实际加载 checkpoint 并检查顶层 key。训练配置写了 EMA 不等于 checkpoint 一定保存 EMA，更不等于官方评测实际读取 EMA。

同时从官方源码确认：

- val/test 用哪套 transforms；
- resize 是直接拉伸还是 letterbox；
- 是否归一化；
- RGB/BGR；
- 原图尺寸是 `(w,h)` 还是 `(h,w)`；
- 后处理是 NMS、soft-NMS 还是 NMS-free；
- 输出坐标在哪个空间；
- 类别 ID 是否需要 remap；
- batch size 是否影响冻结输出。

所有关键判断应变成后端的 fail-fast 校验，不只写在注释里。

### 阶段 C：实现检测器专用后端

建议命名：

```text
<detector>_backend.py
cache_<detector>_detections.py
validate_<detector>_val_consistency.py
verify_<detector>_detection_caches.py
```

当前 `cache_detections.py` 是 Ultralytics YOLO 入口，仍采用递归文件扫描。它可作工程参考，但在没有补齐严格 manifest、实际解码帧、完整 metadata 和集合校验前，不能直接作为新的正式 25 片段入口。

第二个检测器接入时，优先复用以下已经验证的控制逻辑：

- `cache_rtdetr_detections.py` 中的 manifest 读取与 25 片段强校验；
- decoder 线程、批量推理和原子 writer 生命周期；
- `--only` 的显式分片；
- `--resume` 对既有缓存的完整复核；
- 每片段视频/缓存/payload 哈希和耗时汇总；
- `verify_rtdetr_detection_caches.py` 的覆盖、footer、帧号、字段精度和合并报告逻辑。

RT-DETR verifier 中的 checkpoint、EMA、top-300 和 NMS-free 检查是模型专用约束。换模型时应替换成新模型的冻结事实，不能为了“通用”而删除这些检查。

### 阶段 D：逐框数值一致性

正式视频前必须对冻结 val/test 推理结果重跑：

1. 使用相同图片；
2. 使用冻结 batch 和精度；
3. 使用同一后处理；
4. 将新输出转换为与冻结 JSON 相同的 canonical 表示；
5. 按每张图逐行比较框、confidence、class 和顺序；
6. 输出 reference hash、最大误差和 mismatch 列表；
7. 不一致时停止，不运行长视频。

如果某个模型没有冻结逐框预测，应先通过该模型的官方 evaluator 生成一次参考 JSON、记录 SHA-256 并冻结，再开发视频入口。只比较最终 mAP 无法发现 EMA、类别、坐标或后处理的静默错误。

### 阶段 E：manifest 预检和单片段先导

先执行 dry-run，确认 25 个片段和 10,007,059 帧的 manifest：

```bash
python detection_model_benchmark/cache_rtdetr_detections.py --dry-run
```

再把先导结果写入独立目录，避免混入正式集合：

```bash
python detection_model_benchmark/cache_rtdetr_detections.py \
  --device cuda:0 \
  --only LV005:6 \
  --output project_results/detection_cache/pilot_rtdetr_r18_lv005_6 \
  --summary-name detection_cache_manifest_shard0.csv
```

先导至少记录：

- 实际解码帧；
- 检测数量和空帧；
- 完整 footer；
- 字节/帧；
- 解码、推理和整体 FPS；
- GPU 显存；
- 重复运行的 numeric payload hash。

首轮 LV005:6 为 7,258 帧、14,809 个检测、13 个空帧、99.40 字节/帧。该片段检测密集，按它外推约 0.995 GB；正式全量实际为约 0.550 GB，所以先导外推只能用于保守容量判断。

### 阶段 F：正式生成

单进程也使用 shard 文件名，以便最终 verifier 统一处理：

```bash
python detection_model_benchmark/cache_rtdetr_detections.py \
  --device cuda:0 \
  --output project_results/detection_cache/rtdetr_r18_official_v1 \
  --summary-name detection_cache_manifest_shard0.csv
```

恢复中断：

```bash
python detection_model_benchmark/cache_rtdetr_detections.py \
  --device cuda:0 \
  --output project_results/detection_cache/rtdetr_r18_official_v1 \
  --summary-name detection_cache_manifest_shard0.csv \
  --resume
```

`--resume` 不是“文件存在就跳过”。它会验证 cache v2、footer、帧数、视频签名、checkpoint 和 detector config；不一致时失败。

多 GPU/多进程时：

- 每个进程使用互不重叠的 `--only VIDEO_ID:SEGMENT_INDEX`；
- 每个进程使用不同的 `--summary-name`；
- 可以共享 output root，但不能让两个进程写同一片段；
- 每个进程内部仍按 manifest 顺序运行；
- 全部进程成功后必须执行集合 verifier；
- 不要依赖文件系统扫描顺序。

首轮六个 shard 的实际分配如下，保留它是为了断点恢复，不代表其他机器上的最优负载均衡：

| shard | 片段 |
|---|---|
| 0 | `LV003:4`, `LV003:5` |
| 1 | `LV001:1`, `LV003:1` |
| 2 | `LV002:1`, `LV006:1`, `LV006:2` |
| 3 | `LV004:4`, `LV004:5`, `LV005:1` |
| 4 | `LV003:2`, `LV003:3`, `LV004:2`, `LV005:3`, `LV005:6`, `LV006:4`, `LV006:7` |
| 5 | `LV004:1`, `LV004:3`, `LV005:2`, `LV005:4`, `LV005:5`, `LV006:3`, `LV006:5`, `LV006:6` |

### 阶段 G：独立集合校验

RT-DETR 当前命令：

```bash
python detection_model_benchmark/verify_rtdetr_detection_caches.py \
  --cache-root project_results/detection_cache/rtdetr_r18_official_v1 \
  --shard-glob 'detection_cache_manifest_shard*.csv'
```

成功后生成：

```text
detection_cache_manifest.csv
verification_report.json
```

最终报告必须确认：

- manifest 期望路径与实际缓存路径完全相等；
- 25 个 shard 行无重复、无缺失；
- 25 个缓存全部为 cache v2；
- `field_decimals` 完全匹配；
- checkpoint/config/class/postprocess metadata 匹配；
- 每帧数据可完整遍历；
- 每个片段帧数等于 manifest；
- 实际解码帧总数为 10,007,059；
- 每片段文件 SHA 和 payload SHA 与 shard manifest 一致；
- 集合 payload SHA 已生成。

## 6. RT-DETR-R18 当前复核命令

进入项目和环境：

```bash
cd /home/admin1/Projects/ultralytics-main
source /home/admin1/Environments/miniconda3/etc/profile.d/conda.sh
conda activate bench_detection_torch201
```

冻结 val 全量一致性：

```bash
python detection_model_benchmark/validate_rtdetr_val_consistency.py \
  --device cuda:0
```

回归测试：

```bash
python -m unittest discover -s counting_model_benchmark/tests -v
python -m py_compile \
  detection_model_benchmark/rtdetr_r18_backend.py \
  detection_model_benchmark/cache_rtdetr_detections.py \
  detection_model_benchmark/validate_rtdetr_val_consistency.py \
  detection_model_benchmark/verify_rtdetr_detection_caches.py
git diff --check
```

确认 NMS-free 参数不会被误用：

```bash
python detection_model_benchmark/cache_rtdetr_detections.py --iou 0.5
```

预期：argparse 返回 exit code 2，不开始推理。

## 7. 更换为其他检测模型的最小清单

### 7.1 不能直接沿用的 RT-DETR 事实

更换检测器时必须重新确定：

- `ema.module` 还是 `model`；
- 640×640 是否冻结；
- 直接拉伸还是 letterbox；
- 是否有 mean/std；
- RGB 还是 BGR；
- FP32、FP16 还是其他精度；
- batch size；
- 原图尺寸顺序；
- top-k；
- 是否 NMS-free；
- NMS IoU；
- 类别 remap；
- 官方输出顺序；
- reference prediction 的格式与 hash。

### 7.2 必须继续保持的实验公共约束

- 同一 `segments.csv`；
- 同一 25 个物理片段和 manifest 顺序；
- 原图完整画面坐标；
- `Nx6` 字段顺序；
- canonical class 0/1；
- 两类都保留；
- 当前共享缓存下限 0.10；
- cache v2 和字段精度；
- 空帧写出；
- 帧号连续；
- 实际解码帧计数；
- 原子完成和 footer；
- 每片段及集合 payload hash；
- 先一致性、后先导、再全量；
- 新检测器使用新 output root。

### 7.3 新后端验收表

在开始 25 片段前逐项确认：

- [ ] 官方源码 origin 和 commit 已记录；
- [ ] checkpoint SHA-256 已记录；
- [ ] checkpoint 顶层结构已实际打印；
- [ ] 实际加载 state dict 已确认；
- [ ] val transforms 已逐项复现；
- [ ] precision 和 batch 已冻结；
- [ ] 后处理和 NMS 行为已冻结；
- [ ] 原图坐标恢复已从源码确认；
- [ ] class 0/1 保护已实现；
- [ ] reference prediction SHA-256 已冻结；
- [ ] 全量 val 逐框一致性通过；
- [ ] manifest dry-run 通过；
- [ ] 单片段容量与重复 hash 通过；
- [ ] `--iou` 的语义明确处理；
- [ ] 25 片段集合 verifier 已适配新模型事实。

## 8. 常见静默错误

以下错误经常不报异常，但会使整套缓存无效：

| 错误 | 后果 |
|---|---|
| EMA 与裸 model 选错 | 全部框和分数整体变化 |
| letterbox 与直接拉伸混用 | 框坐标系统性偏移 |
| `(w,h)` 与 `(h,w)` 颠倒 | 横纵坐标缩放错误 |
| PIL/RGB 与 OpenCV/BGR 混用 | 检测分布变化 |
| 漏掉或多做 normalization | 全部预测变化 |
| batch 与冻结评测不同 | 某些模型输出数值变化 |
| 额外增加 NMS | NMS-free 模型框集合被改变 |
| 忽略官方 NMS IoU | NMS 模型框集合被改变 |
| class 0/1 对调 | 两个物种计数静默互换 |
| 使用 `CAP_PROP_FRAME_COUNT` 当真值 | 可能接受截断或漏解码 |
| 跳过空帧 | tracker 时间轴错位 |
| 使用 Python `round` | payload hash 跨实现不稳定 |
| 多进程片段重叠 | 同一缓存并发覆盖 |
| 在旧 output root 换权重续跑 | 集合混入两个检测器版本 |

## 9. 移交给 tracker 阶段前

检测缓存通过不代表三层链路已验证。移交前提供：

```text
cache root
detection_cache_manifest.csv
verification_report.json
collection_numeric_payload_sha256
checkpoint_sha256
detector_config_sha256
cache_version
class mapping
confidence policy
NMS policy
actual total frames
```

tracker 运行清单必须引用同一个集合 payload SHA，禁止各算法自行重新检测。

当前 RT-DETR 检测阶段已经完成；下一步仍是协议阶段 2：用同一短片段完成 8/8 tracker 冒烟、重复运行 hash、片段边界重置和计数重放验证。在这些验证通过前，不应直接启动 8 个 tracker 的全量正式计数实验。

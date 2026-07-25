# 8 种多目标跟踪算法对真实诱捕长视频逐物种计数性能影响的 Baseline 实验协议

> 文档用途：导师审阅版。本文不依赖此前聊天上下文，单独阅读即可了解研究目的、实验对象、当前进展、完整参数、评价方法、尚未完成的工程和需要审查的决策。
> 协议版本：v2.0
> 更新日期：2026-07-25
> 当前状态：方案待导师审查，正式全量实验尚未运行。
> 实验性质：面向最终逐物种计数任务的下游跟踪器工程选型，不是标准 MOT 排名实验，也不是独立测试集上的最终论文泛化结论。

---

## 0. 给导师的一页摘要

### 0.1 现在要做什么

本研究已经获得真实诱捕器长视频及每个逻辑长视频结束后的两种实蝇人工数量。当前准备固定同一个目标检测器和同一个圆形进出计数器，只替换中间的多目标跟踪算法，比较 8 种跟踪算法最终造成的逐物种计数误差。

完整链路为：

```text
25 个真实视频片段（组成 6 个逻辑长视频）
              ↓
同一个 RT-DETR-R18 全画面检测器，只推理一次
              ↓
同一份逐帧检测数字缓存
              ↓
8 种不使用 ReID 的在线跟踪算法分别重放
              ↓
同一个固定红圈、有符号进出计数器
              ↓
比较 6 个逻辑长视频、2 个物种的最终计数误差和跟踪速度
```

候选算法为：

```text
SORT、ByteTrack、OC-SORT、SF-SORT、
FastTracker、BoostTrack、HybridSORT、BoT-SORT
```

### 0.2 为什么现在先做这个实验

最终计数误差同时受到检测漏检/误检、轨迹断裂、ID 切换、边界徘徊和类别抖动影响。如果在跟踪器尚未选定前就加入 ROI、边界容差、轨迹拼接、类别投票等计数优化，就无法判断性能提升究竟来自跟踪器还是来自计数补丁。

因此当前阶段先建立一个尽量简单、统一、可重复的计数 baseline：

- 检测器固定；
- 检测结果固定；
- 计数圆固定；
- 计数规则固定；
- 不使用 ROI；
- 不增加任何跟踪器外部轨迹修复；
- 唯一主要变量是 tracker。

先选出在本任务上计数误差最低的 tracker，后续再围绕这个 tracker 单独研究计数逻辑优化和消融实验。

### 0.3 本实验怎样判断哪个算法最好

主指标是 6 个逻辑长视频、2 个物种上的 `Class-MAE`：

```text
Class-MAE
= 所有“逻辑视频 × 物种”绝对计数误差之和
  / (6 × 2)
```

同时报告：

- `Bactrocera dorsalis MAE`；
- `Bactrocera correcta MAE`；
- 每个逻辑长视频、每个物种的预测值；
- 有符号误差和绝对误差；
- 两个物种合计数量的 `Total-MAE`；
- 两个物种同时完全正确的视频比例；
- 产生负计数的视频比例；
- 每种 tracker 的纯跟踪整体 FPS。

选择优先级为：

1. `Class-MAE` 最低；
2. 两个物种的 MAE 更均衡；
3. 完全正确视频比例更高；
4. `Total-MAE` 更低；
5. 准确率接近时选择跟踪 FPS 更高、实现更简单的算法。

### 0.4 已经完成到哪里

已经完成：

- 25 个物理视频片段整理为 6 个逻辑长视频；
- 视频、片段顺序和逐物种目录级真值清单；
- 8 个官方跟踪器的独立源码登记和统一 `Nx6 → Nx8` 接口；
- 统一固定圆计数器；
- 检测、轨迹、计数三级纯数字缓存框架；
- 片段间重置 tracker、逻辑长视频内累计计数；
- 计数评价程序；
- 16 项单元测试和 8/8 跟踪接口冒烟此前均已通过。

尚未完成：

- 原始 RT-DETR-R18 `best.pth` 的逐视频检测缓存后端尚未接入；
- SF-SORT 当前配置没有启用官方示例中的完整在线轨迹找回参数，需要修正；
- 正式运行前的 25 片段严格覆盖校验、缓存内容哈希、结果防覆盖和整体 FPS 汇总尚需补齐；
- 尚未对 10,007,059 帧运行正式 8 算法全量实验；
- 当前没有正式 baseline 精度结果。

### 0.5 希望导师重点审查什么

希望导师重点审查：

1. 用全部 6 个逻辑长视频选择“当前工程 baseline”，而不划分独立测试集，是否符合本阶段目标；
2. 固定 RT-DETR-R18、共享 `conf ≥ 0.10` 数字检测缓存的控制变量设计是否合理；
3. 使用各算法官方默认或官方复现实验参数，而不针对实蝇视频逐算法调参，是否公平；
4. 不使用 ReID、GMC/ECC、光流和离线插值，只比较 detection-only 在线跟踪，是否符合研究问题；
5. `Class-MAE` 是否适合作为 tracker 选择的主指标；
6. 当前固定圆、有符号进出计数是否足以作为第一轮简单 baseline；
7. 在没有逐帧 MOT ID 真值的情况下，只以最终计数误差选择 tracker，结论边界是否表述充分。

---

## 1. 研究背景与本实验在论文中的位置

### 1.1 总体研究任务

本项目面向真实诱捕场景中的两类实蝇检测、跟踪和长视频分类计数。最终目标不是只在单张图片上框出实蝇，而是处理数小时至十余小时的诱捕视频，输出每个逻辑长视频中进入瓶内的两种实蝇最终数量。

目标类别为：

| class_id | 物种学名 | 文中简称 |
|---:|---|---|
| 0 | *Bactrocera dorsalis* | dorsalis |
| 1 | *Bactrocera correcta* | correcta |

### 1.2 当前研究链路

```text
目标检测
  ↓
多目标跟踪
  ↓
固定区域的轨迹跨界事件
  ↓
逐物种长视频计数
```

检测器决定每帧有哪些候选目标；跟踪器决定相邻帧中的检测框是否属于同一只实蝇；计数器根据轨迹中心是否跨越瓶口圆形区域更新数量。

即使检测结果完全相同，不同 tracker 也可能由于以下原因产生不同计数：

- 轨迹过早删除；
- 遮挡后生成新 ID；
- 同一实蝇轨迹被切成多个片段；
- 不同实蝇的 ID 发生交换；
- 低置信度检测能否用于维持轨迹；
- 对快速运动、重叠和短时漏检的处理不同。

因此需要先回答：

> 在固定检测和固定计数规则的条件下，哪一种跟踪算法最适合当前真实诱捕长视频的逐物种计数？

### 1.3 本实验不是标准 MOT 评测

当前真实长视频只有目录级最终逐物种数量，没有逐帧目标框真值和跨帧身份真值。因此本实验不计算：

- HOTA；
- IDF1；
- MOTA；
- IDSW；
- Frag；
- 标准 MOTChallenge 排名。

本实验只能回答：

> 哪一种 tracker 在当前固定检测器和固定计数器下，产生了更低的最终逐物种计数误差？

允许的结论是：

> 算法 A 在本研究的真实诱捕长视频和统一计数协议下获得最低 Class-MAE，更适合作为后续计数优化的工程 tracker baseline。

不允许直接写成：

> 算法 A 的通用多目标跟踪性能优于其他全部算法。

---

## 2. 本轮实验的目标、假设和非目标

### 2.1 主要目标

固定检测器、检测数字输入、计数区域和计数逻辑，仅替换 tracker，从 8 个候选中选择当前真实长视频逐物种计数误差最低的算法。

### 2.2 实验假设

本轮实验采用以下前提：

1. 用户已确认所有逻辑长视频开始时瓶内实蝇数量为 0；
2. 目录级人工数字表示全部物理片段拍摄结束后的最终瓶内数量；
3. 所有片段均按清单中的真实顺序处理；
4. 当前 6 个逻辑长视频全部用于寻找工程 baseline；
5. 不把本轮结果表述为独立测试集泛化性能；
6. 跟踪器只允许使用当前帧检测结果和历史轨迹状态；
7. 两个类别共同进入一个 tracker，类别不参与轨迹匹配。

### 2.3 本轮非目标

本轮不研究：

- ROI 是否能提高检测或计数；
- 圆心和半径的消融；
- 边界容差带；
- 连续多帧进入/退出确认；
- 轨迹拼接或跨 ID 修复；
- 类别投票或类别锁定优化；
- born-inside 特殊处理；
- 负计数截断；
- ReID 或外观特征；
- 标准 MOT 指标；
- 检测器之间的再次比较；
- 树莓派最终部署速度。

上述内容均留到选出 tracker baseline 后再作为独立变量研究。

---

## 3. 实验变量设计

### 3.1 自变量

唯一主要自变量为 tracker：

```text
SORT
ByteTrack
OC-SORT
SF-SORT
FastTracker
BoostTrack
HybridSORT
BoT-SORT
```

### 3.2 控制变量

8 个算法必须完全相同的条件：

- 输入视频字节；
- 成功解码的帧序列；
- RT-DETR-R18 模型结构和权重；
- 图像预处理和推理精度；
- 每帧检测框、置信度和类别；
- 检测缓存最低分数；
- 视频 FPS 和分辨率元数据；
- tracker 每个物理片段重新初始化；
- 两类目标共同进入同一个 tracker；
- 类别不参与匹配；
- 计数圆；
- 计数点；
- 进入/退出定义；
- 逻辑视频初始计数；
- 片段累计顺序；
- 人工真值；
- 评价程序。

### 3.3 因变量

主要因变量：

- `Class-MAE`。

次要因变量：

- dorsalis MAE；
- correcta MAE；
- Total-MAE；
- Exact-Video-Rate；
- Negative-Video-Rate；
- 整体 tracking FPS；
- 每个逻辑视频逐物种有符号误差。

---

## 4. 真实长视频数据

### 4.1 数据组织

当前数据由 25 个 MP4 物理片段组成，按采集语义合并为 6 个逻辑长视频。正式评价单位是逻辑长视频，不是物理片段。

| 逻辑视频 | 物理片段数 | 时长（小时） | 总帧数 | 平均 FPS | dorsalis 真值 | correcta 真值 |
|---|---:|---:|---:|---:|---:|---:|
| LV001 | 1 | 7.479 | 1,508,416 | 56.021 | 132 | 0 |
| LV002 | 1 | 7.307 | 1,448,496 | 55.066 | 208 | 0 |
| LV003 | 5 | 11.165 | 2,248,584 | 55.944 | 236 | 0 |
| LV004 | 5 | 10.411 | 2,074,782 | 55.358 | 244 | 0 |
| LV005 | 6 | 4.668 | 941,188 | 56.002 | 61 | 118 |
| LV006 | 7 | 8.910 | 1,785,593 | 55.665 | 79 | 27 |
| **合计** | **25** | **49.963** | **10,007,059** | — | **960** | **145** |

总时长为：

```text
179,786.922060 秒
= 49 小时 56 分 26.922 秒
```

总视频大小约 175,001,208,713 字节。全部视频均为：

```text
H.264
2304 × 1296
约 55–56 FPS
```

### 4.2 权威清单

正式实验必须以以下清单为准：

- `video_data/manifests/long_videos.csv`：6 个逻辑视频；
- `video_data/manifests/segments.csv`：25 个物理片段、顺序、帧数、大小和 FPS；
- `video_data/manifests/ground_truth_v1.csv`：逻辑视频逐物种真值；
- `video_data/manifests/validation_report_20260725.md`：视频完整性验证。

不得根据文件系统临时扫描顺序猜测片段顺序。正式运行应显式按 `video_id + segment_index` 读取。

### 4.3 真值定义

人工真值为：

```text
全部片段拍摄结束后，瓶内两种实蝇的最终数量
```

用户已明确确认：

```text
每个逻辑长视频开始时瓶内数量 N_start = 0
```

因此算法估计量为：

```text
N_final = 0 + N_enter - N_exit
        = N_enter - N_exit
```

本轮允许预测为负数。负数不截断、不修正，因为负计数是跟踪或计数失败的重要诊断信号。

### 4.4 当前数据的限制

- 只有 6 个逻辑长视频；
- correcta 只在 LV005 和 LV006 中出现正样本；
- 四个逻辑视频的 correcta 真值为 0；
- 没有逐帧 MOT 身份真值；
- 全部 6 个视频用于工程 baseline 选择，没有独立留出的泛化测试视频。

因此本轮适合做工程选型和后续方法起点，不足以单独支撑“普遍适用于所有诱捕场景”的强结论。

---

## 5. 固定目标检测器

### 5.1 检测器选择

本轮跟踪器对比固定使用：

```text
RT-DETR-R18
```

它来自 11 个检测模型的统一 640 输入、100 epoch、seed 42 工程筛选。在修正后的统一检测数据上：

- mixed val `mAP50-95 = 0.779254`；
- mixed test `mAP50-95 = 0.675453`；
- 约 20.084 M 参数；
- 约 61.408 GFLOPs；
- A100 上统一 FP32 测试约 119.36 FPS。

这些检测指标用于说明为什么在本轮 tracker 对比中选择它作为固定前端；本轮不再次比较检测器。

### 5.2 冻结权重

```text
checkpoint:
detection_model_benchmark/results/fruitfly_detection_v1/
benchmark_640_e100_s42/runs/10_rt_detr_r18/checkpoints/best.pth

best epoch:
100

SHA-256:
558b7fd2c7b6488649d64cada5d6c860841301d00b2afec7907f29eeef44cc6f
```

正式运行必须先重新计算权重 SHA-256 并与上述值一致，否则拒绝生成检测缓存。

### 5.3 推理配置

固定为：

| 项目 | 配置 |
|---|---|
| 输入区域 | 原视频完整画面 |
| ROI | 不使用 |
| 裁剪 | 不使用 |
| 原始分辨率 | 2304×1296 |
| 模型输入 | 640×640 |
| 预处理 | 与官方冻结 val 配置一致：Resize、ToImageTensor、ConvertDtype |
| 推理精度 | FP32 |
| TTA | 关闭 |
| 后处理 | RT-DETR 原生 NMS-free postprocessor |
| NMS IoU | 不适用 |
| 原生候选上限 | 每帧 top-300 |
| 共享缓存最低置信度 | `confidence ≥ 0.10` |
| 类别过滤 | 不过滤，保留 class 0 和 class 1 |
| 输出坐标 | 映射回 2304×1296 原始完整画面坐标 |

### 5.4 为什么缓存下限使用 0.10

ByteTrack、SF-SORT、FastTracker、HybridSORT 和 BoT-SORT 等算法具有低置信度检测二阶段关联。如果检测缓存一开始就只保留 `confidence ≥ 0.25` 或 `≥ 0.60` 的框，这些算法的低分框恢复能力会被上游提前删除，比较不再代表算法本身。

因此统一检测缓存保留：

```text
confidence ≥ 0.10
```

该值只是“共享输入缓存下限”，不是所有 tracker 的统一接收阈值。每个 tracker 仍按自己的官方参数决定：

- 哪些是高分检测；
- 哪些只能用于维持旧轨迹；
- 哪些可以建立新轨迹。

SORT 原生没有置信度阈值，因此会使用缓存中全部 `confidence ≥ 0.10` 的检测。这是共享同一输入条件的结果，不为 SORT 额外增加非官方过滤器。

### 5.5 检测缓存

检测器只对 25 个物理片段推理一次，逐帧保存：

```text
[x1, y1, x2, y2, confidence, class_id]
```

统一接口记为：

```text
Nx6 full-frame detections
```

空检测帧也必须写出空数组，不能跳帧。8 个 tracker 必须读取逐字节相同的检测数字缓存。

### 5.6 当前检测端工程状态

冻结 RT-DETR 权重和官方评测加载逻辑已经存在，但当前通用视频缓存脚本仍是 Ultralytics YOLO 加载入口，不能直接加载该原始 `best.pth`。

因此正式实验前必须完成：

1. 将 RT-DETR-R18 官方模型、EMA 权重和 postprocessor 接入视频逐帧推理；
2. 复用冻结 val 的 640×640 预处理和原始尺寸坐标恢复；
3. 保持 NMS-free；
4. 按分数取 top-300 后保存 `confidence ≥ 0.10` 的两个类别；
5. 用已知图片与冻结 val/test 推理结果做数值一致性测试；
6. 通过后再生成 25 个长视频检测缓存。

在此项完成前，正式 8 算法全量实验不能开始。

---

## 6. 统一跟踪输入输出和公平性原则

### 6.1 统一公开接口

所有算法外部统一为：

```text
输入：
Nx6 [x1, y1, x2, y2, confidence, class_id]

输出：
Nx8 [x1, y1, x2, y2, track_id,
     confidence, class_id, detection_index]
```

各官方算法内部需要的 `Nx5`、`boxes + scores` 或 Results-like 对象由适配器转换，计数器不感知 tracker 类型。

### 6.2 类别处理

统一规定：

- `per_class = false`；
- class 0 和 class 1 共同进入一个 tracker；
- `class_id` 不参与关联距离；
- 不允许按类别分别维护两套 ID；
- 类别只在跟踪完成后用于逐物种计数。

大多数官方 tracker 实际只接收：

```text
[x1, y1, x2, y2, confidence]
```

FastTracker 的 Results-like 输入中包含 class，但类别只作为输出元数据，不进入当前关联代价。

### 6.3 统一元数据回填

官方 tracker 通常只输出轨迹框和 ID，没有统一返回当前检测的置信度、类别和 detection index。公共适配器在 tracker 完成关联之后，使用当前轨迹框与当前检测框的一对一 IoU 匹配回填元数据：

```text
metadata_match_iou = 0.10
```

该步骤只用于恢复：

- confidence；
- class_id；
- detection_index。

它发生在原生 tracker 关联之后，不会反向影响 ID 匹配。若当前轨迹输出无法与当前检测达到 0.10 IoU，则：

- `detection_index = -1`；
- confidence 和 class_id 沿用该 track_id 上一次记录。

该共享元数据策略对 8 个算法完全一致，应与原生 tracker 参数分开记录。

### 6.4 允许使用的功能

只要功能完全依赖当前/历史检测结果和轨迹状态，就允许保留：

- 卡尔曼滤波和运动预测；
- IoU、GIoU、HMIoU、形状相似度；
- Mahalanobis 距离；
- 检测置信度融合；
- 高低分框两阶段关联；
- lost track 保存与重新激活；
- 观测中心修正；
- 基于检测框重叠的遮挡判断；
- 基于历史框的速度回滚；
- 基于当前检测数量的动态阈值；
- BoostTrack 的 detection likelihood/shape/soft boosting；
- HybridSORT 的 confidence state 和 height state。

### 6.5 禁止使用的功能

本轮统一禁止：

- ReID；
- 外观 embedding；
- 原图纹理特征；
- GMC；
- ECC；
- 光流；
- 相机运动估计；
- 分割掩膜；
- 深度信息；
- 未来帧；
- 离线轨迹插值；
- tracker 外部轨迹拼接；
- tracker 外部 ID 合并或修复；
- 类别参与关联；
- ROI。

原因是当前统一输入只包含检测数字，不能为部分算法额外提供图像或外观信息。

### 6.6 “官方参数”的采用规则

不同官方仓库对“默认参数”的定义并不一致：

- 有些算法在构造函数中直接给出完整默认值；
- 有些算法的命令行默认值只是空壳，论文核心功能在官方 experiment config 中开启；
- 有些算法提供 MOT17/MOT20/DanceTrack 等数据集专用配置；
- 有些算法完整版本依赖 ReID 或相机运动。

本实验统一采用以下规则：

1. 优先采用作者官方仓库提供的主评测默认值或官方论文复现配置；
2. 如果裸构造器默认会关闭命名算法的核心功能，则使用官方复现实验配置；
3. 多个数据集配置并存时，采用官方通用/MOT17 无 ReID 配置，不根据本实验最终计数结果挑选；
4. 只关闭统一数字缓存无法提供的 ReID、GMC/ECC 等功能；
5. 不凭主观判断额外打开官方默认关闭的混合算法开关；
6. 不针对当前 6 个视频逐算法调参；
7. 所有最终解析参数随结果保存并计算哈希。

例如 OC-SORT 提供 `use_byte` 开关，但官方基础配置默认关闭。若打开，它更接近 “OC-SORT + BYTE 二阶段关联”，因此本轮标准 OC-SORT baseline 保持关闭。HybridSORT 的官方复现配置本身明确开启 BYTE 和 TCM，因此在 HybridSORT 中保留。

---

## 7. 8 个 tracker 的正式参数

### 7.1 官方源码版本

| Tracker | 本地编号 | 固定官方提交 |
|---|---|---|
| SORT | 01_sort | `2236dff5019565958b84df7d871d41cc1db58ac7` |
| ByteTrack | 02_bytetrack | `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99` |
| OC-SORT | 03_ocsort | `8462e7e729a93ccd3bd995c0a79a890336cb3a0b` |
| SF-SORT | 04_sfsort | `b1abdeca5ee361c59a20a7fa3937e830846eea82` |
| FastTracker | 05_fasttracker | `a7e7e65a3e4086c5d2104dca06f6593564eda1ae` |
| BoostTrack | 06_boosttrack | `fb5bfc3a8f067476565e753b3a73df4d757c9d03` |
| HybridSORT | 07_hybridsort | `396f8d30db13304c0cbaf1dcf2e16ded93ce1701` |
| BoT-SORT | 08_botsort | `251985436d6712aaf682aaaf5f71edb4987224bd` |

正式结果中必须同时记录：

- 上述来源提交；
- 主仓库提交；
- Conda 环境和关键依赖版本；
- 实际解析后的 tracker 配置 JSON。

### 7.2 SORT

参数来源：官方构造器默认。

| 参数 | 值 | 含义 |
|---|---:|---|
| `max_age` | 1 | 连续 1 帧未匹配后删除轨迹 |
| `min_hits` | 3 | 稳定输出轨迹所需命中数 |
| `iou_threshold` | 0.30 | IoU 关联阈值 |
| `metadata_match_iou` | 0.10 | 公共适配层元数据回填阈值，非原生跟踪参数 |

启用：

- 卡尔曼运动预测；
- IoU 匹配；
- 匈牙利分配。

不具备：

- 低分框二阶段关联；
- 分数融合；
- 长期 lost-track 重激活；
- ReID/GMC。

特殊说明：SORT 没有原生置信度过滤，因此使用共享缓存中全部 `confidence ≥ 0.10` 检测。

当前配置判断：保持不变。

### 7.3 ByteTrack

参数来源：官方主评测默认。

| 参数 | 值 | 含义 |
|---|---:|---|
| `track_thresh` | 0.60 | 第一阶段高分检测阈值 |
| 内部低分阈值 | 0.10 | 第二阶段低分检测下限 |
| 内部新轨迹阈值 | 0.70 | 官方实现为 `track_thresh + 0.1` |
| `track_buffer` | 30 | 30 FPS 基准下 lost-track 缓冲长度 |
| `match_thresh` | 0.90 | 第一阶段线性分配阈值 |
| 第二阶段匹配阈值 | 0.50 | 官方代码固定值 |
| `mot20` | false | 非 MOT20 模式，开启检测分数与 IoU 融合 |
| `metadata_match_iou` | 0.10 | 公共元数据回填阈值 |

启用：

- 卡尔曼预测；
- 高分检测第一阶段关联；
- 低分检测第二阶段恢复；
- lost-track 重新激活；
- 第一阶段检测分数融合；
- 重复轨迹清理。

帧率说明：官方实现按：

```text
buffer_size = int(fps / 30 × track_buffer)
```

根据本数据约 55–56 FPS，实际缓冲约为 55–56 帧，约等于 1 秒。

当前配置判断：保持不变。

### 7.4 OC-SORT

参数来源：官方主入口与构造器默认。

| 参数 | 值 | 含义 |
|---|---:|---|
| `det_thresh` | 0.60 | 高分检测阈值 |
| `max_age` | 30 | 最大未匹配帧数 |
| `min_hits` | 3 | 最小命中数 |
| `iou_threshold` | 0.30 | 关联阈值 |
| `delta_t` | 3 | 速度方向估计的历史间隔 |
| `asso_func` | `iou` | 关联相似度 |
| `inertia` | 0.20 | 速度方向一致性权重 |
| `use_byte` | false | 不启用可选 BYTE 低分框扩展 |
| `metadata_match_iou` | 0.10 | 公共元数据回填阈值 |

启用：

- Observation-Centric Momentum；
- Observation-Centric Re-Update；
- 基于历史观测的速度方向；
- IoU 与方向一致性关联；
- lost-track 生命周期管理。

关闭：

- `use_byte`，因为官方基础 OC-SORT 默认关闭；
- ReID；
- ECC/GMC。

当前配置判断：保持不变。

### 7.5 SF-SORT

参数来源：官方 `SFSORT_YOLOv8.ipynb` 在线示例，而不是会把恢复时间设为 0 的裸构造器占位默认。

建议正式参数：

| 参数 | 值/公式 | 含义 |
|---|---:|---|
| `dynamic_tuning` | true | 按当前检测数量动态调整阈值 |
| `cth` | 0.50 | 统计有效检测数量的阈值 |
| `high_th` | 0.60 | 高分检测基础阈值 |
| `high_th_m` | 0.10 | 高分阈值动态变化幅度 |
| `match_th_first` | 0.67 | 第一阶段关联阈值 |
| `match_th_first_m` | 0.05 | 第一阶段动态变化幅度 |
| `match_th_second` | 0.20 | 中低分检测第二阶段关联阈值 |
| `low_th` | 0.10 | 最低检测阈值 |
| `new_track_th` | 0.70 | 新建轨迹基础阈值 |
| `new_track_th_m` | 0.08 | 官方 Notebook 写 0.10，但 v4.2 源码有效上限为 0.08 |
| `marginal_timeout` | `floor(0.7 × fps)` | 边缘位置丢失轨迹保留帧数 |
| `central_timeout` | `floor(1.0 × fps)` | 中央区域丢失轨迹保留帧数 |
| `horizontal_margin` | `floor(0.1 × width)` | 左右边缘范围 |
| `vertical_margin` | `floor(0.1 × height)` | 上下边缘范围 |
| `metadata_match_iou` | 0.10 | 公共元数据回填阈值 |

对于 2304×1296、约 55–56 FPS 视频，典型解析值为：

```text
marginal_timeout ≈ 38–39 帧
central_timeout  ≈ 55–56 帧
horizontal_margin = 230 px
vertical_margin   = 129 px
```

启用：

- 高分和中低分两阶段关联；
- 基于检测数量的动态阈值；
- 中央/边缘不同的 lost-track 生命周期；
- 只依赖检测框和历史状态的轨迹找回。

禁止：

- 官方可选的离线后处理和未来帧插值。

当前代码冲突：

- 当前 `dynamic_tuning=false`；
- 当前两个 timeout 均为 0；
- 当前两个 margin 均为 0；
- 当前 `match_th_second=0.30`。

这会使 SF-SORT 的轨迹找回能力没有按官方在线示例发挥。正式 baseline 前必须修改适配器配置字段和配置文件，并按每个片段 FPS 动态解析。

当前配置判断：必须修改后再冻结。

### 7.6 FastTracker

参数来源：官方 Ultralytics `fasttrack.yaml`。

| 参数 | 值 | 含义 |
|---|---:|---|
| `track_high_thresh` | 0.25 | 第一阶段阈值 |
| `track_low_thresh` | 0.10 | 第二阶段最低阈值 |
| `new_track_thresh` | 0.25 | 新轨迹阈值 |
| `track_buffer` | 30 | lost-track 缓冲 |
| `match_thresh` | 0.80 | 第一阶段匹配阈值 |
| `fuse_score` | true | 检测分数与 IoU 融合 |
| `reset_velocity_offset_occ` | 5 | 遮挡时速度回滚历史偏移 |
| `reset_pos_offset_occ` | 3 | 遮挡时位置回滚历史偏移 |
| `enlarge_bbox_occ` | 1.10 | 遮挡搜索框放大系数 |
| `dampen_motion_occ` | 0.50 | 遮挡时运动衰减 |
| `active_occ_to_lost_thresh` | 10 | 遮挡轨迹转 lost 的帧数 |
| `occ_cover_thresh` | 0.70 | 判定遮挡的覆盖比例 |
| `occ_reappear_window` | 40 | 遮挡后重现窗口 |
| `init_iou_suppress` | 0.70 | 与活动轨迹高度重叠时抑制新 ID |
| `metadata_match_iou` | 0.10 | 公共元数据回填阈值 |

启用：

- 高低分框两阶段关联；
- 检测分数融合；
- 仅基于检测框重叠的遮挡判断；
- Kalman 位置/速度回滚；
- 遮挡后重现；
- 新轨迹重复抑制。

这些功能不需要 ReID、图像纹理或未来帧，全部保留。

当前配置判断：保持不变。

### 7.7 BoostTrack

参数来源：官方默认 MOT17 分数配置；关闭数字缓存无法提供的 embedding 和 ECC。

| 参数 | 值 | 含义 |
|---|---:|---|
| `max_age` | 30 | 最大未匹配帧数 |
| `min_hits` | 3 | 最小命中数 |
| `det_thresh` | 0.60 | 检测阈值 |
| `iou_threshold` | 0.30 | 关联阈值 |
| `lambda_iou` | 0.50 | IoU 相似度权重 |
| `lambda_mhd` | 0.25 | Mahalanobis 距离权重 |
| `lambda_shape` | 0.25 | 框形状相似度权重 |
| `use_dlo_boost` | true | Detection Likelihood Boost |
| `use_duo_boost` | true | DUO Boost |
| `dlo_boost_coef` | 0.65 | MOT17 官方系数 |
| `use_rich_s` | true | 丰富相似度 |
| `use_sb` | true | Soft Boost |
| `use_vt` | true | Varying Threshold |
| `use_embedding` | false | 统一数字输入无法提供外观特征 |
| `use_ecc` | false | 统一数字输入无法提供图像运动 |
| `metadata_match_iou` | 0.10 | 公共元数据回填阈值 |

启用：

- IoU、Mahalanobis 和形状组合；
- DLO/DUO；
- rich similarity；
- soft boost；
- varying threshold；
- Kalman 运动状态。

结果中应命名为：

```text
BoostTrack detection-only
```

不能把它表述成使用了完整外观 embedding/ECC 的 BoostTrack。

当前配置判断：保持不变。

### 7.8 HybridSORT

参数来源：官方 canonical MOT17 no-ReID 实验配置。

| 参数 | 值 | 含义 |
|---|---:|---|
| `track_thresh` | 0.60 | 高分检测阈值 |
| `det_thresh` | 0.60 | 建轨/检测阈值 |
| `max_age` | 30 | 最大未匹配帧数 |
| `min_hits` | 3 | 最小命中数 |
| `iou_threshold` | 0.25 | 关联阈值 |
| `delta_t` | 3 | 方向估计间隔 |
| `asso_func` | `Height_Modulated_IoU` | 高度调制 IoU |
| `inertia` | 0.05 | 方向一致性权重 |
| `use_byte` | true | 开启低分检测第二阶段 |
| `TCM_first_step` | true | 第一阶段 confidence-state 匹配 |
| `TCM_byte_step` | true | BYTE 阶段 confidence-state 匹配 |
| `TCM_first_step_weight` | 1.0 | 第一阶段 TCM 权重 |
| `TCM_byte_step_weight` | 1.0 | 第二阶段 TCM 权重 |
| `hybrid_sort_with_reid` | false | 不使用 ReID |
| `with_fastreid` | false | 不使用 FastReID |
| `ECC` | false | 不使用图像相机运动 |
| `metadata_match_iou` | 0.10 | 公共元数据回填阈值 |

启用：

- Observation-Centric 运动信息；
- confidence state；
- height state；
- HMIoU；
- 高低分框两阶段关联；
- 两阶段 TCM。

结果中应命名为：

```text
HybridSORT no-ReID
```

当前配置判断：保持不变。

### 7.9 BoT-SORT

参数来源：官方主评测默认；关闭 ReID 和 CMC。

| 参数 | 值 | 含义 |
|---|---:|---|
| `track_high_thresh` | 0.60 | 第一阶段高分阈值 |
| `track_low_thresh` | 0.10 | 第二阶段最低阈值 |
| `new_track_thresh` | 0.70 | 新建轨迹阈值 |
| `track_buffer` | 30 | 30 FPS 基准缓冲长度 |
| `match_thresh` | 0.80 | 第一阶段匹配阈值 |
| `mot20` | false | 开启分数与 IoU 融合 |
| 第二阶段匹配阈值 | 0.50 | 官方代码固定值 |
| `with_reid` | false | 不使用外观特征 |
| `cmc_method` | `none` | 不使用相机运动补偿 |
| `proximity_thresh` | 0.50 | ReID 关闭时不参与最终代价 |
| `appearance_thresh` | 0.25 | ReID 关闭时不生效 |
| `metadata_match_iou` | 0.10 | 公共元数据回填阈值 |

启用：

- Kalman 预测；
- 高低分框两阶段关联；
- 检测分数融合；
- lost-track 重新激活；
- 重复轨迹移除。

帧率说明：官方实现按实际 FPS 缩放 `track_buffer`，本数据中典型有效缓冲约 55–56 帧。

结果中应命名为：

```text
BoT-SORT no-ReID/no-GMC
```

当前配置判断：保持不变。

---

## 8. 统一计数规则

### 8.1 固定圆

参考画面：

```text
2304 × 1296
```

固定计数圆：

```text
圆心：(1185, 732)
半径：215 px
```

本轮全部视频分辨率相同，因此不发生实际缩放。代码仍保留按画面比例缩放能力，但 baseline 使用上述原始值。

### 8.2 计数点

每条轨迹使用检测/轨迹框中心：

```text
center_x = (x1 + x2) / 2
center_y = (y1 + y2) / 2
```

圆内判定：

```text
sqrt((center_x - 1185)^2 + (center_y - 732)^2) <= 215
```

等于半径时视为圆内。

### 8.3 跨界事件

```text
上一状态 outside，当前状态 inside：enter，delta = +1
上一状态 inside，当前状态 outside：exit， delta = -1
其他情况：delta = 0
```

轨迹第一次出现时只登记 inside/outside 状态，不生成计数事件。

### 8.4 有符号累计

每个逻辑长视频开始：

```python
region_counts = {
    "Bactrocera dorsalis": 0,
    "Bactrocera correcta": 0,
}
```

每次事件：

```python
region_counts[species] += delta
```

允许负数：

```text
不 clamp
不 max(0, count)
不做事后修正
```

### 8.5 类别归属

- 进入事件使用轨迹当前记录的类别；
- 离开事件使用该 track_id 上一次记录的类别；
- 不做类别投票；
- 不锁定整条轨迹的第一次类别；
- 不按类别分别跟踪。

该规则的目的不是声称它最优，而是先建立最简单、统一且无额外优化的 baseline。

### 8.6 多片段逻辑

同一个物理片段内：

- tracker 连续逐帧运行；
- track ID 和轨迹状态连续。

切换到同一逻辑长视频的下一个物理片段时：

- 重新创建 tracker；
- track ID 从头开始；
- 清空 `track_id → inside/outside` 状态；
- 清空轨迹类别状态；
- 保留 `region_counts`，继续累计。

切换到下一个逻辑长视频时：

- tracker 重置；
- 计数重新归零。

这样避免把拍摄不连续的物理文件误当成同一连续轨迹，同时保留目录级最终计数。

---

## 9. 三级数字缓存与重放

### 9.1 分层结构

```text
原始视频
  ↓
检测缓存 D：每帧 Nx6
  ↓
轨迹缓存 T：每帧 Nx8
  ↓
计数结果 C：事件、片段汇总、逻辑视频最终计数
```

### 9.2 为什么必须缓存

当前共有 10,007,059 帧。若每次更换 tracker 或计数参数都重新运行 RT-DETR，会造成大量重复计算，并可能因推理环境或代码变化破坏公平性。

缓存后：

| 改动 | 需要重跑 | 可复用 |
|---|---|---|
| 检测器/权重/预处理/conf | 检测、跟踪、计数 | 原视频 |
| tracker/跟踪参数 | 跟踪、计数 | 检测缓存 |
| 圆/计数逻辑/类别规则 | 计数 | 轨迹缓存 |
| 评价表或可视化 | 评价导出 | 计数或轨迹缓存 |

### 9.3 检测缓存 D

每帧：

```text
[x1, y1, x2, y2, confidence, class_id]
```

必须记录：

- 视频内容 SHA-256；
- 视频大小；
- 宽、高、FPS；
- 清单帧数；
- 实际成功解码帧数；
- RT-DETR checkpoint SHA-256；
- 代码提交；
- 640×640 预处理；
- FP32；
- NMS-free；
- top-300；
- cache floor 0.10；
- 类别映射；
- 每帧数字内容累计 SHA-256；
- complete footer。

### 9.4 轨迹缓存 T

每帧：

```text
[x1, y1, x2, y2,
 track_id, confidence, class_id, detection_index]
```

必须记录：

- 上游检测缓存内容哈希；
- tracker 名称；
- 官方源码提交；
- 主仓库提交；
- 完整解析参数；
- per_class=false；
- 图像依赖关闭项；
- 实际帧数；
- 轨迹数字内容哈希；
- 纯 tracking_seconds；
- complete footer。

### 9.5 缓存失效条件

以下任一项变化，检测缓存必须失效：

- 视频字节；
- RT-DETR 权重；
- 预处理；
- 输入尺寸；
- FP32/FP16；
- top-k；
- 缓存置信度；
- 类别映射；
- 后处理。

以下任一项变化，轨迹缓存必须失效：

- 上游检测缓存内容；
- tracker；
- tracker 参数；
- FPS 解析；
- 适配器实现；
- 官方源码版本；
- 公共元数据回填策略。

圆形参数或计数逻辑变化时，不必重新检测和跟踪，只需从轨迹缓存重放计数。

---

## 10. 正式实验保护

### 10.1 输入完整性

运行前必须严格验证：

```text
逻辑长视频数 = 6
物理片段数   = 25
总帧数       = 10,007,059
分辨率       = 2304×1296
类别映射     = 0 dorsalis / 1 correcta
```

不仅要检查“目录存在”，还要逐项核对：

- `video_id`；
- `segment_index`；
- 相对路径；
- 文件字节大小；
- 视频 SHA-256；
- 报告帧数；
- 实际解码帧数；
- 检测缓存是否完整。

漏一个片段、多一个旧缓存或片段顺序错误都必须拒绝运行。

### 10.2 防止旧缓存混入

正式运行不能只按文件名匹配缓存。每个缓存必须绑定：

```text
video_sha256
detector_checkpoint_sha256
detector_config_sha256
numeric_payload_sha256
```

当前缓存 `cache_id` 主要由头信息生成，尚未覆盖全部数字 payload。正式实验前必须增加内容哈希。

### 10.3 配置冻结

每次正式运行将 8 份解析后配置复制到结果目录：

```text
resolved_configs/
  sort.json
  bytetrack.json
  ocsort.json
  sfsort.json
  fasttracker.json
  boosttrack.json
  hybridsort.json
  botsort.json
  counting.json
  detector.json
```

每个文件计算 SHA-256。结果不能只保存原配置路径，因为原文件以后可能被修改。

### 10.4 结果防覆盖

正式运行 ID 建议：

```text
rtdetr_r18_official_trackers_circle_v1_<YYYYMMDD_HHMMSS>
```

规则：

- 输出目录不存在时才能新建；
- 非空目录默认拒绝运行；
- 不允许自动覆盖同名轨迹、计数和评价文件；
- 如需断点续跑，必须验证所有输入与配置哈希完全相同；
- 失败的临时缓存不能被正式评价读取；
- 每次重跑产生新的 run_id。

### 10.5 运行环境

保存：

- 操作系统；
- Python；
- Conda 环境；
- NumPy、SciPy、PyTorch、OpenCV；
- GPU 型号；
- CUDA/cuDNN；
- CPU；
- 主仓库 commit；
- 8 个官方源码 commit；
- 是否启用确定性算法；
- 每个进程退出码。

### 10.6 时间统计

分开记录：

- 检测耗时；
- 纯 tracker `update()` 耗时；
- 缓存读写和整体 wall time；
- 纯计数重放耗时。

每个 tracker 的主速度指标：

```text
overall_tracking_fps
= 25 个片段总帧数
  / 25 个片段纯 tracker update 总耗时
```

不能直接平均 25 个片段各自的 FPS，因为片段帧数不同。应采用总帧数除以总耗时的加权整体速度。

### 10.7 随机性

这些 tracker 主要为确定性在线算法。正式运行仍应固定：

```text
Python seed
NumPy seed
PyTorch seed
```

同一检测缓存和配置重复短序列运行两次，轨迹缓存数字哈希必须一致。若某算法不能确定性复现，则至少运行 3 次并报告均值和标准差，同时说明随机来源。

---

## 11. 评价指标

设逻辑视频数量 `N=6`，类别集合：

```text
C = {dorsalis, correcta}
```

`y(i,c)` 为视频 `i` 类别 `c` 的人工真值，`p(i,c)` 为预测。

### 11.1 分物种 MAE

```text
MAE(c) = (1/N) × Σ_i |p(i,c) - y(i,c)|
```

分别报告：

- dorsalis MAE；
- correcta MAE。

注意：correcta 只有 2 个逻辑视频为正，其 MAE 会受到 4 个零真值视频的误检影响，应同时查看逐视频结果，不能只看一个平均数。

### 11.2 主指标 Class-MAE

```text
Class-MAE
= (1 / (N × 2))
  × Σ_i Σ_c |p(i,c) - y(i,c)|
```

它直接惩罚逐物种错误，不允许两个类别的误差相互抵消，是本轮 tracker 选择主指标。

### 11.3 有符号误差

```text
Error(i,c) = p(i,c) - y(i,c)
```

- 正值：过计；
- 负值：漏计或净退出过多。

逐视频同时保存：

- prediction；
- ground_truth；
- signed_error；
- absolute_error。

### 11.4 Total-MAE

```text
Total-MAE
= (1/N)
  × Σ_i |Σ_c p(i,c) - Σ_c y(i,c)|
```

Total-MAE 只用于辅助观察总虫数。两个类别一多一少时可能抵消，所以不能替代 Class-MAE。

### 11.5 Exact-Video-Rate

只有同一逻辑视频中两个物种都完全正确才算 exact：

```text
Exact-Video-Rate
= 两类同时完全正确的视频数 / 6
```

### 11.6 Negative-Video-Rate

```text
Negative-Video-Rate
= 至少一个物种预测为负的视频数 / 6
```

负数保留为失败诊断，不做截断。

### 11.7 速度

报告：

- overall tracking FPS；
- 每个物理片段 tracking FPS；
- 总 tracking seconds；
- 总 wall time。

检测器对 8 个算法共享，因此 tracker 主表只比较纯跟踪速度。检测速度单独报告。

---

## 12. Baseline 选择规则

所有 6 个逻辑长视频都进入本轮工程选型。配置冻结后一次性运行完整数据，不能看到最终结果后再修改某个算法参数并仍称为同一轮官方 baseline。

排序规则：

1. `Class-MAE` 升序；
2. 若相同，比较两个物种 MAE 的最大值，较小者优先；
3. 若仍相同，`Exact-Video-Rate` 较高者优先；
4. 若仍相同，`Total-MAE` 较低者优先；
5. 若准确率指标仍接近，overall tracking FPS 较高者优先；
6. 再相同则优先选择依赖更少、实现更简单、部署成本更低的算法。

本轮胜出算法被定义为：

```text
当前 6 个真实逻辑长视频上的计数工程 baseline tracker
```

不能称为：

```text
在未知数据上已经证实最优的通用 tracker
```

如果后续新增独立长视频，应保持本轮配置不变，在新视频上做外部验证。

---

## 13. 正式执行步骤

### 阶段 0：导师审查和协议冻结

1. 审查本协议的研究问题和结论边界；
2. 审查 RT-DETR-R18 固定前端；
3. 审查共享 `conf ≥ 0.10` 缓存；
4. 审查 8 算法参数来源；
5. 审查 SF-SORT 官方示例配置；
6. 审查固定圆和有符号计数；
7. 审查使用全部 6 个视频做工程选型；
8. 确认后生成 `protocol_sha256`，后续不得静默修改。

### 阶段 1：补齐代码

1. 实现 RT-DETR-R18 视频检测缓存后端；
2. 完成原生推理一致性测试；
3. 修改 SF-SORT 完整参数；
4. 增加 25 片段清单强校验；
5. 增加 numeric payload SHA-256；
6. 增加配置快照和哈希；
7. 增加输出目录防覆盖；
8. 增加整体加权 FPS 汇总；
9. 将正式 baseline 的 `N_start=0` 写入运行清单；
10. 去掉正式批处理中的任意非零初始数量覆盖。

### 阶段 2：小规模链路验证

不查看最终 6 视频计数排名，只做工程正确性验证：

1. 单个短帧段生成 RT-DETR 检测缓存；
2. 检查框坐标映射回 2304×1296；
3. 检查 class 0/1 顺序；
4. 8 个 tracker 各运行同一小段；
5. 验证输入检测缓存哈希一致；
6. 验证两次运行轨迹哈希一致；
7. 验证计数圆和跨界单元测试；
8. 验证片段边界 tracker 重置、计数继续；
9. 验证异常中断不产生 complete 缓存。

### 阶段 3：生成统一检测缓存

对 25 个物理片段只运行一次 RT-DETR-R18：

1. 按 manifest 顺序读取；
2. 每帧完整画面推理；
3. 保存 NMS-free top-300 中 `confidence ≥ 0.10` 的检测；
4. 空帧写空数组；
5. 每片段写完整 footer；
6. 汇总实际解码帧数；
7. 验证总帧数 10,007,059；
8. 冻结检测缓存集合哈希。

### 阶段 4：运行 8 个 tracker

每个算法：

1. 读取完全相同的 25 个检测缓存；
2. 每个物理片段新建 tracker；
3. 逐帧在线运行，空帧也更新；
4. 写统一 Nx8 轨迹缓存；
5. 保存解析配置、代码版本和耗时；
6. 完成后校验 25/25；
7. 计算轨迹 payload hash 和 overall tracking FPS。

为避免资源干扰，建议 8 个 tracker 串行运行；如并行，必须分配独立 CPU/GPU 资源，并单独进行串行速度复测。

### 阶段 5：统一计数

对每个 tracker：

1. 按 LV001–LV006 分组；
2. 每个逻辑视频从 0 开始；
3. 按 `segment_index` 顺序重放；
4. 片段间重置 ID 状态；
5. 计数持续累计；
6. 保存全部 enter/exit 事件；
7. 保存片段起止计数；
8. 保存逻辑视频最终逐物种结果。

### 阶段 6：评价与选择

1. 对比 `ground_truth_v1.csv`；
2. 生成逐视频逐物种误差表；
3. 生成 8 算法计数指标表；
4. 生成独立速度表；
5. 按预先冻结规则排序；
6. 选出当前 baseline tracker；
7. 抽查误差最大的视频和事件；
8. 只做失败解释，不回头改变本轮参数；
9. 若要调参，建立新的实验版本，不覆盖本轮官方参数结果。

---

## 14. 预期结果目录

```text
project_results/
  counting_model_benchmark/
    <run_id>/
      protocol/
        MOT算法计数性能对比实验协议.md
        protocol.sha256
      manifests/
        long_videos.csv
        segments.csv
        ground_truth_v1.csv
        input_validation.json
      detector/
        resolved_detector_config.json
        checkpoint.sha256
        detection_cache_manifest.csv
        detection_runtime.csv
      resolved_configs/
        sort.json
        bytetrack.json
        ocsort.json
        sfsort.json
        fasttracker.json
        boosttrack.json
        hybridsort.json
        botsort.json
        counting.json
      sort/
        tracks/
        counts/
        tracking_run.csv
      bytetrack/
      ocsort/
      sfsort/
      fasttracker/
      boosttrack/
      hybridsort/
      botsort/
      evaluation/
        per_video_species_errors.csv
        counting_metrics.csv
        tracking_speed.csv
        baseline_ranking.csv
      environment/
        environment.json
        source_commits.json
        run_log.json
```

大型检测/轨迹缓存可以不提交 Git，但配置、哈希、清单、日志和最终数字结果必须归档。

---

## 15. 论文/汇报结果表模板

### 15.1 计数主表

| Tracker | dorsalis MAE ↓ | correcta MAE ↓ | Class-MAE ↓ | Total-MAE ↓ | Exact-Video-Rate ↑ | Negative-Video-Rate ↓ |
|---|---:|---:|---:|---:|---:|---:|
| SORT |  |  |  |  |  |  |
| ByteTrack |  |  |  |  |  |  |
| OC-SORT |  |  |  |  |  |  |
| SF-SORT |  |  |  |  |  |  |
| FastTracker |  |  |  |  |  |  |
| BoostTrack detection-only |  |  |  |  |  |  |
| HybridSORT no-ReID |  |  |  |  |  |  |
| BoT-SORT no-ReID/no-GMC |  |  |  |  |  |  |

### 15.2 逐视频逐物种表

| Tracker | Video | Species | GT | Prediction | Signed Error | Absolute Error |
|---|---|---|---:|---:|---:|---:|
|  | LV001 | dorsalis | 132 |  |  |  |
|  | LV001 | correcta | 0 |  |  |  |
|  | … | … | … | … | … | … |

### 15.3 速度表

| Tracker | Frames | Tracking Seconds | Overall Tracking FPS ↑ | Wall Time | Input Cache Hash |
|---|---:|---:|---:|---:|---|
| SORT | 10,007,059 |  |  |  |  |
| ByteTrack | 10,007,059 |  |  |  |  |
| OC-SORT | 10,007,059 |  |  |  |  |
| SF-SORT | 10,007,059 |  |  |  |  |
| FastTracker | 10,007,059 |  |  |  |  |
| BoostTrack | 10,007,059 |  |  |  |  |
| HybridSORT | 10,007,059 |  |  |  |  |
| BoT-SORT | 10,007,059 |  |  |  |  |

---

## 16. 风险、偏差和解释边界

### 16.1 检测误差是共同上限

8 个 tracker 使用同一检测缓存保证了公平，但无法消除检测器错误。某帧完全漏检、错误类别或严重偏框，所有 tracker 都可能受到影响。

### 16.2 官方参数未必适合实蝇

大多数 tracker 参数来自行人或通用 MOT 数据集。使用官方参数可以减少人为偏置、形成可复现起点，但不代表这些参数对 55 FPS、小目标、快速飞行的实蝇最优。

因此建议区分：

1. 本协议的“官方/复现参数 baseline”；
2. 后续相同预算的“目标域参数优化实验”。

两者不能混在同一张 baseline 主表中。

### 16.3 只用最终计数无法解释全部 MOT 行为

不同轨迹错误可能在最终计数上相互抵消。例如同一视频既多计又漏计，最终 MAE 可能仍较小。保存完整轨迹缓存和计数事件可以诊断，但没有 MOT ID 真值时不能精确计算 IDSW 或 Frag。

### 16.4 简单圆计数不是最终方法

当前规则没有边界缓冲、时序确认、轨迹拼接和类别投票，可能受到：

- 圆边界徘徊；
- 轨迹碎片；
- 短时误匹配；
- 类别抖动；
- 预测轨迹跨界；
- 目标在片段开始时首次出现在圆内。

本轮故意不修复这些问题，以保持 tracker 对比的计数器一致。选出 baseline 后再逐项优化。

### 16.5 全量工程选型没有独立测试集

用户已决定本轮直接使用全部 6 个逻辑视频选择 baseline。这对当前工程推进是明确且可执行的，但结果属于 all-data engineering selection。

后续若要形成更强论文结论，建议新增独立逻辑长视频，保持检测器、tracker 参数和计数器不变进行外部验证。

---

## 17. 请导师给出建议的问题清单

### 17.1 研究问题

- 以最终逐物种计数误差而不是标准 MOT 指标选择 tracker，是否符合当前论文问题？
- 是否需要额外制作一个较短子集的逐帧 ID 真值，用于解释 Class-MAE 背后的轨迹错误？

### 17.2 数据与评价

- 当前 6 个逻辑视频、correcta 仅 2 个阳性视频，是否足以做阶段性 baseline？
- 是否应在后续新增视频后做独立外部验证？
- `Class-MAE` 作为主指标，Total-MAE、Exact-Video-Rate 和速度作为辅助，是否合理？

### 17.3 检测控制变量

- 固定 RT-DETR-R18 是否合理？
- RT-DETR 采用原生 NMS-free top-300、缓存 `confidence ≥ 0.10` 是否合理？
- 为支持低分框二阶段关联而保留到 0.10，是否会对没有置信度过滤的 SORT 造成不可接受的不利？

### 17.4 跟踪器公平性

- 采用作者官方默认/论文复现参数，而不针对 6 个视频调参，是否是更清楚的第一轮 baseline？
- OC-SORT 保持 `use_byte=false`、HybridSORT 保持官方 `use_byte=true` 是否合理？
- SF-SORT 使用官方 Notebook 完整在线配置，而不是 timeout=0 的裸构造器默认，是否合理？
- BoostTrack、HybridSORT、BoT-SORT 的无 ReID/无 GMC 版本是否应在结果中作为受控变体明确命名？

### 17.5 计数器

- 固定 `(1185,732), R=215`，用 bbox center 瞬时跨界作为第一轮 baseline 是否足够简单且合理？
- 进入用当前类别、离开用上次类别、不做投票是否适合作为初始对比规则？
- 是否应在选出 tracker 后优先研究边界容差、时序确认、轨迹拼接还是类别稳定？

---

## 18. 正式运行前冻结清单

以下项目全部打勾后才能运行全量实验。

### 18.1 协议

- [ ] 导师已审阅研究问题；
- [ ] 导师已审阅结论边界；
- [ ] 协议版本和 SHA-256 已冻结；
- [ ] 后续修改建立新版本，不覆盖 v2.0。

### 18.2 数据

- [ ] 6 个逻辑视频完整；
- [ ] 25 个物理片段完整；
- [ ] 片段顺序与 manifest 一致；
- [ ] 视频内容 SHA-256 已生成；
- [ ] 总帧数为 10,007,059；
- [ ] 真值 class 0/1 映射已验证；
- [ ] 初始计数固定为 0。

### 18.3 检测

- [ ] RT-DETR-R18 权重 SHA-256 一致；
- [ ] 原始 `best.pth` 视频后端已实现；
- [ ] FP32；
- [ ] 640×640；
- [ ] NMS-free；
- [ ] top-300；
- [ ] cache floor 0.10；
- [ ] 全画面坐标恢复验证通过；
- [ ] 25 个检测缓存完整且 payload hash 已生成。

### 18.4 跟踪

- [ ] 8 个官方源码提交已核对；
- [ ] SF-SORT 参数已修正；
- [ ] 8 份正式配置已复制并哈希；
- [ ] per_class=false；
- [ ] ReID/GMC/ECC/光流/离线处理全部关闭；
- [ ] 两类共同进入一个 tracker；
- [ ] 每物理片段重新初始化；
- [ ] 8/8 统一接口冒烟通过；
- [ ] 确定性重跑通过。

### 18.5 计数与评价

- [ ] 圆心 `(1185,732)`；
- [ ] 半径 `215`；
- [ ] bbox center；
- [ ] enter `+1`；
- [ ] exit `-1`；
- [ ] 首次观测不计数；
- [ ] 负数不截断；
- [ ] 片段间计数累计；
- [ ] 6 视频 × 2 物种完整输出；
- [ ] Class-MAE 和速度汇总测试通过；
- [ ] 输出目录防覆盖生效。

---

## 19. 与其他项目文档的关系

- 研究总体背景与论文主线：`research_docs/研究背景与论文主线.md`；
- 当前工程进展：`research_docs/当前进展与下一步.md`；
- 检测、跟踪、计数三级缓存格式：`research_docs/检测-跟踪-计数数值缓存与重放协议.md`；
- 本文件是当前 8 tracker 长视频计数 baseline 的权威实验协议。

若旧文档与本协议在以下两点冲突，以本协议 v2.0 和用户 2026-07-25 最新决定为准：

1. 所有逻辑长视频初始瓶内数量固定为 0，不再作为待核验项；
2. 本轮使用全部 6 个逻辑长视频寻找工程 baseline，不划分 development/validation/test。

代码实现、配置文件和正式运行清单必须在全量实验前同步到本协议，不得把“计划实现”写成“已经完成”，也不得把本轮工程选型结果写成独立测试集泛化结论。

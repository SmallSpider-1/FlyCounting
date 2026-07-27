# 检测模型基准工作区

本工作区创建于 2026-07-17，用于在受控场景 C、真实复杂背景 R1 和真实黑色背景 R2 上开展可复现的两类实蝇检测对比。项目首先准备了官方源码、运行环境、官方预训练权重和冒烟测试；2026-07-18 将用户提供并规范化的 C/R1/R2 划分固化为 `fruitfly_detection_v1`，并接入全部 11 个模型目录。统一 e100 基准现已完成；当前研究状态和结论以 `research_docs/当前进展与下一步.md` 为准。

## 共享长视频检测缓存

`cache_detections.py` 是独立于 tracker 的全画面检测缓存入口，用于后续计数链路。它递归保留逻辑视频目录结构，并写出 `benchmark_common/numeric_cache.py` 定义的通用 `Nx6 [xyxy, confidence, class_id]` JSONL 格式；该入口不执行跟踪、计数或 ROI 优化。

冻结的 RT-DETR-R18 使用独立的官方源码后端 `rtdetr_r18_backend.py` 和 manifest 驱动入口 `cache_rtdetr_detections.py`。生成或替换正式检测缓存前，必须先阅读 [`DETECTION_CACHE_HANDOFF.md`](DETECTION_CACHE_HANDOFF.md)。该文档记录了可复用的后端契约、验证闸门、RT-DETR 精确命令、首轮运行证据以及接入其他检测器时的检查清单。

## 状态汇总

| 序号 | 模型 | 项目目录 | Conda 环境 | 环境来源 | 共用环境的其他项目 | 官方仓库 | Commit SHA | Python | PyTorch | 权重路径 | 数据格式 | CUDA | 单图推理 | 最终状态 |
|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | YOLOv5n | `01_yolov5n` | `bench_detection_torch201` | 新建 | 02, 04, 05, 09, 10 | `ultralytics/yolov5` | `09e765b424d13d01e038b6d8c6f7e15f5384a444` | 3.9.23 | 2.0.1+cu118 | `01_yolov5n/weights/yolov5n.pt` | YOLO | 是（11.8） | 通过 | 就绪 |
| 2 | YOLOv7-tiny | `02_yolov7_tiny` | `bench_detection_torch201` | 新建 | 01, 04, 05, 09, 10 | `WongKinYiu/yolov7` | `a207844b1ce82d204ab36d87d496728d3d2348e7` | 3.9.23 | 2.0.1+cu118 | `02_yolov7_tiny/weights/yolov7-tiny.pt` | YOLO | 是（11.8） | 通过 | 就绪 |
| 3 | YOLOv8n | `03_yolov8n` | `yolov26` | 现有 | 06, 07, 08 | `ultralytics/ultralytics` | `b2a993107584090b3e3c918bb1cefef53c8769d8` | 3.10.20 | 2.6.0+cu124 | `03_yolov8n/weights/yolov8n.pt` | YOLO | 是（12.4） | 通过 | 就绪 |
| 4 | YOLOv9-T | `04_yolov9_t` | `bench_detection_torch201` | 新建 | 01, 02, 05, 09, 10 | `WongKinYiu/yolov9` | `5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff` | 3.9.23 | 2.0.1+cu118 | `04_yolov9_t/weights/yolov9-t-converted.pt` | YOLO | 是（11.8） | 通过 | 就绪 |
| 5 | YOLOv10n | `05_yolov10n` | `bench_detection_torch201` | 新建 | 01, 02, 04, 09, 10 | `THU-MIG/yolov10` | `453c6e38a51e9d1d5a2aa5fb7f1014a711913397` | 3.9.23 | 2.0.1+cu118 | `05_yolov10n/weights/yolov10n.pt` | YOLO | 是（11.8） | 通过 | 就绪 |
| 6 | YOLO11n | `06_yolo11n` | `yolov26` | 现有 | 03, 07, 08 | `ultralytics/ultralytics` | `b2a993107584090b3e3c918bb1cefef53c8769d8` | 3.10.20 | 2.6.0+cu124 | `06_yolo11n/weights/yolo11n.pt` | YOLO | 是（12.4） | 通过 | 就绪 |
| 7 | YOLO12n | `07_yolo12n` | `yolov26` | 现有 | 03, 06, 08 | `sunsmarterjie/yolov12` | `01a22c0603e0eaa6d9bd62120a391e744d92cea2` | 3.10.20 | 2.6.0+cu124 | `07_yolo12n/weights/yolov12n.pt` | YOLO | 是（12.4） | 通过（SDPA 回退） | 就绪 |
| 8 | YOLO26n | `08_yolo26n` | `yolov26` | 现有 | 03, 06, 07 | `ultralytics/ultralytics` | `b2a993107584090b3e3c918bb1cefef53c8769d8` | 3.10.20 | 2.6.0+cu124 | `08_yolo26n/weights/yolo26n.pt` | YOLO | 是（12.4） | 通过 | 就绪 |
| 9 | DEIM-D-FINE-N | `09_deim_dfine_n` | `bench_detection_torch201` | 新建 | 01, 02, 04, 05, 10 | `Intellindust-AI-Lab/DEIM` | `09d35d53d39ee3145a1e61e3a989b28b9468d1dd` | 3.9.23 | 2.0.1+cu118 | `09_deim_dfine_n/weights/deim_dfine_hgnetv2_n_coco_160e.pth` | COCO | 是（11.8） | 通过 | 就绪 |
| 10 | RT-DETR-R18 | `10_rt_detr_r18` | `bench_detection_torch201` | 新建 | 01, 02, 04, 05, 09 | `lyuwenyu/RT-DETR` | `1c8ac3f7ba84f14bd5651ab7b1b70d69a5f55f47` | 3.9.23 | 2.0.1+cu118 | `10_rt_detr_r18/weights/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth` | COCO | 是（11.8） | 通过 | 就绪 |
| 11 | Faster R-CNN R50-FPN | `11_faster_rcnn_r50_fpn` | `pointnext` | 现有 | 无 | `pytorch/vision` | `f23f832d090c868691855cc1261ed907e400c2a2` | 3.10.20 | 2.6.0+cu124 | `11_faster_rcnn_r50_fpn/weights/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth` | COCO | 是（12.4） | 通过 | 就绪 |

YOLOv8n、YOLO11n 和 YOLO26n 使用的三个 Ultralytics 项目副本均固定在同一个 commit。完整权重 URL 和 SHA-256 记录在各项目的 `PROJECT_INFO.md` 以及 `_logs/weight_download_summary.csv` 中。

## 环境决策

初次审计发现了 `MOTIP`、`mammal-paper`、`pointnext`、`proxyformer`、`xanylabeling`、`yolov11`、`yolov26`、外部 `sugar` 环境和 `base`。本项目没有修改 `base`。现有 `yolov26` 环境直接通过了四个模型的冒烟测试；现有 `pointnext` 环境直接通过了 Faster R-CNN 推理及参考训练入口导入测试。

本项目只新建了一个环境：`bench_detection_torch201`。它服务于六个兼容项目，这些项目需要 PyTorch 2.0.1/CUDA 11.8、torchvision 0.15.2、旧版 YOLO 的 NumPy 约束以及 Transformer 检测器依赖。该环境从 `sugar` 中匹配的 torch/torchvision 基线只读克隆而来，没有修改原始 `sugar` 环境。配置期间，通过 `pip uninstall` 清理了克隆环境继承的重复 NumPy 元数据，再按一致版本重新安装；`pip check` 已通过。该环境还配置了优先使用自身 C++ 运行库的激活钩子。

环境导出文件位于 `_env_exports/`；权威环境映射为 `_common/environment_map.yaml`。

## 各项目环境激活命令

```bash
# 01 YOLOv5n
conda activate bench_detection_torch201
# 02 YOLOv7-tiny
conda activate bench_detection_torch201
# 03 YOLOv8n
conda activate yolov26
# 04 YOLOv9-T
conda activate bench_detection_torch201
# 05 YOLOv10n
conda activate bench_detection_torch201
# 06 YOLO11n
conda activate yolov26
# 07 YOLO12n
conda activate yolov26
# 08 YOLO26n
conda activate yolov26
# 09 DEIM-D-FINE-N
conda activate bench_detection_torch201
# 10 RT-DETR-R18
conda activate bench_detection_torch201
# 11 Faster R-CNN ResNet50-FPN
conda activate pointnext
```

具体运行方式应以各项目的 `RUNBOOK.md` 为准。下列命令只负责进入对应环境；训练完成状态和正式结果以研究状态文档及 `results/` 中的验证报告为准。

## 数据集状态

生成的数据集位于 `/home/admin1/Projects/ultralytics-main/datasets/fruitfly_detection_v1`：

- 11,190 张 RGB JPEG 图片、19,540 个标注框和 3,060 张合法空白/背景图片；
- 训练集/验证集/测试集：8,143/1,530/1,517 张；
- 统一类别 ID：`0=Bactrocera dorsalis`、`1=Bactrocera correcta`；
- YOLO 数据根目录：`.../fruitfly_detection_v1/yolo`；
- DEIM/RT-DETR 的 COCO category ID：0/1；
- torchvision Faster R-CNN 的前景 category ID：1/2，其中 0 保留为背景；
- 训练时数据集指纹：`a56fa4a6e10370e963d34e983711c3ed69b5b2f064a0d7771d976d6f1c57b5da`；
- 经用户确认并于 2026-07-22 修正 R2/test 类别 ID 后的评价数据集指纹：`f84b7383bf9028828b36747fd9505cc3afd8807c6b770d99b55d468e0a75cb47`；
- 上述两个指纹对应的训练集和验证集内容逐字节一致；
- 全量图片解码、图片/标签配对、精确重复检查、YOLO 到 COCO 几何转换、划分清单和官方加载器检查均已通过。

实际数据根路径记录在 `_common/dataset_paths.env` 中；每个模型目录均已配置 `dataset_config.yaml`。Faster R-CNN 适配器默认保留空白/背景训练图，使 11 个模型都使用相同的 8,143 张训练图片。

统一 e100 首轮基准的输入尺寸、有效 batch、epoch、workers、seed、增强策略、预训练初始化、checkpoint 选择、设备分配和结果命名均已按当轮协议执行并完成审计；第二篇论文的最终检测器仍未冻结。若后续开展新一轮多 seed、部署或其他检测器比较，应建立新的冻结运行单，不得仅从本 README 反推参数。生成的 `samples.csv` 记录了当前源划分由用户提供；现有 `data/` 目录没有逐图片的源视频/分组清单，因此这部分来源信息仍独立于格式转换验证。

## 只读验证

可随时运行以下只读检查：

```bash
cd /home/admin1/Projects/ultralytics-main/detection_model_benchmark
./verify_all.sh
```

该脚本检查 11 个模型映射、官方 Git origin 与 commit、权重 SHA-256、运行环境、PyTorch/CUDA 以及已通过的冒烟测试记录。它不会安装、升级、删除或修复任何内容。

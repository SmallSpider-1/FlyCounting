# Detection model benchmark workspace

Created on 2026-07-17 for reproducible two-class fruit-fly detection comparisons across controlled scene C, real complex-background R1, and real black-background R2. Source, environments, official pretrained weights, and smoke tests were prepared first; on 2026-07-18 the user-provided normalized C/R1/R2 splits were materialized as `fruitfly_detection_v1` and wired into all 11 project directories. The unified e100 benchmark has since completed; current research status and conclusions are maintained in `research_docs/当前进展与下一步.md`.

## Shared long-video detection cache

`cache_detections.py` is the tracker-independent full-frame cache entry for the
counting pipeline. It recursively preserves logical-video directories and writes
the common `Nx6 [xyxy, confidence, class_id]` JSONL format defined in
`benchmark_common/numeric_cache.py`. It does not run tracking, counting, or ROI
optimization.

The frozen RT-DETR-R18 uses the separate official-source backend
`rtdetr_r18_backend.py` and the manifest-driven entry
`cache_rtdetr_detections.py`. Before generating or replacing a formal detector
cache, read [`DETECTION_CACHE_HANDOFF.md`](DETECTION_CACHE_HANDOFF.md). It records
the reusable backend contract, validation gates, exact RT-DETR commands, first-run
evidence, and the checklist for adding another detector.

## Status summary

| # | Model | Project directory | Conda env | Origin | Other projects sharing env | Official repository | Commit SHA | Python | PyTorch | Weight path | Data | CUDA | Single-image inference | Final |
|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | YOLOv5n | `01_yolov5n` | `bench_detection_torch201` | new | 02, 04, 05, 09, 10 | `ultralytics/yolov5` | `09e765b424d13d01e038b6d8c6f7e15f5384a444` | 3.9.23 | 2.0.1+cu118 | `01_yolov5n/weights/yolov5n.pt` | YOLO | yes (11.8) | PASS | READY |
| 2 | YOLOv7-tiny | `02_yolov7_tiny` | `bench_detection_torch201` | new | 01, 04, 05, 09, 10 | `WongKinYiu/yolov7` | `a207844b1ce82d204ab36d87d496728d3d2348e7` | 3.9.23 | 2.0.1+cu118 | `02_yolov7_tiny/weights/yolov7-tiny.pt` | YOLO | yes (11.8) | PASS | READY |
| 3 | YOLOv8n | `03_yolov8n` | `yolov26` | existing | 06, 07, 08 | `ultralytics/ultralytics` | `b2a993107584090b3e3c918bb1cefef53c8769d8` | 3.10.20 | 2.6.0+cu124 | `03_yolov8n/weights/yolov8n.pt` | YOLO | yes (12.4) | PASS | READY |
| 4 | YOLOv9-T | `04_yolov9_t` | `bench_detection_torch201` | new | 01, 02, 05, 09, 10 | `WongKinYiu/yolov9` | `5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff` | 3.9.23 | 2.0.1+cu118 | `04_yolov9_t/weights/yolov9-t-converted.pt` | YOLO | yes (11.8) | PASS | READY |
| 5 | YOLOv10n | `05_yolov10n` | `bench_detection_torch201` | new | 01, 02, 04, 09, 10 | `THU-MIG/yolov10` | `453c6e38a51e9d1d5a2aa5fb7f1014a711913397` | 3.9.23 | 2.0.1+cu118 | `05_yolov10n/weights/yolov10n.pt` | YOLO | yes (11.8) | PASS | READY |
| 6 | YOLO11n | `06_yolo11n` | `yolov26` | existing | 03, 07, 08 | `ultralytics/ultralytics` | `b2a993107584090b3e3c918bb1cefef53c8769d8` | 3.10.20 | 2.6.0+cu124 | `06_yolo11n/weights/yolo11n.pt` | YOLO | yes (12.4) | PASS | READY |
| 7 | YOLO12n | `07_yolo12n` | `yolov26` | existing | 03, 06, 08 | `sunsmarterjie/yolov12` | `01a22c0603e0eaa6d9bd62120a391e744d92cea2` | 3.10.20 | 2.6.0+cu124 | `07_yolo12n/weights/yolov12n.pt` | YOLO | yes (12.4) | PASS (SDPA fallback) | READY |
| 8 | YOLO26n | `08_yolo26n` | `yolov26` | existing | 03, 06, 07 | `ultralytics/ultralytics` | `b2a993107584090b3e3c918bb1cefef53c8769d8` | 3.10.20 | 2.6.0+cu124 | `08_yolo26n/weights/yolo26n.pt` | YOLO | yes (12.4) | PASS | READY |
| 9 | DEIM-D-FINE-N | `09_deim_dfine_n` | `bench_detection_torch201` | new | 01, 02, 04, 05, 10 | `Intellindust-AI-Lab/DEIM` | `09d35d53d39ee3145a1e61e3a989b28b9468d1dd` | 3.9.23 | 2.0.1+cu118 | `09_deim_dfine_n/weights/deim_dfine_hgnetv2_n_coco_160e.pth` | COCO | yes (11.8) | PASS | READY |
| 10 | RT-DETR-R18 | `10_rt_detr_r18` | `bench_detection_torch201` | new | 01, 02, 04, 05, 09 | `lyuwenyu/RT-DETR` | `1c8ac3f7ba84f14bd5651ab7b1b70d69a5f55f47` | 3.9.23 | 2.0.1+cu118 | `10_rt_detr_r18/weights/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth` | COCO | yes (11.8) | PASS | READY |
| 11 | Faster R-CNN R50-FPN | `11_faster_rcnn_r50_fpn` | `pointnext` | existing | none | `pytorch/vision` | `f23f832d090c868691855cc1261ed907e400c2a2` | 3.10.20 | 2.6.0+cu124 | `11_faster_rcnn_r50_fpn/weights/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth` | COCO | yes (12.4) | PASS | READY |

All three Ultralytics project copies for YOLOv8n/YOLO11n/YOLO26n are pinned to the same commit. Full weight URLs and SHA256 values are in each `PROJECT_INFO.md` and `_logs/weight_download_summary.csv`.

## Environment decisions

The initial audit found `MOTIP`, `mammal-paper`, `pointnext`, `proxyformer`, `xanylabeling`, `yolov11`, `yolov26`, the external `sugar` environment, and `base`. `base` was not modified. Existing `yolov26` directly passed four model smoke tests; existing `pointnext` directly passed Faster R-CNN plus the reference training-entry import.

Only one environment was created: `bench_detection_torch201`. It serves six compatible projects that need PyTorch 2.0.1/CUDA 11.8, torchvision 0.15.2, the legacy YOLO NumPy constraint, and transformer-detector dependencies. It was cloned read-only from the matching torch/torchvision baseline in `sugar`; the original `sugar` environment was not modified. During setup, duplicate NumPy metadata inherited by the clone was cleaned through `pip uninstall` and reinstalled consistently; `pip check` passes. The environment has an activation hook that prefers its own C++ runtime.

Environment exports are under `_env_exports/`. The authoritative mapping is `_common/environment_map.yaml`.

## Activation commands by project

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

Use each project's `RUNBOOK.md`; commands are templates and were not executed as training jobs.

## Dataset status

The generated dataset is `/home/admin1/Projects/ultralytics-main/datasets/fruitfly_detection_v1`:

- 11,190 RGB JPEG images, 19,540 boxes, and 3,060 legitimate empty/background images;
- train/validation/test: 8,143/1,530/1,517 images;
- canonical IDs: `0=Bactrocera dorsalis`, `1=Bactrocera correcta`;
- YOLO root: `.../fruitfly_detection_v1/yolo`;
- DEIM/RT-DETR COCO category IDs: 0/1;
- torchvision Faster R-CNN foreground category IDs: 1/2, with 0 reserved for background;
- training-time dataset fingerprint: `a56fa4a6e10370e963d34e983711c3ed69b5b2f064a0d7771d976d6f1c57b5da`;
- corrected evaluation dataset fingerprint after the user-confirmed 2026-07-22
  R2/test class-ID repair: `f84b7383bf9028828b36747fd9505cc3afd8807c6b770d99b55d468e0a75cb47`;
- train/validation content is byte-identical between those two fingerprints;
- full image decode, image/label pairing, exact duplicate, YOLO-to-COCO geometry, split-list, and official-loader checks all pass.

Actual roots are in `_common/dataset_paths.env`; every model directory now has `dataset_config.yaml`. The Faster R-CNN adapter keeps empty/background training images by default so all 11 models see the same 8,143 training images.

Still to freeze before formal model comparison: image size, effective batch size, epochs, workers, seed, augmentation policy, pretrained initialization policy, checkpoint selection rule, device allocation, and result naming. The generated `samples.csv` records that current source splits were supplied by the user; the current `data/` tree did not include a per-image source-video/group manifest, so that provenance remains separate from the format-conversion validation.

## Verification

Run the read-only checker at any time:

```bash
cd /home/admin1/Projects/ultralytics-main/detection_model_benchmark
./verify_all.sh
```

It checks the 11 mappings, official Git origins and commits, weight SHA256 values, environments, PyTorch/CUDA, and successful smoke-test records. It never installs, upgrades, deletes, or repairs anything.

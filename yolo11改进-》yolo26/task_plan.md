# Task Plan: YOLO26 Migrated Model Training Sweep

## Goal
Run every YOLO26 model config migrated from `/home/admin1/Projects/ultralytics-yolo11-main/ultralytics/cfg/models/11` through `train.py` in the `yolov26` conda environment, with per-run console logs saved.

## Current Phase
Phase 4

## Phases

### Phase 1: Requirements & Discovery
- [x] Confirm user intent: train all migrated improved YOLO26 model configs using `train.py`.
- [x] Identify conda environment: requested `yolo26`; available matching environment is `yolov26`.
- [x] Identify dataset and logging behavior in `train.py`.
- **Status:** complete

### Phase 2: Model Set Definition
- [x] Compare source `yolo11-*.yaml` configs to target `yolo26-*.yaml` configs by filename stem after `yolo11-` to `yolo26-` rename.
- [x] Exclude non-migrated configs such as base `yolo26.yaml` and `yoloe-26*.yaml`.
- [x] Record model count and comparison result.
- **Status:** complete

### Phase 3: Batch Runner Preparation
- [x] Update `train.py` so it can run one model, multiple models, or the migrated sweep from CLI while preserving existing defaults.
- [x] Add robust progress/skip behavior so interrupted sweeps can resume without rerunning completed models unless requested.
- [x] Verify CLI help and migrated-model discovery.
- **Status:** complete

### Phase 4: Execute Training Sweep
- [ ] Launch the migrated sweep under conda env `yolov26`.
- [ ] Ensure console output is logged to `runs/detect/<model>_two_class/train_console_<timestamp>.txt`.
- [ ] Track completed, failed, and skipped models.
- **Status:** in_progress

### Phase 5: Delivery
- [ ] Summarize run command, active process status, output locations, and any failures.
- [ ] Update planning files with final progress.
- **Status:** pending

## Key Questions
1. Which files count as migrated improved models? `yolo26-*.yaml` files whose names map exactly to source `yolo11-*.yaml` files.
2. Which conda env should be used? `yolov26`, because `yolo26` is not present and `yolov26` is the matching local environment.
3. How should logs be recorded? Keep `train.py` tee logging and write one console log per model run under its project directory.

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Use `yolov26` conda env | `conda env list` shows `yolov26` but no `yolo26`; this is the closest and clearly related environment. |
| Define migrated set by exact source/target filename mapping | Avoids running unrelated base, segmentation, pose, YOLOE, or extra YOLO26 configs that were not part of the YOLO11 migration request. |
| Extend `train.py` rather than create a separate trainer | User specifically asked to run using `train.py`; CLI support keeps current single-model workflow intact and adds batch capability. |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| `printf: --: invalid option` while printing a separator in a shell comparison command | 1 | Reran comparison using `printf '%s\n' '--- missing in 26 ---'`. |
| Background `nohup conda run -n yolov26 ...` exited immediately with empty output | 1 | Verified the env's Python directly and relaunched with `/home/admin1/.conda/envs/yolov26/bin/python` under `setsid`. |
| First full sweep would not stop at model 204 and would overlap with GPU1 second-half process | 1 | Stopped the full sweep and replaced it with explicit non-overlapping shards. |
| Ultralytics LOGGER retained a closed per-model tee stream between models | 1 | Updated `train.py` to restore logger streams and make `Tee` tolerate closed log files; restarted final shards with timestamp `20260512_234646`. |

## Notes
- Target migrated config count: 408.
- Current `train.py` training parameters: `data=dataset_two_class.yaml`, `imgsz=640`, `epochs=200`, `batch=4`, `workers=8`, `device=0`, `optimizer=SGD`.
- Final active batch: 8 explicit shards, 51 models each, timestamp `20260512_234646`.
- GPU0 shards: PIDs 8941, 8943, 8945, 8947.
- GPU1 shards: PIDs 8949, 8951, 8953, 8955.
- Final shard logs: `runs/detect/sweep_migrated_yolo26_gpu{0,1}_shard{1..8}_20260512_234646.out`.
- Old non-current artifacts were moved to `runs/detect/_old_noncurrent_20260513_114959`.

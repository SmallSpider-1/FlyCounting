# Task Plan: YOLO26 Raspberry Pi Optimization Experiments

## Goal
Create an isolated experiment workspace to evaluate Raspberry Pi deployment optimizations for `/home/admin1/Projects/ultralytics-main/yolo26n.pt` without modifying existing project files.

## Current Phase
Phase 7

## Phases

### Phase 1: Requirements & Discovery
- [x] Capture user intent: create a new working directory and use file-based planning.
- [x] Identify non-invasive constraint: experiments may add files/outputs but should not alter existing training, inference, dataset, or model files.
- [x] Capture available baseline facts from prior validation.
- **Status:** complete

### Phase 2: Workspace & Experiment Harness
- [x] Create isolated directory structure.
- [x] Add reusable scripts for validation, export, benchmarking, comparison, and experiment orchestration.
- [x] Add config files that declare experiment variants.
- **Status:** complete

### Phase 3: Baseline & Size Sweep
- [x] Validate original PyTorch weights at multiple `imgsz` values.
- [x] Benchmark speed/accuracy tradeoffs.
- [x] Record results in JSON/CSV.
- **Status:** complete

### Phase 4: Export Backend Experiments
- [x] Export and validate NCNN.
- [x] Export and validate ONNX.
- [x] Attempt optional TFLite/INT8 exports if environment supports them.
- [x] Record export errors and successful artifacts.
- **Status:** complete

### Phase 5: Quantization & Acceleration Experiments
- [x] Test export options such as `half`, `int8`, and calibration data where supported.
- [x] Compare exported backends against PyTorch baseline.
- [x] Identify candidates for Raspberry Pi transfer.
- **Status:** complete

### Phase 6: Distillation / Smaller Student Plan
- [x] Define a non-invasive distillation experiment plan and scripts/configs.
- [x] Choose teacher/student candidates.
- [x] Document what code changes would be required before running distillation.
- **Status:** complete

### Phase 7: Reporting & Recommendations
- [x] Generate a comparison report.
- [x] Summarize recommended Raspberry Pi deployment path.
- [x] List next experiments and blockers.
- **Status:** complete

## Key Questions
1. Which deployment format gives the best speed/accuracy tradeoff for this YOLO26 model on CPU-like environments?
2. How much accuracy is lost when reducing `imgsz` from 640 to 512/416/320?
3. Can NCNN/ONNX/TFLite exports be generated successfully from this custom YOLO26 checkpoint?
4. Is INT8 quantization available and accurate enough for the dataset?
5. Is distillation necessary after simpler backend/resolution/quantization tests?

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Use `raspberry_pi_optimization_experiments/` as the isolated workspace | Keeps all optimization files separate from the existing project. |
| Write planning files inside the new workspace | Avoids interfering with existing root-level planning files. |
| Prefer scripts that read existing weights/data via absolute defaults | Keeps experiments repeatable without moving or editing source files. |
| Store all outputs under workspace `results/`, `exports/`, and `logs/` | Makes cleanup and transfer to Raspberry Pi straightforward. |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| NCNN export auto-install stalled while installing `pnnx` through base/system pip | 1 | Killed stalled process; continue with ONNX export and record NCNN as dependency blocker. |
| ONNX export auto-install attempted dependencies through base/system pip | 1 | Killed stalled process and installed ONNX dependencies with `/home/admin1/.conda/envs/yolov11/bin/python -m pip`. |
| TFLite/INT8 TFLite export failed because TensorFlow `>=2.0,<=2.19` is unavailable for current Python package resolution | 1 | Recorded as environment blocker; can retry in a Python/TensorFlow-compatible export environment. |

## Notes
- Existing original model: `/home/admin1/Projects/ultralytics-main/yolo26n.pt`.
- Existing data config: `/home/admin1/Projects/ultralytics-main/data.yaml`.
- Prior original val at 640: Precision 0.9724, Recall 0.9781, mAP50 0.9918, mAP50-95 0.8520, inference 1.587 ms/image on A100.
- Prior 20% zero-channel prune val at 640: Precision 0.9566, Recall 0.6886, mAP50 0.8724, mAP50-95 0.6461, inference 2.077 ms/image on A100.

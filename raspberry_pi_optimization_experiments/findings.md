# Findings & Decisions

## Requirements
- Create a new working directory under `/home/admin1/Projects/ultralytics-main`.
- Use `planning-with-files` inside that directory to plan and track the optimization workflow.
- Do optimization attempts without changing existing project files.
- Try all relevant approaches mentioned previously where feasible: backend export, resolution sweep, quantization, hardware/deployment planning, engineering optimizations, and distillation planning.

## Research Findings
- Ultralytics recommends NCNN for Raspberry Pi because it is optimized for mobile/embedded ARM platforms.
- Ultralytics export supports formats including NCNN, ONNX, TFLite, OpenVINO, MNN, RKNN, ExecuTorch, and accelerator-specific formats.
- Quantization is one of the most important Raspberry Pi/edge optimization routes. INT8 can reduce memory and improve inference speed when the backend supports it.
- Raspberry Pi 5 can use AI HAT / Hailo accelerators for much higher inference throughput, but that requires a different deployment target than pure CPU.
- Existing zero-channel structured pruning does not physically remove channels, so it should not be expected to speed up normal dense inference.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Start with baseline and resolution sweep | Cheap, non-invasive, immediately reveals speed/accuracy tradeoff. |
| Try NCNN before other backends | Most relevant for Raspberry Pi CPU deployment. |
| Keep export artifacts inside workspace | Avoids cluttering project root and makes Raspberry Pi transfer easier. |
| Treat distillation as a separate phase | It is likely useful, but requires a training workflow and teacher/student choice. |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| Ultralytics auto-install used base/system pip for missing export dependencies | Manually installed dependencies with `/home/admin1/.conda/envs/yolov11/bin/python -m pip`. |
| NCNN directory must end with `_ncnn_model` for Ultralytics AutoBackend recognition | Renamed exported NCNN directory and updated export script naming. |
| TFLite export requires TensorFlow `>=2.0,<=2.19`, unavailable in the current package resolution | Marked TFLite/INT8 TFLite as blocked in this environment. |

## Experiment Findings
- Resolution sweep on `yolo26n.pt`: 512 keeps mAP50 high at 0.9912 and mAP50-95 at 0.8326, a modest drop from 640 mAP50-95 0.8520.
- Resolution 416 drops mAP50-95 to 0.7889 but remains usable if speed pressure is high.
- Resolution 320 drops mAP50-95 to 0.6890 and Recall to 0.918, likely too aggressive for this dataset if accuracy matters.
- NCNN export works and preserves accuracy on validation: mAP50 0.9924, mAP50-95 0.8518.
- Static ONNX export works and preserves accuracy: mAP50 0.9919, mAP50-95 0.8527.
- Dynamic ONNX export works and preserves accuracy: mAP50 0.9918, mAP50-95 0.8514.
- On this server CPU, dynamic ONNX validation reported ~19.2 ms/image, static ONNX ~69.2 ms/image, NCNN ~181.6 ms/image. These are not Raspberry Pi timings, but they confirm exported artifact functionality and relative local CPU behavior.

## Resources
- Project root: `/home/admin1/Projects/ultralytics-main`
- Optimization workspace: `/home/admin1/Projects/ultralytics-main/raspberry_pi_optimization_experiments`
- Original weights: `/home/admin1/Projects/ultralytics-main/yolo26n.pt`
- Dataset config: `/home/admin1/Projects/ultralytics-main/data.yaml`
- Existing pruning workspace: `/home/admin1/Projects/ultralytics-main/yolo26_pruning`

## Visual/Browser Findings
- Official Ultralytics Raspberry Pi docs say NCNN provides best inference performance among export formats for Raspberry Pi because it is optimized for ARM/mobile/embedded platforms.
- Official Ultralytics export docs list `half` and `int8` as export options, with INT8 highlighted for edge AI deployments.
- Raspberry Pi AI HAT documentation lists Hailo NPU variants at 13/26 TOPS for AI HAT+ and 40 TOPS for AI HAT+ 2.

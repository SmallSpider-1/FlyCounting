# Distillation Experiment Plan

## Goal
Train a fast Raspberry Pi student model that keeps more of the teacher model's accuracy than simple resizing, quantization, or zero-channel pruning.

## Proposed Teacher
- Primary teacher: `/home/admin1/Projects/ultralytics-main/yolo26n.pt`
- Optional stronger teacher: train or use a larger YOLO26 variant such as `yolo26s` or `yolo26m` on the same dataset, then distill into a smaller student.

## Proposed Students
- Same architecture at lower deployment `imgsz` after finding best resolution from the sweep.
- A custom narrower YOLO26 config if physical speed/size must be reduced.
- The current YOLO26n exported to NCNN/INT8 if backend optimization is enough and distillation is only needed to recover lower-resolution accuracy.

## Loss Ideas
- Standard detection loss from Ultralytics.
- Feature distillation: match intermediate feature maps from teacher to student.
- Logit/objectness distillation: match teacher predictions before NMS.
- Box distillation: encourage student boxes to match teacher high-confidence boxes.

## Non-Invasive Implementation Strategy
- Put distillation code under this workspace or a separate `distillation/` subdirectory.
- Import Ultralytics from the project root.
- Save all runs under `raspberry_pi_optimization_experiments/results/distillation/`.
- Do not edit `train.py`, `data.yaml`, `main.py`, or model YAML files unless a later explicit implementation phase requires a copied student YAML.

## When To Run
Run distillation after:
1. Resolution sweep identifies the lowest acceptable `imgsz`.
2. NCNN export works.
3. INT8 or other quantized export is tested.

Distillation is most useful if speed is acceptable only at a lower resolution or smaller model, but Recall/mAP drops too much.

# Official-Code Optimization Experiment Plan

## Goal

Use official Ultralytics commands/code paths to find the best Raspberry Pi deployment route for `yolo26n.pt`, with minimal custom code and no changes to existing project files.

## Principle

Do not judge optimization on the server alone. Server GPU/CPU results are only sanity checks. The final decision must be made on the Raspberry Pi target device using the same model, same video/image source, and same preprocessing settings.

## Stage 0: Establish A Fair Baseline

### Experiment 0.1: PyTorch `.pt` Validation

Purpose: confirm the original model's accuracy.

```bash
yolo detect val model=/home/admin1/Projects/ultralytics-main/yolo26n.pt data=/home/admin1/Projects/ultralytics-main/data.yaml imgsz=640 batch=4 device=0
```

Record:

- Precision
- Recall
- mAP50
- mAP50-95
- inference ms/img

### Experiment 0.2: PyTorch Prediction Speed On Real Inputs

Purpose: measure real workflow speed, not only validation speed.

```bash
yolo detect predict model=/home/admin1/Projects/ultralytics-main/yolo26n.pt source=/path/to/test/images_or_video imgsz=640 device=cpu
```

Run this on Raspberry Pi as well.

## Stage 1: Input Size Sweep

Purpose: find the fastest acceptable input size before changing model format.

Run:

```bash
for s in 640 512 416 384 320; do
  yolo detect val model=/home/admin1/Projects/ultralytics-main/yolo26n.pt data=/home/admin1/Projects/ultralytics-main/data.yaml imgsz=$s batch=4 device=0 name=val_imgsz_$s
done
```

Decision rule:

- If `512` loses less than about 0.02 mAP50-95 and Recall remains high, prefer `512` for Raspberry Pi.
- Avoid `320` unless speed is far more important than accuracy.

## Stage 2: Official Export Backends

### Experiment 2.1: NCNN Export

Purpose: primary Raspberry Pi CPU candidate.

```bash
yolo export model=/home/admin1/Projects/ultralytics-main/yolo26n.pt format=ncnn imgsz=640
```

Validate:

```bash
yolo detect val model=/home/admin1/Projects/ultralytics-main/yolo26n_ncnn_model data=/home/admin1/Projects/ultralytics-main/data.yaml imgsz=640 device=cpu
```

Repeat at `imgsz=512` during predict/val on Raspberry Pi.

### Experiment 2.2: ONNX Export

Purpose: fallback and comparison backend.

```bash
yolo export model=/home/admin1/Projects/ultralytics-main/yolo26n.pt format=onnx imgsz=640 simplify=True
yolo export model=/home/admin1/Projects/ultralytics-main/yolo26n.pt format=onnx imgsz=640 simplify=True dynamic=True
```

Validate:

```bash
yolo detect val model=/home/admin1/Projects/ultralytics-main/yolo26n.onnx data=/home/admin1/Projects/ultralytics-main/data.yaml imgsz=640 device=cpu
```

Decision rule:

- On Raspberry Pi, compare NCNN vs ONNX using the same source and `imgsz`.
- Keep the faster backend if mAP/Recall is effectively unchanged.

## Stage 3: Quantization

### Experiment 3.1: NCNN FP16

Purpose: test whether smaller/FP16 NCNN helps target device.

```bash
yolo export model=/home/admin1/Projects/ultralytics-main/yolo26n.pt format=ncnn imgsz=640 half=True
```

Note: FP16 benefits depend on backend/hardware support. It may not help CPU-only Raspberry Pi.

### Experiment 3.2: TFLite / INT8 TFLite In A Separate Export Environment

Purpose: test mobile/ARM quantized model.

Current environment failed TensorFlow dependency resolution. Use a clean Python environment compatible with TensorFlow required by Ultralytics.

Commands to try in that environment:

```bash
yolo export model=/home/admin1/Projects/ultralytics-main/yolo26n.pt format=tflite imgsz=640
yolo export model=/home/admin1/Projects/ultralytics-main/yolo26n.pt format=tflite imgsz=640 int8=True data=/home/admin1/Projects/ultralytics-main/data.yaml
```

Decision rule:

- INT8 is only useful if Recall/mAP drop is acceptable and Raspberry Pi inference speed improves.

## Stage 4: Raspberry Pi Real Benchmark

Purpose: make final decision on real hardware.

On Raspberry Pi, test:

- `.pt` at 640 and 512
- NCNN at 640 and 512
- ONNX at 640 and 512
- optional TFLite/INT8 if export succeeds

Record:

- FPS on real video
- CPU usage
- memory usage
- temperature/throttling
- detection quality on representative frames

Recommended command shape:

```bash
yolo detect predict model=/path/to/model source=/path/to/video.mp4 imgsz=512 device=cpu save=False verbose=True
```

## Stage 5: Pipeline-Level Optimization

These often beat model-only changes on Raspberry Pi:

1. Detect every N frames, track between detections.
2. Crop ROI before inference.
3. Use `imgsz=512`.
4. Raise confidence threshold if false positives are not a concern.
5. Disable unnecessary image/video saving.
6. Separate capture, inference, and display/write into different threads.
7. Avoid drawing every frame if only counts are needed.

## Stage 6: Distillation If Needed

Use distillation only after Stage 1-5 prove that speed requires a lower resolution or smaller model that loses too much accuracy.

Candidate setup:

- Teacher: current `yolo26n.pt` or stronger `yolo26s/yolo26m`.
- Student: lower-width YOLO26 config or lower-resolution training/inference model.

Decision rule:

- Distillation is worth it if `imgsz=512` or `416` is fast enough but Recall/mAP is too low.

## Stage 7: Physical Pruning Only After Backend Tests

The previous zero-channel pruning is not enough because it does not physically remove channels. If pruning is revisited:

- Use physical channel pruning with model graph surgery.
- Fine-tune after pruning.
- Export the compact model to NCNN/ONNX.
- Compare against the simpler baseline: original model + NCNN + `imgsz=512`.

## Recommended Experiment Order

1. `imgsz` sweep: 640, 512, 416, 384, 320.
2. NCNN export and Raspberry Pi benchmark at 640/512.
3. ONNX static/dynamic benchmark at 640/512.
4. Pipeline optimization: frame skip + tracking + ROI.
5. TFLite/INT8 in a separate TensorFlow-compatible environment.
6. Distillation if lower resolution loses too much accuracy.
7. Physical pruning only if still necessary.

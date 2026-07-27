# YOLOv8n runbook

Conda environment: existing `yolov26`, reused without changes. Activate with `conda activate yolov26`; verify with `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.

Dataset format is YOLO. Copy `dataset_config.example` to `dataset_config.yaml` after class order is frozen. Weight: `weights/yolov8n.pt`. Set `PYTHONPATH` so this pinned source commit, not the installed package copy, is used.

From `src/`:

```bash
PYTHONPATH="$PWD" yolo detect train model=../weights/yolov8n.pt data=../dataset_config.yaml imgsz="$IMG_SIZE" batch="$BATCH_SIZE" epochs="$EPOCHS" device="$DEVICE" project="$OUTPUT_ROOT/03_yolov8n" name=train
PYTHONPATH="$PWD" yolo detect val model="$TRAINED_WEIGHTS" data=../dataset_config.yaml imgsz="$IMG_SIZE" batch="$BATCH_SIZE" device="$DEVICE"
PYTHONPATH="$PWD" yolo detect train resume="$LAST_CHECKPOINT"
```

Still required: final dataset path, class order/names, image size, batch, epochs, device, output root, and checkpoint paths.

# YOLOv5n runbook

Conda environment: `bench_detection_torch201` (new, shared). Activate with `conda activate bench_detection_torch201`; verify with `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.

Dataset format is YOLO detection format. After the split and class order are frozen, copy `dataset_config.example` to `dataset_config.yaml` and fill paths/names. Pretrained weight: `weights/yolov5n.pt`.

From `src/`, the official command templates are:

```bash
python train.py --data ../dataset_config.yaml --weights ../weights/yolov5n.pt --img "$IMG_SIZE" --batch "$BATCH_SIZE" --epochs "$EPOCHS" --device "$DEVICE" --project "$OUTPUT_ROOT/01_yolov5n"
python val.py --data ../dataset_config.yaml --weights "$TRAINED_WEIGHTS" --img "$IMG_SIZE" --batch "$BATCH_SIZE" --device "$DEVICE"
python train.py --resume "$LAST_CHECKPOINT"
```

Default output root in the template is `$OUTPUT_ROOT/01_yolov5n`. User-supplied variables: final dataset path, class 0/1 order and names, `IMG_SIZE`, `BATCH_SIZE`, `EPOCHS`, `DEVICE`, `OUTPUT_ROOT`, and checkpoint paths. No values are frozen here.

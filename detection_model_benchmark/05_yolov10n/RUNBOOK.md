# YOLOv10n runbook

Conda environment: `bench_detection_torch201` (new, shared). Activate with `conda activate bench_detection_torch201`; verify with `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.

Dataset format is YOLO. Finalize `dataset_config.yaml` from the example. Weight: `weights/yolov10n.pt`. Always set `PYTHONPATH` to use the THU-MIG fork in this directory.

From `src/`:

```bash
PYTHONPATH="$PWD" yolo detect train model=../weights/yolov10n.pt data=../dataset_config.yaml imgsz="$IMG_SIZE" batch="$BATCH_SIZE" epochs="$EPOCHS" device="$DEVICE" project="$OUTPUT_ROOT/05_yolov10n" name=train
PYTHONPATH="$PWD" yolo detect val model="$TRAINED_WEIGHTS" data=../dataset_config.yaml imgsz="$IMG_SIZE" batch="$BATCH_SIZE" device="$DEVICE"
PYTHONPATH="$PWD" yolo detect train resume="$LAST_CHECKPOINT"
```

Still required: final dataset path, class order/names, image size, batch, epochs, device, output root, and checkpoint paths.

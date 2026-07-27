# YOLOv7-tiny runbook

Conda environment: `bench_detection_torch201` (new, shared). Activate with `conda activate bench_detection_torch201`; verify with `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.

Dataset format is YOLO detection format. Copy and finalize `dataset_config.example` as `dataset_config.yaml`. Pretrained weight: `weights/yolov7-tiny.pt`; use a relative weight path because the legacy downloader lowercases absolute paths.

From `src/`, official templates are:

```bash
python train.py --workers "$WORKERS" --device "$DEVICE" --batch-size "$BATCH_SIZE" --data ../dataset_config.yaml --img "$IMG_SIZE" "$IMG_SIZE" --cfg cfg/training/yolov7-tiny.yaml --weights ../weights/yolov7-tiny.pt --hyp data/hyp.scratch.tiny.yaml --epochs "$EPOCHS" --project "$OUTPUT_ROOT/02_yolov7_tiny" --name train
python test.py --data ../dataset_config.yaml --img-size "$IMG_SIZE" --batch-size "$BATCH_SIZE" --device "$DEVICE" --weights "$TRAINED_WEIGHTS" --task val
python train.py --resume "$LAST_CHECKPOINT"
```

Default output root: `$OUTPUT_ROOT/02_yolov7_tiny`. Still required: dataset path, class order/names, workers, image size, batch, epochs, device, output root, and checkpoint paths.

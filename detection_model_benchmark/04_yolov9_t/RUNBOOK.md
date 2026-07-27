# YOLOv9-T runbook

Conda environment: `bench_detection_torch201` (new, shared). Activate with `conda activate bench_detection_torch201`; verify with `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.

Dataset format is YOLO. Copy `dataset_config.example` to `dataset_config.yaml`. The official T architecture is `models/detect/yolov9-t.yaml`; weight is `weights/yolov9-t-converted.pt`.

From `src/`:

```bash
python train_dual.py --workers "$WORKERS" --device "$DEVICE" --batch "$BATCH_SIZE" --data ../dataset_config.yaml --img "$IMG_SIZE" --cfg models/detect/yolov9-t.yaml --weights ../weights/yolov9-t-converted.pt --hyp hyp.scratch-high.yaml --epochs "$EPOCHS" --project "$OUTPUT_ROOT/04_yolov9_t" --name train --min-items 0
python val.py --data ../dataset_config.yaml --img "$IMG_SIZE" --batch "$BATCH_SIZE" --device "$DEVICE" --weights "$TRAINED_CONVERTED_WEIGHTS"
python train_dual.py --resume "$LAST_CHECKPOINT"
```

The training command is a template only; confirm the intended converted/unconverted checkpoint flow on the validation split before the formal run. Still required: dataset paths, class order, workers, image size, batch, epochs, device, output root, and checkpoints.

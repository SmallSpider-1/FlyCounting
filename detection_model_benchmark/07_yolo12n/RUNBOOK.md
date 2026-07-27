# YOLO12n runbook

Conda environment: existing `yolov26`, reused without changes. Activate with `conda activate yolov26`; verify with `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.

Dataset format is YOLO. Finalize `dataset_config.yaml` from the example. Official Turbo weight: `weights/yolov12n.pt`. This environment uses the repository's PyTorch SDPA fallback because FlashAttention is not installed; this was GPU-smoke-tested.

From `src/`:

```bash
PYTHONPATH="$PWD" yolo detect train model=../weights/yolov12n.pt data=../dataset_config.yaml imgsz="$IMG_SIZE" batch="$BATCH_SIZE" epochs="$EPOCHS" device="$DEVICE" project="$OUTPUT_ROOT/07_yolo12n" name=train
PYTHONPATH="$PWD" yolo detect val model="$TRAINED_WEIGHTS" data=../dataset_config.yaml imgsz="$IMG_SIZE" batch="$BATCH_SIZE" device="$DEVICE"
PYTHONPATH="$PWD" yolo detect train resume="$LAST_CHECKPOINT"
```

Still required: dataset path, class order/names, image size, batch, epochs, device, output root, and checkpoint paths. Reassess FlashAttention only if it becomes an explicit speed-study factor.

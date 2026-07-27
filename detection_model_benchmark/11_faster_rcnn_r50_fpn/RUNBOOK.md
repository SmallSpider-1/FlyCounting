# Faster R-CNN ResNet50-FPN runbook

Conda environment: existing `pointnext`, reused without changes. Activate with `conda activate pointnext`; verify with `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.

Use standard COCO layout: `train2017/`, `val2017/`, `annotations/instances_train2017.json`, and `annotations/instances_val2017.json`. Foreground category IDs must be frozen and remapped to contiguous `1..N`; background is added internally. The runtime uses the installed official torchvision 0.21.0 binary while `src/references/detection` supplies the official training reference. The local `train_custom.py` adapter keeps empty/background training images by default so Faster R-CNN sees the same samples as the other ten models; pass `--drop-empty-images` only for an explicitly separate ablation. Weight: `weights/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth`.

From the project directory:

```bash
python train_custom.py --dataset coco --data-path "$DATASET_COCO_ROOT" --model fasterrcnn_resnet50_fpn --num-foreground-classes "$NUM_CLASSES" --initial-weights-file weights/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth --device cuda --batch-size "$BATCH_SIZE" --epochs "$EPOCHS" --workers "$WORKERS" --output-dir "$OUTPUT_ROOT/11_faster_rcnn_r50_fpn"
python train_custom.py --dataset coco --data-path "$DATASET_COCO_ROOT" --model fasterrcnn_resnet50_fpn --num-foreground-classes "$NUM_CLASSES" --device cuda --resume "$TRAINED_CHECKPOINT" --test-only
python train_custom.py --dataset coco --data-path "$DATASET_COCO_ROOT" --model fasterrcnn_resnet50_fpn --num-foreground-classes "$NUM_CLASSES" --device cuda --resume "$LAST_CHECKPOINT" --output-dir "$OUTPUT_ROOT/11_faster_rcnn_r50_fpn"
```

Still required: COCO root, category ID/name order, number of foreground classes, batch, epochs, workers, output root, and checkpoint paths. `train_custom.py` only adapts the official reference to a non-COCO class count and local shape-compatible pretrained state; it does not alter torchvision source.

# RT-DETR-R18 runbook

Conda environment: `bench_detection_torch201` (new, shared). Activate with `conda activate bench_detection_torch201`; verify with `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.

Use COCO detection JSON. Copy `dataset_config.example` to `src/rtdetr_pytorch/configs/dataset/fruitfly_detection.yml`, then compose a custom config from `configs/rtdetr/rtdetr_r18vd_6x_coco.yml` plus the custom dataset config. Set `remap_mscoco_category: false`. Weight: `weights/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth`.

From `src/rtdetr_pytorch/`:

```bash
CUDA_VISIBLE_DEVICES="$DEVICE" python tools/train.py -c "$CUSTOM_CONFIG" -t ../../weights/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth --amp --seed "$SEED"
CUDA_VISIBLE_DEVICES="$DEVICE" python tools/train.py -c "$CUSTOM_CONFIG" --test-only -r "$TRAINED_CHECKPOINT"
CUDA_VISIBLE_DEVICES="$DEVICE" python tools/train.py -c "$CUSTOM_CONFIG" -r "$LAST_CHECKPOINT" --amp
```

Set the config output directory to `$OUTPUT_ROOT/10_rt_detr_r18`. Still required: COCO paths, category ID/name order, batch/workers, seed, device, output root, and checkpoints. This is the original R18 config, not an RT-DETRv2/v3 substitute.

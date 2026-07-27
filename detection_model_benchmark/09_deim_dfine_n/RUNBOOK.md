# DEIM-D-FINE-N runbook

Conda environment: `bench_detection_torch201` (new, shared). Activate with `conda activate bench_detection_torch201`; verify with `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.

Dataset format is COCO detection JSON. After categories and splits are frozen, copy `dataset_config.example` to `src/configs/dataset/fruitfly_detection.yml`, then create a custom model config that includes `configs/deim_dfine/deim_hgnetv2_n_coco.yml` followed by the custom dataset config. Keep `remap_mscoco_category: false`. Weight: `weights/deim_dfine_hgnetv2_n_coco_160e.pth`.

From `src/`, with `CUSTOM_CONFIG` pointing to that composed config:

```bash
CUDA_VISIBLE_DEVICES="$DEVICE" python train.py -c "$CUSTOM_CONFIG" -t ../weights/deim_dfine_hgnetv2_n_coco_160e.pth --use-amp --seed "$SEED"
CUDA_VISIBLE_DEVICES="$DEVICE" python train.py -c "$CUSTOM_CONFIG" --test-only -r "$TRAINED_CHECKPOINT"
CUDA_VISIBLE_DEVICES="$DEVICE" python train.py -c "$CUSTOM_CONFIG" -r "$LAST_CHECKPOINT" --use-amp
```

The official config controls its output directory; set it to `$OUTPUT_ROOT/09_deim_dfine_n` in the composed config. Still required: COCO paths, category ID/name order, batch/workers, seed, device, output root, and checkpoint paths.

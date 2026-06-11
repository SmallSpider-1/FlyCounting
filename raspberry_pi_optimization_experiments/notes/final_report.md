# YOLO26 Raspberry Pi Optimization Experiment Report

## Scope

All files were created under:

`/home/admin1/Projects/ultralytics-main/raspberry_pi_optimization_experiments`

The original project files, model YAML, `data.yaml`, `train.py`, `main.py`, and `yolo26n.pt` were not intentionally modified. Temporary root-level export files produced by Ultralytics were copied into the workspace and then removed.

## Baseline Resolution Sweep

| Model              | imgsz |  mAP50 | mAP50-95 | Inference ms/img | Notes                                             |
| ------------------ | ----: | -----: | -------: | ---------------: | ------------------------------------------------- |
| PyTorch yolo26n.pt |   640 | 0.9918 |   0.8520 |           1.6268 | Best accuracy baseline on A100 validation run     |
| PyTorch yolo26n.pt |   512 | 0.9912 |   0.8326 |           1.7169 | Small accuracy drop; good candidate               |
| PyTorch yolo26n.pt |   416 | 0.9832 |   0.7889 |           1.7184 | Larger accuracy drop                              |
| PyTorch yolo26n.pt |   320 | 0.9730 |   0.6890 |           1.6815 | Too much mAP50-95 loss for accuracy-sensitive use |

Local GPU timings are not representative of Raspberry Pi. Accuracy trends are still useful.

## Exported Backend Validation

| Artifact     |  mAP50 | mAP50-95 | CPU Inference ms/img | Notes                                        |
| ------------ | -----: | -------: | -------------------: | -------------------------------------------- |
| NCNN         | 0.9924 |   0.8518 |             181.6072 | Works; directory must end with `_ncnn_model` |
| ONNX static  | 0.9919 |   0.8527 |              69.1992 | Works                                        |
| ONNX dynamic | 0.9918 |   0.8514 |              19.1939 | Works; fastest on this server CPU            |

Server CPU timing is not Raspberry Pi timing. Repeat on target hardware before final choice.

## Blocked / Failed Experiments

| Experiment         | Result | Reason                                                                 |
| ------------------ | ------ | ---------------------------------------------------------------------- |
| TFLite export      | Failed | TensorFlow `>=2.0,<=2.19` could not be resolved in current environment |
| INT8 TFLite export | Failed | Same TensorFlow blocker; also needs dataset calibration config         |

## Current Recommendation

1. Transfer these candidates to the Raspberry Pi and benchmark on the actual device:
   - `exports/yolo26n_ncnn_1_20260514_200356_ncnn_model/`
   - `exports/yolo26n_onnx_2_20260514_200250/yolo26n.onnx`
2. Test `imgsz=512` first. It keeps most accuracy while reducing input workload for the target device.
3. If Raspberry Pi CPU is still too slow, prioritize:
   - NCNN on Pi
   - Lower `imgsz` such as 512 or 416
   - ROI cropping and frame-skipping with tracking
   - Raspberry Pi AI HAT / Hailo acceleration
4. Use distillation if a lower resolution or smaller student model loses too much Recall/mAP.

## Key Commands

Run resolution sweep:

```bash
conda activate yolov11
python raspberry_pi_optimization_experiments/scripts/run_resolution_sweep.py
```

Export NCNN/ONNX:

```bash
python raspberry_pi_optimization_experiments/scripts/export_backends.py --formats ncnn onnx
```

Validate exported artifacts:

```bash
python raspberry_pi_optimization_experiments/scripts/validate_exported_models.py --device cpu --batch 4 --imgsz 640
```

Summarize results:

```bash
python raspberry_pi_optimization_experiments/scripts/compare_results.py --output-prefix final_summary
```

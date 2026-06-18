# Findings & Decisions

## Requirements

- Use the conda environment requested as `yolo26`; local environment discovery found `yolov26`.
- Train the improved YOLO26 model configs migrated from `/home/admin1/Projects/ultralytics-yolo11-main/ultralytics/cfg/models/11`.
- Use `/home/admin1/Projects/ultralytics-main/train.py` for training.
- Record training console output logs.
- Use `planning-with-files` to plan and track the experiment.

## Research Findings

- `/home/admin1/Projects/ultralytics-main/ultralytics/cfg/models/26` contains 411 yaml files.
- The source directory `/home/admin1/Projects/ultralytics-yolo11-main/ultralytics/cfg/models/11` contains 408 `yolo11-*.yaml` files.
- Mapping target names by replacing `yolo26-` with `yolo11-` shows 0 extra and 0 missing files for the migrated `yolo26-*.yaml` set.
- `train.py` already tees stdout and stderr into a timestamped `train_console_YYYYMMDD_HHMMSS.txt` file under the model run directory.
- `train.py` currently defaults to `MODEL_NAME = "yolo12"` but builds `MODEL_CONFIGS` from all `ultralytics/cfg/models/26/yolo26*.yaml` files.
- `dataset_two_class.yaml` points to the two-class Bactrocera dataset in `/home/admin1/Projects/ultralytics-yolo11-main/dataset_two_class`.
- GPU check after initial launch showed GPU0 around 43% utilization and GPU1 idle.
- A second process was launched on GPU1 with the latter 204 model names, starting at `yolo26-C3k2-WTConv` and ending at `yolo26-wConv`.
- The original full `--migrated` process would not stop automatically at model 204; it was stopped and replaced by explicit non-overlapping shards.
- Final active shard layout uses 8 processes: 4 on GPU0 and 4 on GPU1, 51 models per process.

## Technical Decisions

| Decision                                  | Rationale                                                                                                              |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Run the exact migrated set of 408 configs | This matches the user's wording and avoids unrelated configs in the target directory.                                  |
| Use CLI arguments in `train.py`           | Enables one command to run all migrated models while still using the requested script.                                 |
| Preserve existing defaults                | Existing open tabs show prior single-model runs; keeping default behavior avoids breaking the user's current workflow. |

## Issues Encountered

| Issue                                                                                          | Resolution                                                                                                                                                                       |
| ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Requested conda env `yolo26` does not exist                                                    | Use local env `yolov26`, which is present and semantically matches the request.                                                                                                  |
| The full sweep is very large at 408 models x 200 epochs                                        | Prepare resumable batch execution and per-model logs.                                                                                                                            |
| `nohup conda run -n yolov26 ...` exited immediately with no output                             | Use `/home/admin1/.conda/envs/yolov26/bin/python` directly with `setsid` for the background sweep.                                                                               |
| Single GPU utilization is modest because each run uses small models and batch=4                | Run another independent training process on GPU1 using the non-overlapping second half of the model list.                                                                        |
| Ultralytics LOGGER can keep a closed tee stream between models in a multi-model Python process | Updated `train.py` to restore LOGGER stream state and make `Tee.write/flush` tolerate closed log files.                                                                          |
| Old attempts made active directories confusing                                                 | Moved non-current artifacts into `runs/detect/_old_noncurrent_20260513_114959`; current active locations now retain final shard outputs and current model logs/result dirs only. |

## Resources

- Training script: `/home/admin1/Projects/ultralytics-main/train.py`
- Target model directory: `/home/admin1/Projects/ultralytics-main/ultralytics/cfg/models/26`
- Source model directory: `/home/admin1/Projects/ultralytics-yolo11-main/ultralytics/cfg/models/11`
- Dataset config: `/home/admin1/Projects/ultralytics-main/dataset_two_class.yaml`
- Runs/log root: `/home/admin1/Projects/ultralytics-main/runs/detect`
- Final shard outputs: `/home/admin1/Projects/ultralytics-main/runs/detect/sweep_migrated_yolo26_gpu{0,1}_shard{1..8}_20260512_234646.out`
- Cleanup archive: `/home/admin1/Projects/ultralytics-main/runs/detect/_old_noncurrent_20260513_114959`

## Visual/Browser Findings

- No browser or image sources used.

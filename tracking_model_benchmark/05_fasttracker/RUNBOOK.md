# FastTracker runbook

Status: the official Ultralytics v8.4.105 no-appearance adaptation is integrated and minimal-smoke ready in `bench_tracking`.

Run the registered smoke from the repository root:

```bash
conda activate bench_tracking
python tracking_model_benchmark/_common/smoke_tests.py fasttracker
```

The runtime configuration is `/home/admin1/Projects/ultralytics-main/ultralytics/cfg/trackers/fasttrack.yaml`. The smoke constructs `FASTTracker` directly from cached synthetic detections; it does not load a detector, video, or weight.

Formal adapter outline:

1. Read the common cached AABB detections and convert them to the Ultralytics Results-like `xywh/conf/cls` interface.
2. Construct one `FASTTracker` per video and keep `per_class=false`.
3. Do not add ReID, GMC/ECC/optical flow, future frames, or offline reconnection.
4. Tune only on validation data, then record frozen parameters under `configs/`.
5. Preserve the v8.4.105 upstream provenance and isolated compatibility base recorded in `src/SOURCE.md`.

The unified full-cache entry is now enabled through `_common/run_tracker.py --tracker fasttracker`; use `configs/counting_candidate_v1.json` only as a pre-tuning candidate.

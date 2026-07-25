# SORT runbook

Status: source archived and minimal smoke ready in the independent `bench_tracking` environment.

Activate the dedicated environment and run the minimal check from `src/`:

```bash
conda activate bench_tracking
PYTHONPATH="$PWD" MPLBACKEND=Agg python -c "from sort import Sort; tracker=Sort(max_age=30, min_hits=3, iou_threshold=0.3); print(type(tracker).__name__)"
```

Formal adapter outline:

1. Read the frozen per-frame AABB detection cache; do not invoke a detector.
2. Pass one `Nx5 [x1,y1,x2,y2,score]` array per frame, with all classes combined (`per_class=false`).
3. Reset `Sort` for every video and preserve returned IDs without offline reconnection.
4. Keep ReID, GMC/ECC/optical flow, future frames, and post-processing disabled.
5. Freeze validation-set parameters into `configs/` before any test evaluation.

The unified full-cache entry is now enabled through `_common/run_tracker.py --tracker sort`; use `configs/counting_candidate_v1.json` only as a pre-tuning candidate.

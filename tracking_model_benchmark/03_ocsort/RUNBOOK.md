# OC-SORT runbook

Status: source archived and minimal smoke ready in the independent `bench_tracking` environment.

Run the minimal check from `src/`:

```bash
conda activate bench_tracking
PYTHONPATH="$PWD" python -c "from trackers.ocsort_tracker.ocsort import OCSort; tracker=OCSort(det_thresh=0.6, max_age=30, min_hits=3, iou_threshold=0.3, delta_t=3, asso_func='iou', inertia=0.2, use_byte=False); print(type(tracker).__name__)"
```

Formal adapter outline:

1. Consume the shared cached AABB detections instead of repository detector code.
2. Use `OCSort.update`, not the category-gated `update_public` path.
3. Associate the combined two-species stream (`per_class=false`) with no ReID or GMC.
4. Disable interpolation and all other offline/future-frame processing.
5. Tune only on validation data, then save the frozen configuration in `configs/`.

The unified full-cache entry is now enabled through `_common/run_tracker.py --tracker ocsort`; use `configs/counting_candidate_v1.json` only as a pre-tuning candidate.

# HybridSORT runbook

Conda environment: independent `bench_tracking`; the pure-motion core and non-empty update are smoke-ready.

From `src/`, the constructor check is:

```bash
PYTHONPATH="$PWD" python - <<'PY'
from types import SimpleNamespace
from trackers.hybrid_sort_tracker.hybrid_sort import Hybrid_Sort

args = SimpleNamespace(
    track_thresh=0.6,
    TCM_first_step=False,
    TCM_byte_step=False,
    TCM_byte_step_weight=0.0,
)
tracker = Hybrid_Sort(args=args, det_thresh=0.1)
print(type(tracker).__name__)
PY
```

Formal adapter outline:

1. Use `hybrid_sort.py::Hybrid_Sort`, never `hybrid_sort_reid.py`.
2. Use the independent `bench_tracking` environment and keep every pre-existing environment read-only.
3. Feed all cached AABB detections together (`per_class=false`) and keep ReID/ECC/GMC disabled.
4. Exclude category-gated public-update behavior, interpolation, future frames, and offline post-processing.
5. Reset state per video and freeze validation-selected parameters under `configs/`.

See `logs/smoke_test.log` for the import, constructor, two-frame association, and empty-frame pass.

The unified full-cache entry is `_common/run_tracker.py --tracker hybridsort`; `configs/counting_candidate_v1.json` is only a pre-tuning candidate.

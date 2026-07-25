# SFSORT runbook

Conda environment: independent `bench_tracking`. Activate with `conda activate bench_tracking`; existing environments must remain read-only.

From `src/`, the constructor-level check is:

```bash
PYTHONPATH="$PWD" python -c "from SFSORT import SFSORT; tracker=SFSORT({}); print(type(tracker).__name__)"
```

Formal adapter outline:

1. Build the validation-frozen argument dictionary from `configs/`.
2. Feed one combined `boxes`/`scores` batch from the common cache for each frame (`per_class=false`).
3. Use the online `SFSORT.update` path only; do not use future-frame gap filling or offline mode.
4. Reset tracker state for every video and map output tracks back to current detections for uniform downstream class handling.
5. Do not load the upstream-tracked `src/best.pt`; it is unrelated to cached-detection association.

The unified full-cache entry is now enabled through `_common/run_tracker.py --tracker sfsort`; use `configs/counting_candidate_v1.json` only as a pre-tuning candidate.

# BoT-SORT runbook

Status: source archived and constrained no-ReID/no-GMC smoke ready in `bench_tracking`. The official vendored FastReID uses `torch._six`, so use the reproducible external shim in `_common/smoke_tests.py`; the official `src/` tree remains unmodified.

Run the registered no-ReID/no-GMC smoke from the benchmark root:

```bash
conda activate bench_tracking
python _common/smoke_tests.py botsort
```

Formal adapter outline:

1. Keep `with_reid=false` and `cmc_method=none`; never load ReID weights.
2. Feed the unified cached AABB detections once per frame (`per_class=false`).
3. Exclude interpolation, future frames, offline post-processing, and repository detector runners.
4. Pin NumPy `<1.24` for the archived `np.float` paths before runtime validation.
5. Reset per video and freeze validation-selected parameters in `configs/`.

This constrained variant must be reported as “BoT-SORT (no ReID, no GMC),” not as the paper's default system.

The unified full-cache entry is `_common/run_tracker.py --tracker botsort`; `configs/counting_candidate_v1.json` is only a pre-tuning candidate.

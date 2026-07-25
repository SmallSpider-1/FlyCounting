# ByteTrack runbook

Status: source archived and minimal smoke ready in `bench_tracking`; NumPy is pinned to 1.23.5 for the archived `np.float` paths.

Activate the dedicated environment and run this constructor check from `src/`:

```bash
conda activate bench_tracking
PYTHONPATH="$PWD" python - <<'PY'
from types import SimpleNamespace
from yolox.tracker.byte_tracker import BYTETracker

args = SimpleNamespace(track_thresh=0.6, track_buffer=30, match_thresh=0.8, mot20=False)
tracker = BYTETracker(args, frame_rate=30)
print(type(tracker).__name__)
PY
```

Formal adapter outline:

1. Read the common low-confidence AABB cache and never call the repository's YOLOX detector runners.
2. Pass all classes together once per frame (`per_class=false`); do not use ReID tutorial variants.
3. Configure the high/low association thresholds only on validation data.
4. Reset tracker state per video and disable interpolation, future frames, and offline post-processing.
5. Record the frozen no-ReID/no-GMC configuration under `configs/`.

The cached-detection path uses no detector or ReID weight inside the tracker process.

The unified full-cache entry is `_common/run_tracker.py --tracker bytetrack`; `configs/counting_candidate_v1.json` is only a pre-tuning candidate.

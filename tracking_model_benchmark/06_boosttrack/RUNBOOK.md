# BoostTrack runbook

Status: source archived and minimal no-ReID/no-ECC smoke ready in `bench_tracking`. The official module still imports the ReID package at module load, but the disabled path does not construct or invoke it.

Run from `src/` (both `src` and `src/external` must be on `PYTHONPATH`):

```bash
conda activate bench_tracking
PYTHONPATH="$PWD:$PWD/external" python - <<'PY'
from default_settings import GeneralSettings

GeneralSettings.values["use_embedding"] = False
GeneralSettings.values["use_ecc"] = False
from tracker.boost_track import BoostTrack

tracker = BoostTrack(video_name=None)
assert tracker.embedder is None and tracker.ecc is None
print(type(tracker).__name__)
PY
```

Formal adapter outline:

1. Set `use_embedding=false` and `use_ecc=false` before construction; do not load ReID weights.
2. Consume only the common AABB detection cache, with all classes combined (`per_class=false`).
3. Do not run the repository's detector/evaluation `main.py` or its interpolation/post-processing path.
4. Decide on validation data whether BoostTrack++ extras remain enabled; record the choice because this is a constrained ablation.
5. Reset state per video and freeze the final configuration under `configs/`.

The unified full-cache entry is now enabled through `_common/run_tracker.py --tracker boosttrack`; use `configs/counting_candidate_v1.json` only as a pre-tuning candidate.

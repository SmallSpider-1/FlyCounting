# BoostTrack project record

- Tracker: BoostTrack
- Official repository: https://github.com/vukasin-stanojevic/BoostTrack.git
- Git branch: `master`
- Git commit: `fb5bfc3a8f067476565e753b3a73df4d757c9d03`
- Clone date: 2026-07-23
- Conda environment: independent `bench_tracking`
- Environment source: cloned from `bench_detection_torch201`; existing environments remained unchanged
- Python / PyTorch / torchvision: 3.9.23 / 2.0.1+cu118 / 0.15.2+cu118
- Official license: MIT
- Core source: `src/tracker/boost_track.py::BoostTrack`
- Local implementation relation: a maintained adaptation exists at `boxmot.trackers.boosttrack.boosttrack.BoostTrack`, whose default config enables ReID and ECC and therefore cannot be reused unchanged.
- Experiment variant: set `use_embedding=false` and `use_ecc=false`, disable offline post-processing/future frames, and use one combined detection stream (`per_class=false`). This is a constrained ablation, not the paper's default configuration.
- Weights: no detector or ReID weight was downloaded; disabled ReID must not initialize a model.
- Minimum validation: official core import, no-ReID/no-ECC construction, two-frame stable ID, and empty-frame update passed; see `logs/smoke_test.log`. The unchanged module still requires `torchreid` at import time but never constructs it in this variant.

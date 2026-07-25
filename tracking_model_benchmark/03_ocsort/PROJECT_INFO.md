# OC-SORT project record

- Tracker: OC-SORT
- Official repository: https://github.com/noahcao/OC_SORT.git
- Git branch: `master`
- Git commit: `8462e7e729a93ccd3bd995c0a79a890336cb3a0b`
- Clone date: 2026-07-23
- Conda environment: independent `bench_tracking`
- Environment source: cloned from `bench_detection_torch201`; existing environments remained unchanged
- Python / PyTorch / torchvision: 3.9.23 / 2.0.1+cu118 / 0.15.2+cu118
- Official license: MIT
- Core source: `src/trackers/ocsort_tracker/ocsort.py::OCSort`
- Local implementation relation: a maintained adaptation exists at `boxmot.trackers.ocsort.ocsort.OcSort`; this archive keeps the official pure-motion source.
- Experiment variant: use `OCSort.update`, not category-gated `update_public`; no ReID, no GMC, no interpolation/offline post-processing, and `per_class=false`.
- Weights: none required.
- Minimum validation: pure-motion core import, construction, two-frame stable ID, and empty-frame update passed; see `logs/smoke_test.log`.

# SORT project record

- Tracker: SORT
- Official repository: https://github.com/abewley/sort.git
- Git branch: `master`
- Git commit: `2236dff5019565958b84df7d871d41cc1db58ac7`
- Clone date: 2026-07-23
- Conda environment: independent `bench_tracking`
- Environment source: cloned from `bench_detection_torch201`; existing environments remained unchanged
- Python / PyTorch / torchvision: 3.9.23 / 2.0.1+cu118 / 0.15.2+cu118
- Official license: GPL-3.0
- Core source: `src/sort.py::Sort`
- Local implementation relation: no equivalent plain SORT backend was found in local BoxMOT or Ultralytics; StrongSORT internals are not this baseline.
- Experiment variant: native geometry-only Kalman + IoU tracker; no ReID, no GMC, no class splitting (`per_class=false`).
- Weights: none required.
- Minimum validation: core import, construction, two-frame stable ID, and empty-frame update passed; see `logs/smoke_test.log`.

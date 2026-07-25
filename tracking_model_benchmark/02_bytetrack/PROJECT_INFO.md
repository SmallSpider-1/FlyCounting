# ByteTrack project record

- Tracker: ByteTrack
- Official repository: https://github.com/ifzhang/ByteTrack.git
- Git branch: `main`
- Git commit: `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99`
- Clone date: 2026-07-23
- Conda environment: independent `bench_tracking`
- Environment source: cloned from `bench_detection_torch201`; existing environments remained unchanged
- Python / PyTorch / torchvision: 3.9.23 / 2.0.1+cu118 / 0.15.2+cu118
- Official license: MIT
- Core source: `src/yolox/tracker/byte_tracker.py::BYTETracker`
- Local implementation relation: related maintained backends exist at `boxmot.trackers.bytetrack.ByteTrack` and `ultralytics.trackers.byte_tracker.BYTETracker`; this archive pins the official repository independently.
- Experiment variant: official two-stage confidence/IoU association, natively without ReID or GMC; all classes must be associated together (`per_class=false`), and detector runners/interpolation are excluded.
- Weights: no detector weight is needed for cached-detection tracking and none was downloaded separately.
- Minimum validation: official core import, construction, two-frame stable ID, and empty-frame update passed with NumPy 1.23.5; see `logs/smoke_test.log`.

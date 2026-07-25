# FastTracker project record

- Tracker: Ultralytics no-appearance FastTracker adaptation
- Official repository: https://github.com/ultralytics/ultralytics.git
- Upstream tag / commit: `v8.4.105` / `a7e7e65a3e4086c5d2104dca06f6593564eda1ae`
- Local source pointer: `/home/admin1/Projects/ultralytics-main/ultralytics/trackers/`
- Local base branch / commit: `main` / `b06fe71473c5541c2bfa06222e30ce4db379b4f9` (Ultralytics 8.4.20 tracker tree before the adaptation)
- Archive update date: 2026-07-25
- Conda environment: independent `bench_tracking`
- Python / PyTorch / torchvision: 3.9.23 / 2.0.1+cu118 / 0.15.2+cu118
- Core source: `ultralytics/trackers/fast_tracker.py::FASTTracker`
- Local implementation relation: official v8.4.105 FastTracker and its helper closure were added to the local Ultralytics package. Its newer ByteTrack base is isolated as `fasttracker_byte_tracker.py`, so the pre-existing local ByteTrack and BoT-SORT implementations are not overwritten.
- Experiment variant: the Ultralytics implementation has no ReID encoder, GMC, ECC, or optical flow. One tracker handles the combined class stream (`per_class=false`).
- Source policy: the traffic-oriented standalone repository was not cloned; `src/SOURCE.md` remains a provenance pointer and no duplicate source is stored here.
- Weights: none required or downloaded.
- Minimum validation: core import, construction, two-frame stable ID across a class change, and empty-frame update passed; see `logs/smoke_test.log`.

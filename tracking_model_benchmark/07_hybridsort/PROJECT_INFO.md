# HybridSORT project record

- Tracker: HybridSORT
- Official repository: https://github.com/ymzis69/HybridSORT.git
- Git branch: `master`
- Git commit: `396f8d30db13304c0cbaf1dcf2e16ded93ce1701`
- Clone date: 2026-07-23
- Conda environment: independent `bench_tracking`
- Environment source: cloned from `bench_detection_torch201`; existing environments remained unchanged
- Python / PyTorch / torchvision: 3.9.23 / 2.0.1+cu118 / 0.15.2+cu118
- PyTorch CUDA runtime: 11.8; CUDA available: yes
- Official license: MIT
- Core source: `src/trackers/hybrid_sort_tracker/hybrid_sort.py::Hybrid_Sort`
- Local implementation relation: BoxMOT provides `HybridSort`, but that integration is a ReID/ECC variant; strict experiments should use the official pure-motion class, not `hybrid_sort_reid.py`.
- Experiment variant: pure-motion class, no ReID, no GMC, no interpolation/offline processing, and `per_class=false`.
- Weights: none required for the selected pure-motion class.
- Minimum validation: pure-motion import, construction, two-frame stable ID, and empty-frame update passed with ReID/ECC flags disabled; see `logs/smoke_test.log`.

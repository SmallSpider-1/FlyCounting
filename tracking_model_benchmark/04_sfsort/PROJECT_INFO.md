# SFSORT project record

- Tracker: SFSORT
- Official repository: https://github.com/gitmehrdad/SFSORT.git
- Git branch: `main`
- Git commit: `b1abdeca5ee361c59a20a7fa3937e830846eea82`
- Clone date: 2026-07-23
- Conda environment: independent `bench_tracking`
- Environment source: cloned from `bench_detection_torch201`; existing environments remained unchanged
- Python / PyTorch / torchvision: 3.9.23 / 2.0.1+cu118 / 0.15.2+cu118
- PyTorch CUDA runtime: 11.8; CUDA available: yes
- Official license: MIT
- Core source: `src/SFSORT.py::SFSORT`
- Local implementation relation: a maintained adaptation exists at `boxmot.trackers.sfsort.sfsort.SFSORT`; the official single-file source is used for this archive.
- Experiment variant: online geometry-only mode; no ReID, no GMC, no future-frame/offline filling, and one combined stream for `per_class=false`.
- Weight note: the upstream Git repository itself tracks `src/best.pt`; it remains untouched as part of the clean source archive, is not used, and no separate weight was downloaded or stored.
- Minimum validation: core import, construction, two-frame stable ID, and empty-frame update passed; see `logs/smoke_test.log`.

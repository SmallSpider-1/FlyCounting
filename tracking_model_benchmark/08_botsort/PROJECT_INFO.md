# BoT-SORT project record

- Tracker: BoT-SORT
- Official repository: https://github.com/NirAharon/BoT-SORT.git
- Git branch: `main`
- Git commit: `251985436d6712aaf682aaaf5f71edb4987224bd`
- Clone date: 2026-07-23
- Conda environment: independent `bench_tracking`
- Environment source: cloned from `bench_detection_torch201`; existing environments remained unchanged
- Python / PyTorch / torchvision: 3.9.23 / 2.0.1+cu118 / 0.15.2+cu118
- Official license: MIT
- Core source: `src/tracker/bot_sort.py::BoTSORT`
- Local implementation relation: related backends exist at `boxmot.trackers.botsort.BotSort` and `ultralytics.trackers.bot_sort.BOTSORT`; the Ultralytics implementation can explicitly use an identity GMC path, while BoxMOT would need careful no-CMC adaptation.
- Experiment variant: `with_reid=false`, `cmc_method=none`, no interpolation/offline processing, and a single combined stream (`per_class=false`). This is a constrained no-ReID/no-GMC ablation, not the paper's default configuration.
- Weights: no detector or ReID weight was downloaded and ReID weight loading is forbidden for this variant.
- Minimum validation: no-ReID/no-GMC construction, two-frame stable ID, and empty-frame update passed with NumPy 1.23.5. The unchanged vendored FastReID import needs a narrow `torch._six.string_classes` shim on PyTorch 2.0; see `logs/smoke_test.log`.

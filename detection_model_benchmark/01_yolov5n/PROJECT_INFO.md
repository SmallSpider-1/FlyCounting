# YOLOv5n project record

- Model: YOLOv5n
- Official repository: https://github.com/ultralytics/yolov5.git
- Git branch/tag: `master` (`v7.0-516-g09e765b4`)
- Git commit: `09e765b424d13d01e038b6d8c6f7e15f5384a444`
- Clone date: 2026-07-17
- Conda environment: `bench_detection_torch201`
- Environment source: created for this benchmark (compatible baseline cloned read-only from `sugar`, then isolated dependencies installed)
- Shared with: YOLOv7-tiny, YOLOv9-T, YOLOv10n, DEIM-D-FINE-N, RT-DETR-R18
- Python / PyTorch / torchvision: 3.9.23 / 2.0.1+cu118 / 0.15.2+cu118
- PyTorch CUDA runtime: 11.8; CUDA available: yes
- Pretrained weight: `weights/yolov5n.pt`
- Official weight URL: https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt
- Weight SHA256: `4f180cf23ba0717ada0badd6c685026d73d48f184d00fc159c2641284b2ac0a3`
- Official license: AGPL-3.0
- Minimum validation: core import, official weight load and GPU single-image inference passed; see `logs/smoke_test.log`.

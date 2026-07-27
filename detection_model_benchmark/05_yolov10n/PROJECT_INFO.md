# YOLOv10n project record

- Model: YOLOv10n
- Official repository: https://github.com/THU-MIG/yolov10.git
- Git branch/tag: `main` (`v1.1-61-g453c6e38`)
- Git commit: `453c6e38a51e9d1d5a2aa5fb7f1014a711913397`
- Clone date: 2026-07-17
- Conda environment: `bench_detection_torch201`
- Environment source: created for this benchmark
- Shared with: YOLOv5n, YOLOv7-tiny, YOLOv9-T, DEIM-D-FINE-N, RT-DETR-R18
- Python / PyTorch / torchvision: 3.9.23 / 2.0.1+cu118 / 0.15.2+cu118
- PyTorch CUDA runtime: 11.8; CUDA available: yes
- Pretrained weight: `weights/yolov10n.pt`
- Official weight URL: https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt
- Weight SHA256: `61b91ffc99b284792dca49bf40216945833cc2a515e1a742954e6e9327cfc19e`
- Official license: AGPL-3.0
- Minimum validation: official YOLOv10 core import, weight load and GPU single-image inference passed; see `logs/smoke_test.log`.

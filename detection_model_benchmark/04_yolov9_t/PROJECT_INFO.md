# YOLOv9-T project record

- Model: YOLOv9-T (`models/detect/yolov9-t.yaml`, not another YOLOv9 scale)
- Official repository: https://github.com/WongKinYiu/yolov9.git
- Git branch/tag: `main` (`v0.1-104-g5b1ea9a`)
- Git commit: `5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff`
- Clone date: 2026-07-17
- Conda environment: `bench_detection_torch201`
- Environment source: created for this benchmark
- Shared with: YOLOv5n, YOLOv7-tiny, YOLOv10n, DEIM-D-FINE-N, RT-DETR-R18
- Python / PyTorch / torchvision: 3.9.23 / 2.0.1+cu118 / 0.15.2+cu118
- PyTorch CUDA runtime: 11.8; CUDA available: yes
- Pretrained weight: `weights/yolov9-t-converted.pt`
- Official weight URL: https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-t-converted.pt
- Weight SHA256: `61e080e964e65e32b884477c5e6344c607c7e02103d64649de810edaeb869803`
- Official license: GPL-3.0
- Minimum validation: core import, official converted T weight load and GPU single-image inference passed; see `logs/smoke_test.log`.

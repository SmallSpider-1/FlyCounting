# YOLOv7-tiny project record

- Model: YOLOv7-tiny
- Official repository: https://github.com/WongKinYiu/yolov7.git
- Git branch/tag: `main` (`v0.1-128-ga207844`)
- Git commit: `a207844b1ce82d204ab36d87d496728d3d2348e7`
- Clone date: 2026-07-17
- Conda environment: `bench_detection_torch201`
- Environment source: created for this benchmark
- Shared with: YOLOv5n, YOLOv9-T, YOLOv10n, DEIM-D-FINE-N, RT-DETR-R18
- Python / PyTorch / torchvision: 3.9.23 / 2.0.1+cu118 / 0.15.2+cu118
- PyTorch CUDA runtime: 11.8; CUDA available: yes
- Pretrained weight: `weights/yolov7-tiny.pt`
- Official weight URL: https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
- Weight SHA256: `0fa1423606133c06362ec716029dade1169d5631e83195b75d8efa66d2ee0b77`
- Official license: GPL-3.0
- Minimum validation: core import, local official weight load and GPU single-image inference passed with a relative weight path; see `logs/smoke_test.log`.

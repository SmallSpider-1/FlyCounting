# YOLO26n project record

- Model: YOLO26n
- Official repository: https://github.com/ultralytics/ultralytics.git
- Git branch/tag: `main` / `v8.4.98`
- Git commit: `b2a993107584090b3e3c918bb1cefef53c8769d8`
- Clone date: 2026-07-17
- Conda environment: `yolov26`
- Environment source: existing, reused without dependency changes
- Shared with: YOLOv8n, YOLO11n, YOLO12n
- Python / PyTorch / torchvision: 3.10.20 / 2.6.0+cu124 / 0.21.0+cu124
- PyTorch CUDA runtime: 12.4; CUDA available: yes
- Pretrained weight: `weights/yolo26n.pt`
- Official weight URL: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt
- Weight SHA256: `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`
- Official license: AGPL-3.0
- Minimum validation: local v8.4.98 core import, official weight load and GPU single-image inference passed; see `logs/smoke_test.log`.

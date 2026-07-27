# YOLO12n project record

- Model: YOLO12n Turbo (repository default)
- Official repository: https://github.com/sunsmarterjie/yolov12.git
- Git branch/tag: `main` (`cls-8-g01a22c0`)
- Git commit: `01a22c0603e0eaa6d9bd62120a391e744d92cea2`
- Clone date: 2026-07-17
- Conda environment: `yolov26`
- Environment source: existing, reused without dependency changes
- Shared with: YOLOv8n, YOLO11n, YOLO26n
- Python / PyTorch / torchvision: 3.10.20 / 2.6.0+cu124 / 0.21.0+cu124
- PyTorch CUDA runtime: 12.4; CUDA available: yes
- Pretrained weight: `weights/yolov12n.pt`
- Official weight URL: https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12n.pt
- Weight SHA256: `37080c2891b94c62998f0bfb552dd70c32f9f2ee36618b9e7b3da49b49e150ac`
- Official license: AGPL-3.0
- Minimum validation: official core and weight ran on GPU; FlashAttention was unavailable and the repository's SDPA fallback was used successfully. See `logs/smoke_test.log`.

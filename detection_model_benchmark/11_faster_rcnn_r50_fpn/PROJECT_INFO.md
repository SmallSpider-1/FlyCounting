# Faster R-CNN ResNet50-FPN project record

- Model: torchvision `fasterrcnn_resnet50_fpn` (V1 weights, not V2)
- Official repository: https://github.com/pytorch/vision.git
- Git branch/tag: `main`
- Git commit: `f23f832d090c868691855cc1261ed907e400c2a2`
- Clone date: 2026-07-17
- Conda environment: `pointnext`
- Environment source: existing, reused without dependency changes
- Shared with: none
- Python / PyTorch / torchvision: 3.10.20 / 2.6.0+cu124 / 0.21.0+cu124
- PyTorch CUDA runtime: 12.4; CUDA available: yes
- Pretrained weight: `weights/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth`
- Official weight URL: https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
- Weight SHA256: `258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384`
- Official license: BSD-3-Clause
- Minimum validation: torchvision model import, local state_dict load, GPU single-image inference, and the official reference training entry import passed; see `logs/smoke_test.log`.

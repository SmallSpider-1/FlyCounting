# DEIM-D-FINE-N project record

- Model: DEIM-D-FINE-N (`configs/deim_dfine/deim_hgnetv2_n_coco.yml`), not plain D-FINE-N
- Official repository: https://github.com/Intellindust-AI-Lab/DEIM.git
- Git branch/tag: `main`
- Git commit: `09d35d53d39ee3145a1e61e3a989b28b9468d1dd`
- Clone date: 2026-07-17
- Conda environment: `bench_detection_torch201`
- Environment source: created for this benchmark
- Shared with: YOLOv5n, YOLOv7-tiny, YOLOv9-T, YOLOv10n, RT-DETR-R18
- Python / PyTorch / torchvision: 3.9.23 / 2.0.1+cu118 / 0.15.2+cu118
- PyTorch CUDA runtime: 11.8; CUDA available: yes
- Pretrained weight: `weights/deim_dfine_hgnetv2_n_coco_160e.pth`
- Official listing URL: https://drive.google.com/file/d/1ZPEhiU9nhW4M5jLnYOFwTSLQC1Ugf62e/view?usp=sharing
- Automated official download URL: https://drive.usercontent.google.com/download?id=1ZPEhiU9nhW4M5jLnYOFwTSLQC1Ugf62e&export=download&confirm=t
- Weight SHA256: `8536a97a57bad2951cb4679e8578be225e4a19db678ab568a44e538f4f0fc917`
- Official license: Apache-2.0
- Minimum validation: official torch inference entry loaded the DEIM checkpoint and produced `logs/torch_results.jpg` on GPU; see `logs/smoke_test.log`.

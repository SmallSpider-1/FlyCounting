# RT-DETR-R18 project record

- Model: original PyTorch RT-DETR-R18 (`rtdetr_r18vd_6x_coco.yml`); not L, v2-S or v3-R18
- Official repository: https://github.com/lyuwenyu/RT-DETR.git
- Git branch/tag: `main` (`v1.0-45-g1c8ac3f`)
- Git commit: `1c8ac3f7ba84f14bd5651ab7b1b70d69a5f55f47`
- Clone date: 2026-07-17
- Conda environment: `bench_detection_torch201`
- Environment source: created for this benchmark
- Shared with: YOLOv5n, YOLOv7-tiny, YOLOv9-T, YOLOv10n, DEIM-D-FINE-N
- Python / PyTorch / torchvision: 3.9.23 / 2.0.1+cu118 / 0.15.2+cu118
- PyTorch CUDA runtime: 11.8; CUDA available: yes
- Pretrained weight: `weights/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth`
- Official weight URL: https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth
- Weight SHA256: `3ba8b5c909c9a1c4f21e96d0a7251ab1a485093955ca327d0061fef8d33c66f0`
- Official license: Apache-2.0
- Minimum validation: official PyTorch inference entry loaded the R18 checkpoint and produced `logs/results_0.jpg` on GPU. The official ResNet-18 backbone cache was also downloaded by repository code. See `logs/smoke_test.log`.

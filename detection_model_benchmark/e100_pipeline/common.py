#!/usr/bin/env python3
"""Shared definitions for the fixed-budget fruit-fly detector benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path("/home/admin1/Projects/ultralytics-main")
BENCHMARK_ROOT = PROJECT_ROOT / "detection_model_benchmark"
DATASET_ROOT = PROJECT_ROOT / "datasets/fruitfly_detection_v1"
RESULT_ROOT = BENCHMARK_ROOT / "results/fruitfly_detection_v1/benchmark_640_e100_s42"
PIPELINE_ROOT = BENCHMARK_ROOT / "e100_pipeline"
EXPECTED_DATASET_FINGERPRINT = "a56fa4a6e10370e963d34e983711c3ed69b5b2f064a0d7771d976d6f1c57b5da"
TRAIN_IMAGES = 8143
VAL_IMAGES = 1530
TEST_IMAGES = 1517
TARGET_EPOCHS = 100
TARGET_EFFECTIVE_BATCH = 16
TARGET_STEPS_PER_EPOCH = math.ceil(TRAIN_IMAGES / TARGET_EFFECTIVE_BATCH)
COCO_TRAIN_IMAGES = 118287


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "yolov5n": {
        "index": 1,
        "display": "YOLOv5n",
        "run_name": "01_yolov5n",
        "project": "01_yolov5n",
        "family": "legacy_yolo5",
        "env": "bench_detection_torch201",
        "weight": "01_yolov5n/weights/yolov5n.pt",
        "source": "01_yolov5n/src",
        "commit": "09e765b424d13d01e038b6d8c6f7e15f5384a444",
        "official_epochs": 100,
        "optimizer": "SGD",
        "base_lr": 0.01,
        "reference_batch": 64,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "lrf": 0.01,
        "scheduler": "linear",
        "warmup_epochs": 3.0,
    },
    "yolov7_tiny": {
        "index": 2,
        "display": "YOLOv7-tiny",
        "run_name": "02_yolov7_tiny",
        "project": "02_yolov7_tiny",
        "family": "legacy_yolo7",
        "env": "bench_detection_torch201",
        "weight": "02_yolov7_tiny/weights/yolov7-tiny.pt",
        "source": "02_yolov7_tiny/src",
        "commit": "a207844b1ce82d204ab36d87d496728d3d2348e7",
        "official_epochs": 300,
        "optimizer": "SGD",
        "base_lr": 0.01,
        "reference_batch": 64,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "lrf": 0.01,
        "scheduler": "cosine_one_cycle",
        "warmup_epochs": 3.0,
    },
    "yolov8n": {
        "index": 3,
        "display": "YOLOv8n",
        "run_name": "03_yolov8n",
        "project": "03_yolov8n",
        "family": "ultralytics_current",
        "env": "yolov26",
        "weight": "03_yolov8n/weights/yolov8n.pt",
        "source": "03_yolov8n/src",
        "commit": "b2a993107584090b3e3c918bb1cefef53c8769d8",
        "official_epochs": 500,
        "optimizer": "SGD",
        "base_lr": 0.01,
        "reference_batch": 64,
        "momentum": 0.937,
        "weight_decay": 0.001,
        "lrf": 0.01,
        "scheduler": "linear",
        "warmup_epochs": 3.0,
    },
    "yolov9_t": {
        "index": 4,
        "display": "YOLOv9-T",
        "run_name": "04_yolov9_t",
        "project": "04_yolov9_t",
        "family": "legacy_yolo9",
        "env": "bench_detection_torch201",
        "weight": "04_yolov9_t/weights/yolov9-t-converted.pt",
        "source": "04_yolov9_t/src",
        "commit": "5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff",
        "official_epochs": 100,
        "optimizer": "SGD",
        "base_lr": 0.01,
        "reference_batch": 64,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "lrf": 0.01,
        "scheduler": "linear",
        "warmup_epochs": 3.0,
    },
    "yolov10n": {
        "index": 5,
        "display": "YOLOv10n",
        "run_name": "05_yolov10n",
        "project": "05_yolov10n",
        "family": "ultralytics_yolov10",
        "env": "bench_detection_torch201",
        "weight": "05_yolov10n/weights/yolov10n.pt",
        "source": "05_yolov10n/src",
        "commit": "453c6e38a51e9d1d5a2aa5fb7f1014a711913397",
        "official_epochs": 500,
        "optimizer": "SGD",
        "base_lr": 0.01,
        "reference_batch": 64,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "lrf": 0.01,
        "scheduler": "linear",
        "warmup_epochs": 3.0,
        "postprocess": "NMS_FREE",
    },
    "yolo11n": {
        "index": 6,
        "display": "YOLO11n",
        "run_name": "06_yolo11n",
        "project": "06_yolo11n",
        "family": "ultralytics_current",
        "env": "yolov26",
        "weight": "06_yolo11n/weights/yolo11n.pt",
        "source": "06_yolo11n/src",
        "commit": "b2a993107584090b3e3c918bb1cefef53c8769d8",
        "official_epochs": 600,
        "optimizer": "SGD",
        "base_lr": 0.01,
        "reference_batch": 64,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "lrf": 0.01,
        "scheduler": "linear",
        "warmup_epochs": 3.0,
    },
    "yolo12n": {
        "index": 7,
        "display": "YOLO12n",
        "run_name": "07_yolo12n",
        "project": "07_yolo12n",
        "family": "ultralytics_yolo12",
        "env": "yolov26",
        "weight": "07_yolo12n/weights/yolov12n.pt",
        "source": "07_yolo12n/src",
        "commit": "01a22c0603e0eaa6d9bd62120a391e744d92cea2",
        "official_epochs": 600,
        "optimizer": "SGD",
        "base_lr": 0.01,
        "reference_batch": 64,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "lrf": 0.01,
        "scheduler": "linear",
        "warmup_epochs": 3.0,
    },
    "yolo26n": {
        "index": 8,
        "display": "YOLO26n",
        "run_name": "08_yolo26n",
        "project": "08_yolo26n",
        "family": "ultralytics_current",
        "env": "yolov26",
        "weight": "08_yolo26n/weights/yolo26n.pt",
        "source": "08_yolo26n/src",
        "commit": "b2a993107584090b3e3c918bb1cefef53c8769d8",
        "official_epochs": 245,
        "optimizer": "MuSGD",
        "base_lr": 0.0054,
        "reference_batch": 64,
        "momentum": 0.94676,
        "weight_decay": 0.00064,
        "lrf": 0.04952,
        "scheduler": "linear",
        "warmup_epochs": 0.98124,
        "postprocess": "NMS_FREE",
    },
    "deim_dfine_n": {
        "index": 9,
        "display": "DEIM-D-FINE-N",
        "run_name": "09_deim_dfine_n",
        "project": "09_deim_dfine_n",
        "family": "deim",
        "env": "bench_detection_torch201",
        "weight": "09_deim_dfine_n/weights/deim_dfine_hgnetv2_n_coco_160e.pth",
        "source": "09_deim_dfine_n/src",
        "commit": "09d35d53d39ee3145a1e61e3a989b28b9468d1dd",
        "official_epochs": 160,
        "optimizer": "AdamW",
        "base_lr": 0.0008,
        "backbone_lr": 0.0004,
        "reference_batch": 128,
        "betas": [0.9, 0.999],
        "weight_decay": 0.0001,
        "scheduler": "flat_cosine_gamma_1_constant",
        "warmup_steps": 2000,
        "ema_decay": 0.9999,
        "ema_warmup_steps": 1000,
        "ema_restart_epoch": 148,
        "postprocess": "NMS_FREE",
    },
    "rt_detr_r18": {
        "index": 10,
        "display": "RT-DETR-R18",
        "run_name": "10_rt_detr_r18",
        "project": "10_rt_detr_r18",
        "family": "rt_detr",
        "env": "bench_detection_torch201",
        "weight": "10_rt_detr_r18/weights/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth",
        "source": "10_rt_detr_r18/src/rtdetr_pytorch",
        "commit": "1c8ac3f7ba84f14bd5651ab7b1b70d69a5f55f47",
        "official_epochs": 72,
        "optimizer": "AdamW",
        "base_lr": 0.0001,
        "backbone_lr": 0.00001,
        "reference_batch": 16,
        "betas": [0.9, 0.999],
        "weight_decay": 0.0001,
        "scheduler": "multistep_no_in_budget_decay",
        "official_milestones": [1000],
        "ema_decay": 0.9999,
        "ema_warmup_steps": 2000,
        "postprocess": "NMS_FREE",
    },
    "faster_rcnn_r50_fpn": {
        "index": 11,
        "display": "Faster R-CNN R50-FPN",
        "run_name": "11_faster_rcnn_r50_fpn",
        "project": "11_faster_rcnn_r50_fpn",
        "family": "faster_rcnn",
        "env": "pointnext",
        "weight": "11_faster_rcnn_r50_fpn/weights/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
        "source": "11_faster_rcnn_r50_fpn/src",
        "commit": "f23f832d090c868691855cc1261ed907e400c2a2",
        "official_epochs": 26,
        "optimizer": "SGD",
        "base_lr": 0.02,
        "reference_batch": 16,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "scheduler": "multistep",
        "official_milestones": [16, 22],
        "official_warmup_steps": 1000,
    },
}


INITIAL_ASSIGNMENTS = {0: "faster_rcnn_r50_fpn", 1: "rt_detr_r18"}
FIFO_MODELS = [
    "deim_dfine_n",
    "yolov9_t",
    "yolov7_tiny",
    "yolo12n",
    "yolo26n",
    "yolov10n",
    "yolo11n",
    "yolov8n",
    "yolov5n",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def run_dir(model: str) -> Path:
    return RESULT_ROOT / "runs" / MODEL_SPECS[model]["run_name"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_write_yaml(path: Path, value: Any) -> None:
    atomic_write_text(path, yaml.safe_dump(value, allow_unicode=True, sort_keys=False))


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def append_csv(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def mapped_epoch(official_epoch: float, official_total: int) -> int:
    scaled = official_epoch / official_total * TARGET_EPOCHS
    return min(TARGET_EPOCHS, max(1, math.floor(scaled + 0.5)))


def mapped_steps(official_steps: int, official_epochs: int, official_batch: int) -> int:
    official_steps_per_epoch = math.ceil(COCO_TRAIN_IMAGES / official_batch)
    official_total_steps = official_steps_per_epoch * official_epochs
    return max(1, round(official_steps / official_total_steps * TARGET_STEPS_PER_EPOCH * TARGET_EPOCHS))


def resolved_lr(spec: dict[str, Any]) -> float:
    return spec["base_lr"] * TARGET_EFFECTIVE_BATCH / spec["reference_batch"]


def load_resolved_config(model: str) -> dict[str, Any]:
    return read_yaml(run_dir(model) / "resolved_config.yaml")


def expected_config_hash(model: str) -> str:
    return sha256_file(run_dir(model) / "resolved_config.yaml")

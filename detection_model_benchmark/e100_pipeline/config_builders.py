#!/usr/bin/env python3
"""Build exact framework-native runtime configuration files from the frozen protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from common import BENCHMARK_ROOT, load_resolved_config, run_dir


def legacy_hyp(model: str, config: dict[str, Any]) -> dict[str, Any]:
    source_dir = Path(config["source_dir"])
    source_hyp = {
        "yolov5n": source_dir / "data/hyps/hyp.scratch-low.yaml",
        "yolov7_tiny": source_dir / "data/hyp.scratch.tiny.yaml",
        "yolov9_t": source_dir / "data/hyps/hyp.scratch-high.yaml",
    }[model]
    with source_hyp.open(encoding="utf-8") as stream:
        hyp = yaml.safe_load(stream)
    hyp.update(
        {
            "lr0": float(config["resolved_base_lr"]),
            "lrf": float(config["lrf"]),
            "momentum": float(config["momentum"]),
            "weight_decay": float(config["resolved_weight_decay"]),
            "warmup_epochs": float(config["scheduler"]["mapped_warmup_epochs"]),
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "degrees": 0.0,
            "translate": 0.0,
            "scale": 0.0,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 0.0,
            "mixup": 0.0,
        }
    )
    if "copy_paste" in hyp:
        hyp["copy_paste"] = 0.0
    return hyp


def ultralytics_overrides(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    return {
        "data": config["training_dataset_config"],
        "epochs": config["epochs"],
        "imgsz": config["imgsz"],
        "batch": config["micro_batch"],
        "workers": config["workers"],
        "device": 0,
        "seed": config["seed"],
        "deterministic": False,
        "amp": config["amp"],
        "optimizer": config["optimizer"],
        "lr0": config["resolved_base_lr"],
        "lrf": config["lrf"],
        "momentum": config["momentum"],
        "weight_decay": config["resolved_weight_decay"],
        "warmup_epochs": config["scheduler"]["mapped_warmup_epochs"],
        "nbs": 16,
        "cos_lr": False,
        "patience": 1000,
        "val": True,
        "save": True,
        "save_period": -1,
        "plots": False,
        "cache": False,
        "rect": False,
        "multi_scale": False,
        "close_mosaic": 0,
        "project": str(output_dir / "native"),
        "name": "train",
        "exist_ok": True,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "bgr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
        "verbose": True,
    }


def deim_runtime_config(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    return {
        "__include__": [
            str(BENCHMARK_ROOT / "09_deim_dfine_n/src/configs/deim_dfine/deim_hgnetv2_n_coco.yml"),
            config["dataset_config"],
        ],
        "output_dir": str(output_dir / "native/train"),
        "device": "cuda",
        "sync_bn": False,
        "find_unused_parameters": False,
        "use_amp": True,
        "use_ema": True,
        "epoches": config["epochs"],
        "HGNetv2": {"pretrained": False},
        "lrsheduler": "flatcosine",
        "lr_gamma": 1.0,
        "warmup_iter": config["scheduler"]["mapped_warmup_steps"],
        "flat_epoch": config["epochs"],
        "no_aug_epoch": 0,
        "optimizer": {
            "type": "AdamW",
            "params": [
                {"params": "^(?=.*backbone)(?!.*norm|bn).*$", "lr": config["resolved_backbone_lr"]},
                {
                    "params": "^(?=.*backbone)(?=.*norm|bn).*$",
                    "lr": config["resolved_backbone_lr"],
                    "weight_decay": 0.0,
                },
                {"params": "^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$", "weight_decay": 0.0},
            ],
            "lr": config["resolved_base_lr"],
            "betas": config["betas"],
            "weight_decay": config["resolved_weight_decay"],
        },
        "ema": {
            "type": "ModelEMA",
            "decay": 0.9999,
            "warmups": config["scheduler"]["mapped_ema_warmup_steps"],
            "start": 0,
        },
        "train_dataloader": {
            "total_batch_size": config["micro_batch"],
            "num_workers": config["workers"],
            "drop_last": False,
            "dataset": {
                "transforms": {
                    "ops": [
                        {"type": "RandomHorizontalFlip", "p": 0.5},
                        {"type": "Resize", "size": [640, 640]},
                        {"type": "SanitizeBoundingBoxes", "min_size": 1},
                        {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
                        {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
                    ],
                    "policy": {"name": "stop_epoch", "epoch": [], "ops": []},
                    "mosaic_prob": 0.0,
                }
            },
            "collate_fn": {
                "type": "BatchImageCollateFunction",
                "base_size": 640,
                "base_size_repeat": None,
                "mixup_prob": 0.0,
                "mixup_epochs": [0, 0],
                "stop_epoch": config["scheduler"]["mapped_ema_restart_epoch"],
                "ema_restart_decay": 0.9999,
            },
        },
        "val_dataloader": {
            "total_batch_size": config["micro_batch"],
            "num_workers": config["workers"],
            "drop_last": False,
            "dataset": {
                "transforms": {
                    "ops": [
                        {"type": "Resize", "size": [640, 640]},
                        {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
                    ]
                }
            },
        },
    }


def rtdetr_runtime_config(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    return {
        "__include__": [
            str(BENCHMARK_ROOT / "10_rt_detr_r18/src/rtdetr_pytorch/configs/rtdetr/rtdetr_r18vd_6x_coco.yml"),
            config["dataset_config"],
        ],
        "output_dir": str(output_dir / "native/train"),
        "device": "cuda",
        "sync_bn": False,
        "find_unused_parameters": False,
        "use_amp": True,
        "use_ema": True,
        "epoches": config["epochs"],
        "PResNet": {"depth": 18, "freeze_at": -1, "freeze_norm": False, "pretrained": False},
        "optimizer": {
            "type": "AdamW",
            "params": [
                {"params": "^(?=.*backbone)(?=.*norm).*$", "lr": config["resolved_backbone_lr"], "weight_decay": 0.0},
                {"params": "^(?=.*backbone)(?!.*norm).*$", "lr": config["resolved_backbone_lr"]},
                {"params": "^(?=.*(?:encoder|decoder))(?=.*(?:norm|bias)).*$", "weight_decay": 0.0},
            ],
            "lr": config["resolved_base_lr"],
            "betas": config["betas"],
            "weight_decay": config["resolved_weight_decay"],
        },
        "lr_scheduler": {"type": "MultiStepLR", "milestones": [], "gamma": 0.1},
        "ema": {"type": "ModelEMA", "decay": 0.9999, "warmups": config["scheduler"]["mapped_ema_warmup_steps"]},
        "train_dataloader": {
            "batch_size": config["micro_batch"],
            "num_workers": config["workers"],
            "drop_last": False,
            "dataset": {
                "transforms": {
                    "ops": [
                        {"type": "RandomHorizontalFlip", "p": 0.5},
                        {"type": "Resize", "size": [640, 640]},
                        {"type": "ToImageTensor"},
                        {"type": "ConvertDtype"},
                        {"type": "SanitizeBoundingBox", "min_size": 1},
                        {"type": "ConvertBox", "out_fmt": "cxcywh", "normalize": True},
                    ]
                }
            },
        },
        "val_dataloader": {
            "batch_size": config["micro_batch"],
            "num_workers": config["workers"],
            "drop_last": False,
            "dataset": {
                "transforms": {
                    "ops": [
                        {"type": "Resize", "size": [640, 640]},
                        {"type": "ToImageTensor"},
                        {"type": "ConvertDtype"},
                    ]
                }
            },
        },
    }


def faster_runtime_config(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    return {
        "model": "fasterrcnn_resnet50_fpn",
        "dataset": "coco",
        "data_path": config["training_dataset_root"],
        "epochs": config["epochs"],
        "batch_size": config["micro_batch"],
        "workers": config["workers"],
        "imgsz": 640,
        "optimizer": "SGD",
        "lr": config["resolved_base_lr"],
        "momentum": config["momentum"],
        "weight_decay": config["resolved_weight_decay"],
        "lr_steps": config["scheduler"]["mapped_milestones"],
        "warmup_iters": config["scheduler"]["mapped_warmup_steps"],
        "aspect_ratio_group_factor": 3,
        "drop_last": False,
        "data_augmentation": "hflip",
        "amp": True,
        "seed": config["seed"],
        "output_dir": str(output_dir / "native/train"),
    }


def runtime_config_for(model: str, config: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    config = config or load_resolved_config(model)
    output_dir = run_dir(model)
    family = config["family"]
    if family.startswith("legacy_yolo"):
        return "runtime_hyp.yaml", legacy_hyp(model, config)
    if family in {"ultralytics_current", "ultralytics_yolov10", "ultralytics_yolo12"}:
        return "runtime_overrides.yaml", ultralytics_overrides(config, output_dir)
    if family == "deim":
        return "runtime_config.yaml", deim_runtime_config(config, output_dir)
    if family == "rt_detr":
        return "runtime_config.yaml", rtdetr_runtime_config(config, output_dir)
    if family == "faster_rcnn":
        return "runtime_config.yaml", faster_runtime_config(config, output_dir)
    raise KeyError(f"Unsupported family: {family}")

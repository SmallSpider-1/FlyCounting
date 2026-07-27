#!/usr/bin/env python3
"""Generate, validate, hash, and freeze all fixed-budget benchmark configurations."""

from __future__ import annotations

import csv
import math
import os
import stat
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (
    BENCHMARK_ROOT,
    DATASET_ROOT,
    EXPECTED_DATASET_FINGERPRINT,
    FIFO_MODELS,
    INITIAL_ASSIGNMENTS,
    MODEL_SPECS,
    RESULT_ROOT,
    TARGET_EFFECTIVE_BATCH,
    TARGET_EPOCHS,
    TARGET_STEPS_PER_EPOCH,
    TEST_IMAGES,
    TRAIN_IMAGES,
    VAL_IMAGES,
    atomic_write_json,
    atomic_write_text,
    atomic_write_yaml,
    mapped_epoch,
    mapped_steps,
    now_iso,
    resolved_lr,
    run_dir,
    sha256_file,
)
from config_builders import runtime_config_for


def official_schedule(spec):
    schedule = {
        "model": spec["display"],
        "source_commit": spec["commit"],
        "official_total_epochs": spec["official_epochs"],
        "optimizer": spec["optimizer"],
        "official_base_lr": spec["base_lr"],
        "official_reference_effective_batch": spec["reference_batch"],
        "official_weight_decay": spec["weight_decay"],
        "scheduler": spec["scheduler"],
    }
    for key in (
        "backbone_lr",
        "momentum",
        "betas",
        "lrf",
        "warmup_epochs",
        "warmup_steps",
        "official_warmup_steps",
        "official_milestones",
        "ema_decay",
        "ema_warmup_steps",
        "ema_restart_epoch",
    ):
        if key in spec:
            schedule[key] = spec[key]
    return schedule


def mapped_schedule(spec):
    schedule = {
        "epochs": TARGET_EPOCHS,
        "scheduler": spec["scheduler"],
        "validation_every_epoch": True,
        "early_stopping": False,
        "strong_augmentation_stage": "N/A_ALREADY_DISABLED",
        "mapped_no_aug": "N/A_ALREADY_DISABLED",
    }
    if "warmup_epochs" in spec:
        schedule["mapped_warmup_epochs"] = mapped_epoch(spec["warmup_epochs"], spec["official_epochs"])
    if spec["family"] == "deim":
        schedule.update(
            {
                "mapped_warmup_steps": mapped_steps(spec["warmup_steps"], spec["official_epochs"], spec["reference_batch"]),
                "mapped_ema_warmup_steps": mapped_steps(spec["ema_warmup_steps"], spec["official_epochs"], spec["reference_batch"]),
                "mapped_flat_epoch": TARGET_EPOCHS,
                "mapped_ema_restart_epoch": mapped_epoch(spec["ema_restart_epoch"], spec["official_epochs"]),
                "mapping_notes": "Official flat_epoch=7800 is outside the 160-epoch run and lr_gamma=1.0, so the in-budget post-warmup LR is constant.",
            }
        )
    elif spec["family"] == "rt_detr":
        schedule.update(
            {
                "mapped_milestones": [],
                "mapped_ema_warmup_steps": mapped_steps(spec["ema_warmup_steps"], spec["official_epochs"], spec["reference_batch"]),
                "mapping_notes": "Official milestone 1000 is outside the 72-epoch budget; preserving official behavior means no in-budget LR decay.",
            }
        )
    elif spec["family"] == "faster_rcnn":
        schedule.update(
            {
                "mapped_milestones": [mapped_epoch(value, spec["official_epochs"]) for value in spec["official_milestones"]],
                "mapped_warmup_steps": mapped_steps(spec["official_warmup_steps"], spec["official_epochs"], spec["reference_batch"]),
            }
        )
    return schedule


def resolved_config(model, spec, mapped):
    micro_batch = 16
    accumulation = 1
    config = {
        "model_key": model,
        "model": spec["display"],
        "family": spec["family"],
        "environment": spec["env"],
        "source_dir": str(BENCHMARK_ROOT / spec["source"]),
        "source_commit": spec["commit"],
        "weight": str(BENCHMARK_ROOT / spec["weight"]),
        "dataset_root": str(DATASET_ROOT),
        "dataset_fingerprint": EXPECTED_DATASET_FINGERPRINT,
        "dataset_config": str(BENCHMARK_ROOT / spec["project"] / "dataset_config.yaml"),
        "training_dataset_config": str(
            run_dir(model) / "train_val_dataset.yaml"
            if spec["family"].startswith("legacy_yolo")
            or spec["family"] in {"ultralytics_current", "ultralytics_yolov10", "ultralytics_yolo12"}
            else BENCHMARK_ROOT / spec["project"] / "dataset_config.yaml"
        ),
        "training_dataset_root": str(
            RESULT_ROOT / "preflight/train_val_coco"
            if spec["family"] == "faster_rcnn"
            else DATASET_ROOT
        ),
        "imgsz": 640,
        "epochs": TARGET_EPOCHS,
        "seed": 42,
        "amp": True,
        "workers": 8,
        "micro_batch": micro_batch,
        "accumulation": accumulation,
        "effective_batch": micro_batch * accumulation,
        "nbs": 16 if "yolo" in spec["family"] else None,
        "drop_last": False,
        "train_images": TRAIN_IMAGES,
        "val_images": VAL_IMAGES,
        "test_images": TEST_IMAGES,
        "optimizer": spec["optimizer"],
        "official_reference_batch": spec["reference_batch"],
        "official_base_lr": spec["base_lr"],
        "resolved_base_lr": resolved_lr(spec),
        "momentum": spec.get("momentum"),
        "betas": spec.get("betas"),
        "official_weight_decay": spec["weight_decay"],
        "resolved_weight_decay": spec["weight_decay"],
        "scheduler": mapped,
        "augmentation": {
            "resize": True,
            "padding": True,
            "normalize": True,
            "horizontal_flip_probability": 0.5,
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "random_erasing": 0.0,
            "vertical_flip": 0.0,
            "multi_scale": False,
            "test_time_augmentation": False,
            "offline_augmentation": False,
            "hsv": False,
            "random_crop": False,
            "random_zoom": False,
            "photometric_distort": False,
        },
        "validation_every_epoch": True,
        "early_stopping": False,
        "best_metric": "val_mAP50_95",
        "periodic_checkpoint_epochs": list(range(10, 101, 10)),
        "determinism": {
            "python_seed": 42,
            "numpy_seed": 42,
            "torch_seed": 42,
            "cuda_seed": 42,
            "force_deterministic_algorithms": False,
        },
        "postprocess_type": spec.get("postprocess", "NMS"),
        "nms_iou": None if spec.get("postprocess") == "NMS_FREE" else 0.7,
        "score_export_threshold": 0.001,
        "max_det": 300,
    }
    if "backbone_lr" in spec:
        config["official_backbone_lr"] = spec["backbone_lr"]
        config["resolved_backbone_lr"] = spec["backbone_lr"] * TARGET_EFFECTIVE_BATCH / spec["reference_batch"]
    if "lrf" in spec:
        config["lrf"] = spec["lrf"]
    return config


def lr_rows(spec, mapped, resolved):
    rows = []
    base = resolved["resolved_base_lr"]
    for epoch in range(1, TARGET_EPOCHS + 1):
        if spec["scheduler"] == "cosine_one_cycle":
            factor = ((1 - math.cos(math.pi * (epoch - 1) / TARGET_EPOCHS)) / 2) * (spec["lrf"] - 1) + 1
        elif spec["scheduler"] == "linear":
            factor = max(1 - (epoch - 1) / TARGET_EPOCHS, 0) * (1 - spec["lrf"]) + spec["lrf"]
        elif spec["scheduler"] == "multistep":
            factor = 1.0
            for milestone in mapped["mapped_milestones"]:
                if epoch > milestone:
                    factor *= 0.1
        else:
            factor = 1.0
        rows.append({"epoch": epoch, "lr": base * factor, "factor": factor})
    return rows


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare():
    RESULT_ROOT.mkdir(parents=True, exist_ok=False)
    for directory in ("preflight", "scheduler", "runs", "summary"):
        (RESULT_ROOT / directory).mkdir(parents=True, exist_ok=True)

    train_val_yolo = {
        "path": str(DATASET_ROOT / "yolo"),
        "train": "images/train",
        "val": "images/val",
        "nc": 2,
        "names": {0: "Bactrocera dorsalis", 1: "Bactrocera correcta"},
    }
    atomic_write_yaml(RESULT_ROOT / "preflight/train_val_yolo.yaml", train_val_yolo)
    coco_view = RESULT_ROOT / "preflight/train_val_coco"
    (coco_view / "annotations").mkdir(parents=True, exist_ok=True)
    for name, target in {
        "train2017": DATASET_ROOT / "coco/train2017",
        "val2017": DATASET_ROOT / "coco/val2017",
        "annotations/instances_train2017.json": DATASET_ROOT / "coco/annotations/instances_train2017.json",
        "annotations/instances_val2017.json": DATASET_ROOT / "coco/annotations/instances_val2017.json",
    }.items():
        link = coco_view / name
        link.symlink_to(target)

    protocol = {
        "protocol_version": 3,
        "created_at": now_iso(),
        "budget_type": "fixed_100_epoch_screening",
        "description": "统一 100 epoch 固定预算、单随机种子的首轮工程筛选",
        "imgsz": 640,
        "epochs": TARGET_EPOCHS,
        "seed": 42,
        "amp": True,
        "workers": 8,
        "target_effective_batch": TARGET_EFFECTIVE_BATCH,
        "drop_last": False,
        "validation_every_epoch": True,
        "early_stopping": False,
        "train_images": TRAIN_IMAGES,
        "val_images": VAL_IMAGES,
        "test_images": TEST_IMAGES,
        "dataset_fingerprint": EXPECTED_DATASET_FINGERPRINT,
        "test_access": "forbidden_until_all_terminal_and_checkpoints_frozen",
    }
    atomic_write_yaml(RESULT_ROOT / "protocol.yaml", protocol)

    decisions = """# Protocol decisions

- AdamW models use the same linear LR scaling rule as SGD models, as explicitly required by the frozen prompt.
- The official nominal weight decay is retained after setting actual effective batch to 16; YOLO nbs is set to 16 to prevent hidden accumulation and weight-decay rescaling.
- YOLOv8n, YOLOv10n, YOLO11n and YOLO12n checkpoint metadata records their official recipe as `optimizer=auto`; the corresponding official source rules resolve long COCO schedules to SGD, so the benchmark explicitly freezes SGD. YOLO26n metadata explicitly records MuSGD and retains it.
- Model-specific official recipe metadata in the local official checkpoint takes precedence over a later repository-wide default when it is available (notably YOLOv8n weight decay and YOLO26n optimizer/LR).
- DEIM-N has `flat_epoch=7800` with a 160-epoch budget and `lr_gamma=1.0`; its effective official in-budget schedule is warmup followed by constant LR. The mapped schedule preserves that behavior and maps only the active EMA restart stage.
- RT-DETR-R18 has milestone 1000 with a 72-epoch budget; no LR decay occurs in the official run, so the mapped 100-epoch schedule contains no in-budget milestone.
- Step-based warmup and EMA ramps are mapped by their fraction of official total COCO optimizer steps, using 118,287 official COCO training images and the documented official effective batch.
- Faster R-CNN keeps aspect grouping for full batches and combines residual groups only for the final partial batch, ensuring all 8,143 images are used exactly once without duplication or dropping.
- Strong augmentation is disabled from epoch 1 for every model. All corresponding no-augmentation nodes are `N/A_ALREADY_DISABLED`; LR and EMA stages that remain active are still mapped.
- Test annotations and metrics remain inaccessible to training workers. Test evaluation is a separate post-freeze stage.
"""
    atomic_write_text(RESULT_ROOT / "protocol_decisions.md", decisions)
    atomic_write_text(RESULT_ROOT / "summary/environment_changes.md", "# Environment changes\n\nNone at protocol creation.\n")
    atomic_write_text(RESULT_ROOT / "summary/failures_and_repairs.md", "# Failures and repairs\n\nNone at protocol creation.\n")

    scheduler_rows = []
    batch_rows = []
    manifest_files = [
        RESULT_ROOT / "protocol.yaml",
        RESULT_ROOT / "protocol_decisions.md",
        RESULT_ROOT / "preflight/train_val_yolo.yaml",
        DATASET_ROOT / "coco/annotations/instances_train2017.json",
        DATASET_ROOT / "coco/annotations/instances_val2017.json",
    ]
    for model, spec in MODEL_SPECS.items():
        directory = run_dir(model)
        for child in ("checkpoints", "predictions", "evaluation", "speed", "native"):
            (directory / child).mkdir(parents=True, exist_ok=True)
        model_dataset_files = []
        if spec["family"].startswith("legacy_yolo") or spec["family"] in {
            "ultralytics_current", "ultralytics_yolov10", "ultralytics_yolo12"
        }:
            train_list = directory / "train_images.txt"
            val_list = directory / "val_images.txt"
            train_paths = sorted(path for path in (DATASET_ROOT / "yolo/images/train").iterdir() if path.is_file())
            val_paths = sorted(path for path in (DATASET_ROOT / "yolo/images/val").iterdir() if path.is_file())
            atomic_write_text(train_list, "\n".join(map(str, train_paths)) + "\n")
            atomic_write_text(val_list, "\n".join(map(str, val_paths)) + "\n")
            if spec["family"].startswith("legacy_yolo"):
                dataset_view = {
                    "train": str(train_list),
                    "val": str(val_list),
                    "nc": 2,
                    "names": {0: "Bactrocera dorsalis", 1: "Bactrocera correcta"},
                }
            else:
                view_root = directory / "dataset_view"
                (view_root / "images").mkdir(parents=True, exist_ok=True)
                for split in ("train", "val"):
                    (view_root / f"images/{split}").symlink_to(DATASET_ROOT / f"yolo/images/{split}", target_is_directory=True)
                    label_view = view_root / f"labels/{split}"
                    label_view.mkdir(parents=True, exist_ok=True)
                    for label in sorted((DATASET_ROOT / f"yolo/labels/{split}").glob("*.txt")):
                        (label_view / label.name).symlink_to(label)
                dataset_view = {
                    "path": str(view_root),
                    "train": "images/train",
                    "val": "images/val",
                    "nc": 2,
                    "names": {0: "Bactrocera dorsalis", 1: "Bactrocera correcta"},
                }
            atomic_write_yaml(directory / "train_val_dataset.yaml", dataset_view)
            model_dataset_files = [train_list, val_list, directory / "train_val_dataset.yaml"]
        official = official_schedule(spec)
        mapped = mapped_schedule(spec)
        resolved = resolved_config(model, spec, mapped)
        resolution = {
            "official_base_lr": spec["base_lr"],
            "official_reference_effective_batch": spec["reference_batch"],
            "target_effective_batch": TARGET_EFFECTIVE_BATCH,
            "scaling_ratio": TARGET_EFFECTIVE_BATCH / spec["reference_batch"],
            "target_base_lr": resolved["resolved_base_lr"],
            "scaling_source": "local official config/checkpoint metadata and documented reference batch",
            "planned_micro_batch": resolved["micro_batch"],
            "planned_accumulation": resolved["accumulation"],
            "framework_auto_scaled_weight_decay": False,
            "official_weight_decay": spec["weight_decay"],
            "resolved_weight_decay": spec["weight_decay"],
            "resolution_notes": "nbs=16 for YOLO; no secondary LR or weight-decay scaling is allowed at runtime.",
        }
        if "backbone_lr" in spec:
            resolution["official_backbone_lr"] = spec["backbone_lr"]
            resolution["target_backbone_lr"] = resolved["resolved_backbone_lr"]
        atomic_write_yaml(directory / "official_schedule.yaml", official)
        atomic_write_yaml(directory / "mapped_schedule_e100.yaml", mapped)
        atomic_write_yaml(directory / "resolved_config.yaml", resolved)
        atomic_write_yaml(directory / "batch_lr_resolution.yaml", resolution)
        runtime_filename, runtime_config = runtime_config_for(model, resolved)
        atomic_write_yaml(directory / runtime_filename, runtime_config)

        rows = lr_rows(spec, mapped, resolved)
        write_csv(directory / "lr_schedule.csv", ["epoch", "lr", "factor"], rows)
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.plot([row["epoch"] for row in rows], [row["lr"] for row in rows])
        axis.set(title=f"{spec['display']} mapped LR", xlabel="Epoch", ylabel="Learning rate")
        axis.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(directory / "lr_schedule.png", dpi=160)
        plt.close(figure)

        command = (
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            f"exec /home/admin1/.conda/envs/bench_detection_torch201/bin/python {BENCHMARK_ROOT / 'e100_pipeline/train_worker.py'} "
            f"--model {model} --physical-gpu \"${{PHYSICAL_GPU_ID:?}}\"\n"
        )
        atomic_write_text(directory / "command.sh", command)
        os.chmod(directory / "command.sh", os.stat(directory / "command.sh").st_mode | stat.S_IXUSR)
        atomic_write_text(directory / "source_commit.txt", spec["commit"] + "\n")
        weight_path = BENCHMARK_ROOT / spec["weight"]
        atomic_write_text(directory / "weight_sha256.txt", f"{sha256_file(weight_path)}  {weight_path}\n")
        atomic_write_json(
            directory / "status.json",
            {
                "model": model,
                "display": spec["display"],
                "status": "pending_preflight",
                "updated_at": now_iso(),
                "rng_state_saved": True,
            },
        )
        manifest_files.extend(
            [
                directory / "official_schedule.yaml",
                directory / "mapped_schedule_e100.yaml",
                directory / "resolved_config.yaml",
                directory / "batch_lr_resolution.yaml",
                directory / runtime_filename,
                directory / "lr_schedule.csv",
                *model_dataset_files,
            ]
        )
        scheduler_rows.append(
            {
                "model": spec["display"],
                "official_total_epochs": spec["official_epochs"],
                "official_scheduler": spec["scheduler"],
                "official_milestones": spec.get("official_milestones", ""),
                "mapped_milestones": mapped.get("mapped_milestones", ""),
                "official_warmup": spec.get("warmup_epochs", spec.get("warmup_steps", "")),
                "mapped_warmup": mapped.get("mapped_warmup_epochs", mapped.get("mapped_warmup_steps", "")),
                "official_no_aug": spec.get("no_aug_epoch", "strong augment in official recipe"),
                "mapped_no_aug": "N/A_ALREADY_DISABLED",
                "official_ema_stage": spec.get("ema_restart_epoch", spec.get("ema_warmup_steps", "")),
                "mapped_ema_stage": mapped.get("mapped_ema_restart_epoch", mapped.get("mapped_ema_warmup_steps", "")),
                "mapping_notes": mapped.get("mapping_notes", "relative stage mapping with minimum one warmup epoch"),
            }
        )
        batch_rows.append(
            {
                "model": spec["display"],
                "official_reference_batch": spec["reference_batch"],
                "official_base_lr": spec["base_lr"],
                "target_effective_batch": TARGET_EFFECTIVE_BATCH,
                "target_base_lr": resolved["resolved_base_lr"],
                "scaling_ratio": TARGET_EFFECTIVE_BATCH / spec["reference_batch"],
                "micro_batch": resolved["micro_batch"],
                "accumulation": resolved["accumulation"],
                "official_weight_decay": spec["weight_decay"],
                "resolved_weight_decay": spec["weight_decay"],
            }
        )

    write_csv(
        RESULT_ROOT / "scheduler_mapping.csv",
        [
            "model", "official_total_epochs", "official_scheduler", "official_milestones", "mapped_milestones",
            "official_warmup", "mapped_warmup", "official_no_aug", "mapped_no_aug", "official_ema_stage",
            "mapped_ema_stage", "mapping_notes",
        ],
        scheduler_rows,
    )
    write_csv(
        RESULT_ROOT / "batch_lr_mapping.csv",
        [
            "model", "official_reference_batch", "official_base_lr", "target_effective_batch", "target_base_lr",
            "scaling_ratio", "micro_batch", "accumulation", "official_weight_decay", "resolved_weight_decay",
        ],
        batch_rows,
    )
    manifest_files.extend([RESULT_ROOT / "scheduler_mapping.csv", RESULT_ROOT / "batch_lr_mapping.csv"])
    manifest_lines = []
    for path in sorted(manifest_files):
        label = path.relative_to(RESULT_ROOT) if path.is_relative_to(RESULT_ROOT) else path
        manifest_lines.append(f"{sha256_file(path)}  {label}")
    atomic_write_text(RESULT_ROOT / "configuration_manifest.sha256", "\n".join(manifest_lines) + "\n")

    queue = {
        "schema_version": 1,
        "created_at": now_iso(),
        "initial_assignments": {str(key): value for key, value in INITIAL_ASSIGNMENTS.items()},
        "fifo": FIFO_MODELS,
        "models": {model: {"status": "pending_preflight", "claims": 0} for model in MODEL_SPECS},
    }
    atomic_write_json(RESULT_ROOT / "scheduler/queue.json", queue)
    atomic_write_json(
        RESULT_ROOT / "scheduler/pipeline_status.json",
        {"status": "preflight", "updated_at": now_iso(), "completed": [], "running": [], "pending": list(MODEL_SPECS), "failed": []},
    )
    for gpu in (0, 1):
        atomic_write_json(
            RESULT_ROOT / f"scheduler/gpu_{gpu}_state.json",
            {"physical_gpu_id": gpu, "status": "idle", "model": None, "pid": None, "updated_at": now_iso()},
        )


if __name__ == "__main__":
    prepare()

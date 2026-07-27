#!/usr/bin/env python3
"""Static preflight validation for the frozen 11-model protocol."""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

from common import (
    DATASET_ROOT,
    EXPECTED_DATASET_FINGERPRINT,
    MODEL_SPECS,
    RESULT_ROOT,
    TARGET_EFFECTIVE_BATCH,
    TARGET_EPOCHS,
    TRAIN_IMAGES,
    VAL_IMAGES,
    atomic_write_json,
    load_resolved_config,
    now_iso,
    read_yaml,
    run_dir,
    sha256_file,
)
from config_builders import runtime_config_for


def check_manifest() -> list[str]:
    failures = []
    for line in (RESULT_ROOT / "configuration_manifest.sha256").read_text(encoding="utf-8").splitlines():
        expected, label = line.split(maxsplit=1)
        path = Path(label) if label.startswith("/") else RESULT_ROOT / label
        if not path.exists() or sha256_file(path) != expected:
            failures.append(f"manifest:{label}")
    return failures


def git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate() -> dict[str, Any]:
    failures = check_manifest()
    dataset_validation = json.loads((DATASET_ROOT / "metadata/validation_report.json").read_text(encoding="utf-8"))
    dataset_stats = json.loads((DATASET_ROOT / "metadata/dataset_stats.json").read_text(encoding="utf-8"))
    dataset_checks = {
        "validation_pass": dataset_validation["status"] == "PASS" and all(dataset_validation["checks"].values()),
        "fingerprint": dataset_stats["dataset_fingerprint_sha256"] == EXPECTED_DATASET_FINGERPRINT,
        "train_images": dataset_stats["by_split"]["train"]["images"] == TRAIN_IMAGES,
        "val_images": dataset_stats["by_split"]["val"]["images"] == VAL_IMAGES,
        "classes": dataset_stats["class_names"] == {"0": "Bactrocera dorsalis", "1": "Bactrocera correcta"},
    }
    failures.extend(f"dataset:{key}" for key, passed in dataset_checks.items() if not passed)

    isolated_yolo = read_yaml(RESULT_ROOT / "preflight/train_val_yolo.yaml")
    isolation_checks = {
        "yolo_has_no_test_key": "test" not in isolated_yolo,
        "faster_view_has_no_test": not any("test" in path.name.lower() for path in (RESULT_ROOT / "preflight/train_val_coco").rglob("*")),
        "faster_train_exists": (RESULT_ROOT / "preflight/train_val_coco/train2017").exists(),
        "faster_val_exists": (RESULT_ROOT / "preflight/train_val_coco/val2017").exists(),
    }
    failures.extend(f"isolation:{key}" for key, passed in isolation_checks.items() if not passed)

    model_reports = {}
    for model, spec in MODEL_SPECS.items():
        config = load_resolved_config(model)
        runtime_name, derived_runtime = runtime_config_for(model, config)
        frozen_runtime = read_yaml(run_dir(model) / runtime_name)
        checks = {
            "source_commit": git_head(Path(config["source_dir"])) == spec["commit"],
            "weight_exists": Path(config["weight"]).is_file(),
            "weight_hash_record": sha256_file(Path(config["weight"])) in (run_dir(model) / "weight_sha256.txt").read_text(),
            "runtime_config_exact": frozen_runtime == derived_runtime,
            "epochs": config["epochs"] == TARGET_EPOCHS,
            "imgsz": config["imgsz"] == 640,
            "seed": config["seed"] == 42,
            "effective_batch": config["micro_batch"] * config["accumulation"] == TARGET_EFFECTIVE_BATCH,
            "accumulation": config["accumulation"] == 1,
            "drop_last": config["drop_last"] is False,
            "base_lr": math.isclose(
                config["resolved_base_lr"],
                config["official_base_lr"] * TARGET_EFFECTIVE_BATCH / config["official_reference_batch"],
                rel_tol=0,
                abs_tol=1e-15,
            ),
            "weight_decay": config["resolved_weight_decay"] == config["official_weight_decay"],
            "weak_augmentation_only": all(
                config["augmentation"][key] in (False, 0, 0.0)
                for key in (
                    "mosaic", "mixup", "copy_paste", "random_erasing", "vertical_flip", "multi_scale",
                    "test_time_augmentation", "offline_augmentation", "hsv", "random_crop", "random_zoom",
                    "photometric_distort",
                )
            ) and config["augmentation"]["horizontal_flip_probability"] == 0.5,
            "environment_exists": (Path("/home/admin1/.conda/envs") / spec["env"] / "bin/python").is_file(),
            "no_formal_checkpoint": not any((run_dir(model) / "checkpoints").iterdir()),
        }
        if spec["family"].startswith("legacy_yolo") or spec["family"] in {
            "ultralytics_current", "ultralytics_yolov10", "ultralytics_yolo12"
        }:
            training_data = read_yaml(Path(config["training_dataset_config"]))
            checks["test_isolated"] = (
                Path(config["training_dataset_config"]).parent == run_dir(model)
                and "test" not in training_data
                and (
                    (
                        Path(training_data["train"]).parent == run_dir(model)
                        and Path(training_data["val"]).parent == run_dir(model)
                    )
                    or Path(training_data.get("path", "")).parent == run_dir(model)
                )
            )
        elif spec["family"] == "faster_rcnn":
            checks["test_isolated"] = config["training_dataset_root"] == str(RESULT_ROOT / "preflight/train_val_coco")
        else:
            source_data = read_yaml(Path(config["training_dataset_config"]))
            checks["test_isolated"] = "test_dataloader" not in source_data and "test" not in source_data
        failures.extend(f"{model}:{key}" for key, passed in checks.items() if not passed)
        model_reports[model] = checks

    gpu_output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()
    gpu_rows = []
    for line in gpu_output:
        index, name, total, free, utilization, temperature = [item.strip() for item in line.split(",")]
        gpu_rows.append(
            {
                "index": int(index),
                "name": name,
                "memory_total_mib": int(total),
                "memory_free_mib": int(free),
                "utilization_percent": int(utilization),
                "temperature_c": None if "N/A" in temperature else int(temperature),
            }
        )
    gpu_checks = {
        "two_gpus": len(gpu_rows) == 2,
        "idle": all(row["utilization_percent"] <= 5 for row in gpu_rows),
        "gpu0_capacity": gpu_rows[0]["memory_free_mib"] >= 70000,
        "gpu1_capacity": gpu_rows[1]["memory_free_mib"] >= 35000,
    }
    failures.extend(f"gpu:{key}" for key, passed in gpu_checks.items() if not passed)
    report = {
        "checked_at": now_iso(),
        "all_passed": not failures,
        "failures": failures,
        "dataset": dataset_checks,
        "test_isolation": isolation_checks,
        "models": model_reports,
        "gpus": gpu_rows,
        "gpu_checks": gpu_checks,
        "numactl_available": bool(subprocess.run(["bash", "-lc", "command -v numactl"], capture_output=True).stdout.strip()),
        "formal_test_access": "forbidden",
    }
    atomic_write_json(RESULT_ROOT / "preflight/static_validation.json", report)
    return report


if __name__ == "__main__":
    result = validate()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["all_passed"] else 1)

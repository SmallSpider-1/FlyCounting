#!/usr/bin/env python3
"""Create train/val-only tiny data and run one real train/validation batch per model."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from common import (
    BENCHMARK_ROOT,
    DATASET_ROOT,
    MODEL_SPECS,
    PIPELINE_ROOT,
    RESULT_ROOT,
    atomic_write_json,
    atomic_write_yaml,
    expected_config_hash,
    load_resolved_config,
    now_iso,
    read_json,
    read_yaml,
    run_dir,
)
from config_builders import deim_runtime_config, rtdetr_runtime_config


ENV_ROOT = Path("/home/admin1/.conda/envs")
DRYRUN_ROOT = RESULT_ROOT / "preflight/dryruns"
TINY_ROOT = RESULT_ROOT / "preflight/tiny_train_val"


def subset_coco(source: Path, destination: Path, count: int = 16) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    images = payload["images"][:count]
    ids = {image["id"] for image in images}
    payload["images"] = images
    payload["annotations"] = [annotation for annotation in payload["annotations"] if annotation["image_id"] in ids]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, destination)


def create_tiny_data() -> dict[str, Path]:
    TINY_ROOT.mkdir(parents=True, exist_ok=True)
    train_images = sorted((DATASET_ROOT / "yolo/images/train").glob("*.jpg"))[:16]
    val_images = sorted((DATASET_ROOT / "yolo/images/val").glob("*.jpg"))[:16]
    if len(train_images) != 16 or len(val_images) != 16:
        raise RuntimeError("Tiny dry-run requires exactly 16 selected train and val images")
    train_list = TINY_ROOT / "train.txt"
    val_list = TINY_ROOT / "val.txt"
    train_list.write_text("\n".join(map(str, train_images)) + "\n", encoding="utf-8")
    val_list.write_text("\n".join(map(str, val_images)) + "\n", encoding="utf-8")
    yolo_yaml = TINY_ROOT / "yolo_train_val.yaml"
    atomic_write_yaml(
        yolo_yaml,
        {
            "train": str(train_list),
            "val": str(val_list),
            "nc": 2,
            "names": {0: "Bactrocera dorsalis", 1: "Bactrocera correcta"},
        },
    )

    annotations = TINY_ROOT / "annotations"
    subset_coco(DATASET_ROOT / "coco/annotations/instances_train.json", annotations / "instances_train.json")
    subset_coco(DATASET_ROOT / "coco/annotations/instances_val.json", annotations / "instances_val.json")
    subset_coco(DATASET_ROOT / "coco/annotations/instances_train2017.json", annotations / "instances_train2017.json")
    subset_coco(DATASET_ROOT / "coco/annotations/instances_val2017.json", annotations / "instances_val2017.json")

    faster_root = TINY_ROOT / "faster_coco"
    (faster_root / "annotations").mkdir(parents=True, exist_ok=True)
    for name, target in {
        "train2017": DATASET_ROOT / "coco/train2017",
        "val2017": DATASET_ROOT / "coco/val2017",
        "annotations/instances_train2017.json": annotations / "instances_train2017.json",
        "annotations/instances_val2017.json": annotations / "instances_val2017.json",
    }.items():
        link = faster_root / name
        if not link.exists():
            link.symlink_to(target)

    deim_data = TINY_ROOT / "deim_data.yaml"
    atomic_write_yaml(
        deim_data,
        {
            "task": "detection",
            "evaluator": {"type": "CocoEvaluator", "iou_types": ["bbox"]},
            "num_classes": 2,
            "remap_mscoco_category": False,
            "train_dataloader": {
                "type": "DataLoader",
                "dataset": {
                    "type": "CocoDetection",
                    "img_folder": str(DATASET_ROOT / "coco/train"),
                    "ann_file": str(annotations / "instances_train.json"),
                    "return_masks": False,
                    "transforms": {"type": "Compose", "ops": None},
                },
                "shuffle": True,
                "num_workers": 8,
                "drop_last": False,
                "collate_fn": {"type": "BatchImageCollateFunction"},
            },
            "val_dataloader": {
                "type": "DataLoader",
                "dataset": {
                    "type": "CocoDetection",
                    "img_folder": str(DATASET_ROOT / "coco/val"),
                    "ann_file": str(annotations / "instances_val.json"),
                    "return_masks": False,
                    "transforms": {"type": "Compose", "ops": None},
                },
                "shuffle": False,
                "num_workers": 8,
                "drop_last": False,
                "collate_fn": {"type": "BatchImageCollateFunction"},
            },
        },
    )
    rtdetr_data = TINY_ROOT / "rtdetr_data.yaml"
    atomic_write_yaml(
        rtdetr_data,
        {
            "task": "detection",
            "num_classes": 2,
            "remap_mscoco_category": False,
            "train_dataloader": {
                "type": "DataLoader",
                "dataset": {
                    "type": "CocoDetection",
                    "img_folder": str(DATASET_ROOT / "coco/train"),
                    "ann_file": str(annotations / "instances_train.json"),
                    "transforms": {"type": "Compose", "ops": None},
                },
                "shuffle": True,
                "batch_size": 16,
                "num_workers": 8,
                "drop_last": False,
            },
            "val_dataloader": {
                "type": "DataLoader",
                "dataset": {
                    "type": "CocoDetection",
                    "img_folder": str(DATASET_ROOT / "coco/val"),
                    "ann_file": str(annotations / "instances_val.json"),
                    "transforms": {"type": "Compose", "ops": None},
                },
                "shuffle": False,
                "batch_size": 16,
                "num_workers": 8,
                "drop_last": False,
            },
        },
    )
    return {"yolo": yolo_yaml, "faster": faster_root, "deim": deim_data, "rtdetr": rtdetr_data}


def child_env(gpu: int, model: str) -> dict[str, str]:
    env = os.environ.copy()
    conda_lib = str(ENV_ROOT / MODEL_SPECS[model]["env"] / "lib")
    inherited_library_path = env.get("LD_LIBRARY_PATH", "")
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONNOUSERSITE": "1",
            "PIP_NO_INDEX": "1",
            "YOLO_AUTOINSTALL": "false",
            "WANDB_MODE": "disabled",
            "WANDB_DISABLED": "true",
            "COMET_MODE": "DISABLED",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HTTP_PROXY": "http://127.0.0.1:9",
            "HTTPS_PROXY": "http://127.0.0.1:9",
            "ALL_PROXY": "socks5://127.0.0.1:9",
            "http_proxy": "http://127.0.0.1:9",
            "https_proxy": "http://127.0.0.1:9",
            "all_proxy": "socks5://127.0.0.1:9",
            "LD_LIBRARY_PATH": conda_lib + (":" + inherited_library_path if inherited_library_path else ""),
        }
    )
    if MODEL_SPECS[model]["family"] != "ultralytics_yolov10":
        env["YOLO_OFFLINE"] = "true"
    else:
        env.pop("YOLO_OFFLINE", None)
    return env


def transformer_dry_config(model: str, data_config: Path, output_dir: Path) -> Path:
    config = load_resolved_config(model)
    runtime = deim_runtime_config(config, run_dir(model)) if model == "deim_dfine_n" else rtdetr_runtime_config(config, run_dir(model))
    runtime["__include__"][1] = str(data_config)
    runtime["output_dir"] = str(output_dir / "native/train")
    runtime["epoches"] = 1
    if model == "deim_dfine_n":
        runtime["flat_epoch"] = 1
        runtime["train_dataloader"]["collate_fn"]["stop_epoch"] = 1
        runtime["train_dataloader"]["collate_fn"]["mixup_epochs"] = [0, 0]
    path = output_dir / "runtime_config.yaml"
    atomic_write_yaml(path, runtime)
    return path


def model_yolo_data(shared_yaml: Path, output_dir: Path, directory_view: bool) -> Path:
    shared = read_yaml(shared_yaml)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_list = data_dir / "train_images.txt"
    val_list = data_dir / "val_images.txt"
    train_list.write_text(Path(shared["train"]).read_text(encoding="utf-8"), encoding="utf-8")
    val_list.write_text(Path(shared["val"]).read_text(encoding="utf-8"), encoding="utf-8")
    config = data_dir / "train_val.yaml"
    if directory_view:
        view_root = data_dir / "dataset_view"
        for split, paths in (("train", train_list), ("val", val_list)):
            image_dir = view_root / f"images/{split}"
            label_dir = view_root / f"labels/{split}"
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            for image_path in paths.read_text(encoding="utf-8").splitlines():
                image = Path(image_path)
                (image_dir / image.name).symlink_to(image)
                label = Path(str(image).replace(f"/images/{split}/", f"/labels/{split}/")).with_suffix(".txt")
                if label.exists():
                    (label_dir / label.name).symlink_to(label)
        payload = {"path": str(view_root), "train": "images/train", "val": "images/val", "nc": 2, "names": shared["names"]}
    else:
        payload = {"train": str(train_list), "val": str(val_list), "nc": 2, "names": shared["names"]}
    atomic_write_yaml(config, payload)
    return config


def dryrun_command(model: str, data: dict[str, Path], output_dir: Path) -> tuple[list[str], Path, int]:
    config = load_resolved_config(model)
    spec = MODEL_SPECS[model]
    python = ENV_ROOT / spec["env"] / "bin/python"
    family = spec["family"]
    gpu = 0 if family == "faster_rcnn" else 1
    yolo_data = model_yolo_data(data["yolo"], output_dir, not family.startswith("legacy_yolo")) if "yolo" in model else None
    if family in {"ultralytics_current", "ultralytics_yolov10", "ultralytics_yolo12"}:
        command = [
            str(python), str(PIPELINE_ROOT / "ultralytics_adapter.py"), "--model", model,
            "--data", str(yolo_data), "--output-dir", str(output_dir), "--epochs", "1",
            "--expected-images", "16", "--no-canonical",
        ]
        cwd = Path(config["source_dir"])
    elif family.startswith("legacy_yolo"):
        command = [
            str(python), str(PIPELINE_ROOT / "legacy_yolo_adapter.py"), "--model", model,
            "--data", str(yolo_data), "--output-dir", str(output_dir), "--epochs", "1",
            "--expected-images", "16",
        ]
        cwd = Path(config["source_dir"])
    elif family in {"deim", "rt_detr"}:
        runtime = transformer_dry_config(model, data["deim" if family == "deim" else "rtdetr"], output_dir)
        command = [
            str(python), str(PIPELINE_ROOT / "transformer_adapter.py"), "--model", model,
            "--runtime-config", str(runtime), "--output-dir", str(output_dir),
            "--expected-images", "16", "--no-canonical",
        ]
        cwd = Path(config["source_dir"])
    else:
        command = [
            str(python), str(BENCHMARK_ROOT / "11_faster_rcnn_r50_fpn/train_custom.py"),
            "--data-path", str(data["faster"]), "--dataset", "coco", "--model", "fasterrcnn_resnet50_fpn",
            "--device", "cuda", "--batch-size", "16", "--epochs", "1", "--workers", "8",
            "--opt", "sgd", "--lr", str(config["resolved_base_lr"]), "--momentum", str(config["momentum"]),
            "--weight-decay", str(config["resolved_weight_decay"]), "--lr-scheduler", "multisteplr",
            "--lr-steps", "62", "85", "--lr-gamma", "0.1", "--print-freq", "1",
            "--output-dir", str(output_dir / "native/train"), "--aspect-ratio-group-factor", "3",
            "--data-augmentation", "hflip", "--amp", "--num-foreground-classes", "2",
            "--initial-weights-file", config["weight"], "--benchmark-run-dir", str(output_dir),
            "--resolved-config-hash", expected_config_hash(model), "--warmup-iters", "0",
            "--min-size", "640", "--max-size", "640", "--seed", "42",
        ]
        cwd = BENCHMARK_ROOT / spec["project"]
    return command, cwd, gpu


def environment_snapshots() -> dict[str, Any]:
    reports = {}
    for env_name in sorted({spec["env"] for spec in MODEL_SPECS.values()}):
        python = ENV_ROOT / env_name / "bin/python"
        script = (
            "import json,sys,torch,torchvision,yaml; "
            "print(json.dumps({'python':sys.version,'torch':torch.__version__,'torchvision':torchvision.__version__,"
            "'yaml':yaml.__version__,'cuda_available':torch.cuda.is_available(),'cuda_count':torch.cuda.device_count()}))"
        )
        result = subprocess.run([str(python), "-c", script], check=True, capture_output=True, text=True)
        reports[env_name] = json.loads(result.stdout.strip().splitlines()[-1])
        freeze = subprocess.run([str(python), "-m", "pip", "freeze", "--all"], check=True, capture_output=True, text=True)
        (RESULT_ROOT / f"preflight/{env_name}_pip_freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    atomic_write_json(RESULT_ROOT / "preflight/environment_snapshots.json", reports)
    return reports


def validate_dryrun(model: str, output_dir: Path, return_code: int, gpu: int) -> dict[str, Any]:
    summary = read_json(output_dir / "adapter_summary.json", {})
    runtime = read_json(output_dir / "runtime_config_validation.json", {})
    metrics_path = output_dir / "metrics_history.csv"
    steps_path = output_dir / "optimizer_steps.csv"
    metrics = list(csv.DictReader(metrics_path.open(newline="", encoding="utf-8"))) if metrics_path.exists() else []
    steps = list(csv.DictReader(steps_path.open(newline="", encoding="utf-8"))) if steps_path.exists() else []
    checks = {
        "process_exit_zero": return_code == 0,
        "one_metric_epoch": len(metrics) == 1 and metrics[0].get("epoch") == "1",
        "one_optimizer_epoch": len(steps) == 1 and steps[0].get("epoch") == "1",
        "sixteen_images_seen": len(steps) == 1 and steps[0].get("images_seen") == "16",
        "one_microbatch": len(steps) == 1 and steps[0].get("micro_batches") == "1",
        "one_optimizer_step": len(steps) == 1 and steps[0].get("optimizer_steps") == "1",
        "runtime_checks": bool(runtime.get("checks")) and all(runtime["checks"].values()),
        "peak_memory_recorded": float(summary.get("peak_cuda_memory_mib", 0)) > 0,
    }
    return {
        "model": model,
        "gpu": gpu,
        "return_code": return_code,
        "passed": all(checks.values()),
        "checks": checks,
        "peak_cuda_memory_mib": summary.get("peak_cuda_memory_mib"),
        "runtime": runtime.get("actual", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=list(MODEL_SPECS))
    args = parser.parse_args()
    static = read_json(RESULT_ROOT / "preflight/static_validation.json", {})
    if not static.get("all_passed"):
        raise RuntimeError("Static validation must pass before GPU dry-runs")
    data = create_tiny_data()
    environments = environment_snapshots()
    DRYRUN_ROOT.mkdir(parents=True, exist_ok=True)
    reports = read_json(RESULT_ROOT / "preflight/dryrun_results.json", {})
    models = [args.only] if args.only else list(MODEL_SPECS)
    for model in models:
        if reports.get(model, {}).get("passed"):
            continue
        output_dir = DRYRUN_ROOT / model
        if output_dir.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir.rename(DRYRUN_ROOT / f"{model}.failed_{timestamp}")
        output_dir.mkdir(parents=True)
        command, cwd, gpu = dryrun_command(model, data, output_dir)
        log_path = output_dir / "dryrun.log"
        with log_path.open("w", encoding="utf-8", buffering=1) as log:
            log.write(f"[{now_iso()}] command={command!r}\n")
            process = subprocess.run(command, cwd=cwd, env=child_env(gpu, model), stdout=log, stderr=subprocess.STDOUT)
        report = validate_dryrun(model, output_dir, process.returncode, gpu)
        reports[model] = report
        atomic_write_json(RESULT_ROOT / "preflight/dryrun_results.json", reports)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        if not report["passed"]:
            print(f"Dry-run failed for {model}; see {log_path}", file=sys.stderr)
            return 1

    all_models_passed = all(reports.get(model, {}).get("passed") for model in MODEL_SPECS)
    final_report = {
        "created_at": now_iso(),
        "all_passed": bool(static.get("all_passed")) and all_models_passed,
        "static_validation": static,
        "environment_snapshots": environments,
        "dryruns": reports,
        "test_accessed_by_workers": False,
        "environment_changes": [],
    }
    atomic_write_json(RESULT_ROOT / "preflight/report.json", final_report)
    if final_report["all_passed"]:
        queue_path = RESULT_ROOT / "scheduler/queue.json"
        queue = read_json(queue_path)
        for model in MODEL_SPECS:
            queue["models"][model]["status"] = "ready"
            status = read_json(run_dir(model) / "status.json", {})
            status.update({"status": "ready", "updated_at": now_iso(), "preflight_passed": True})
            atomic_write_json(run_dir(model) / "status.json", status)
        queue["status"] = "ready"
        queue["updated_at"] = now_iso()
        atomic_write_json(queue_path, queue)
        atomic_write_json(
            RESULT_ROOT / "scheduler/pipeline_status.json",
            {"status": "ready", "updated_at": now_iso(), "completed": [], "running": [], "pending": list(MODEL_SPECS), "failed": []},
        )
    return 0 if final_report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

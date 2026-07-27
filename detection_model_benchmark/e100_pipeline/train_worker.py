#!/usr/bin/env python3
"""Execute one benchmark model on one physical GPU with strict resume semantics."""

from __future__ import annotations

import argparse
import csv
import fcntl
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from common import (
    BENCHMARK_ROOT,
    MODEL_SPECS,
    PIPELINE_ROOT,
    RESULT_ROOT,
    atomic_write_json,
    expected_config_hash,
    load_resolved_config,
    now_iso,
    run_dir,
    sha256_file,
)


CONDA_ROOT = Path("/home/admin1/.conda/envs")


def verify_manifest() -> None:
    manifest = RESULT_ROOT / "configuration_manifest.sha256"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing frozen configuration manifest: {manifest}")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = RESULT_ROOT / relative
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"Frozen configuration changed: {relative}; expected {expected}, got {actual}")


def status_path(model: str) -> Path:
    return run_dir(model) / "status.json"


def write_status(model: str, status: str, **extra: Any) -> None:
    payload = {
        "model": model,
        "display": MODEL_SPECS[model]["display"],
        "status": status,
        "updated_at": now_iso(),
        **extra,
    }
    atomic_write_json(status_path(model), payload)


def common_child_env(physical_gpu: int, model: str) -> dict[str, str]:
    environment = os.environ.copy()
    conda_lib = str(CONDA_ROOT / MODEL_SPECS[model]["env"] / "lib")
    inherited_library_path = environment.get("LD_LIBRARY_PATH", "")
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(physical_gpu),
            "BENCH_PHYSICAL_GPU_ID": str(physical_gpu),
            "BENCH_VISIBLE_GPU_ID": "0",
            "BENCH_MODEL": model,
            "PYTHONNOUSERSITE": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
            "YOLO_AUTOINSTALL": "false",
            "WANDB_MODE": "disabled",
            "WANDB_DISABLED": "true",
            "COMET_MODE": "DISABLED",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "NO_PROXY": "localhost,127.0.0.1",
            "no_proxy": "localhost,127.0.0.1",
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
        environment["YOLO_OFFLINE"] = "true"
    else:
        environment.pop("YOLO_OFFLINE", None)
    return environment


def faster_command(model: str, config: dict[str, Any], python: Path) -> list[str]:
    directory = run_dir(model)
    command = [
        str(python),
        str(BENCHMARK_ROOT / "11_faster_rcnn_r50_fpn/train_custom.py"),
        "--data-path", config["training_dataset_root"],
        "--dataset", "coco",
        "--model", "fasterrcnn_resnet50_fpn",
        "--device", "cuda",
        "--batch-size", str(config["micro_batch"]),
        "--epochs", str(config["epochs"]),
        "--workers", str(config["workers"]),
        "--opt", "sgd",
        "--lr", str(config["resolved_base_lr"]),
        "--momentum", str(config["momentum"]),
        "--weight-decay", str(config["resolved_weight_decay"]),
        "--lr-scheduler", "multisteplr",
        "--lr-steps", *[str(value) for value in config["scheduler"]["mapped_milestones"]],
        "--lr-gamma", "0.1",
        "--print-freq", "50",
        "--output-dir", str(directory / "native/train"),
        "--aspect-ratio-group-factor", "3",
        "--data-augmentation", "hflip",
        "--amp",
        "--num-foreground-classes", "2",
        "--initial-weights-file", str(config["weight"]),
        "--benchmark-run-dir", str(directory),
        "--resolved-config-hash", expected_config_hash(model),
        "--warmup-iters", str(config["scheduler"]["mapped_warmup_steps"]),
        "--min-size", "640",
        "--max-size", "640",
        "--seed", str(config["seed"]),
    ]
    canonical_last = directory / "checkpoints/last.pth"
    if canonical_last.exists():
        command.extend(["--resume", str(canonical_last)])
    return command


def build_command(model: str) -> tuple[list[str], Path]:
    spec = MODEL_SPECS[model]
    config = load_resolved_config(model)
    python = CONDA_ROOT / spec["env"] / "bin/python"
    if not python.exists():
        raise FileNotFoundError(f"Missing frozen environment interpreter: {python}")
    family = spec["family"]
    if family in {"ultralytics_current", "ultralytics_yolov10", "ultralytics_yolo12"}:
        command = [str(python), str(PIPELINE_ROOT / "ultralytics_adapter.py"), "--model", model]
        cwd = Path(config["source_dir"])
    elif family.startswith("legacy_yolo"):
        command = [str(python), str(PIPELINE_ROOT / "legacy_yolo_adapter.py"), "--model", model]
        cwd = Path(config["source_dir"])
    elif family in {"deim", "rt_detr"}:
        command = [str(python), str(PIPELINE_ROOT / "transformer_adapter.py"), "--model", model]
        cwd = Path(config["source_dir"])
    elif family == "faster_rcnn":
        command = faster_command(model, config, python)
        cwd = BENCHMARK_ROOT / spec["project"]
    else:
        raise KeyError(f"Unsupported model family: {family}")
    return command, cwd


def validate_completion(model: str) -> dict[str, Any]:
    directory = run_dir(model)
    metrics_path = directory / "metrics_history.csv"
    steps_path = directory / "optimizer_steps.csv"
    with metrics_path.open(newline="", encoding="utf-8") as stream:
        metrics = list(csv.DictReader(stream))
    with steps_path.open(newline="", encoding="utf-8") as stream:
        steps = list(csv.DictReader(stream))
    metric_epochs = [int(row["epoch"]) for row in metrics]
    step_epochs = [int(row["epoch"]) for row in steps]
    suffix = ".pth" if MODEL_SPECS[model]["family"] in {"deim", "rt_detr", "faster_rcnn"} else ".pt"
    required = [directory / f"checkpoints/{name}{suffix}" for name in ("last", "best")]
    required.extend(directory / f"checkpoints/epoch_{epoch:03d}{suffix}" for epoch in range(10, 101, 10))
    missing = [str(path) for path in required if not path.exists()]
    runtime = __import__("json").loads((directory / "runtime_config_validation.json").read_text(encoding="utf-8"))
    failed_checks = [key for key, passed in runtime["checks"].items() if not passed]
    if metric_epochs != list(range(1, 101)):
        raise RuntimeError(f"Metrics epochs are incomplete or duplicated: {metric_epochs[:5]} ... {metric_epochs[-5:]}")
    if step_epochs != list(range(1, 101)):
        raise RuntimeError(f"Optimizer audit epochs are incomplete or duplicated: {step_epochs[:5]} ... {step_epochs[-5:]}")
    if missing:
        raise RuntimeError(f"Missing canonical checkpoints: {missing}")
    if failed_checks:
        raise RuntimeError(f"Runtime validation contains failed checks: {failed_checks}")
    return {
        "epochs": 100,
        "best_val_map50_95": max(float(row["val_map50_95"]) for row in metrics),
        "global_optimizer_steps": sum(int(row["optimizer_steps"]) for row in steps),
        "checkpoint_suffix": suffix,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_SPECS))
    parser.add_argument("--physical-gpu", required=True, type=int, choices=[0, 1])
    args = parser.parse_args()
    model = args.model
    directory = run_dir(model)
    lock_path = directory / ".worker.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_stream = lock_path.open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f"A worker already owns {model}")

    child: subprocess.Popen | None = None

    def forward_signal(signum, _frame):
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGTERM, forward_signal)
    signal.signal(signal.SIGINT, forward_signal)
    try:
        verify_manifest()
        config = load_resolved_config(model)
        if expected_config_hash(model) != sha256_file(directory / "resolved_config.yaml"):
            raise RuntimeError("Resolved config hash is not stable")
        command, cwd = build_command(model)
        assignment = {
            "model": model,
            "physical_gpu_id": args.physical_gpu,
            "visible_gpu_id_inside_process": 0,
            "worker_pid": os.getpid(),
            "environment": MODEL_SPECS[model]["env"],
            "command": command,
            "cwd": str(cwd),
            "started_at": now_iso(),
        }
        atomic_write_json(directory / "gpu_assignment.json", assignment)
        write_status(
            model,
            "running",
            physical_gpu_id=args.physical_gpu,
            visible_gpu_id=0,
            worker_pid=os.getpid(),
            config_hash=expected_config_hash(model),
        )
        log_path = directory / "train.log"
        environment = common_child_env(args.physical_gpu, model)
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            log.write(f"\n[{now_iso()}] worker_pid={os.getpid()} command={command!r}\n")
            child = subprocess.Popen(
                command,
                cwd=cwd,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=False,
            )
            write_status(
                model,
                "running",
                physical_gpu_id=args.physical_gpu,
                visible_gpu_id=0,
                worker_pid=os.getpid(),
                training_pid=child.pid,
                config_hash=expected_config_hash(model),
            )
            return_code = child.wait()
            log.write(f"[{now_iso()}] training process return_code={return_code}\n")
        if return_code != 0:
            suffix = ".pth" if config["family"] in {"deim", "rt_detr", "faster_rcnn"} else ".pt"
            resumable = (directory / f"checkpoints/last{suffix}").exists()
            write_status(
                model,
                "resume_required" if resumable else "failed",
                return_code=return_code,
                resumable=resumable,
                physical_gpu_id=args.physical_gpu,
            )
            return return_code
        completion = validate_completion(model)
        write_status(model, "complete", physical_gpu_id=args.physical_gpu, **completion)
        return 0
    except Exception as error:
        with (directory / "train.log").open("a", encoding="utf-8") as log:
            log.write(traceback.format_exc())
        suffix = ".pth" if MODEL_SPECS[model]["family"] in {"deim", "rt_detr", "faster_rcnn"} else ".pt"
        resumable = (directory / f"checkpoints/last{suffix}").exists()
        write_status(
            model,
            "resume_required" if resumable else "failed",
            error=repr(error),
            resumable=resumable,
            physical_gpu_id=args.physical_gpu,
        )
        return 1
    finally:
        fcntl.flock(lock_stream, fcntl.LOCK_UN)
        lock_stream.close()


if __name__ == "__main__":
    sys.exit(main())

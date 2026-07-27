#!/usr/bin/env python3
"""Rerun frozen test inference into an isolated audit result root.

The original benchmark result tree is treated as read-only.  Checkpoints and
frozen runtime configuration are exposed to the audit worker through symlinks;
all predictions, native framework artifacts, caches, metadata, and logs are
written below the caller-provided audit root.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PIPELINE_ROOT = Path(__file__).resolve().parent
ORIGINAL_ROOT = Path(
    "/home/admin1/Projects/ultralytics-main/detection_model_benchmark/"
    "results/fruitfly_detection_v1/benchmark_640_e100_s42"
)
ENV_ROOT = Path("/home/admin1/.conda/envs")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def expose_frozen_inputs(audit_root: Path, model_key: str, run_name: str) -> None:
    source = ORIGINAL_ROOT / "runs" / run_name
    destination = audit_root / "runs" / run_name
    destination.mkdir(parents=True, exist_ok=True)
    for relative in ("checkpoints", "resolved_config.yaml", "runtime_config.yaml"):
        source_path = source / relative
        if not source_path.exists():
            continue
        destination_path = destination / relative
        if destination_path.is_symlink():
            if destination_path.resolve() != source_path.resolve():
                raise RuntimeError(f"Unexpected audit symlink: {destination_path}")
        elif destination_path.exists():
            raise RuntimeError(f"Audit input path already exists and is not a symlink: {destination_path}")
        else:
            destination_path.symlink_to(source_path, target_is_directory=source_path.is_dir())


def worker(audit_root: Path, model_key: str) -> None:
    sys.path.insert(0, str(PIPELINE_ROOT))
    import common

    common.RESULT_ROOT = audit_root.resolve()
    config = common.load_resolved_config(model_key)
    family = common.MODEL_SPECS[model_key]["family"]
    source_dir = Path(config["source_dir"])
    sys.path.insert(0, str(source_dir))
    os.chdir(source_dir)

    # Evaluation entry points in several YOLO generations reinterpret
    # device=0 as a physical index and overwrite the worker's CUDA mask.
    # Preload and patch only their device selector so cuda:0 remains the one
    # device exposed by the controller.
    if family.startswith("legacy_yolo"):
        import utils.torch_utils as torch_utils_module

        original_select_device = torch_utils_module.select_device

        def audit_select_device(device="", *select_args: Any, **select_kwargs: Any) -> Any:
            return original_select_device("", *select_args, **select_kwargs)

        torch_utils_module.select_device = audit_select_device
    elif family in {"ultralytics_current", "ultralytics_yolov10", "ultralytics_yolo12"}:
        import ultralytics.engine.validator as validator_module
        import ultralytics.utils.torch_utils as torch_utils_module

        original_select_device = torch_utils_module.select_device

        def audit_select_device(device="", *select_args: Any, **select_kwargs: Any) -> Any:
            return original_select_device("", *select_args, **select_kwargs)

        torch_utils_module.select_device = audit_select_device
        validator_module.select_device = audit_select_device

    import postprocess_worker

    checkpoint = postprocess_worker.best_checkpoint(model_key)
    output = postprocess_worker.prediction_path(model_key, "test")
    print(
        f"INFERENCE_START model={model_key} split=test checkpoint={checkpoint} "
        f"checkpoint_sha256={common.sha256_file(checkpoint)} output={output}",
        flush=True,
    )
    postprocess_worker.predict(model_key, "test", force=True)
    prediction_rows = len(json.loads(output.read_text(encoding="utf-8")))
    print(
        f"INFERENCE_DONE model={model_key} split=test rows={prediction_rows} "
        f"bytes={output.stat().st_size} prediction_sha256={common.sha256_file(output)}",
        flush=True,
    )


def controller(audit_root: Path, physical_gpu: int, resume: bool) -> int:
    sys.path.insert(0, str(PIPELINE_ROOT))
    import common

    if audit_root.exists() and not resume:
        raise FileExistsError(f"Audit root already exists; use --resume to continue: {audit_root}")
    audit_root.mkdir(parents=True, exist_ok=True)
    logs = audit_root / "logs"
    commands = audit_root / "commands"
    logs.mkdir(exist_ok=True)
    commands.mkdir(exist_ok=True)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "Rerun all 11 frozen best checkpoints on corrected test images without overwriting original outputs",
        "started_at": now_iso(),
        "original_result_root": str(ORIGINAL_ROOT),
        "audit_root": str(audit_root.resolve()),
        "physical_gpu": physical_gpu,
        "models": {},
    }
    manifest_path = audit_root / "inference_manifest.json"
    atomic_json(manifest_path, manifest)

    failed = False
    for model_key, spec in common.MODEL_SPECS.items():
        run_name = spec["run_name"]
        expose_frozen_inputs(audit_root, model_key, run_name)
        output = audit_root / "runs" / run_name / "predictions" / "test_mixed.json"
        if resume and output.exists():
            manifest["models"][model_key] = {
                "status": "kept_existing_on_resume",
                "output": str(output),
                "completed_at": now_iso(),
            }
            atomic_json(manifest_path, manifest)
            continue

        python = ENV_ROOT / spec["env"] / "bin" / "python"
        if not python.exists():
            raise FileNotFoundError(python)
        command = [
            str(python),
            "-u",
            str(Path(__file__).resolve()),
            "--worker",
            "--audit-root",
            str(audit_root.resolve()),
            "--model",
            model_key,
        ]
        command_path = commands / f"{run_name}.sh"
        command_path.write_text(" ".join(command) + "\n", encoding="utf-8")
        log_path = logs / f"{run_name}_test_inference.log"
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(physical_gpu),
                "BENCH_PHYSICAL_GPU_ID": str(physical_gpu),
                "BENCH_VISIBLE_GPU_ID": "0",
                "PYTHONUNBUFFERED": "1",
                "YOLO_AUTOINSTALL": "false",
                "YOLO_OFFLINE": "true",
                "WANDB_MODE": "disabled",
                "COMET_MODE": "DISABLED",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        environment["LD_LIBRARY_PATH"] = str(ENV_ROOT / spec["env"] / "lib") + (
            ":" + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else ""
        )
        started_at = now_iso()
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8", buffering=1) as stream:
            stream.write(f"AUDIT_MODEL={model_key}\n")
            stream.write(f"DISPLAY_NAME={spec['display']}\n")
            stream.write(f"STARTED_AT={started_at}\n")
            stream.write(f"PHYSICAL_GPU={physical_gpu}\n")
            stream.write(f"CUDA_VISIBLE_DEVICES={physical_gpu}\n")
            stream.write(f"CONDA_ENV={spec['env']}\n")
            stream.write(f"PYTHON={python}\n")
            stream.write(f"COMMAND={' '.join(command)}\n")
            stream.write("--- MODEL OUTPUT BEGIN ---\n")
            process = subprocess.run(
                command,
                cwd=str(PIPELINE_ROOT),
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            elapsed = time.monotonic() - started
            stream.write("--- MODEL OUTPUT END ---\n")
            stream.write(f"RETURN_CODE={process.returncode}\n")
            stream.write(f"ELAPSED_SECONDS={elapsed:.3f}\n")
            stream.write(f"FINISHED_AT={now_iso()}\n")
        status = "complete" if process.returncode == 0 and output.exists() else "failed"
        failed = failed or status != "complete"
        manifest["models"][model_key] = {
            "display": spec["display"],
            "status": status,
            "environment": spec["env"],
            "physical_gpu": physical_gpu,
            "started_at": started_at,
            "completed_at": now_iso(),
            "elapsed_seconds": round(elapsed, 3),
            "return_code": process.returncode,
            "command_file": str(command_path),
            "log": str(log_path),
            "output": str(output),
            "output_bytes": output.stat().st_size if output.exists() else None,
        }
        atomic_json(manifest_path, manifest)

    manifest["completed_at"] = now_iso()
    manifest["status"] = "failed" if failed else "complete"
    atomic_json(manifest_path, manifest)
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model")
    args = parser.parse_args()
    if args.worker:
        if not args.model:
            parser.error("--model is required with --worker")
        worker(args.audit_root, args.model)
    else:
        raise SystemExit(controller(args.audit_root.resolve(), args.physical_gpu, args.resume))


if __name__ == "__main__":
    main()

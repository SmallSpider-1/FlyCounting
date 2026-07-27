#!/usr/bin/env python3
"""Persistent two-GPU FIFO scheduler for the fixed benchmark queue."""

from __future__ import annotations

import csv
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from common import (
    FIFO_MODELS,
    INITIAL_ASSIGNMENTS,
    MODEL_SPECS,
    PIPELINE_ROOT,
    RESULT_ROOT,
    atomic_write_json,
    now_iso,
    read_json,
    run_dir,
)


OUTER_PYTHON = Path("/home/admin1/.conda/envs/bench_detection_torch201/bin/python")
SERVICE_NAME = "fruitfly-benchmark-e100-s42.service"
MIN_FREE_MIB = {
    "faster_rcnn_r50_fpn": 30000,
    "rt_detr_r18": 30000,
    "deim_dfine_n": 30000,
    "yolov7_tiny": 14000,
    "yolov9_t": 16000,
    "yolo12n": 18000,
    "yolo26n": 14000,
    "yolov10n": 14000,
    "yolo11n": 12000,
    "yolov8n": 12000,
    "yolov5n": 10000,
}
MAX_CONSECUTIVE_FAILURES = 3
READY_STATES = {"ready", "pending", "pending_preflight", "resume_required"}
FAILED_STATES = {"failed", "env_conflict_blocked", "protocol_resolution_blocked", "invalid_protocol"}
FORMAL_PROCESS_MARKERS = (
    "e100_pipeline/train_worker.py",
    "e100_pipeline/ultralytics_adapter.py",
    "e100_pipeline/legacy_yolo_adapter.py",
    "e100_pipeline/transformer_adapter.py",
    "11_faster_rcnn_r50_fpn/train_custom.py",
)


def scheduler_log(event: str, **details: Any) -> None:
    payload = {"at": now_iso(), "event": event, "scheduler_pid": os.getpid(), **details}
    with (RESULT_ROOT / "scheduler/scheduler.log").open("a", encoding="utf-8", buffering=1) as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def process_command(pid: int | None) -> str:
    if not pid:
        return ""
    try:
        return (Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return ""


def pid_matches_worker(pid: int | None, model: str) -> bool:
    command = process_command(pid)
    return "e100_pipeline/train_worker.py" in command and f"--model {model}" in command


def formal_processes() -> dict[int, str]:
    processes: dict[int, str] = {}
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        pid = int(item.name)
        if pid == os.getpid():
            continue
        command = process_command(pid)
        if any(marker in command for marker in FORMAL_PROCESS_MARKERS):
            processes[pid] = command
    return processes


def free_gpu_memory() -> dict[int, int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    values = {}
    for line in result.stdout.splitlines():
        index, memory = [item.strip() for item in line.split(",")]
        values[int(index)] = int(memory)
    return values


def state_for(model: str) -> str:
    return read_json(run_dir(model) / "status.json", {}).get("status", "pending")


def checkpoint_suffix(model: str) -> str:
    return ".pth" if MODEL_SPECS[model]["family"] in {"deim", "rt_detr", "faster_rcnn"} else ".pt"


def recorded_epochs(model: str) -> int:
    path = run_dir(model) / "metrics_history.csv"
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as stream:
        epochs = [int(row["epoch"]) for row in csv.DictReader(stream)]
    return max(epochs, default=0)


def completion_errors(model: str) -> list[str]:
    """Validate enough evidence to safely skip a model marked complete."""
    directory = run_dir(model)
    errors: list[str] = []
    for filename in ("metrics_history.csv", "optimizer_steps.csv"):
        path = directory / filename
        if not path.exists():
            errors.append(f"missing {filename}")
            continue
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        epochs = [int(row["epoch"]) for row in rows]
        if epochs != list(range(1, 101)):
            errors.append(f"{filename} does not contain exactly epochs 1..100")
    runtime = read_json(directory / "runtime_config_validation.json", {})
    if not runtime or not runtime.get("checks") or not all(runtime["checks"].values()):
        errors.append("runtime_config_validation is missing or failed")
    suffix = checkpoint_suffix(model)
    names = ["last", "best", *[f"epoch_{epoch:03d}" for epoch in range(10, 101, 10)]]
    for name in names:
        path = directory / "checkpoints" / f"{name}{suffix}"
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"missing or empty checkpoint {path.name}")
    return errors


def write_gpu_state(gpu: int, status: str, **extra: Any) -> None:
    atomic_write_json(
        RESULT_ROOT / f"scheduler/gpu_{gpu}_state.json",
        {"physical_gpu_id": gpu, "status": status, "updated_at": now_iso(), **extra},
    )


def write_daemon_state(status: str, **extra: Any) -> None:
    path = RESULT_ROOT / "scheduler/daemon.json"
    current = read_json(path, {})
    payload = {
        "pid": os.getpid(),
        "service": SERVICE_NAME,
        "started_at": current.get("started_at", now_iso()),
        "heartbeat_at": now_iso(),
        "status": status,
        **extra,
    }
    atomic_write_json(path, payload)


class Scheduler:
    def __init__(self) -> None:
        self.stop = False
        self.queue_path = RESULT_ROOT / "scheduler/queue.json"
        self.queue = read_json(self.queue_path)
        if not self.queue or "models" not in self.queue:
            raise RuntimeError(f"Missing or invalid queue: {self.queue_path}")
        self.active: dict[int, dict[str, Any]] = {}

    def save_queue(self, status: str = "training") -> None:
        self.queue["status"] = status
        self.queue["updated_at"] = now_iso()
        atomic_write_json(self.queue_path, self.queue)

    def touch_model_heartbeat(self, model: str, worker_pid: int) -> None:
        path = run_dir(model) / "status.json"
        payload = read_json(path, {})
        if payload.get("status") == "running" and payload.get("worker_pid") == worker_pid:
            payload["heartbeat_at"] = now_iso()
            payload["scheduler_pid"] = os.getpid()
            atomic_write_json(path, payload)

    def pipeline_snapshot(self, status: str = "training") -> None:
        states = {model: state_for(model) for model in MODEL_SPECS}
        payload = {
            "status": status,
            "scheduler_pid": os.getpid(),
            "service": SERVICE_NAME,
            "updated_at": now_iso(),
            "completed": [model for model, value in states.items() if value == "complete"],
            "running": [model for model, value in states.items() if value == "running"],
            "pending": [model for model, value in states.items() if value in READY_STATES - {"resume_required"}],
            "resume_required": [model for model, value in states.items() if value == "resume_required"],
            "failed": [model for model, value in states.items() if value in FAILED_STATES],
        }
        atomic_write_json(RESULT_ROOT / "scheduler/pipeline_status.json", payload)
        write_daemon_state("running", active={str(gpu): data["model"] for gpu, data in self.active.items()})

    def _mark_stale_running(self, model: str, payload: dict[str, Any]) -> str:
        suffix = checkpoint_suffix(model)
        resumable = (run_dir(model) / "checkpoints" / f"last{suffix}").exists()
        replacement = "resume_required" if resumable else "ready"
        payload.update(
            {
                "status": replacement,
                "resumable": resumable,
                "updated_at": now_iso(),
                "reconciled_from_stale_running": True,
                "stale_worker_pid": payload.get("worker_pid"),
            }
        )
        payload.pop("worker_pid", None)
        payload.pop("training_pid", None)
        atomic_write_json(run_dir(model) / "status.json", payload)
        scheduler_log("stale_model_state_reconciled", model=model, replacement=replacement, resumable=resumable)
        return replacement

    def reconcile_disk_state(self) -> None:
        """Rebuild queue/GPU state from model status, artifacts and real PIDs."""
        known_processes: set[int] = set()
        for model in MODEL_SPECS:
            payload = read_json(run_dir(model) / "status.json", {})
            state = payload.get("status", "pending")
            queue_state = self.queue["models"].setdefault(model, {"claims": 0, "status": "ready"})
            queue_state.setdefault("claims", 0)
            queue_state.setdefault("consecutive_failures", 0)
            if state == "complete":
                errors = completion_errors(model)
                if errors:
                    raise RuntimeError(f"Refusing to skip incomplete model {model}: {errors}")
                queue_state["status"] = "complete"
                queue_state["consecutive_failures"] = 0
                continue
            if state == "running":
                worker_pid = payload.get("worker_pid")
                training_pid = payload.get("training_pid")
                if pid_matches_worker(worker_pid, model):
                    gpu = payload.get("physical_gpu_id")
                    if gpu not in (0, 1):
                        raise RuntimeError(f"Running model {model} has invalid GPU {gpu}")
                    if gpu in self.active:
                        raise RuntimeError(f"Two model states claim physical GPU {gpu}")
                    self.active[gpu] = {"model": model, "worker_pid": worker_pid, "process": None}
                    known_processes.add(worker_pid)
                    if pid_alive(training_pid):
                        known_processes.add(training_pid)
                    queue_state["status"] = "running"
                    write_gpu_state(gpu, "running", model=model, worker_pid=worker_pid, recovered=True)
                    scheduler_log("running_worker_recovered", model=model, gpu=gpu, worker_pid=worker_pid)
                    continue
                if pid_alive(training_pid):
                    raise RuntimeError(
                        f"Orphan training process {training_pid} exists for {model}; refusing duplicate launch"
                    )
                state = self._mark_stale_running(model, payload)
            if state in READY_STATES:
                queue_state["status"] = "resume_required" if state == "resume_required" else "ready"
            elif state in FAILED_STATES:
                queue_state["status"] = state
            else:
                raise RuntimeError(f"Unknown model state {state!r} for {model}")

        unknown = {pid: command for pid, command in formal_processes().items() if pid not in known_processes}
        if unknown:
            raise RuntimeError(f"Untracked formal benchmark processes found; refusing launch: {unknown}")
        for gpu in (0, 1):
            if gpu not in self.active:
                write_gpu_state(gpu, "idle", model=None, worker_pid=None, reconciled=True)
        self.save_queue()
        scheduler_log(
            "disk_state_reconciled",
            completed=[model for model in MODEL_SPECS if state_for(model) == "complete"],
            active={str(gpu): data["model"] for gpu, data in self.active.items()},
        )

    def eligible_model(self, gpu: int, free_mib: int) -> str | None:
        active_models = {entry["model"] for entry in self.active.values()}
        initial = INITIAL_ASSIGNMENTS.get(gpu)
        if initial:
            queue_state = self.queue["models"][initial]
            if (
                initial not in active_models
                and queue_state["status"] in {"ready", "resume_required"}
                and queue_state["consecutive_failures"] < MAX_CONSECUTIVE_FAILURES
                and state_for(initial) in {"ready", "resume_required"}
                and free_mib >= MIN_FREE_MIB[initial]
            ):
                return initial
        for model in FIFO_MODELS:
            queue_state = self.queue["models"][model]
            if model in active_models:
                continue
            if queue_state["status"] not in {"ready", "resume_required"}:
                continue
            if state_for(model) not in {"ready", "resume_required"}:
                continue
            if queue_state["consecutive_failures"] >= MAX_CONSECUTIVE_FAILURES:
                continue
            if free_mib >= MIN_FREE_MIB[model]:
                return model
        return None

    def launch(self, gpu: int, model: str) -> None:
        if gpu in self.active:
            raise RuntimeError(f"Physical GPU {gpu} already has an active worker")
        if model in {entry["model"] for entry in self.active.values()}:
            raise RuntimeError(f"Model {model} is already active")
        command = [
            str(OUTER_PYTHON),
            str(PIPELINE_ROOT / "train_worker.py"),
            "--model",
            model,
            "--physical-gpu",
            str(gpu),
        ]
        log_path = RESULT_ROOT / f"scheduler/worker_{model}.log"
        log = log_path.open("a", encoding="utf-8", buffering=1)
        log.write(f"\n[{now_iso()}] scheduler_pid={os.getpid()} launch={command!r}\n")
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        queue_state = self.queue["models"][model]
        queue_state["claims"] += 1
        queue_state["status"] = "running"
        queue_state["last_gpu"] = gpu
        queue_state["last_claimed_at"] = now_iso()
        queue_state["last_claim_start_epoch"] = recorded_epochs(model)
        self.active[gpu] = {"model": model, "worker_pid": process.pid, "process": process, "log": log}
        write_gpu_state(gpu, "running", model=model, worker_pid=process.pid, claim=queue_state["claims"])
        self.save_queue()
        scheduler_log("worker_launched", model=model, gpu=gpu, worker_pid=process.pid, claim=queue_state["claims"])

    def reap(self) -> None:
        for gpu, active in list(self.active.items()):
            process = active.get("process")
            worker_pid = active["worker_pid"]
            finished = process.poll() is not None if process is not None else not pid_matches_worker(worker_pid, active["model"])
            if not finished:
                write_gpu_state(gpu, "running", model=active["model"], worker_pid=worker_pid)
                self.touch_model_heartbeat(active["model"], worker_pid)
                continue
            if active.get("log"):
                active["log"].close()
            model = active["model"]
            state = state_for(model)
            claims = self.queue["models"][model]["claims"]
            queue_state = self.queue["models"][model]
            if state == "complete":
                errors = completion_errors(model)
                if errors:
                    raise RuntimeError(f"Worker marked {model} complete with invalid artifacts: {errors}")
                replacement = "complete"
                queue_state["consecutive_failures"] = 0
            else:
                start_epoch = int(queue_state.get("last_claim_start_epoch", 0))
                current_epoch = recorded_epochs(model)
                made_progress = current_epoch > start_epoch
                previous_failures = int(queue_state.get("consecutive_failures", 0))
                consecutive_failures = 1 if made_progress else previous_failures + 1
                queue_state["consecutive_failures"] = consecutive_failures
                queue_state["last_exit_epoch"] = current_epoch
                queue_state["last_exit_made_progress"] = made_progress
                replacement = (
                    "resume_required"
                    if consecutive_failures < MAX_CONSECUTIVE_FAILURES
                    else "failed"
                )
                payload = read_json(run_dir(model) / "status.json", {})
                payload.update(
                    {
                        "status": replacement,
                        "updated_at": now_iso(),
                        "scheduler_retry": claims + 1,
                        "consecutive_failures": consecutive_failures,
                        "last_claim_start_epoch": start_epoch,
                        "last_exit_epoch": current_epoch,
                    }
                )
                atomic_write_json(run_dir(model) / "status.json", payload)
            self.queue["models"][model]["status"] = replacement
            return_code = process.returncode if process is not None else None
            write_gpu_state(gpu, "idle", model=None, worker_pid=None, previous_model=model)
            del self.active[gpu]
            self.save_queue()
            scheduler_log(
                "worker_exited",
                model=model,
                gpu=gpu,
                worker_pid=worker_pid,
                return_code=return_code,
                resulting_status=replacement,
            )

    def run(self) -> int:
        preflight = read_json(RESULT_ROOT / "preflight/report.json", {})
        if not preflight.get("all_passed"):
            raise RuntimeError("Formal scheduler refuses to start before all preflight checks and dry-runs pass")
        self.reconcile_disk_state()
        self.pipeline_snapshot()
        while not self.stop:
            self.reap()
            states = [state_for(model) for model in MODEL_SPECS]
            if all(value == "complete" or value in FAILED_STATES for value in states):
                all_complete = all(value == "complete" for value in states)
                final_status = "training_complete" if all_complete else "training_terminal_with_failures"
                self.save_queue(final_status)
                self.pipeline_snapshot(final_status)
                return 0 if all_complete else 2
            memory = free_gpu_memory()
            for gpu in (0, 1):
                if gpu in self.active:
                    continue
                model = self.eligible_model(gpu, memory.get(gpu, 0))
                if model:
                    self.launch(gpu, model)
                else:
                    write_gpu_state(gpu, "idle", model=None, worker_pid=None, free_memory_mib=memory.get(gpu))
            self.pipeline_snapshot()
            time.sleep(10)
        self.save_queue("scheduler_stopping")
        self.pipeline_snapshot("scheduler_stopping")
        return 0


def main() -> int:
    lock_path = RESULT_ROOT / "scheduler/scheduler.lock"
    lock_stream = lock_path.open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Scheduler is already running", file=sys.stderr)
        return 1
    scheduler = Scheduler()
    scheduler.stop = False
    signal.signal(signal.SIGTERM, lambda *_: setattr(scheduler, "stop", True))
    signal.signal(signal.SIGINT, lambda *_: setattr(scheduler, "stop", True))
    started_at = now_iso()
    atomic_write_json(
        RESULT_ROOT / "scheduler/daemon.json",
        {
            "pid": os.getpid(),
            "service": SERVICE_NAME,
            "started_at": started_at,
            "heartbeat_at": started_at,
            "status": "starting",
        },
    )
    scheduler_log("scheduler_started", service=SERVICE_NAME)
    exit_code = 1
    error: Exception | None = None
    try:
        exit_code = scheduler.run()
        return exit_code
    except Exception as caught:
        error = caught
        scheduler_log("scheduler_crashed", error=repr(caught), traceback=traceback.format_exc())
        return 1
    finally:
        final_status = "crashed" if error else ("stopped" if scheduler.stop else ("completed" if exit_code == 0 else "stopped"))
        write_daemon_state(
            final_status,
            exit_code=exit_code,
            stopped_at=now_iso(),
            error=repr(error) if error else None,
        )
        scheduler_log("scheduler_exited", status=final_status, exit_code=exit_code, error=repr(error) if error else None)
        fcntl.flock(lock_stream, fcntl.LOCK_UN)
        lock_stream.close()


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Protocol adapter for DEIM-D-FINE-N and RT-DETR-R18."""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common import atomic_write_json, expected_config_hash, load_resolved_config, read_yaml, run_dir
from config_builders import deim_runtime_config, rtdetr_runtime_config


METRIC_FIELDS = ["epoch", "val_map50_95", "val_ap50", "val_ap75"]
STEP_FIELDS = [
    "epoch",
    "images_seen",
    "micro_batches",
    "optimizer_steps",
    "micro_batch",
    "accumulation",
    "effective_batch",
    "last_batch_size",
]


def torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def append_csv_once(path: Path, fields: list[str], row: dict[str, Any]) -> None:
    if any(item.get("epoch") == str(row["epoch"]) for item in csv_rows(path)):
        return
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def bbox_stats(test_stats: dict[str, Any]) -> list[float]:
    for key, values in test_stats.items():
        if "bbox" in key.lower():
            return [float(value) for value in values]
    if len(test_stats) == 1:
        return [float(value) for value in next(iter(test_stats.values()))]
    raise KeyError(f"Cannot identify COCO bbox stats in {list(test_stats)}")


def restore_rng(checkpoint: dict[str, Any]) -> None:
    state = checkpoint.get("rng_state")
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])


class ProtocolRecorder:
    def __init__(self, model: str, output_dir: Path, expected_images: int, canonical: bool):
        self.model = model
        self.config = load_resolved_config(model)
        self.output_dir = output_dir
        self.expected_images = expected_images
        self.canonical = canonical
        self.best = max(
            (float(row["val_map50_95"]) for row in csv_rows(output_dir / "metrics_history.csv")),
            default=-1.0,
        )

    def validate_runtime(self, solver: Any) -> None:
        loader = solver.train_dataloader
        size = len(loader.dataset)
        batch = int(loader.batch_size)
        lrs = [float(group["lr"]) for group in solver.optimizer.param_groups]
        initial_lrs = [float(group.get("initial_lr", group["lr"])) for group in solver.optimizer.param_groups]
        weight_decays = [float(group.get("weight_decay", 0.0)) for group in solver.optimizer.param_groups]
        expected_main = float(self.config["resolved_base_lr"])
        expected_backbone = float(self.config["resolved_backbone_lr"])
        checks = {
            "train_images": size == self.expected_images,
            "micro_batch": batch == int(self.config["micro_batch"]),
            "dataloader_batches": len(loader) == int(np.ceil(size / batch)),
            "drop_last": not bool(loader.drop_last),
            "optimizer": type(solver.optimizer).__name__.lower() == "adamw",
            "main_lr": any(abs(value - expected_main) < 1e-12 for value in initial_lrs),
            "backbone_lr": any(abs(value - expected_backbone) < 1e-12 for value in initial_lrs),
            "weight_decay": abs(max(weight_decays) - float(self.config["resolved_weight_decay"])) < 1e-12,
        }
        audit = {
            "model": self.model,
            "checks": checks,
            "actual": {
                "optimizer_class": type(solver.optimizer).__name__,
                "lrs": lrs,
                "initial_lrs": initial_lrs,
                "weight_decays": weight_decays,
                "train_images": size,
                "batch_size": batch,
                "dataloader_batches": len(loader),
                "drop_last": bool(loader.drop_last),
                "ema_decay": float(solver.ema.decay) if solver.ema is not None else None,
                "ema_warmups": int(solver.ema.warmups) if solver.ema is not None and hasattr(solver.ema, "warmups") else None,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            },
        }
        atomic_write_json(self.output_dir / "runtime_config_validation.json", audit)
        failed = [key for key, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(f"Runtime protocol validation failed: {failed}")

    def record(self, solver: Any, epoch: int, stats: list[float], state: dict[str, Any]) -> None:
        actual_epoch = epoch + 1
        append_csv_once(
            self.output_dir / "metrics_history.csv",
            METRIC_FIELDS,
            {"epoch": actual_epoch, "val_map50_95": stats[0], "val_ap50": stats[1], "val_ap75": stats[2]},
        )
        loader = solver.train_dataloader
        append_csv_once(
            self.output_dir / "optimizer_steps.csv",
            STEP_FIELDS,
            {
                "epoch": actual_epoch,
                "images_seen": self.expected_images,
                "micro_batches": len(loader),
                "optimizer_steps": len(loader),
                "micro_batch": self.config["micro_batch"],
                "accumulation": 1,
                "effective_batch": self.config["effective_batch"],
                "last_batch_size": self.expected_images % int(self.config["micro_batch"]) or int(self.config["micro_batch"]),
            },
        )
        global_steps = sum(int(row["optimizer_steps"]) for row in csv_rows(self.output_dir / "optimizer_steps.csv"))
        state.update(
            {
                "resolved_config_hash": expected_config_hash(self.model),
                "global_optimizer_step": global_steps,
                "benchmark_best_metric": "val_mAP50_95",
                "benchmark_val_mAP50_95": stats[0],
                "rng_state": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all(),
                },
            }
        )
        native_dir = self.output_dir / "native/train"
        atomic_torch_save(state, native_dir / "last.pth")
        if not self.canonical:
            return
        checkpoints = self.output_dir / "checkpoints"
        atomic_torch_save(state, checkpoints / "last.pth")
        if stats[0] > self.best:
            self.best = stats[0]
            atomic_torch_save(state, checkpoints / "best.pth")
        if actual_epoch % 10 == 0:
            atomic_torch_save(state, checkpoints / f"epoch_{actual_epoch:03d}.pth")


def deim_fit(solver: Any, recorder: ProtocolRecorder) -> None:
    from engine.misc import dist_utils, stats as model_stats
    from engine.optim.lr_scheduler import FlatCosineLRScheduler
    from engine.solver.det_engine import evaluate, train_one_epoch

    solver.train()
    if solver.cfg.resume:
        restore_rng(torch_load(Path(solver.cfg.resume)))
    recorder.validate_runtime(solver)
    _, description = model_stats(solver.cfg)
    print(description)
    args = solver.cfg
    solver.self_lr_scheduler = True
    solver.lr_scheduler = FlatCosineLRScheduler(
        solver.optimizer,
        args.lr_gamma,
        len(solver.train_dataloader),
        total_epochs=args.epoches,
        warmup_iter=args.warmup_iter,
        flat_epochs=args.flat_epoch,
        no_aug_epochs=args.no_aug_epoch,
    )
    start = time.time()
    stage_epoch = int(solver.train_dataloader.collate_fn.stop_epoch)
    for epoch in range(solver.last_epoch + 1, args.epoches):
        solver.train_dataloader.set_epoch(epoch)
        if dist_utils.is_dist_available_and_initialized():
            solver.train_dataloader.sampler.set_epoch(epoch)
        if epoch == stage_epoch and recorder.canonical:
            stage_best = recorder.output_dir / "checkpoints/best.pth"
            if stage_best.exists():
                solver.load_resume_state(str(stage_best))
                solver.ema.decay = solver.train_dataloader.collate_fn.ema_restart_decay
                print(f"Mapped DEIM EMA restart from pure stage-1 best at epoch {epoch}")
        train_one_epoch(
            True,
            solver.lr_scheduler,
            solver.model,
            solver.criterion,
            solver.train_dataloader,
            solver.optimizer,
            solver.device,
            epoch,
            max_norm=args.clip_max_norm,
            print_freq=args.print_freq,
            ema=solver.ema,
            scaler=solver.scaler,
            lr_warmup_scheduler=solver.lr_warmup_scheduler,
            writer=solver.writer,
        )
        solver.last_epoch = epoch
        module = solver.ema.module if solver.ema else solver.model
        test_stats, _ = evaluate(
            module,
            solver.criterion,
            solver.postprocessor,
            solver.val_dataloader,
            solver.evaluator,
            solver.device,
        )
        recorder.record(solver, epoch, bbox_stats(test_stats), solver.state_dict())
    print(f"Training time: {time.time() - start:.1f}s")


def rtdetr_fit(solver: Any, recorder: ProtocolRecorder) -> None:
    from src.data import get_coco_api_from_dataset
    from src.misc import dist
    from src.solver.det_engine import evaluate, train_one_epoch

    solver.train()
    if solver.cfg.resume:
        restore_rng(torch_load(Path(solver.cfg.resume)))
    recorder.validate_runtime(solver)
    args = solver.cfg
    base_ds = get_coco_api_from_dataset(solver.val_dataloader.dataset)
    start = time.time()
    for epoch in range(solver.last_epoch + 1, args.epoches):
        if dist.is_dist_available_and_initialized():
            solver.train_dataloader.sampler.set_epoch(epoch)
        train_one_epoch(
            solver.model,
            solver.criterion,
            solver.train_dataloader,
            solver.optimizer,
            solver.device,
            epoch,
            args.clip_max_norm,
            print_freq=args.log_step,
            ema=solver.ema,
            scaler=solver.scaler,
        )
        solver.lr_scheduler.step()
        solver.last_epoch = epoch
        module = solver.ema.module if solver.ema else solver.model
        test_stats, _ = evaluate(
            module,
            solver.criterion,
            solver.postprocessor,
            solver.val_dataloader,
            base_ds,
            solver.device,
            solver.output_dir,
        )
        recorder.record(solver, epoch, bbox_stats(test_stats), solver.state_dict(epoch))
    print(f"Training time: {time.time() - start:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["deim_dfine_n", "rt_detr_r18"])
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-images", type=int)
    parser.add_argument("--no-canonical", action="store_true")
    args = parser.parse_args()

    config = load_resolved_config(args.model)
    output_dir = args.output_dir or run_dir(args.model)
    output_dir.mkdir(parents=True, exist_ok=True)

    # The default file_descriptor tensor-sharing backend can fail while the
    # DEIM validation loader transfers batches with
    # "received 0 items of ancdata". The file_system backend preserves the
    # same 8-worker data pipeline and tensors without Unix FD handoff.
    torch.multiprocessing.set_sharing_strategy("file_system")
    runtime_path = args.runtime_config or (run_dir(args.model) / "runtime_config.yaml")
    if args.runtime_config is None:
        derived = (
            deim_runtime_config(config, run_dir(args.model))
            if args.model == "deim_dfine_n"
            else rtdetr_runtime_config(config, run_dir(args.model))
        )
        if read_yaml(runtime_path) != derived:
            raise RuntimeError(f"Frozen transformer config differs from derived protocol: {runtime_path}")

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    source_dir = Path(config["source_dir"])
    sys.path.insert(0, str(source_dir))
    os.chdir(source_dir)
    recorder = ProtocolRecorder(
        args.model,
        output_dir,
        args.expected_images or int(config["train_images"]),
        not args.no_canonical,
    )
    canonical_last = output_dir / "checkpoints/last.pth"
    resume = str(canonical_last) if not args.no_canonical and canonical_last.exists() else None
    tuning = None if resume else config["weight"]

    if args.model == "deim_dfine_n":
        from engine.core import YAMLConfig
        from engine.misc import dist_utils
        from engine.solver import TASKS

        dist_utils.setup_distributed(0, "builtin", seed=int(config["seed"]))
        cfg = YAMLConfig(
            str(runtime_path),
            resume=resume,
            tuning=tuning,
            use_amp=True,
            output_dir=str(output_dir / "native/train"),
            device="cuda",
            seed=int(config["seed"]),
        )
        solver = TASKS[cfg.yaml_cfg["task"]](cfg)
        deim_fit(solver, recorder)
        dist_utils.cleanup()
    else:
        from src.core import YAMLConfig
        from src.misc import dist
        from src.solver import TASKS

        dist.init_distributed()
        dist.set_seed(int(config["seed"]))
        cfg = YAMLConfig(str(runtime_path), resume=resume, tuning=tuning, use_amp=True)
        cfg.output_dir = str(output_dir / "native/train")
        solver = TASKS[cfg.yaml_cfg["task"]](cfg)
        rtdetr_fit(solver, recorder)
    atomic_write_json(
        output_dir / "adapter_summary.json",
        {
            "model": args.model,
            "epochs_requested": int(cfg.epoches),
            "expected_images": recorder.expected_images,
            "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
            "canonical": recorder.canonical,
        },
    )


if __name__ == "__main__":
    main()

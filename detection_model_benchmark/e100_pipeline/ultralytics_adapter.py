#!/usr/bin/env python3
"""Protocol adapter for the five Ultralytics-based benchmark models."""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common import atomic_write_json, expected_config_hash, load_resolved_config, read_yaml, run_dir
from config_builders import ultralytics_overrides


METRIC_FIELDS = ["epoch", "val_map50_95", "val_ap50", "val_precision", "val_recall"]
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
    """Load a trusted local training checkpoint across PyTorch default changes."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def append_csv_once(path: Path, fields: list[str], row: dict[str, Any]) -> None:
    existing = read_csv_rows(path)
    epoch = str(row["epoch"])
    if any(item.get("epoch") == epoch for item in existing):
        return
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def metric_value(metrics: dict[str, Any], *candidates: str) -> float:
    for key in candidates:
        if key in metrics:
            value = metrics[key]
            if hasattr(value, "item"):
                value = value.item()
            return float(value)
    raise KeyError(f"None of the metric keys {candidates} exist; got {sorted(metrics)}")


def dataset_length(loader: Any) -> int:
    dataset = loader.dataset
    while hasattr(dataset, "dataset") and not hasattr(dataset, "im_files"):
        dataset = dataset.dataset
    return len(dataset)


class ProtocolCallbacks:
    """Audit runtime semantics and create framework-independent checkpoints."""

    def __init__(self, model_key: str, output_dir: Path, canonical: bool, expected_images: int, target_epochs: int):
        self.model_key = model_key
        self.config = load_resolved_config(model_key)
        self.output_dir = output_dir
        self.canonical = canonical
        self.expected_images = expected_images
        self.target_epochs = target_epochs
        self.hash = expected_config_hash(model_key)
        self.epoch_optimizer_steps = 0
        self.global_optimizer_steps = sum(
            int(row["optimizer_steps"]) for row in read_csv_rows(output_dir / "optimizer_steps.csv")
        )
        historical = read_csv_rows(output_dir / "metrics_history.csv")
        self.best_map = max((float(row["val_map50_95"]) for row in historical), default=-1.0)

    def register(self, model: Any) -> None:
        model.add_callback("on_pretrain_routine_end", self.on_pretrain_routine_end)
        model.add_callback("on_train_epoch_start", self.on_train_epoch_start)
        model.add_callback("on_fit_epoch_end", self.on_fit_epoch_end)

    def on_pretrain_routine_end(self, trainer: Any) -> None:
        actual_train_images = dataset_length(trainer.train_loader)
        actual_batch = int(getattr(trainer.train_loader, "batch_size", 0) or trainer.batch_size)
        actual_accumulation = int(trainer.accumulate)
        optimizer_name = type(trainer.optimizer).__name__
        initial_lrs = [float(group.get("initial_lr", group["lr"])) for group in trainer.optimizer.param_groups]
        weight_decays = [float(group.get("weight_decay", 0.0)) for group in trainer.optimizer.param_groups]
        momentums = [
            float(group["momentum"] if "momentum" in group else group.get("betas", (0.0, 0.0))[0])
            for group in trainer.optimizer.param_groups
        ]
        expected_optimizer = str(self.config["optimizer"]).lower()
        optimizer_ok = expected_optimizer in optimizer_name.lower()
        if expected_optimizer == "musgd":
            optimizer_ok = "musgd" in optimizer_name.lower()
        checks = {
            "train_images": actual_train_images == self.expected_images,
            "micro_batch": actual_batch == int(self.config["micro_batch"]),
            "accumulation": actual_accumulation == int(self.config["accumulation"]),
            "optimizer": optimizer_ok,
            "base_lr": any(abs(value - float(self.config["resolved_base_lr"])) < 1e-12 for value in initial_lrs),
            "weight_decay": abs(max(weight_decays) - float(self.config["resolved_weight_decay"])) < 1e-12,
            "nbs": int(trainer.args.nbs) == 16,
            "drop_last": not bool(getattr(trainer.train_loader, "drop_last", False)),
        }
        audit = {
            "model": self.model_key,
            "checks": checks,
            "actual": {
                "optimizer_class": optimizer_name,
                "initial_lrs": initial_lrs,
                "weight_decays": weight_decays,
                "momentums_or_beta1": momentums,
                "train_images": actual_train_images,
                "micro_batch": actual_batch,
                "accumulation": actual_accumulation,
                "dataloader_batches": len(trainer.train_loader),
                "drop_last": bool(getattr(trainer.train_loader, "drop_last", False)),
                "torch_cudnn_benchmark": torch.backends.cudnn.benchmark,
                "torch_cudnn_deterministic": torch.backends.cudnn.deterministic,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            },
        }
        atomic_write_json(self.output_dir / "runtime_config_validation.json", audit)
        failures = [key for key, passed in checks.items() if not passed]
        if failures:
            raise RuntimeError(f"Runtime protocol validation failed: {failures}; details in {self.output_dir}")

        original_optimizer_step = trainer.optimizer_step

        def counted_optimizer_step() -> None:
            original_optimizer_step()
            self.epoch_optimizer_steps += 1
            self.global_optimizer_steps += 1

        trainer.optimizer_step = counted_optimizer_step

    def on_train_epoch_start(self, trainer: Any) -> None:
        self.epoch_optimizer_steps = 0

    def on_fit_epoch_end(self, trainer: Any) -> None:
        # Ultralytics calls this callback once more during final best-checkpoint validation.
        epoch = int(trainer.epoch) + 1
        if epoch < 1 or epoch > self.target_epochs:
            return
        metrics = dict(trainer.metrics)
        val_map = metric_value(metrics, "metrics/mAP50-95(B)", "metrics/mAP50-95")
        val_ap50 = metric_value(metrics, "metrics/mAP50(B)", "metrics/mAP50")
        precision = metric_value(metrics, "metrics/precision(B)", "metrics/precision")
        recall = metric_value(metrics, "metrics/recall(B)", "metrics/recall")
        append_csv_once(
            self.output_dir / "metrics_history.csv",
            METRIC_FIELDS,
            {
                "epoch": epoch,
                "val_map50_95": val_map,
                "val_ap50": val_ap50,
                "val_precision": precision,
                "val_recall": recall,
            },
        )
        micro_batches = len(trainer.train_loader)
        micro_batch = int(self.config["micro_batch"])
        last_batch = self.expected_images % micro_batch or micro_batch
        append_csv_once(
            self.output_dir / "optimizer_steps.csv",
            STEP_FIELDS,
            {
                "epoch": epoch,
                "images_seen": self.expected_images,
                "micro_batches": micro_batches,
                "optimizer_steps": self.epoch_optimizer_steps,
                "micro_batch": micro_batch,
                "accumulation": self.config["accumulation"],
                "effective_batch": self.config["effective_batch"],
                "last_batch_size": last_batch,
            },
        )

        if not self.canonical:
            return
        native_last = Path(trainer.last)
        if not native_last.exists():
            raise FileNotFoundError(f"Native checkpoint was not saved before callback: {native_last}")
        checkpoint = torch_load(native_last)
        checkpoint.update(
            {
                "resolved_config_hash": self.hash,
                "global_optimizer_step": self.global_optimizer_steps,
                "benchmark_best_metric": "val_mAP50_95",
                "benchmark_val_mAP50_95": val_map,
                "rng_state": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all(),
                },
            }
        )
        checkpoints = self.output_dir / "checkpoints"
        atomic_torch_save(checkpoint, checkpoints / "last.pt")
        if val_map > self.best_map:
            self.best_map = val_map
            atomic_torch_save(checkpoint, checkpoints / "best.pt")
        if epoch % 10 == 0:
            atomic_torch_save(checkpoint, checkpoints / f"epoch_{epoch:03d}.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--expected-images", type=int)
    parser.add_argument("--no-canonical", action="store_true")
    args = parser.parse_args()

    config = load_resolved_config(args.model)
    source_dir = Path(config["source_dir"])
    sys.path.insert(0, str(source_dir))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    if config["family"] != "ultralytics_yolov10":
        os.environ.setdefault("YOLO_OFFLINE", "true")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("COMET_MODE", "DISABLED")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from ultralytics import YOLO
    import ultralytics.data.dataset as dataset_module
    import ultralytics.engine.trainer as trainer_module
    import ultralytics.utils.torch_utils as torch_utils_module
    import ultralytics.data.augment as augment_module

    # Some bundled Ultralytics generations interpret device=0 as a physical
    # index and overwrite the worker's CUDA_VISIBLE_DEVICES mask. Inside a
    # single-GPU worker, device=0 is the framework-visible cuda:0; preserve the
    # external physical mask while leaving the frozen training argument intact.
    original_select_device = torch_utils_module.select_device

    def benchmark_select_device(device="", *select_args, **select_kwargs):
        if "BENCH_PHYSICAL_GPU_ID" in os.environ:
            return original_select_device("", *select_args, **select_kwargs)
        return original_select_device(device, *select_args, **select_kwargs)

    torch_utils_module.select_device = benchmark_select_device
    trainer_module.select_device = benchmark_select_device

    class BenchmarkNoOpAlbumentations:
        """Disable optional hidden blur/gray transforms outside the protocol."""

        def __init__(self, *unused_args, **unused_kwargs):
            self.transform = None

        def __call__(self, labels, *unused_args, **unused_kwargs):
            return labels

    augment_module.Albumentations = BenchmarkNoOpAlbumentations

    # The official AMP self-test tries to download a separate reference model. The benchmark already
    # performs a real AMP train/val dry-run with the target model, so keep formal workers fully offline.
    trainer_module.check_amp = lambda _model: True

    def dependency_free_results_csv(trainer: Any) -> dict[str, list[Any]]:
        if not Path(trainer.csv).exists():
            return {}
        with Path(trainer.csv).open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            return {}
        return {key.strip(): [row[key] for row in rows] for key in rows[0]}

    # YOLO12's checkpoint helper imports pandas only to transpose a tiny CSV. Keep the frozen
    # environment unchanged and use the equivalent standard-library implementation for every fork.
    trainer_module.BaseTrainer.read_results_csv = dependency_free_results_csv

    output_dir = args.output_dir or run_dir(args.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = args.data or Path(config["training_dataset_config"])
    epochs = args.epochs or int(config["epochs"])
    expected_images = args.expected_images or int(config["train_images"])
    canonical = not args.no_canonical

    # Ultralytics label caches are NumPy pickles. A cache written by NumPy 2
    # references ``numpy._core`` and cannot be loaded by the frozen NumPy 1.x
    # environment used by YOLOv10. This fork resolves the model's image symlink
    # back to the shared dataset before choosing its cache path, so redirect
    # both cache reads and writes to a model-local namespace. The shared dataset
    # and its caches remain strictly read-only.
    original_load_dataset_cache_file = dataset_module.load_dataset_cache_file
    original_save_dataset_cache_file = dataset_module.save_dataset_cache_file
    local_cache_dir = output_dir / "dataset_cache"

    def benchmark_cache_path(path: str | Path) -> Path:
        cache_path = Path(path)
        if args.model == "yolov10n" and cache_path.name in {"train.cache", "val.cache"}:
            return local_cache_dir / cache_path.name
        return cache_path

    def benchmark_load_dataset_cache_file(path: str | Path) -> dict[str, Any]:
        cache_path = benchmark_cache_path(path)
        try:
            return original_load_dataset_cache_file(cache_path)
        except ModuleNotFoundError as error:
            archive = output_dir / "recovery_attempts" / "numpy_cache_incompat_20260721"
            archive.mkdir(parents=True, exist_ok=True)
            destination = archive / cache_path.name
            if cache_path.exists() and not destination.exists():
                shutil.move(str(cache_path), destination)
            print(
                f"Benchmark cache repair: archived incompatible model-local cache {cache_path} "
                f"under {archive}; rebuilding with the frozen environment."
            )
            raise FileNotFoundError(cache_path) from error

    def benchmark_save_dataset_cache_file(prefix: str, path: str | Path, payload: dict[str, Any]) -> None:
        cache_path = benchmark_cache_path(path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if Path(path) != cache_path:
            print(f"Benchmark cache isolation: {path} -> {cache_path}")
        original_save_dataset_cache_file(prefix, cache_path, payload)

    dataset_module.load_dataset_cache_file = benchmark_load_dataset_cache_file
    dataset_module.save_dataset_cache_file = benchmark_save_dataset_cache_file
    last = output_dir / "checkpoints/last.pt"
    starting_weight = last if canonical and last.exists() else Path(config["weight"])
    model = YOLO(str(starting_weight))
    callbacks = ProtocolCallbacks(args.model, output_dir, canonical, expected_images, epochs)
    callbacks.register(model)

    derived_args = ultralytics_overrides(config, run_dir(args.model))
    frozen_overrides = run_dir(args.model) / "runtime_overrides.yaml"
    if frozen_overrides.exists() and read_yaml(frozen_overrides) != derived_args:
        raise RuntimeError(f"Frozen runtime overrides differ from derived protocol: {frozen_overrides}")
    train_args = dict(derived_args)
    train_args.update(
        {
            "data": str(data),
            "epochs": epochs,
            "project": str(output_dir / "native"),
        }
    )
    if canonical and last.exists():
        train_args["resume"] = True
    results = model.train(**train_args)
    if not canonical:
        native_last = output_dir / "native/train/weights/last.pt"
        if native_last.exists():
            shutil.copy2(native_last, output_dir / "dryrun_last.pt")
    if results is None:
        raise RuntimeError("Ultralytics training returned no result")
    atomic_write_json(
        output_dir / "adapter_summary.json",
        {
            "model": args.model,
            "epochs_requested": epochs,
            "expected_images": expected_images,
            "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
            "canonical": canonical,
        },
    )


if __name__ == "__main__":
    main()

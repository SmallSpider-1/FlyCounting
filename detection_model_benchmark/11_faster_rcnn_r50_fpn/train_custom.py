#!/usr/bin/env python3
"""Thin adapter around torchvision's official references/detection/train.py for custom COCO classes."""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision


PROJECT_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = PROJECT_DIR / "src" / "references" / "detection"
sys.path.insert(0, str(REFERENCE_DIR))

import train as reference_train  # noqa: E402
import coco_utils as reference_coco_utils  # noqa: E402
import utils as reference_utils  # noqa: E402


class NoDropGroupedBatchSampler(torch.utils.data.BatchSampler):
    """Group full batches by aspect ratio, then emit every residual sample exactly once."""

    def __init__(self, sampler, group_ids, batch_size):
        self.sampler = sampler
        self.group_ids = group_ids
        self.batch_size = batch_size

    def __iter__(self):
        buffers = defaultdict(list)
        residual = []
        for index in self.sampler:
            group_id = self.group_ids[index]
            buffers[group_id].append(index)
            if len(buffers[group_id]) == self.batch_size:
                yield buffers.pop(group_id)
        for group_id in sorted(buffers):
            residual.extend(buffers[group_id])
        for offset in range(0, len(residual), self.batch_size):
            yield residual[offset : offset + self.batch_size]

    def __len__(self):
        return math.ceil(len(self.sampler) / self.batch_size)


def atomic_torch_save(payload, path: Path) -> None:
    """Atomically save a checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def benchmark_train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    """Official torchvision loop with a protocol-mapped warmup and optimizer-step audit."""
    run_dir = Path(os.environ["BENCH_RUN_DIR"])
    model.train()
    metric_logger = reference_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", reference_utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    warmup_scheduler = None
    warmup_iters = int(os.environ.get("BENCH_WARMUP_ITERS", "0"))
    if epoch == 0 and warmup_iters > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0 / 1000, total_iters=min(warmup_iters, len(data_loader) - 1)
        )

    images_seen = 0
    micro_batches = 0
    optimizer_steps = 0
    last_batch_size = 0
    runtime_validation = run_dir / "runtime_config_validation.json"
    if not runtime_validation.exists():
        actual_batch_size = int(getattr(data_loader.batch_sampler, "batch_size", None) or os.environ["BENCH_MICRO_BATCH"])
        initial_lrs = [float(group.get("initial_lr", group["lr"])) for group in optimizer.param_groups]
        weight_decays = [float(group.get("weight_decay", 0.0)) for group in optimizer.param_groups]
        checks = {
            "train_images": len(data_loader.dataset) == int(os.environ["BENCH_EXPECTED_IMAGES"]),
            "micro_batch": actual_batch_size == int(os.environ["BENCH_MICRO_BATCH"]),
            "dataloader_batches": len(data_loader) == math.ceil(len(data_loader.dataset) / actual_batch_size),
            "optimizer": type(optimizer).__name__ == "SGD",
            "base_lr": any(abs(value - float(os.environ["BENCH_BASE_LR"])) < 1e-12 for value in initial_lrs),
            "weight_decay": abs(max(weight_decays) - float(os.environ["BENCH_WEIGHT_DECAY"])) < 1e-12,
        }
        temporary = runtime_validation.with_suffix(".json.tmp")
        import json

        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(
                {
                    "model": "faster_rcnn_r50_fpn",
                    "checks": checks,
                    "actual": {
                        "optimizer_class": type(optimizer).__name__,
                        "initial_lrs": initial_lrs,
                        "weight_decays": weight_decays,
                        "train_images": len(data_loader.dataset),
                        "micro_batch": actual_batch_size,
                        "dataloader_batches": len(data_loader),
                        "cudnn_benchmark": torch.backends.cudnn.benchmark,
                        "cudnn_deterministic": torch.backends.cudnn.deterministic,
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    },
                },
                stream,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
        os.replace(temporary, runtime_validation)
        failures = [key for key, passed in checks.items() if not passed]
        if failures:
            raise RuntimeError(f"Runtime protocol validation failed: {failures}")
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        last_batch_size = len(images)
        images_seen += last_batch_size
        micro_batches += 1
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in target.items()} for target in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        reduced = reference_utils.reduce_dict(loss_dict)
        reduced_losses = sum(loss for loss in reduced.values())
        loss_value = reduced_losses.item()
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Non-finite Faster R-CNN loss at epoch {epoch}: {loss_value}")
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()
        optimizer_steps += 1
        if warmup_scheduler is not None:
            warmup_scheduler.step()
        metric_logger.update(loss=reduced_losses, **reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    steps_file = run_dir / "optimizer_steps.csv"
    write_header = not steps_file.exists()
    with steps_file.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "epoch", "images_seen", "micro_batches", "optimizer_steps", "micro_batch",
                "accumulation", "effective_batch", "last_batch_size",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "epoch": epoch + 1,
                "images_seen": images_seen,
                "micro_batches": micro_batches,
                "optimizer_steps": optimizer_steps,
                "micro_batch": int(os.environ["BENCH_MICRO_BATCH"]),
                "accumulation": 1,
                "effective_batch": int(os.environ["BENCH_MICRO_BATCH"]),
                "last_batch_size": last_batch_size,
            }
        )
    return metric_logger


def get_dataset_with_custom_classes(*, original_get_dataset, is_train, args):
    """Keep negative training images so every benchmark model sees the same samples."""
    if not is_train or args.drop_empty_images:
        dataset, _ = original_get_dataset(is_train=is_train, args=args)
        return dataset, args.num_foreground_classes + 1

    original_filter = reference_coco_utils._coco_remove_images_without_annotations
    reference_coco_utils._coco_remove_images_without_annotations = lambda dataset: dataset
    try:
        dataset, _ = original_get_dataset(is_train=is_train, args=args)
    finally:
        reference_coco_utils._coco_remove_images_without_annotations = original_filter
    print(f"Keeping all {len(dataset)} training images, including empty/background images")
    return dataset, args.num_foreground_classes + 1


def main() -> None:
    parser = reference_train.get_args_parser()
    parser.add_argument(
        "--num-foreground-classes",
        type=int,
        required=True,
        help="Foreground class count; Faster R-CNN adds the background class internally.",
    )
    parser.add_argument(
        "--initial-weights-file",
        type=Path,
        default=None,
        help="Local torchvision Faster R-CNN COCO state_dict used for shape-compatible initialization.",
    )
    parser.add_argument(
        "--drop-empty-images",
        action="store_true",
        help=(
            "Use torchvision reference behavior and remove empty training images. "
            "The benchmark default keeps them so all 11 models see the same training set."
        ),
    )
    parser.add_argument("--benchmark-run-dir", type=Path, required=True)
    parser.add_argument("--resolved-config-hash", required=True)
    parser.add_argument("--warmup-iters", type=int, default=0)
    parser.add_argument("--min-size", type=int, default=640)
    parser.add_argument("--max-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.num_foreground_classes < 1:
        parser.error("--num-foreground-classes must be positive")

    original_get_dataset = reference_train.get_dataset
    original_get_model = torchvision.models.get_model
    original_evaluate = reference_train.evaluate

    os.environ["BENCH_RUN_DIR"] = str(args.benchmark_run_dir)
    os.environ["BENCH_WARMUP_ITERS"] = str(args.warmup_iters)
    os.environ["BENCH_MICRO_BATCH"] = str(args.batch_size)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    def get_model_with_local_initialization(name, **kwargs):
        kwargs["weights"] = None
        kwargs["min_size"] = args.min_size
        kwargs["max_size"] = args.max_size
        model = original_get_model(name, **kwargs)
        if args.initial_weights_file:
            state = torch.load(args.initial_weights_file, map_location="cpu", weights_only=True)
            current = model.state_dict()
            compatible = {key: value for key, value in state.items() if key in current and current[key].shape == value.shape}
            missing, unexpected = model.load_state_dict(compatible, strict=False)
            print(
                f"Loaded {len(compatible)} shape-compatible tensors from {args.initial_weights_file}; "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
        return model

    metrics_file = args.benchmark_run_dir / "metrics_history.csv"
    if metrics_file.exists():
        with metrics_file.open(newline="", encoding="utf-8") as stream:
            best_metric = max((float(row["val_map50_95"]) for row in csv.DictReader(stream)), default=-1.0)
    else:
        best_metric = -1.0
    global_optimizer_steps = 0

    def evaluate_and_checkpoint(*evaluate_args, **evaluate_kwargs):
        nonlocal best_metric, global_optimizer_steps
        evaluator = original_evaluate(*evaluate_args, **evaluate_kwargs)
        checkpoint_path = Path(args.output_dir) / "checkpoint.pth"
        if not checkpoint_path.exists():
            return evaluator
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        epoch = int(checkpoint["epoch"]) + 1
        steps_file = args.benchmark_run_dir / "optimizer_steps.csv"
        with steps_file.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        global_optimizer_steps = sum(int(row["optimizer_steps"]) for row in rows)
        metric = float(evaluator.coco_eval["bbox"].stats[0])
        checkpoint.update(
            {
                "resolved_config_hash": args.resolved_config_hash,
                "global_optimizer_step": global_optimizer_steps,
                "rng_state": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all(),
                },
            }
        )
        checkpoints_dir = args.benchmark_run_dir / "checkpoints"
        atomic_torch_save(checkpoint, checkpoints_dir / "last.pth")
        checkpoint_sha_metric = metric
        if metric > best_metric:
            best_metric = metric
            atomic_torch_save(checkpoint, checkpoints_dir / "best.pth")
        if epoch % 10 == 0:
            atomic_torch_save(checkpoint, checkpoints_dir / f"epoch_{epoch:03d}.pth")

        metrics_file = args.benchmark_run_dir / "metrics_history.csv"
        write_header = not metrics_file.exists()
        with metrics_file.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["epoch", "val_map50_95", "val_ap50", "val_ap75"])
            if write_header:
                writer.writeheader()
            stats = evaluator.coco_eval["bbox"].stats
            writer.writerow({"epoch": epoch, "val_map50_95": stats[0], "val_ap50": stats[1], "val_ap75": stats[2]})
        print(
            f"BENCHMARK_EPOCH epoch={epoch} val_map50_95={checkpoint_sha_metric:.8f} "
            f"optimizer_steps={global_optimizer_steps}"
        )
        return evaluator

    reference_train.get_dataset = lambda *, is_train, args: get_dataset_with_custom_classes(
        original_get_dataset=original_get_dataset,
        is_train=is_train,
        args=args,
    )
    reference_train.GroupedBatchSampler = NoDropGroupedBatchSampler
    reference_train.train_one_epoch = benchmark_train_one_epoch
    reference_train.evaluate = evaluate_and_checkpoint
    torchvision.models.get_model = get_model_with_local_initialization
    import json

    with (Path(args.data_path) / "annotations/instances_train2017.json").open(encoding="utf-8") as stream:
        os.environ["BENCH_EXPECTED_IMAGES"] = str(len(json.load(stream)["images"]))
    os.environ["BENCH_BASE_LR"] = str(args.lr)
    os.environ["BENCH_WEIGHT_DECAY"] = str(args.weight_decay)
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        rng_state = resume_checkpoint.get("rng_state")
        if rng_state:
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.set_rng_state(rng_state["torch"])
            if torch.cuda.is_available() and rng_state.get("cuda"):
                torch.cuda.set_rng_state_all(rng_state["cuda"])
    original_torch_load = torch.load

    def trusted_benchmark_load(*load_args, **load_kwargs):
        load_kwargs["weights_only"] = False
        return original_torch_load(*load_args, **load_kwargs)

    torch.load = trusted_benchmark_load
    try:
        reference_train.main(args)
    finally:
        torch.load = original_torch_load
    import json

    summary_path = args.benchmark_run_dir / "adapter_summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "model": "faster_rcnn_r50_fpn",
                "epochs_requested": args.epochs,
                "expected_images": int(os.environ["BENCH_EXPECTED_IMAGES"]),
                "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
                "canonical": True,
            },
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
    os.replace(temporary, summary_path)


if __name__ == "__main__":
    main()

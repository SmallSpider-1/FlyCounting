#!/usr/bin/env python3
"""Run YOLOv5, YOLOv7 and YOLOv9 without modifying their frozen source trees."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from common import expected_config_hash, load_resolved_config, run_dir
from config_builders import legacy_hyp


HELPER_SOURCE = r'''
def _bench_rows(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _bench_append_once(path, fields, row):
    if any(item.get("epoch") == str(row["epoch"]) for item in _bench_rows(path)):
        return
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _bench_atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _bench_atomic_torch_save(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _bench_epoch(ckpt, epoch, results, nb, dataset_size, optimizer, scaler, scheduler):
    run_dir = Path(os.environ["BENCH_RUN_DIR"])
    actual_epoch = int(epoch) + 1
    metric_path = run_dir / "metrics_history.csv"
    step_path = run_dir / "optimizer_steps.csv"
    results = [float(value) for value in results]
    _bench_append_once(
        metric_path,
        ["epoch", "val_map50_95", "val_ap50", "val_precision", "val_recall"],
        {
            "epoch": actual_epoch,
            "val_map50_95": results[3],
            "val_ap50": results[2],
            "val_precision": results[0],
            "val_recall": results[1],
        },
    )
    micro_batch = int(os.environ["BENCH_MICRO_BATCH"])
    accumulation = int(os.environ["BENCH_ACCUMULATION"])
    optimizer_steps = int(math.ceil(int(nb) / accumulation))
    _bench_append_once(
        step_path,
        ["epoch", "images_seen", "micro_batches", "optimizer_steps", "micro_batch", "accumulation", "effective_batch", "last_batch_size"],
        {
            "epoch": actual_epoch,
            "images_seen": int(dataset_size),
            "micro_batches": int(nb),
            "optimizer_steps": optimizer_steps,
            "micro_batch": micro_batch,
            "accumulation": accumulation,
            "effective_batch": micro_batch * accumulation,
            "last_batch_size": int(dataset_size) % micro_batch or micro_batch,
        },
    )
    previous_steps = sum(int(row["optimizer_steps"]) for row in _bench_rows(step_path))
    ckpt.update(
        {
            "resolved_config_hash": os.environ["BENCH_CONFIG_HASH"],
            "global_optimizer_step": previous_steps,
            "benchmark_best_metric": "val_mAP50_95",
            "benchmark_val_mAP50_95": results[3],
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
            },
        }
    )
    checkpoints = run_dir / "checkpoints"
    _bench_atomic_torch_save(ckpt, checkpoints / "last.pt")
    history = _bench_rows(metric_path)
    previous_best = max((float(row["val_map50_95"]) for row in history if int(row["epoch"]) < actual_epoch), default=-1.0)
    if results[3] > previous_best:
        _bench_atomic_torch_save(ckpt, checkpoints / "best.pt")
    if actual_epoch % 10 == 0:
        _bench_atomic_torch_save(ckpt, checkpoints / f"epoch_{actual_epoch:03d}.pt")

    validation_path = run_dir / "runtime_config_validation.json"
    if not validation_path.exists():
        groups = optimizer.param_groups
        initial_lrs = [float(group.get("initial_lr", group["lr"])) for group in groups]
        weight_decays = [float(group.get("weight_decay", 0.0)) for group in groups]
        expected_lr = float(os.environ["BENCH_BASE_LR"])
        expected_wd = float(os.environ["BENCH_WEIGHT_DECAY"])
        checks = {
            "train_images": int(dataset_size) == int(os.environ["BENCH_EXPECTED_IMAGES"]),
            "micro_batches": int(nb) == math.ceil(int(dataset_size) / micro_batch),
            "accumulation": accumulation == 1,
            "optimizer": os.environ["BENCH_OPTIMIZER"].lower() in type(optimizer).__name__.lower(),
            "base_lr": any(abs(value - expected_lr) < 1e-12 for value in initial_lrs),
            "weight_decay": abs(max(weight_decays) - expected_wd) < 1e-12,
        }
        _bench_atomic_json(
            validation_path,
            {
                "model": os.environ["BENCH_MODEL"],
                "checks": checks,
                "actual": {
                    "optimizer_class": type(optimizer).__name__,
                    "initial_lrs": initial_lrs,
                    "weight_decays": weight_decays,
                    "train_images": int(dataset_size),
                    "micro_batches": int(nb),
                    "accumulation": accumulation,
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                },
            },
        )
        failed = [key for key, value in checks.items() if not value]
        if failed:
            raise RuntimeError(f"Runtime protocol validation failed: {failed}")
'''


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {label} replacement, found {count}")
    return source.replace(old, new, 1)


def transform_source(model: str, source: str) -> str:
    imports = "\nimport csv\nimport json\n" + HELPER_SOURCE + "\n"
    source = imports + source
    source = replace_once(
        source,
        "nbs = 64  # nominal batch size",
        'nbs = int(os.environ["BENCH_NBS"])  # benchmark nominal batch size',
        "nbs",
    )
    if model == "yolov7_tiny":
        source = replace_once(
            source,
            "init_seeds(2 + rank)",
            'init_seeds(int(os.environ["BENCH_SEED"]) + 1 + rank)',
            "seed",
        )
        source = replace_once(
            source,
            "fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]",
            "fi = float(results[3])  # benchmark best metric: pure val mAP@.5:.95",
            "pure best metric",
        )
        injection = """                _bench_epoch(ckpt, epoch, results, nb, len(dataset), optimizer, scaler, scheduler)\n                del ckpt"""
        source = replace_once(source, "                del ckpt", injection, "epoch hook")
        source = source.replace("                strip_optimizer(f)  # strip optimizers", "                pass  # keep optimizer state for benchmark resume")
    else:
        quote = '"' if model == "yolov5n" else "'"
        fitness_line = (
            f"fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of "
            "[P, R, mAP@.5, mAP@.5-.95]"
        )
        source = replace_once(
            source,
            fitness_line,
            "fi = float(results[3])  # benchmark best metric: pure val mAP@.5:.95",
            "pure best metric",
        )
        if model == "yolov5n":
            source = replace_once(source, "        check_git_status()", "        pass  # benchmark offline mode", "git network check")
            source = replace_once(
                source,
                '        check_requirements(ROOT / "requirements.txt")',
                "        pass  # dependency installation is forbidden during formal training",
                "automatic dependency check",
            )
            block = '''                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f"epoch{epoch}.pt")
                del ckpt'''
            replacement = '''                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f"epoch{epoch}.pt")
                _bench_epoch(ckpt, epoch, results, nb, len(dataset), optimizer, scaler, scheduler)
                del ckpt'''
            source = replace_once(source, block, replacement, "epoch hook")
            source = source.replace("                strip_optimizer(f)  # strip optimizers", "                pass  # keep optimizer state for benchmark resume")
            source = replace_once(source, "deterministic=True", "deterministic=False", "determinism flag")
        else:
            block = """                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt"""
            replacement = """                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                _bench_epoch(ckpt, epoch, results, nb, len(train_loader.dataset), optimizer, scaler, scheduler)
                del ckpt"""
            source = replace_once(source, block, replacement, "epoch hook")
            source = source.replace("                strip_optimizer(f)  # strip optimizers", "                pass  # keep optimizer state for benchmark resume")
            source = replace_once(source, "deterministic=True", "deterministic=False", "determinism flag")
    return source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["yolov5n", "yolov7_tiny", "yolov9_t"])
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--expected-images", type=int)
    args = parser.parse_args()

    config = load_resolved_config(args.model)
    source_dir = Path(config["source_dir"])
    output_dir = args.output_dir or run_dir(args.model)
    native_project = output_dir / "native"
    native_last = native_project / "train/weights/last.pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    hyp = legacy_hyp(args.model, config)
    hyp_path = output_dir / "runtime_hyp.yaml"
    if hyp_path.exists():
        with hyp_path.open(encoding="utf-8") as stream:
            if yaml.safe_load(stream) != hyp:
                raise RuntimeError(f"Frozen runtime hyp differs from derived protocol: {hyp_path}")
    else:
        with hyp_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(hyp, stream, allow_unicode=True, sort_keys=False)

    os.environ.update(
        {
            "BENCH_RUN_DIR": str(output_dir),
            "BENCH_CONFIG_HASH": expected_config_hash(args.model),
            "BENCH_MODEL": args.model,
            "BENCH_NBS": "16",
            "BENCH_MICRO_BATCH": str(config["micro_batch"]),
            "BENCH_ACCUMULATION": str(config["accumulation"]),
            "BENCH_SEED": str(config["seed"]),
            "BENCH_BASE_LR": str(config["resolved_base_lr"]),
            "BENCH_WEIGHT_DECAY": str(config["resolved_weight_decay"]),
            "BENCH_OPTIMIZER": str(config["optimizer"]),
            "BENCH_EXPECTED_IMAGES": str(args.expected_images or config["train_images"]),
            "WANDB_MODE": "disabled",
            "COMET_MODE": "DISABLED",
            "HF_HUB_OFFLINE": "1",
        }
    )
    sys.path.insert(0, str(source_dir))
    os.chdir(source_dir)

    # Legacy YOLO select_device("0") rewrites CUDA_VISIBLE_DEVICES to physical
    # GPU 0, even when the worker has already isolated physical GPU 1 and the
    # framework-visible device is correctly cuda:0. Keep the frozen logical
    # device argument, but preserve the worker's external physical-GPU mask.
    import utils.torch_utils as legacy_torch_utils

    original_select_device = legacy_torch_utils.select_device

    def benchmark_select_device(device="", *select_args, **select_kwargs):
        if "BENCH_PHYSICAL_GPU_ID" in os.environ:
            return original_select_device("", *select_args, **select_kwargs)
        return original_select_device(device, *select_args, **select_kwargs)

    legacy_torch_utils.select_device = benchmark_select_device

    # YOLOv5/9 instantiate a hidden Albumentations pipeline whenever that
    # optional package happens to be installed. Its default blur/gray/CLAHE
    # transforms are outside the frozen weak-augmentation protocol. Replace
    # only this optional layer; native resize/pad and hflip=0.5 remain active.
    try:
        import utils.augmentations as legacy_augmentations

        class BenchmarkNoOpAlbumentations:
            def __init__(self, *unused_args, **unused_kwargs):
                self.transform = None

            def __call__(self, image, labels, *unused_args, **unused_kwargs):
                return image, labels

        legacy_augmentations.Albumentations = BenchmarkNoOpAlbumentations
    except ImportError:
        pass
    epochs = args.epochs or int(config["epochs"])
    data = args.data or Path(config["training_dataset_config"])

    if native_last.exists():
        command = ["train", "--resume", str(native_last)]
    elif args.model == "yolov7_tiny":
        command = [
            "train",
            "--workers", str(config["workers"]),
            "--device", "0",
            "--batch-size", str(config["micro_batch"]),
            "--data", str(data),
            "--img-size", str(config["imgsz"]), str(config["imgsz"]),
            "--cfg", "cfg/training/yolov7-tiny.yaml",
            "--weights", "../weights/yolov7-tiny.pt",
            "--hyp", str(hyp_path),
            "--epochs", str(epochs),
            "--project", str(native_project),
            "--name", "train",
            "--exist-ok",
        ]
    else:
        command = [
            "train",
            "--workers", str(config["workers"]),
            "--device", "0",
            "--batch-size", str(config["micro_batch"]),
            "--data", str(data),
            "--imgsz", str(config["imgsz"]),
            "--weights", str(config["weight"]),
            "--hyp", str(hyp_path),
            "--epochs", str(epochs),
            "--optimizer", "SGD",
            "--patience", "1000",
            "--project", str(native_project),
            "--name", "train",
            "--exist-ok",
            "--noplots",
            "--seed", str(config["seed"]),
        ]
        if args.model == "yolov9_t":
            command.extend(["--cfg", "models/detect/yolov9-t.yaml", "--min-items", "0", "--close-mosaic", "0"])

    script = source_dir / ("train_dual.py" if args.model == "yolov9_t" else "train.py")
    transformed = transform_source(args.model, script.read_text(encoding="utf-8"))
    previous_argv = sys.argv
    sys.argv = command
    namespace = {"__name__": "__main__", "__file__": str(script), "__package__": None}
    try:
        exec(compile(transformed, str(script), "exec"), namespace)
    finally:
        sys.argv = previous_argv
    summary_path = output_dir / "adapter_summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    import torch

    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "model": args.model,
                "epochs_requested": epochs,
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

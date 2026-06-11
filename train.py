import argparse
import csv
import contextlib
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')


ROOT = Path(__file__).resolve().parent

# 以后切换模型时，只需要修改这一行。
MODEL_NAME = "yolo12"
DATASET_PATH = ROOT / "dataset_two_class.yaml"
RUNS_DIR = ROOT / "runs/detect"

MODEL_CONFIGS = {
    "yolo5": ROOT / "ultralytics/cfg/models/v5/yolov5.yaml",
    "yolo8": ROOT / "ultralytics/cfg/models/v8/yolov8.yaml",
    "yolo11": ROOT / "ultralytics/cfg/models/11/yolo11.yaml",
    "yolo12": ROOT / "ultralytics/cfg/models/12/yolo12.yaml",
    "yolo26": ROOT / "ultralytics/cfg/models/26/yolo26.yaml",
}
MODEL_CONFIGS.update({p.stem: p for p in (ROOT / "ultralytics/cfg/models/26").glob("yolo26*.yaml")})
SOURCE_YOLO11_MODELS_DIR = Path("/home/admin1/Projects/ultralytics-yolo11-main/ultralytics/cfg/models/11")


class Tee:
    """Write terminal output to both console and a log file."""

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, data):
        self.stream.write(data)
        try:
            self.log_file.write(data)
        except ValueError:
            pass

    def flush(self):
        self.stream.flush()
        try:
            self.log_file.flush()
        except ValueError:
            pass


def get_log_path(project_dir):
    log_dir = Path(project_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"train_console_{timestamp}.txt"


def get_dataset_suffix(dataset_path):
    stem = Path(dataset_path).stem
    return stem.removeprefix("dataset_")


def get_project_path(model_name, dataset_path):
    dataset_suffix = get_dataset_suffix(dataset_path)
    return RUNS_DIR / f"{model_name}_{dataset_suffix}"


def parse_args():
    parser = argparse.ArgumentParser(description="Train one or more Ultralytics model configs.")
    parser.add_argument(
        "models",
        nargs="*",
        help="Model names to train, e.g. yolo26-C3k2-Star. Defaults to MODEL_NAME when --migrated is not set.",
    )
    parser.add_argument(
        "--migrated",
        action="store_true",
        help="Train every yolo26-* config that maps to a yolo11-* config in the source YOLO11 model directory.",
    )
    parser.add_argument(
        "--source-model-dir",
        type=Path,
        default=SOURCE_YOLO11_MODELS_DIR,
        help="Source directory used to define migrated yolo11-* configs.",
    )
    parser.add_argument("--data", type=Path, default=DATASET_PATH, help="Dataset yaml path.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=4, help="Training batch size.")
    parser.add_argument("--workers", type=int, default=8, help="Training workers.")
    parser.add_argument("--device", default="0", help="Training device.")
    parser.add_argument("--optimizer", default="SGD", help="Optimizer name.")
    parser.add_argument("--force", action="store_true", help="Run even if a completed weights/best.pt exists.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected model names without starting training.",
    )
    return parser.parse_args()


def get_migrated_model_names(source_model_dir):
    names = []
    missing = []
    for source_config in sorted(source_model_dir.glob("yolo11-*.yaml")):
        target_name = "yolo26-" + source_config.stem.removeprefix("yolo11-")
        if target_name in MODEL_CONFIGS:
            names.append(target_name)
        else:
            missing.append(target_name)
    if missing:
        raise FileNotFoundError(f"迁移模型缺失: {', '.join(missing[:20])}")
    return names


def has_completed_weights(project):
    return any(project.glob("**/weights/best.pt"))


def write_sweep_summary(rows, summary_path):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=["model", "status", "log_path", "project", "message"])
        writer.writeheader()
        writer.writerows(rows)


def set_ultralytics_logger_stream(stream):
    try:
        from ultralytics.utils import LOGGER
    except Exception:
        return []

    previous_streams = []
    for handler in LOGGER.handlers:
        if hasattr(handler, "stream"):
            previous_streams.append((handler, handler.stream))
            handler.setStream(stream)
    return previous_streams


def restore_logger_streams(previous_streams):
    for handler, stream in previous_streams:
        try:
            handler.setStream(stream)
        except Exception:
            pass


def train_one(model_name, args):
    if model_name not in MODEL_CONFIGS:
        valid_models = ", ".join(sorted(MODEL_CONFIGS))
        raise ValueError(f"未知模型: {model_name}，可选模型: {valid_models}")

    model_config = MODEL_CONFIGS[model_name]
    project = get_project_path(model_name, args.data)
    if not args.force and has_completed_weights(project):
        print(f"跳过已完成模型: {model_name} ({project})")
        return {"model": model_name, "status": "skipped", "log_path": "", "project": str(project), "message": "completed weights found"}

    log_path = get_log_path(project)
    from ultralytics import YOLO

    with open(log_path, "w", encoding="utf-8", buffering=1) as log_file:
        stdout_tee = Tee(sys.stdout, log_file)
        stderr_tee = Tee(sys.stderr, log_file)

        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"训练日志保存到: {log_path}")
            print(f"模型名称: {model_name}")
            print(f"模型配置: {model_config}")

            previous_logger_streams = set_ultralytics_logger_stream(stdout_tee)
            try:
                model = YOLO(model_config)
                model.train(data=args.data,
                            project=project,
                            imgsz=args.imgsz,
                            epochs=args.epochs,
                            single_cls=False,  #如果是Ture的话只会将所有目标作为单一类别进行训练和检测
                            batch=args.batch,
                            workers=args.workers,
                            device=args.device,
                            optimizer=args.optimizer #指定训练使用的优化器。SGD 表示随机梯度下降法（Stochastic Gradient Descent），这是一种常用的优化方法。
                            )
            finally:
                restore_logger_streams(previous_logger_streams)
    return {"model": model_name, "status": "completed", "log_path": str(log_path), "project": str(project), "message": ""}


if __name__ == '__main__':
    args = parse_args()
    model_names = get_migrated_model_names(args.source_model_dir) if args.migrated else (args.models or [MODEL_NAME])

    if args.dry_run:
        print(f"将训练 {len(model_names)} 个模型:")
        for name in model_names:
            print(name)
        sys.exit(0)

    sweep_started = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = RUNS_DIR / f"sweep_migrated_yolo26_{sweep_started}_pid{os.getpid()}.csv"
    rows = []
    failures = 0
    print(f"本次将训练 {len(model_names)} 个模型，汇总文件: {summary_path}")

    for index, model_name in enumerate(model_names, start=1):
        print(f"\n===== [{index}/{len(model_names)}] {model_name} =====")
        try:
            row = train_one(model_name, args)
        except Exception as exc:
            failures += 1
            project = get_project_path(model_name, args.data)
            log_path = get_log_path(project)
            with open(log_path, "w", encoding="utf-8", buffering=1) as log_file:
                log_file.write(f"训练失败: {model_name}\n")
                log_file.write(traceback.format_exc())
            traceback.print_exc()
            row = {
                "model": model_name,
                "status": "failed",
                "log_path": str(log_path),
                "project": str(project),
                "message": repr(exc),
            }
        rows.append(row)
        write_sweep_summary(rows, summary_path)

    print(f"\n训练批次结束: total={len(rows)}, failures={failures}, summary={summary_path}")
    sys.exit(1 if failures else 0)

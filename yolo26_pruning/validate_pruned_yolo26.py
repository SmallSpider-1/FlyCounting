import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

PRUNE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PRUNE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_WEIGHTS = PRUNE_ROOT / "yolo26_structured_prune" / "yolo26n_structured_pruned_0.20_20260514_192659.pt"
DEFAULT_DATA = PROJECT_ROOT / "data.yaml"
DEFAULT_OUTPUT_DIR = PRUNE_ROOT / "performance_results"


def parse_args():
    parser = argparse.ArgumentParser(description="Validate pruned YOLO26 weights on the configured dataset.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="需要验证的剪枝权重 .pt 路径。")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="数据集 data.yaml 路径。")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="验证结果保存目录。")
    parser.add_argument("--imgsz", type=int, default=640, help="验证图片尺寸。")
    parser.add_argument("--batch", type=int, default=4, help="验证 batch size。")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers。")
    parser.add_argument("--device", default="0", help="验证设备，例如 0 或 cpu。")
    parser.add_argument("--conf", type=float, default=None, help="验证置信度阈值，默认使用 Ultralytics 默认值。")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值。")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="验证数据划分。")
    return parser.parse_args()


def ensure_path(path, label):
    if not path.exists():
        raise FileNotFoundError(f"{label}不存在: {path}")


def metric_value(obj, name, default=0.0):
    return float(getattr(obj, name, default) or 0.0)


def collect_metrics(metrics):
    box = getattr(metrics, "box", None)
    speed = getattr(metrics, "speed", {}) or {}
    results = {
        "precision": metric_value(box, "mp"),
        "recall": metric_value(box, "mr"),
        "map50": metric_value(box, "map50"),
        "map50_95": metric_value(box, "map"),
        "fitness": metric_value(metrics, "fitness"),
        "speed_ms": {k: float(v) for k, v in speed.items()},
    }
    if box is not None:
        results["class_precision"] = [float(x) for x in getattr(box, "p", [])]
        results["class_recall"] = [float(x) for x in getattr(box, "r", [])]
        results["class_map50"] = [float(x) for x in getattr(box, "ap50", [])]
        results["class_map50_95"] = [float(x) for x in getattr(box, "ap", [])]
    return results


def main():
    args = parse_args()
    ensure_path(args.weights, "权重文件")
    ensure_path(args.data, "数据集配置")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    print(f"验证权重: {args.weights}")
    print(f"数据配置: {args.data}")
    print(f"数据划分: {args.split}")
    print(f"imgsz={args.imgsz}, batch={args.batch}, device={args.device}")

    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        split=args.split,
        project=str(args.output_dir),
        name=f"val_{args.weights.stem}",
        exist_ok=True,
        verbose=True,
    )

    summary = {
        "weights": str(args.weights),
        "data": str(args.data),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": collect_metrics(metrics),
        "save_dir": str(getattr(metrics, "save_dir", "")),
    }

    output_json = args.output_dir / f"{args.weights.stem}_metrics.json"
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    m = summary["metrics"]
    print("\n验证结果")
    print(f"Precision:   {m['precision']:.4f}")
    print(f"Recall:      {m['recall']:.4f}")
    print(f"mAP50:       {m['map50']:.4f}")
    print(f"mAP50-95:    {m['map50_95']:.4f}")
    print(f"Fitness:     {m['fitness']:.4f}")
    if m["speed_ms"]:
        print("Speed(ms):   " + ", ".join(f"{k}={v:.3f}" for k, v in m["speed_ms"].items()))
    print(f"结果已保存: {output_json}")


if __name__ == "__main__":
    main()

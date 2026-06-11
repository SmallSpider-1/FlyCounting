import argparse
import traceback
from pathlib import Path

from common import RESULTS_DIR, collect_val_metrics, load_config, timestamp, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Validate YOLO weights across multiple image sizes.")
    parser.add_argument("--config", type=Path, default=None, help="Experiment matrix JSON.")
    parser.add_argument("--weights", type=Path, default=None, help="Override model weights.")
    parser.add_argument("--data", type=Path, default=None, help="Override data yaml.")
    parser.add_argument("--imgsz", type=int, nargs="*", default=None, help="Image sizes to test.")
    parser.add_argument("--device", default=None, help="Device, e.g. 0 or cpu.")
    parser.add_argument("--batch", type=int, default=None, help="Batch size.")
    parser.add_argument("--workers", type=int, default=None, help="Workers.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    weights = Path(args.weights or cfg["weights"])
    data = Path(args.data or cfg["data"])
    sizes = args.imgsz or cfg["resolution_sweep"]
    device = args.device if args.device is not None else cfg["device"]
    batch = args.batch if args.batch is not None else cfg["batch"]
    workers = args.workers if args.workers is not None else cfg["workers"]

    from ultralytics import YOLO

    run_id = timestamp()
    results = {
        "run_id": run_id,
        "weights": str(weights),
        "data": str(data),
        "device": device,
        "batch": batch,
        "workers": workers,
        "results": [],
    }

    for size in sizes:
        item = {"imgsz": int(size), "status": "pending"}
        try:
            print(f"\n=== Validate imgsz={size} ===")
            model = YOLO(str(weights))
            metrics = model.val(
                data=str(data),
                imgsz=int(size),
                batch=batch,
                workers=workers,
                device=device,
                project=str(RESULTS_DIR / "val_resolution_sweep"),
                name=f"{weights.stem}_imgsz{size}",
                exist_ok=True,
                verbose=False,
            )
            item.update({"status": "ok", "metrics": collect_val_metrics(metrics)})
        except Exception as exc:
            item.update({"status": "error", "error": repr(exc), "traceback": traceback.format_exc()})
        results["results"].append(item)
        write_json(RESULTS_DIR / f"resolution_sweep_{weights.stem}_{run_id}.json", results)

    print(f"\nSaved: {RESULTS_DIR / f'resolution_sweep_{weights.stem}_{run_id}.json'}")


if __name__ == "__main__":
    main()

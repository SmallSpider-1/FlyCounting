import argparse
import traceback
from pathlib import Path

from common import EXPORTS_DIR, RESULTS_DIR, collect_val_metrics, load_config, timestamp, write_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate exported YOLO artifacts from the workspace exports directory."
    )
    parser.add_argument("--config", type=Path, default=None, help="Experiment matrix JSON.")
    parser.add_argument(
        "--artifacts", nargs="*", type=Path, default=None, help="Specific exported model files/directories."
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--device", default=None, help="Device, e.g. cpu or 0.")
    parser.add_argument("--batch", type=int, default=None, help="Batch size.")
    parser.add_argument("--workers", type=int, default=None, help="Workers.")
    return parser.parse_args()


def discover_artifacts():
    artifacts = []
    for path in sorted(EXPORTS_DIR.iterdir()):
        if path.is_dir() and path.name.endswith("_ncnn_model") and (path / "metadata.yaml").exists():
            artifacts.append(path)
    for suffix in ("*.onnx", "*.tflite", "*.engine", "*.xml"):
        artifacts.extend(sorted(EXPORTS_DIR.rglob(suffix)))
    return artifacts


def main():
    args = parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    data = Path(cfg["data"])
    artifacts = args.artifacts or discover_artifacts()
    device = args.device if args.device is not None else cfg["device"]
    batch = args.batch if args.batch is not None else cfg["batch"]
    workers = args.workers if args.workers is not None else cfg["workers"]
    run_id = timestamp()

    from ultralytics import YOLO

    results = {
        "run_id": run_id,
        "data": str(data),
        "imgsz": args.imgsz,
        "device": device,
        "batch": batch,
        "workers": workers,
        "results": [],
    }

    for artifact in artifacts:
        item = {"artifact": str(artifact), "status": "pending"}
        try:
            print(f"\n=== Validate artifact: {artifact} ===")
            model = YOLO(str(artifact), task="detect")
            metrics = model.val(
                data=str(data),
                imgsz=args.imgsz,
                batch=batch,
                workers=workers,
                device=device,
                project=str(RESULTS_DIR / "val_exported_models"),
                name=artifact.name,
                exist_ok=True,
                verbose=False,
            )
            item.update({"status": "ok", "metrics": collect_val_metrics(metrics)})
        except Exception as exc:
            item.update({"status": "error", "error": repr(exc), "traceback": traceback.format_exc()})
        results["results"].append(item)
        write_json(RESULTS_DIR / f"validate_exported_models_{run_id}.json", results)

    print(f"\nSaved: {RESULTS_DIR / f'validate_exported_models_{run_id}.json'}")


if __name__ == "__main__":
    main()

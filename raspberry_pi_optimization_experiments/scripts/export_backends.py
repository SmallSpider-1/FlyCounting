import argparse
import shutil
import traceback
from pathlib import Path

from common import EXPORTS_DIR, RESULTS_DIR, load_config, timestamp, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLO weights to deployment backends.")
    parser.add_argument("--config", type=Path, default=None, help="Experiment matrix JSON.")
    parser.add_argument("--weights", type=Path, default=None, help="Override model weights.")
    parser.add_argument("--formats", nargs="*", default=None, help="Formats to export, e.g. ncnn onnx tflite.")
    parser.add_argument("--device", default=None, help="Export device.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    weights = Path(args.weights or cfg["weights"])
    selected = set(args.formats) if args.formats else None
    device = args.device if args.device is not None else cfg["device"]
    export_specs = [x for x in cfg["export_formats"] if selected is None or x["format"] in selected]

    from ultralytics import YOLO

    run_id = timestamp()
    results = {"run_id": run_id, "weights": str(weights), "exports": []}

    for index, spec in enumerate(export_specs, start=1):
        fmt = spec["format"]
        export_args = dict(spec.get("args", {}))
        export_args["format"] = fmt
        export_args.setdefault("device", device)
        item = {"format": fmt, "args": export_args, "status": "pending"}
        try:
            print(f"\n=== Export {index}/{len(export_specs)}: {fmt} {export_args} ===")
            model = YOLO(str(weights))
            exported = Path(model.export(**export_args))
            suffix = "_ncnn_model" if fmt == "ncnn" else ""
            target_dir = EXPORTS_DIR / f"{weights.stem}_{fmt}_{index}_{run_id}{suffix}"
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            if exported.is_dir():
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                shutil.copytree(exported, target_dir)
                stored = target_dir
            else:
                target_dir.mkdir(parents=True, exist_ok=True)
                stored = target_dir / exported.name
                shutil.copy2(exported, stored)
            item.update({"status": "ok", "exported": str(exported), "stored": str(stored)})
        except Exception as exc:
            item.update({"status": "error", "error": repr(exc), "traceback": traceback.format_exc()})
        results["exports"].append(item)
        write_json(RESULTS_DIR / f"export_backends_{weights.stem}_{run_id}.json", results)

    print(f"\nSaved: {RESULTS_DIR / f'export_backends_{weights.stem}_{run_id}.json'}")


if __name__ == "__main__":
    main()

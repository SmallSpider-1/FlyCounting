import argparse
import csv
import json
from pathlib import Path

from common import RESULTS_DIR, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Collect optimization experiment JSON files into CSV/JSON summaries.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="Directory containing result JSON files.")
    parser.add_argument("--output-prefix", default="summary", help="Output filename prefix.")
    return parser.parse_args()


def load_json_files(results_dir):
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "results" in payload:
            for item in payload["results"]:
                metrics = item.get("metrics", {})
                speed = metrics.get("speed_ms", {})
                rows.append(
                    {
                        "source_file": path.name,
                        "kind": "validation",
                        "weights": payload.get("weights", ""),
                        "imgsz": item.get("imgsz", ""),
                        "status": item.get("status", ""),
                        "precision": metrics.get("precision", ""),
                        "recall": metrics.get("recall", ""),
                        "map50": metrics.get("map50", ""),
                        "map50_95": metrics.get("map50_95", ""),
                        "inference_ms": speed.get("inference", ""),
                        "preprocess_ms": speed.get("preprocess", ""),
                        "postprocess_ms": speed.get("postprocess", ""),
                        "artifact": "",
                        "error": item.get("error", ""),
                    }
                )
        if "exports" in payload:
            for item in payload["exports"]:
                rows.append(
                    {
                        "source_file": path.name,
                        "kind": "export",
                        "weights": payload.get("weights", ""),
                        "imgsz": item.get("args", {}).get("imgsz", ""),
                        "status": item.get("status", ""),
                        "precision": "",
                        "recall": "",
                        "map50": "",
                        "map50_95": "",
                        "inference_ms": "",
                        "preprocess_ms": "",
                        "postprocess_ms": "",
                        "artifact": item.get("stored", ""),
                        "error": item.get("error", ""),
                    }
                )
    return rows


def main():
    args = parse_args()
    rows = load_json_files(args.results_dir)
    json_path = args.results_dir / f"{args.output_prefix}.json"
    csv_path = args.results_dir / f"{args.output_prefix}.csv"
    write_json(json_path, {"rows": rows})

    fieldnames = [
        "source_file",
        "kind",
        "weights",
        "imgsz",
        "status",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "inference_ms",
        "preprocess_ms",
        "postprocess_ms",
        "artifact",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()

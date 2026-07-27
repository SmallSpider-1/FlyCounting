#!/usr/bin/env python3
"""Compare an isolated test-inference audit with the frozen benchmark output."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PIPELINE_ROOT = Path(__file__).resolve().parent
ORIGINAL_ROOT = Path(
    "/home/admin1/Projects/ultralytics-main/detection_model_benchmark/"
    "results/fruitfly_detection_v1/benchmark_640_e100_s42"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    args = parser.parse_args()
    audit_root = args.audit_root.resolve()

    sys.path.insert(0, str(PIPELINE_ROOT))
    import common
    from finalize_benchmark import DOMAINS, coco_metrics, precision_recall, validate_prediction_ids

    comparison_root = audit_root / "comparison"
    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    image_validation: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "models": {},
    }
    all_hashes_equal = True
    maximum_delta = 0.0

    for model_key, spec in common.MODEL_SPECS.items():
        run_name = spec["run_name"]
        old_prediction = ORIGINAL_ROOT / "runs" / run_name / "predictions/test_mixed.json"
        new_prediction = audit_root / "runs" / run_name / "predictions/test_mixed.json"
        old_hash = sha256(old_prediction)
        new_hash = sha256(new_prediction)
        old_data = json.loads(old_prediction.read_text(encoding="utf-8"))
        new_data = json.loads(new_prediction.read_text(encoding="utf-8"))
        hashes_equal = old_hash == new_hash
        all_hashes_equal = all_hashes_equal and hashes_equal
        prediction_rows.append(
            {
                "model": model_key,
                "display": spec["display"],
                "old_sha256": old_hash,
                "new_sha256": new_hash,
                "byte_identical": hashes_equal,
                "old_rows": len(old_data),
                "new_rows": len(new_data),
                "row_count_identical": len(old_data) == len(new_data),
            }
        )

        original_metrics = json.loads(
            (ORIGINAL_ROOT / "runs" / run_name / "evaluation/central_metrics.json").read_text(encoding="utf-8")
        )
        threshold = float(original_metrics["val_f1_threshold"]["threshold"])
        new_pr = precision_recall(DOMAINS["mixed"], new_prediction, threshold)
        new_domains: dict[str, Any] = {}
        image_validation["models"][model_key] = {}
        for domain, truth_path in DOMAINS.items():
            truth = json.loads(truth_path.read_text(encoding="utf-8"))
            if domain == "mixed":
                domain_prediction = new_prediction
                domain_data = new_data
            else:
                valid_ids = {int(row["id"]) for row in truth["images"]}
                domain_data = [row for row in new_data if int(row["image_id"]) in valid_ids]
                domain_prediction = audit_root / "runs" / run_name / "predictions" / f"test_{domain}.json"
                domain_prediction.parent.mkdir(parents=True, exist_ok=True)
                domain_prediction.write_text(
                    json.dumps(domain_data, ensure_ascii=False, separators=(",", ":")) + "\n",
                    encoding="utf-8",
                )
            image_validation["models"][model_key][domain] = validate_prediction_ids(domain_data, truth)
            metrics = coco_metrics(truth_path, domain_prediction)
            new_domains[domain] = metrics
            old = original_metrics["test_domains"][domain]
            row: dict[str, Any] = {"model": model_key, "display": spec["display"], "domain": domain}
            for key in ("mAP50_95", "AP50", "AP75", "AR"):
                old_value = float(old[key])
                new_value = float(metrics[key])
                delta = new_value - old_value
                maximum_delta = max(maximum_delta, abs(delta))
                row[f"old_{key}"] = old_value
                row[f"new_{key}"] = new_value
                row[f"delta_{key}"] = delta
            metric_rows.append(row)

        old_pr = original_metrics["test_precision_recall"]
        for key in ("precision", "recall", "f1"):
            maximum_delta = max(maximum_delta, abs(float(new_pr[key]) - float(old_pr[key])))
        atomic_json(
            audit_root / "runs" / run_name / "evaluation/central_metrics.json",
            {
                "model": model_key,
                "source": "independent inference rerun after R2/test class repair",
                "frozen_val_threshold": threshold,
                "test_precision_recall": new_pr,
                "test_domains": new_domains,
            },
        )

    write_csv(
        comparison_root / "prediction_comparison.csv",
        [
            "model", "display", "old_sha256", "new_sha256", "byte_identical",
            "old_rows", "new_rows", "row_count_identical",
        ],
        prediction_rows,
    )
    metric_fields = ["model", "display", "domain"]
    for key in ("mAP50_95", "AP50", "AP75", "AR"):
        metric_fields.extend([f"old_{key}", f"new_{key}", f"delta_{key}"])
    write_csv(comparison_root / "metric_comparison.csv", metric_fields, metric_rows)
    atomic_json(comparison_root / "image_id_validation.json", image_validation)
    summary = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "models": len(prediction_rows),
        "domains_per_model": 4,
        "all_prediction_files_byte_identical": all_hashes_equal,
        "all_prediction_row_counts_identical": all(row["row_count_identical"] for row in prediction_rows),
        "maximum_absolute_metric_delta": maximum_delta,
        "all_metrics_identical": maximum_delta == 0.0,
        "canonical_ground_truth": {domain: str(path) for domain, path in DOMAINS.items()},
        "prediction_comparison": str(comparison_root / "prediction_comparison.csv"),
        "metric_comparison": str(comparison_root / "metric_comparison.csv"),
        "image_id_validation": str(comparison_root / "image_id_validation.json"),
    }
    atomic_json(comparison_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not all_hashes_equal or maximum_delta != 0.0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare the video backend numerically against frozen val_mixed predictions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection_model_benchmark.rtdetr_r18_backend import (
    DEFAULT_RUNTIME_CONFIG,
    DEFAULT_WEIGHTS,
    RTDETRR18Backend,
    RUN_DIR,
    require_benchmark_environment,
    sha256_file,
)


DEFAULT_ANNOTATIONS = ROOT / "datasets" / "fruitfly_detection_v1" / "coco" / "annotations" / "instances_val.json"
DEFAULT_IMAGES = ROOT / "datasets" / "fruitfly_detection_v1" / "coco" / "val"
DEFAULT_FROZEN = RUN_DIR / "predictions" / "val_mixed.json"
DEFAULT_OUTPUT = (
    ROOT
    / "project_results"
    / "detection_cache"
    / "rtdetr_r18_official_v1_validation"
    / "val_consistency.json"
)
EXPECTED_FROZEN_SHA256 = "25d0b96e3b22554a8fde18b73f39f8b4422236c55e1ff206d1b4301b6d9ddffe"
EXPORT_CONFIDENCE = 0.001
FROZEN_BATCH_SIZE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RT-DETR-R18 frozen val numerical consistency test.")
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def canonical_row(image_id: int, row: list[float | int]) -> dict[str, Any]:
    x1, y1, x2, y2, score, class_id = row
    return {
        "image_id": int(image_id),
        "category_id": int(class_id),
        "bbox": [
            round(float(x1), 4),
            round(float(y1), 4),
            round(max(0.0, float(x2) - float(x1)), 4),
            round(max(0.0, float(y2) - float(y1)), 4),
        ],
        "score": float(score),
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    require_benchmark_environment()
    frozen_sha256 = sha256_file(args.frozen)
    if frozen_sha256 != EXPECTED_FROZEN_SHA256:
        raise ValueError(
            f"Frozen val prediction SHA-256 mismatch: expected {EXPECTED_FROZEN_SHA256}, got {frozen_sha256}"
        )
    annotation = json.loads(args.annotations.read_text(encoding="utf-8"))
    records = annotation["images"]
    frozen_rows = json.loads(args.frozen.read_text(encoding="utf-8"))
    frozen_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in frozen_rows:
        frozen_by_image[int(row["image_id"])].append(row)

    backend = RTDETRR18Backend(args.weights, args.runtime_config, args.device)
    from PIL import Image

    images_compared = 0
    boxes_compared = 0
    exact_rows = 0
    mismatched_images: list[dict[str, Any]] = []
    max_bbox_abs_error = 0.0
    max_confidence_abs_error = 0.0
    for offset in range(0, len(records), FROZEN_BATCH_SIZE):
        batch_records = records[offset : offset + FROZEN_BATCH_SIZE]
        pil_images = []
        for record in batch_records:
            path = args.images / record["file_name"]
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                if rgb.size != (int(record["width"]), int(record["height"])):
                    raise ValueError(f"Annotation image size mismatch: {path}")
                pil_images.append(rgb.copy())
        predicted_batches = backend.predict_pil(pil_images, EXPORT_CONFIDENCE)
        for record, predicted in zip(batch_records, predicted_batches):
            actual = [canonical_row(int(record["id"]), row) for row in predicted]
            expected = frozen_by_image[int(record["id"])]
            images_compared += 1
            boxes_compared += min(len(actual), len(expected))
            image_exact_rows = 0
            labels_and_order_match = len(actual) == len(expected)
            for actual_row, expected_row in zip(actual, expected):
                if actual_row == expected_row:
                    exact_rows += 1
                    image_exact_rows += 1
                if actual_row["category_id"] != expected_row["category_id"]:
                    labels_and_order_match = False
                bbox_error = max(
                    abs(float(left) - float(right))
                    for left, right in zip(actual_row["bbox"], expected_row["bbox"])
                )
                score_error = abs(float(actual_row["score"]) - float(expected_row["score"]))
                max_bbox_abs_error = max(max_bbox_abs_error, bbox_error)
                max_confidence_abs_error = max(max_confidence_abs_error, score_error)
            if (
                len(actual) != len(expected)
                or image_exact_rows != len(expected)
                or not labels_and_order_match
            ):
                mismatched_images.append(
                    {
                        "image_id": int(record["id"]),
                        "file_name": record["file_name"],
                        "expected_boxes": len(expected),
                        "actual_boxes": len(actual),
                        "exact_rows": image_exact_rows,
                        "labels_and_order_match": labels_and_order_match,
                    }
                )
        print(
            f"val consistency {min(offset + FROZEN_BATCH_SIZE, len(records))}/{len(records)}",
            flush=True,
        )

    passed = (
        not mismatched_images
        and images_compared == len(records)
        and exact_rows == len(frozen_rows)
        and max_bbox_abs_error == 0.0
        and max_confidence_abs_error == 0.0
    )
    report = {
        "passed": passed,
        "comparison": "exact canonical rows after frozen 4-decimal xywh export",
        "images_compared": images_compared,
        "frozen_rows": len(frozen_rows),
        "boxes_compared": boxes_compared,
        "exact_rows": exact_rows,
        "mismatched_image_count": len(mismatched_images),
        "mismatched_images_first_20": mismatched_images[:20],
        "max_bbox_abs_error": max_bbox_abs_error,
        "max_confidence_abs_error": max_confidence_abs_error,
        "batch_size": FROZEN_BATCH_SIZE,
        "confidence_export_threshold": EXPORT_CONFIDENCE,
        "checkpoint_state": backend.checkpoint_state,
        "checkpoint_sha256": backend.checkpoint_sha256,
        "runtime_config_sha256": backend.runtime_config_sha256,
        "source_commit": backend.source_commit,
        "frozen_prediction": str(args.frozen.resolve()),
        "frozen_prediction_sha256": frozen_sha256,
    }
    atomic_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not passed:
        raise SystemExit("RT-DETR val numerical consistency FAILED")


if __name__ == "__main__":
    main()

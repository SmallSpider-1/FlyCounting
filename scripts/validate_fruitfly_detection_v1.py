#!/usr/bin/env python3
"""Validate the generated fruitfly_detection_v1 dataset end to end."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from PIL import Image


DOMAINS = ("C", "R1", "R2")
SPLITS = ("train", "val", "test")
TIMEZONE = ZoneInfo("Asia/Shanghai")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_samples(root: Path) -> list[dict[str, str]]:
    with (root / "metadata" / "samples.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def parse_yolo(path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        fields = raw_line.split()
        if len(fields) != 5:
            raise RuntimeError(f"{path}:{line_number}: expected five fields")
        class_id = int(fields[0])
        cx, cy, width, height = map(float, fields[1:])
        if class_id not in (0, 1):
            raise RuntimeError(f"{path}:{line_number}: invalid class {class_id}")
        values = (cx, cy, width, height)
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"{path}:{line_number}: non-finite bbox")
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise RuntimeError(f"{path}:{line_number}: invalid normalized bbox")
        boxes.append((class_id, cx, cy, width, height))
    return boxes


def scan_image(root: Path, sample: dict[str, str]) -> dict[str, Any]:
    path = root / sample["target_image"]
    payload = path.read_bytes()
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        width, height = image.size
        image_format = image.format
        image_mode = image.mode
    return {
        "path": path,
        "sha256": sha256_bytes(payload),
        "width": width,
        "height": height,
        "format": image_format,
        "mode": image_mode,
    }


def compare_coco(
    root: Path,
    split: str,
    samples: list[dict[str, str]],
    category_offset: int,
) -> tuple[int, int]:
    suffix = "2017" if category_offset else ""
    path = root / "coco" / "annotations" / f"instances_{split}{suffix}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_categories = {category_offset, category_offset + 1}
    actual_categories = {int(row["id"]) for row in payload["categories"]}
    if actual_categories != expected_categories:
        raise RuntimeError(
            f"{path}: categories {actual_categories} != {expected_categories}"
        )

    split_samples = [sample for sample in samples if sample["split"] == split]
    sample_by_id = {int(sample["image_id"]): sample for sample in split_samples}
    image_by_id = {int(image["id"]): image for image in payload["images"]}
    if set(sample_by_id) != set(image_by_id):
        raise RuntimeError(f"{path}: COCO image IDs differ from samples.csv")

    expected_boxes: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = defaultdict(list)
    for image_id, sample in sample_by_id.items():
        image = image_by_id[image_id]
        if image["file_name"] != sample["target_filename"]:
            raise RuntimeError(f"{path}: filename mismatch for image {image_id}")
        width = int(sample["width"])
        height = int(sample["height"])
        yolo_path = root / sample["target_label"]
        for class_id, cx, cy, box_width, box_height in parse_yolo(yolo_path):
            bbox = (
                (cx - box_width / 2) * width,
                (cy - box_height / 2) * height,
                box_width * width,
                box_height * height,
            )
            expected_boxes[image_id].append((class_id + category_offset, bbox))

    actual_boxes: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = defaultdict(list)
    annotation_ids = set()
    for annotation in payload["annotations"]:
        annotation_id = int(annotation["id"])
        if annotation_id in annotation_ids:
            raise RuntimeError(f"{path}: duplicate annotation ID {annotation_id}")
        annotation_ids.add(annotation_id)
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        bbox = tuple(map(float, annotation["bbox"]))
        if category_id not in expected_categories or len(bbox) != 4:
            raise RuntimeError(f"{path}: invalid annotation {annotation_id}")
        x, y, width, height = bbox
        expected_polygon = [
            [x, y, x + width, y, x + width, y + height, x, y + height]
        ]
        segmentation = annotation.get("segmentation")
        if not isinstance(segmentation, list) or len(segmentation) != 1 or len(segmentation[0]) != 8:
            raise RuntimeError(f"{path}: invalid rectangular segmentation in annotation {annotation_id}")
        if any(
            not math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-7)
            for left, right in zip(segmentation[0], expected_polygon[0])
        ):
            raise RuntimeError(f"{path}: segmentation differs from bbox in annotation {annotation_id}")
        if not math.isclose(float(annotation["area"]), bbox[2] * bbox[3], rel_tol=1e-9):
            raise RuntimeError(f"{path}: area mismatch in annotation {annotation_id}")
        actual_boxes[image_id].append((category_id, bbox))

    if set(expected_boxes) != set(actual_boxes):
        # Empty images are absent from both dictionaries, so this compares only positive images.
        raise RuntimeError(f"{path}: positive-image annotation coverage differs")
    for image_id in expected_boxes:
        expected = expected_boxes[image_id]
        actual = actual_boxes[image_id]
        if len(expected) != len(actual):
            raise RuntimeError(f"{path}: box count mismatch for image {image_id}")
        for expected_box, actual_box in zip(expected, actual):
            if expected_box[0] != actual_box[0] or any(
                not math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-7)
                for left, right in zip(expected_box[1], actual_box[1])
            ):
                raise RuntimeError(f"{path}: bbox mismatch for image {image_id}")
    return len(payload["images"]), len(payload["annotations"])


def validate(root: Path, write_report: bool) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"dataset root does not exist: {root}")
    samples = load_samples(root)
    if not samples:
        raise RuntimeError("samples.csv is empty")
    if len({sample["sample_id"] for sample in samples}) != len(samples):
        raise RuntimeError("duplicate sample_id in samples.csv")
    if len({sample["image_id"] for sample in samples}) != len(samples):
        raise RuntimeError("duplicate image_id in samples.csv")

    errors: list[str] = []
    image_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(scan_image, root, sample): sample for sample in samples}
        for done, future in enumerate(as_completed(futures), start=1):
            sample = futures[future]
            try:
                result = future.result()
                if result["sha256"] != sample["image_sha256"]:
                    raise RuntimeError("image SHA-256 differs from samples.csv")
                if result["width"] != int(sample["width"]) or result["height"] != int(
                    sample["height"]
                ):
                    raise RuntimeError("image dimensions differ from samples.csv")
                if result["format"] != sample["image_format"] or result["mode"] != sample["image_mode"]:
                    raise RuntimeError("image format/mode differs from samples.csv")
                image_results.append(result)
            except Exception as exc:
                errors.append(f"{sample['target_image']}: {exc}")
            if done % 1000 == 0:
                print(f"validated images {done}/{len(samples)}", flush=True)
    if errors:
        raise RuntimeError("image validation failures: " + "; ".join(errors[:10]))

    target_hashes = Counter(result["sha256"] for result in image_results)
    duplicate_hashes = {digest: count for digest, count in target_hashes.items() if count > 1}
    if duplicate_hashes:
        raise RuntimeError(f"duplicate target image hashes: {list(duplicate_hashes.items())[:5]}")

    totals = Counter()
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    by_domain: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        label_path = root / sample["target_label"]
        label_payload = label_path.read_bytes()
        if sha256_bytes(label_payload) != sample["target_label_sha256"]:
            raise RuntimeError(f"target label SHA-256 mismatch: {label_path}")
        boxes = parse_yolo(label_path)
        classes = Counter(box[0] for box in boxes)
        expected = (
            int(sample["object_count"]),
            int(sample["class_0_count"]),
            int(sample["class_1_count"]),
            int(sample["is_empty"]),
        )
        actual = (len(boxes), classes[0], classes[1], int(not boxes))
        if actual != expected:
            raise RuntimeError(f"label statistics mismatch: {label_path}")
        row = Counter(
            images=1,
            boxes=len(boxes),
            class_0=classes[0],
            class_1=classes[1],
            empty_images=int(not boxes),
        )
        totals.update(row)
        by_split[sample["split"]].update(row)
        by_domain[sample["domain"]].update(row)

    coco_results = {}
    for split in SPLITS:
        coco_results[f"{split}_zero_based"] = compare_coco(
            root, split, samples, category_offset=0
        )
        coco_results[f"{split}_faster_rcnn"] = compare_coco(
            root, split, samples, category_offset=1
        )

    for split in SPLITS:
        coco_link = root / "coco" / split
        faster_link = root / "coco" / f"{split}2017"
        if not coco_link.is_symlink() or not faster_link.is_symlink():
            raise RuntimeError(f"missing COCO image symlink for {split}")
        for domain in DOMAINS:
            split_list = root / "yolo" / "splits" / f"{split}_{domain}.txt"
            paths = [Path(line) for line in split_list.read_text(encoding="utf-8").splitlines() if line]
            expected_count = sum(
                sample["split"] == split and sample["domain"] == domain for sample in samples
            )
            if len(paths) != expected_count or any(not path.is_file() for path in paths):
                raise RuntimeError(f"invalid split list: {split_list}")

    stats = json.loads((root / "metadata" / "dataset_stats.json").read_text(encoding="utf-8"))
    expected_totals = {
        key: int(stats["totals"][key])
        for key in ("images", "boxes", "class_0", "class_1", "empty_images")
    }
    actual_totals = {key: totals[key] for key in expected_totals}
    if actual_totals != expected_totals:
        raise RuntimeError(f"dataset_stats totals mismatch: {actual_totals} != {expected_totals}")

    report = {
        "dataset": "fruitfly_detection_v1",
        "validated_at": datetime.now(TIMEZONE).isoformat(timespec="seconds"),
        "root": str(root),
        "status": "PASS",
        "checks": {
            "sample_ids_unique": True,
            "image_ids_unique": True,
            "all_images_full_decode": True,
            "all_image_hashes_match_source_manifest": True,
            "all_target_images_sha256_unique": True,
            "all_labels_valid_and_match_manifest": True,
            "zero_based_coco_matches_yolo": True,
            "faster_rcnn_coco_matches_yolo": True,
            "domain_split_lists_valid": True,
            "coco_image_symlinks_valid": True,
        },
        "totals": actual_totals,
        "by_split": {key: dict(value) for key, value in by_split.items()},
        "by_domain": {key: dict(value) for key, value in by_domain.items()},
        "coco_main_files": {
            key: {"images": value[0], "annotations": value[1]}
            for key, value in coco_results.items()
        },
        "dataset_fingerprint_sha256": stats["dataset_fingerprint_sha256"],
    }
    if write_report:
        (root / "metadata" / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/fruitfly_detection_v1"),
    )
    parser.add_argument("--write-report", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    try:
        result = validate(arguments.dataset_root, arguments.write_report)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as error:
        print(f"VALIDATION FAILED: {error}", file=sys.stderr)
        raise

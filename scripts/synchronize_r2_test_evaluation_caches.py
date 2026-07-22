#!/usr/bin/env python3
"""Synchronize derived native test-label caches with the current YOLO labels.

This utility intentionally does not run inference or metric evaluation. It rebuilds
only the cached class/box arrays from the current label text files, while refusing
to write if image counts, box geometry, or the expected corrected class totals do
not match.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


EXPECTED_IMAGES = 1517
EXPECTED_ALL_COUNTS = {0: 1220, 1: 1486}
EXPECTED_R2_IMAGES = 405
EXPECTED_R2_COUNTS = {0: 335, 1: 1065}


def label_path_for_image(image_path: str) -> Path:
    path = Path(image_path)
    parts = list(path.parts)
    indexes = [index for index, part in enumerate(parts) if part == "images"]
    if not indexes:
        raise ValueError(f"Cannot map image path to label path: {path}")
    parts[indexes[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def load_yolo_rows(path: Path, np):
    rows: list[list[float]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5 or fields[0] not in {"0", "1"}:
            raise ValueError(f"{path}:{line_number}: invalid two-class YOLO row")
        row = [float(value) for value in fields]
        rows.append(row)
    if not rows:
        return np.empty((0, 5), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32).reshape(-1, 5)


def add_counts(counts: dict[int, int], classes) -> None:
    for value in classes.reshape(-1):
        class_id = int(value)
        if class_id not in counts:
            raise ValueError(f"Unexpected class ID {class_id}")
        counts[class_id] += 1


def check_geometry(cached, current, np, image_path: str) -> None:
    if cached.shape != current.shape or not np.array_equal(cached.astype(np.float32), current):
        raise ValueError(f"Cached box geometry does not match current label: {image_path}")


def check_totals(image_count: int, counts: dict[int, int], r2_images: int, r2_counts: dict[int, int]) -> None:
    if image_count != EXPECTED_IMAGES:
        raise ValueError(f"Expected {EXPECTED_IMAGES} images, found {image_count}")
    if counts != EXPECTED_ALL_COUNTS:
        raise ValueError(f"Expected mixed-test counts {EXPECTED_ALL_COUNTS}, found {counts}")
    if r2_images != EXPECTED_R2_IMAGES:
        raise ValueError(f"Expected {EXPECTED_R2_IMAGES} R2 images, found {r2_images}")
    if r2_counts != EXPECTED_R2_COUNTS:
        raise ValueError(f"Expected R2 counts {EXPECTED_R2_COUNTS}, found {r2_counts}")


def atomic_numpy_save(path: Path, payload, np) -> None:
    mode = path.stat().st_mode
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        np.save(stream, payload)
        temporary = Path(stream.name)
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def synchronize_ultralytics(path: Path) -> dict[str, object]:
    import numpy as np

    cache = np.load(path, allow_pickle=True).item()
    labels = cache.get("labels")
    if not isinstance(labels, list):
        raise TypeError(f"Unsupported Ultralytics cache structure: {path}")

    before_counts = {0: 0, 1: 0}
    before_r2_counts = {0: 0, 1: 0}
    after_counts = {0: 0, 1: 0}
    after_r2_counts = {0: 0, 1: 0}
    r2_images = 0
    for entry in labels:
        image_path = str(entry["im_file"])
        is_r2 = Path(image_path).name.startswith("R2__")
        old_classes = np.asarray(entry["cls"])
        add_counts(before_counts, old_classes)
        if is_r2:
            r2_images += 1
            add_counts(before_r2_counts, old_classes)

        current = load_yolo_rows(label_path_for_image(image_path), np)
        check_geometry(np.asarray(entry["bboxes"]), current[:, 1:5], np, image_path)
        entry["cls"] = current[:, 0:1].astype(old_classes.dtype, copy=False)
        add_counts(after_counts, entry["cls"])
        if is_r2:
            add_counts(after_r2_counts, entry["cls"])

    check_totals(len(labels), after_counts, r2_images, after_r2_counts)
    atomic_numpy_save(path, cache, np)
    return {
        "format": "ultralytics_numpy_cache",
        "images": len(labels),
        "before_counts": before_counts,
        "before_r2_counts": before_r2_counts,
        "after_counts": after_counts,
        "after_r2_counts": after_r2_counts,
    }


def synchronize_yolov7(path: Path) -> dict[str, object]:
    import numpy as np
    import torch

    cache = torch.load(path, map_location="cpu")
    image_keys = [key for key in cache if isinstance(key, str) and key not in {"hash", "results", "version", "msgs"}]
    before_counts = {0: 0, 1: 0}
    before_r2_counts = {0: 0, 1: 0}
    after_counts = {0: 0, 1: 0}
    after_r2_counts = {0: 0, 1: 0}
    r2_images = 0
    for image_path in image_keys:
        value = cache[image_path]
        if not isinstance(value, (list, tuple)) or not value:
            raise TypeError(f"Unsupported YOLOv7 cache entry: {image_path}")
        cached_rows = np.asarray(value[0])
        if cached_rows.ndim != 2 or cached_rows.shape[1] != 5:
            raise ValueError(f"Unexpected YOLOv7 label array: {image_path} {cached_rows.shape}")
        is_r2 = Path(image_path).name.startswith("R2__")
        add_counts(before_counts, cached_rows[:, 0:1])
        if is_r2:
            r2_images += 1
            add_counts(before_r2_counts, cached_rows[:, 0:1])

        current = load_yolo_rows(label_path_for_image(image_path), np)
        check_geometry(cached_rows[:, 1:5], current[:, 1:5], np, image_path)
        replacement = current.astype(cached_rows.dtype, copy=False)
        if isinstance(value, tuple):
            cache[image_path] = (replacement, *value[1:])
        else:
            value[0] = replacement
        add_counts(after_counts, replacement[:, 0:1])
        if is_r2:
            add_counts(after_r2_counts, replacement[:, 0:1])

    check_totals(len(image_keys), after_counts, r2_images, after_r2_counts)
    mode = path.stat().st_mode
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    torch.save(cache, temporary)
    os.chmod(temporary, mode)
    os.replace(temporary, path)
    return {
        "format": "yolov7_torch_cache",
        "images": len(image_keys),
        "before_counts": before_counts,
        "before_r2_counts": before_r2_counts,
        "after_counts": after_counts,
        "after_r2_counts": after_r2_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=("ultralytics", "yolov7"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    path = args.cache.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if args.format == "ultralytics":
        result = synchronize_ultralytics(path)
    else:
        result = synchronize_yolov7(path)
    result["cache"] = str(path)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

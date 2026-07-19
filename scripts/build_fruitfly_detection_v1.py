#!/usr/bin/env python3
"""Build the unified C/R1/R2 fruit-fly detection benchmark dataset.

The source ``data/`` tree is treated as read-only.  The builder preserves the
source train/val/test assignments, namespaces image names, emits a canonical
YOLO view, and derives two COCO views:

* category IDs 0/1 for DEIM and RT-DETR (``remap_mscoco_category: false``)
* category IDs 1/2 for torchvision Faster R-CNN (background is 0)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from PIL import Image


DOMAINS = ("C", "R1", "R2")
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {
    0: "Bactrocera dorsalis",
    1: "Bactrocera correcta",
}
# The current data/ tree was normalized by the user before this builder was run.
SOURCE_CLASS_MAP = {
    "C": {0: 0, 1: 1},
    "R1": {0: 0},
    "R2": {0: 0, 1: 1},
}
TIMEZONE = ZoneInfo("Asia/Shanghai")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def discover_pairs(source_root: Path, domain: str, split: str) -> list[tuple[Path, Path]]:
    image_dir = source_root / domain / "images" / split
    label_dir = source_root / domain / "labels" / split
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise RuntimeError(f"missing source directory for {domain}/{split}")

    images: dict[str, Path] = {}
    for path in image_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stem in images:
                raise RuntimeError(f"duplicate image stem in {image_dir}: {path.stem}")
            images[path.stem] = path

    labels = {
        path.stem: path
        for path in label_dir.glob("*.txt")
        if path.name.lower() != "classes.txt"
    }
    missing = sorted(images.keys() - labels.keys())
    orphan = sorted(labels.keys() - images.keys())
    if missing or orphan:
        raise RuntimeError(
            f"pairing failure in {domain}/{split}: "
            f"missing_labels={missing[:10]}, orphan_labels={orphan[:10]}"
        )
    return [(images[stem], labels[stem]) for stem in sorted(images)]


def parse_label(path: Path, domain: str) -> tuple[list[dict[str, Any]], str]:
    mapping = SOURCE_CLASS_MAP[domain]
    boxes: list[dict[str, Any]] = []
    normalized_lines: list[str] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise RuntimeError(f"{path}:{line_number}: expected 5 fields")
        try:
            source_class = int(fields[0])
            cx, cy, width, height = map(float, fields[1:])
        except ValueError as exc:
            raise RuntimeError(f"{path}:{line_number}: non-numeric label") from exc
        if str(source_class) != fields[0] or source_class not in mapping:
            raise RuntimeError(f"{path}:{line_number}: unsupported class {fields[0]}")
        values = (cx, cy, width, height)
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"{path}:{line_number}: non-finite bbox")
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise RuntimeError(f"{path}:{line_number}: invalid normalized bbox")
        if (
            cx - width / 2 < -1e-6
            or cy - height / 2 < -1e-6
            or cx + width / 2 > 1 + 1e-6
            or cy + height / 2 > 1 + 1e-6
        ):
            raise RuntimeError(f"{path}:{line_number}: bbox crosses image boundary")

        target_class = mapping[source_class]
        boxes.append(
            {
                "class_id": target_class,
                "cx": cx,
                "cy": cy,
                "width": width,
                "height": height,
            }
        )
        normalized_lines.append(f"{target_class} {' '.join(fields[1:])}")

    text = "\n".join(normalized_lines)
    if text:
        text += "\n"
    return boxes, text


def materialize_image(source: Path, target: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(source, target)
    elif mode == "hardlink":
        os.link(source, target)
    else:  # pragma: no cover - argparse protects this branch
        raise ValueError(mode)


def coco_payload(
    images: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    category_offset: int,
    description: str,
    generated_at: str,
) -> dict[str, Any]:
    converted_annotations = []
    for annotation in annotations:
        converted = dict(annotation)
        converted["category_id"] = annotation["category_id"] + category_offset
        converted_annotations.append(converted)
    return {
        "info": {
            "description": description,
            "version": "1.0",
            "year": 2026,
            "date_created": generated_at,
        },
        "licenses": [],
        "images": images,
        "annotations": converted_annotations,
        "categories": [
            {
                "id": class_id + category_offset,
                "name": name,
                "supercategory": "fruit_fly",
            }
            for class_id, name in CLASS_NAMES.items()
        ],
    }


def write_yaml_files(build_root: Path, final_root: Path) -> None:
    yolo_root = final_root / "yolo"
    names_block = "\n".join(f"  {key}: {value}" for key, value in CLASS_NAMES.items())
    dataset_yaml = (
        "# Unified C/R1/R2 two-class benchmark dataset.\n"
        f"path: {yolo_root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 2\n"
        "names:\n"
        f"{names_block}\n"
    )
    (build_root / "yolo" / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")

    for domain in DOMAINS:
        domain_yaml = (
            f"# Mixed training set; {domain} validation/test view.\n"
            f"path: {yolo_root}\n"
            "train: images/train\n"
            f"val: splits/val_{domain}.txt\n"
            f"test: splits/test_{domain}.txt\n"
            "nc: 2\n"
            "names:\n"
            f"{names_block}\n"
        )
        (build_root / "yolo" / f"dataset_{domain}.yaml").write_text(
            domain_yaml, encoding="utf-8"
        )

    classes_yaml = (
        "dataset: fruitfly_detection_v1\n"
        "status: user_provided_normalized_data\n"
        "canonical_yolo_and_deim_rtdetr_ids:\n"
        f"{names_block}\n"
        "torchvision_faster_rcnn_ids:\n"
        "  0: __background__\n"
        "  1: Bactrocera dorsalis\n"
        "  2: Bactrocera correcta\n"
        "source_class_mapping:\n"
        "  C: {0: 0, 1: 1}\n"
        "  R1: {0: 0}\n"
        "  R2: {0: 0, 1: 1}\n"
        "class_order_basis: Current data tree supplied as normalized by the user on 2026-07-18.\n"
    )
    (build_root / "metadata" / "classes.yaml").write_text(classes_yaml, encoding="utf-8")


def write_readme(build_root: Path, final_root: Path, stats: dict[str, Any]) -> None:
    text = f"""# fruitfly_detection_v1

Unified two-class C/R1/R2 detection dataset generated from the read-only source
tree `{stats['source_root']}`.

## Frozen class order

- `0`: Bactrocera dorsalis
- `1`: Bactrocera correcta

DEIM and RT-DETR use `coco/annotations/instances_{{split}}.json` with category
IDs 0/1. Torchvision Faster R-CNN uses
`coco/annotations/instances_{{split}}2017.json` with foreground category IDs
1/2 because category 0 is reserved for background.

## Dataset roots

- YOLO: `{final_root / 'yolo'}`
- COCO: `{final_root / 'coco'}`

Images are copied once under `yolo/images/`. COCO image directories are relative
symbolic links to those copies. Source images and labels are not modified.

## Counts

- Images: {stats['totals']['images']}
- Bounding boxes: {stats['totals']['boxes']}
- Empty/background images: {stats['totals']['empty_images']}
- Exact duplicate image groups: {stats['duplicates']['groups']}
- Dataset fingerprint: `{stats['dataset_fingerprint_sha256']}`

See `metadata/dataset_stats.json`, `metadata/samples.csv`, and
`metadata/validation_report.json` for the auditable record.
"""
    (build_root / "README.md").write_text(text, encoding="utf-8")


def build(source_root: Path, output_root: Path, materialization: str) -> None:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if not source_root.is_dir():
        raise RuntimeError(f"source root does not exist: {source_root}")
    if output_root.exists():
        raise RuntimeError(f"refusing to overwrite existing target: {output_root}")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.building-", dir=output_root.parent)
    )
    generated_at = datetime.now(TIMEZONE).isoformat(timespec="seconds")
    print(f"building in {temp_root}", flush=True)

    for split in SPLITS:
        (temp_root / "yolo" / "images" / split).mkdir(parents=True, exist_ok=True)
        (temp_root / "yolo" / "labels" / split).mkdir(parents=True, exist_ok=True)
    (temp_root / "yolo" / "splits").mkdir(parents=True)
    (temp_root / "coco" / "annotations").mkdir(parents=True)
    (temp_root / "metadata").mkdir(parents=True)

    samples: list[dict[str, Any]] = []
    images_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    annotations_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    image_hash_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    counters: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    bbox_widths: list[float] = []
    bbox_heights: list[float] = []
    image_id = 0
    annotation_id = 0

    all_pairs: list[tuple[str, str, Path, Path]] = []
    for domain in DOMAINS:
        for split in SPLITS:
            all_pairs.extend(
                (domain, split, image_path, label_path)
                for image_path, label_path in discover_pairs(source_root, domain, split)
            )

    for index, (domain, split, source_image, source_label) in enumerate(all_pairs, start=1):
        image_id += 1
        target_filename = f"{domain}__{image_id:08d}{source_image.suffix.lower()}"
        target_image = temp_root / "yolo" / "images" / split / target_filename
        target_label = temp_root / "yolo" / "labels" / split / f"{Path(target_filename).stem}.txt"

        boxes, normalized_label = parse_label(source_label, domain)
        image_sha = sha256_file(source_image)
        label_source_sha = sha256_file(source_label)
        with Image.open(source_image) as image:
            image.load()
            width, height = image.size
            image_format = image.format
            image_mode = image.mode
        if width <= 0 or height <= 0:
            raise RuntimeError(f"invalid image dimensions: {source_image}")

        materialize_image(source_image, target_image, materialization)
        target_label.write_text(normalized_label, encoding="utf-8")
        label_target_sha = sha256_file(target_label)

        image_record = {
            "id": image_id,
            "file_name": target_filename,
            "width": width,
            "height": height,
            "domain": domain,
        }
        images_by_split[split].append(image_record)

        class_counts = Counter(box["class_id"] for box in boxes)
        for box in boxes:
            annotation_id += 1
            pixel_width = box["width"] * width
            pixel_height = box["height"] * height
            pixel_x = (box["cx"] - box["width"] / 2) * width
            pixel_y = (box["cy"] - box["height"] / 2) * height
            bbox_widths.append(pixel_width)
            bbox_heights.append(pixel_height)
            annotations_by_split[split].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": box["class_id"],
                    "bbox": [pixel_x, pixel_y, pixel_width, pixel_height],
                    "area": pixel_width * pixel_height,
                    "iscrowd": 0,
                    # Torchvision's official reference loader always converts
                    # polygons to masks, even for bbox-only detection.  A
                    # rectangle exactly matching the AABB keeps both targets
                    # geometrically identical.
                    "segmentation": [
                        [
                            pixel_x,
                            pixel_y,
                            pixel_x + pixel_width,
                            pixel_y,
                            pixel_x + pixel_width,
                            pixel_y + pixel_height,
                            pixel_x,
                            pixel_y + pixel_height,
                        ]
                    ],
                }
            )

        target_image_final = output_root / "yolo" / "images" / split / target_filename
        target_label_final = output_root / "yolo" / "labels" / split / target_label.name
        relative_source_image = source_image.relative_to(source_root)
        relative_source_label = source_label.relative_to(source_root)
        sample = {
            "sample_id": f"FFD1-{image_id:08d}",
            "image_id": image_id,
            "domain": domain,
            "split": split,
            "source_group": "",
            "split_origin": "user_provided",
            "source_image": str(relative_source_image),
            "source_label": str(relative_source_label),
            "target_image": str(target_image_final.relative_to(output_root)),
            "target_label": str(target_label_final.relative_to(output_root)),
            "target_filename": target_filename,
            "width": width,
            "height": height,
            "image_format": image_format,
            "image_mode": image_mode,
            "image_bytes": source_image.stat().st_size,
            "image_sha256": image_sha,
            "source_label_sha256": label_source_sha,
            "target_label_sha256": label_target_sha,
            "object_count": len(boxes),
            "class_0_count": class_counts[0],
            "class_1_count": class_counts[1],
            "is_empty": int(not boxes),
        }
        samples.append(sample)
        image_hash_groups[image_sha].append(
            {
                "domain": domain,
                "split": split,
                "source_image": str(relative_source_image),
            }
        )
        counter = counters[(domain, split)]
        counter["images"] += 1
        counter["boxes"] += len(boxes)
        counter["class_0"] += class_counts[0]
        counter["class_1"] += class_counts[1]
        counter["empty_images"] += int(not boxes)
        counter["image_bytes"] += source_image.stat().st_size
        if index % 500 == 0 or index == len(all_pairs):
            print(f"materialized {index}/{len(all_pairs)} images", flush=True)

    duplicate_groups = {
        digest: rows for digest, rows in image_hash_groups.items() if len(rows) > 1
    }
    cross_split_groups = {
        digest: rows
        for digest, rows in duplicate_groups.items()
        if len({row["split"] for row in rows}) > 1
    }
    if cross_split_groups:
        first = next(iter(cross_split_groups.items()))
        raise RuntimeError(f"cross-split exact duplicate detected: {first}")

    # Dataset fingerprint excludes generated timestamps and absolute output paths.
    fingerprint = hashlib.sha256()
    for sample in samples:
        fingerprint.update(
            (
                f"{sample['sample_id']}|{sample['domain']}|{sample['split']}|"
                f"{sample['source_image']}|{sample['image_sha256']}|"
                f"{sample['target_label_sha256']}\n"
            ).encode("utf-8")
        )
    dataset_fingerprint = fingerprint.hexdigest()

    for split in SPLITS:
        images = images_by_split[split]
        annotations = annotations_by_split[split]
        zero_payload = coco_payload(
            images,
            annotations,
            category_offset=0,
            description=f"fruitfly_detection_v1 {split} for DEIM/RT-DETR",
            generated_at=generated_at,
        )
        faster_payload = coco_payload(
            images,
            annotations,
            category_offset=1,
            description=f"fruitfly_detection_v1 {split} for torchvision Faster R-CNN",
            generated_at=generated_at,
        )
        json_dump(temp_root / "coco" / "annotations" / f"instances_{split}.json", zero_payload)
        json_dump(
            temp_root / "coco" / "annotations" / f"instances_{split}2017.json",
            faster_payload,
        )

        for domain in DOMAINS:
            domain_image_ids = {image["id"] for image in images if image["domain"] == domain}
            domain_images = [image for image in images if image["id"] in domain_image_ids]
            domain_annotations = [
                annotation
                for annotation in annotations
                if annotation["image_id"] in domain_image_ids
            ]
            json_dump(
                temp_root / "coco" / "annotations" / f"instances_{split}_{domain}.json",
                coco_payload(
                    domain_images,
                    domain_annotations,
                    category_offset=0,
                    description=f"fruitfly_detection_v1 {domain}/{split} for DEIM/RT-DETR",
                    generated_at=generated_at,
                ),
            )
            json_dump(
                temp_root
                / "coco"
                / "annotations"
                / f"instances_{split}2017_{domain}.json",
                coco_payload(
                    domain_images,
                    domain_annotations,
                    category_offset=1,
                    description=f"fruitfly_detection_v1 {domain}/{split} for torchvision Faster R-CNN",
                    generated_at=generated_at,
                ),
            )

    for split in SPLITS:
        os.symlink(f"../yolo/images/{split}", temp_root / "coco" / split)
        os.symlink(split, temp_root / "coco" / f"{split}2017")

    for split in SPLITS:
        for domain in DOMAINS:
            rows = [
                sample
                for sample in samples
                if sample["split"] == split and sample["domain"] == domain
            ]
            list_path = temp_root / "yolo" / "splits" / f"{split}_{domain}.txt"
            list_path.write_text(
                "".join(
                    f"{output_root / sample['target_image']}\n" for sample in rows
                ),
                encoding="utf-8",
            )

    sample_fields = list(samples[0].keys())
    with (temp_root / "metadata" / "samples.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=sample_fields)
        writer.writeheader()
        writer.writerows(samples)

    with (temp_root / "metadata" / "duplicate_report.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = [
            "duplicate_group_id",
            "sha256",
            "domain",
            "split",
            "source_image",
            "cross_split",
            "cross_domain",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group_id, (digest, rows) in enumerate(sorted(duplicate_groups.items()), start=1):
            cross_split = len({row["split"] for row in rows}) > 1
            cross_domain = len({row["domain"] for row in rows}) > 1
            for row in rows:
                writer.writerow(
                    {
                        "duplicate_group_id": group_id,
                        "sha256": digest,
                        **row,
                        "cross_split": int(cross_split),
                        "cross_domain": int(cross_domain),
                    }
                )

    def aggregate(rows: list[Counter[str]]) -> dict[str, int]:
        total: Counter[str] = Counter()
        for row in rows:
            total.update(row)
        return dict(total)

    stats = {
        "dataset": "fruitfly_detection_v1",
        "generated_at": generated_at,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "materialization": materialization,
        "class_names": {str(key): value for key, value in CLASS_NAMES.items()},
        "class_mapping_status": "user_provided_normalized_data",
        "split_status": "preserved_from_user_provided_source_tree",
        "source_group_metadata_status": "not_present_in_source_tree",
        "dataset_fingerprint_sha256": dataset_fingerprint,
        "totals": aggregate(list(counters.values())),
        "by_split": {
            split: aggregate([counters[(domain, split)] for domain in DOMAINS])
            for split in SPLITS
        },
        "by_domain": {
            domain: aggregate([counters[(domain, split)] for split in SPLITS])
            for domain in DOMAINS
        },
        "by_domain_and_split": {
            domain: {split: dict(counters[(domain, split)]) for split in SPLITS}
            for domain in DOMAINS
        },
        "image_dimensions": {
            "min_width": min(int(sample["width"]) for sample in samples),
            "max_width": max(int(sample["width"]) for sample in samples),
            "min_height": min(int(sample["height"]) for sample in samples),
            "max_height": max(int(sample["height"]) for sample in samples),
        },
        "bbox_dimensions_pixels": {
            "min_width": min(bbox_widths),
            "max_width": max(bbox_widths),
            "mean_width": sum(bbox_widths) / len(bbox_widths),
            "min_height": min(bbox_heights),
            "max_height": max(bbox_heights),
            "mean_height": sum(bbox_heights) / len(bbox_heights),
        },
        "duplicates": {
            "groups": len(duplicate_groups),
            "cross_split_groups": len(cross_split_groups),
            "unique_image_sha256": len(image_hash_groups),
        },
    }
    json_dump(temp_root / "metadata" / "dataset_stats.json", stats)
    json_dump(
        temp_root / "metadata" / "build_info.json",
        {
            "builder": str(Path(__file__).resolve()),
            "generated_at": generated_at,
            "python": sys.version,
            "pillow": Image.__version__,
            "source_root": str(source_root),
            "output_root": str(output_root),
            "materialization": materialization,
            "source_files_modified": False,
            "dataset_fingerprint_sha256": dataset_fingerprint,
        },
    )
    write_yaml_files(temp_root, output_root)
    write_readme(temp_root, output_root, stats)

    temp_root.rename(output_root)
    print(f"build complete: {output_root}")
    print(f"dataset fingerprint: {dataset_fingerprint}")
    print(json.dumps(stats["totals"], ensure_ascii=False, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data"),
        help="Read-only C/R1/R2 source root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/fruitfly_detection_v1"),
    )
    parser.add_argument(
        "--materialization",
        choices=("copy", "hardlink"),
        default="copy",
        help="How to place the single canonical image copy under yolo/images",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    try:
        build(arguments.source_root, arguments.output_root, arguments.materialization)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise

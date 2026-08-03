#!/usr/bin/env python3
"""Cache one frozen detector's predictions for the standardized fruitfly MOT frames."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_common.numeric_cache import NumericCacheWriter, iter_cache_frames, read_cache_header
from detection_model_benchmark.rtdetr_r18_backend import (
    DEFAULT_RUNTIME_CONFIG,
    DEFAULT_WEIGHTS,
    EXPECTED_CLASS_NAMES,
    RTDETRR18Backend,
    sha256_file,
)


DEFAULT_DATASET = ROOT / "datasets" / "fruitfly_mot_v1"
DEFAULT_OUTPUT = (
    ROOT
    / "project_results"
    / "fruitfly_mot_v1"
    / "rtdetr_r18_official_defaults_all_v1"
    / "detections"
    / "numeric_cache"
)
DEFAULT_CONFIDENCE = 0.10
DEFAULT_BATCH_SIZE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_sequences(dataset_root: Path) -> list[dict[str, str]]:
    manifest = dataset_root / "manifests" / "sequences.csv"
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 15 or len({row["sequence_id"] for row in rows}) != len(rows):
        raise ValueError(f"Expected 15 unique sequences in {manifest}, got {len(rows)}")
    return sorted(rows, key=lambda row: row["sequence_id"])


def image_files(dataset_root: Path, row: dict[str, str]) -> list[Path]:
    image_root = dataset_root / row["source_frame_dir"]
    paths = sorted(image_root.glob("*.jpg"))
    expected = int(row["num_frames"])
    if len(paths) != expected:
        raise ValueError(f"{row['sequence_id']} image count mismatch: expected {expected}, got {len(paths)}")
    expected_names = [f"{index:06d}.jpg" for index in range(1, expected + 1)]
    actual_names = [path.name for path in paths]
    if actual_names != expected_names:
        raise ValueError(f"{row['sequence_id']} image names are not contiguous MOT frame numbers")
    return paths


def file_hashes(path: Path) -> tuple[str, str]:
    whole = hashlib.sha256()
    payload = hashlib.sha256()
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream):
            whole.update(line)
            if line_number > 0 and line.startswith(b'{"f":'):
                payload.update(line)
    return whole.hexdigest(), payload.hexdigest()


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_existing(path: Path, row: dict[str, str], producer: dict[str, Any]) -> dict[str, Any]:
    header = read_cache_header(path, "detections")
    sequence_id = row["sequence_id"]
    if header.get("manifest", {}).get("sequence_id") != sequence_id:
        raise ValueError(f"Existing cache sequence mismatch: {path}")
    if header.get("producer", {}).get("detector_config_sha256") != producer["detector_config_sha256"]:
        raise ValueError(f"Existing cache detector config mismatch: {path}")
    frames = 0
    detections = 0
    per_class = {0: 0, 1: 0}
    for _frame_index, rows in iter_cache_frames(path, "detections", require_complete=True):
        frames += 1
        detections += len(rows)
        for detection in rows:
            class_id = int(detection[5])
            if class_id not in per_class:
                raise ValueError(f"Unexpected class {class_id} in existing cache: {path}")
            per_class[class_id] += 1
    expected_frames = int(row["num_frames"])
    if frames != expected_frames:
        raise ValueError(f"Existing cache frame mismatch for {sequence_id}: {frames} != {expected_frames}")
    file_sha256, payload_sha256 = file_hashes(path)
    return {
        "sequence_id": sequence_id,
        "frames": frames,
        "detections": detections,
        "class_0_detections": per_class[0],
        "class_1_detections": per_class[1],
        "cache_id": header["cache_id"],
        "numeric_payload_sha256": payload_sha256,
        "cache_file_sha256": file_sha256,
        "cache_bytes": path.stat().st_size,
        "inference_seconds": "",
        "wall_seconds": "",
        "inference_fps": "",
        "status": "validated_existing",
        "cache_path": str(path),
    }


def process_sequence(
    backend: RTDETRR18Backend,
    dataset_root: Path,
    output_root: Path,
    row: dict[str, str],
    batch_size: int,
    confidence: float,
    producer: dict[str, Any],
) -> dict[str, Any]:
    from PIL import Image

    sequence_id = row["sequence_id"]
    paths = image_files(dataset_root, row)
    destination = output_root / f"{sequence_id}.detections.jsonl"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing cache: {destination}")
    width, height = int(row["width"]), int(row["height"])
    video_path = dataset_root / row["source_video"]
    writer = NumericCacheWriter(
        destination,
        "detections",
        {
            "video": {
                "path": str(video_path.resolve()),
                "size": int(row["video_bytes"]),
                "sha256": row["video_sha256"],
            },
            "image_sequence": {
                "path": str((dataset_root / row["source_frame_dir"]).resolve()),
                "format": "MOTChallenge img1 JPEG",
                "frame_indexing": "one_based",
            },
            "manifest": {
                "sequence_id": sequence_id,
                "split": row["split"],
                "frame_count": int(row["num_frames"]),
            },
            "frame_size": [width, height],
            "fps": float(row["fps"]),
            "reported_frame_count": int(row["num_frames"]),
            "coordinate_space": "full_frame_xyxy",
            "class_names": {str(key): value for key, value in EXPECTED_CLASS_NAMES.items()},
            "producer": {
                "stage": "fruitfly_mot_v1_rtdetr_r18_detection_cache",
                "source": "authoritative_MOTChallenge_JPEG_frames",
                **producer,
            },
        },
    )
    total_detections = 0
    per_class = {0: 0, 1: 0}
    inference_seconds = 0.0
    wall_start = time.perf_counter()
    try:
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = []
            for path in batch_paths:
                with Image.open(path) as image:
                    rgb = image.convert("RGB")
                    if rgb.size != (width, height):
                        raise ValueError(f"Image size mismatch: {path}: {rgb.size} != {(width, height)}")
                    images.append(rgb.copy())
            infer_start = time.perf_counter()
            results = backend.predict_pil(images, confidence)
            if backend.device.type == "cuda":
                backend.torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - infer_start
            for offset, detections in enumerate(results):
                frame_index = start + offset + 1
                writer.write_frame(frame_index, detections)
                total_detections += len(detections)
                for detection in detections:
                    class_id = int(detection[5])
                    if class_id not in per_class:
                        raise ValueError(f"Unexpected class {class_id} at {sequence_id} frame {frame_index}")
                    per_class[class_id] += 1
            print(
                f"[{sequence_id}] {min(start + batch_size, len(paths))}/{len(paths)} frames",
                end="\r",
                flush=True,
            )
        writer.close()
    except Exception:
        writer.abort()
        raise
    print(" " * 80, end="\r")
    wall_seconds = time.perf_counter() - wall_start
    file_sha256, payload_sha256 = file_hashes(destination)
    result = {
        "sequence_id": sequence_id,
        "frames": len(paths),
        "detections": total_detections,
        "class_0_detections": per_class[0],
        "class_1_detections": per_class[1],
        "cache_id": writer.cache_id,
        "numeric_payload_sha256": payload_sha256,
        "cache_file_sha256": file_sha256,
        "cache_bytes": destination.stat().st_size,
        "inference_seconds": round(inference_seconds, 6),
        "wall_seconds": round(wall_seconds, 6),
        "inference_fps": round(len(paths) / inference_seconds, 6),
        "status": "generated",
        "cache_path": str(destination),
    }
    print(
        f"[{sequence_id}] complete frames={len(paths)} detections={total_detections} "
        f"inference_fps={result['inference_fps']}"
    )
    return result


def main() -> int:
    args = parse_args()
    if args.batch_size != DEFAULT_BATCH_SIZE:
        raise ValueError(f"Batch size is frozen at {DEFAULT_BATCH_SIZE}")
    if args.confidence != DEFAULT_CONFIDENCE:
        raise ValueError(f"Cache confidence is frozen at {DEFAULT_CONFIDENCE}")
    dataset_root = args.dataset_root.resolve()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = load_sequences(dataset_root)
    backend = RTDETRR18Backend(args.weights, args.runtime_config, args.device)
    producer = backend.metadata(args.batch_size, args.confidence)
    summary_rows: list[dict[str, Any]] = []
    summary_path = output_root.parent / "detection_run.csv"
    for row in rows:
        destination = output_root / f"{row['sequence_id']}.detections.jsonl"
        if destination.exists() and args.resume:
            result = validate_existing(destination, row, producer)
            print(f"[{row['sequence_id']}] validated existing {destination}")
        else:
            result = process_sequence(
                backend,
                dataset_root,
                output_root,
                row,
                args.batch_size,
                args.confidence,
                producer,
            )
        summary_rows.append(result)
        atomic_write_csv(summary_path, summary_rows)
    metadata = {
        "status": "complete",
        "dataset_root": str(dataset_root),
        "dataset_fingerprint": (dataset_root / "quality_reports" / "dataset_fingerprint.txt")
        .read_text(encoding="utf-8")
        .strip(),
        "sequences": len(summary_rows),
        "frames": sum(int(item["frames"]) for item in summary_rows),
        "weights": str(Path(args.weights).resolve()),
        "weights_sha256": sha256_file(Path(args.weights)),
        "runtime_config": str(Path(args.runtime_config).resolve()),
        "confidence": args.confidence,
        "batch_size": args.batch_size,
        "device": args.device,
        "source": "authoritative MOTChallenge JPEG frames",
    }
    (output_root.parent / "detection_protocol.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Completed {metadata['sequences']} sequences / {metadata['frames']} frames: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

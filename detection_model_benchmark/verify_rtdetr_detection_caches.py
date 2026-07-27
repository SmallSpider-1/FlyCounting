#!/usr/bin/env python3
"""Strictly verify and merge the formal 25-segment RT-DETR cache collection."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_common.numeric_cache import CACHE_VERSION, FIELD_DECIMALS, iter_cache_frames, read_cache_header
from detection_model_benchmark.cache_rtdetr_detections import (
    CACHE_CONFIDENCE,
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT,
    DEFAULT_VIDEO_ROOT,
    FORMAL_FRAMES,
    FORMAL_SEGMENTS,
    cache_hashes,
    cache_path,
    load_manifest,
)
from detection_model_benchmark.rtdetr_r18_backend import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CLASS_NAMES,
    EXPECTED_SOURCE_COMMIT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the complete formal RT-DETR detection-cache collection.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shard-glob", default="detection_cache_manifest_shard*.csv")
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_shard_rows(cache_root: Path, pattern: str) -> dict[str, dict[str, str]]:
    paths = sorted(cache_root.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No shard manifests matching {pattern!r} in {cache_root}")
    rows: dict[str, dict[str, str]] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                key = f"{row['video_id']}:{int(row['segment_index'])}"
                if key in rows:
                    raise ValueError(f"Duplicate shard-manifest row for {key}")
                rows[key] = row
    return rows


def main() -> None:
    args = parse_args()
    cache_root = args.cache_root.resolve()
    segments = load_manifest(args.manifest, args.video_root)
    shard_rows = load_shard_rows(cache_root, args.shard_glob)
    expected_paths = {cache_path(cache_root, segment).resolve() for segment in segments}
    actual_paths = {path.resolve() for path in cache_root.rglob("*.detections.jsonl")}
    missing = sorted(map(str, expected_paths - actual_paths))
    extra = sorted(map(str, actual_paths - expected_paths))
    if missing or extra:
        raise ValueError(f"Cache path coverage mismatch: missing={missing}, extra={extra}")
    if len(shard_rows) != FORMAL_SEGMENTS:
        raise ValueError(f"Expected {FORMAL_SEGMENTS} shard rows, got {len(shard_rows)}")

    total_frames = 0
    total_detections = 0
    total_empty_frames = 0
    ordered_rows: list[dict[str, Any]] = []
    verified: list[dict[str, Any]] = []
    for ordinal, segment in enumerate(segments, 1):
        path = cache_path(cache_root, segment).resolve()
        header = read_cache_header(path, "detections")
        producer = header.get("producer", {})
        expected_decimals = {
            field: FIELD_DECIMALS[field]
            for field in ("x1", "y1", "x2", "y2", "confidence", "class_id")
        }
        checks = {
            "cache_version": header.get("cache_version") == CACHE_VERSION == 2,
            "field_decimals": header.get("field_decimals") == expected_decimals,
            "checkpoint_sha256": producer.get("checkpoint_sha256") == EXPECTED_CHECKPOINT_SHA256,
            "checkpoint_state": producer.get("checkpoint_state") == "ema.module",
            "source_commit": producer.get("source_commit") == EXPECTED_SOURCE_COMMIT,
            "precision": producer.get("precision") == "FP32" and producer.get("amp") is False,
            "input_size": producer.get("input_size") == [640, 640],
            "top_queries": int(producer.get("top_queries", -1)) == 300,
            "confidence": float(producer.get("confidence", -1)) == CACHE_CONFIDENCE,
            "nms_free": producer.get("nms") is False
            and producer.get("nms_iou") == "not_applicable",
            "class_filter": producer.get("class_filter") == "none",
            "class_names": header.get("class_names")
            == {str(key): value for key, value in EXPECTED_CLASS_NAMES.items()},
            "orig_size_order": producer.get("orig_target_sizes_order") == ["width", "height"],
            "manifest_key": header.get("manifest", {}).get("video_id") == segment.video_id
            and int(header.get("manifest", {}).get("segment_index", -1)) == segment.segment_index,
            "manifest_frames": int(header.get("reported_frame_count", -1)) == segment.frame_count,
            "video_size": int(header.get("video", {}).get("size", -1)) == segment.size_bytes,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"Header validation failed for {segment.key}: {failed}")

        frames = 0
        detections = 0
        empty_frames = 0
        per_class = {0: 0, 1: 0}
        for frame_index, rows in iter_cache_frames(path, "detections", require_complete=True):
            frames += 1
            if frame_index != frames:
                raise AssertionError("iter_cache_frames returned a non-consecutive frame")
            if not rows:
                empty_frames += 1
            detections += len(rows)
            for row in rows:
                class_id = row[5]
                if type(class_id) is not int or class_id not in per_class:
                    raise ValueError(
                        f"Invalid class_id at {segment.key} frame {frame_index}: {class_id!r}"
                    )
                if not 0.0 <= float(row[4]) <= 1.0:
                    raise ValueError(
                        f"Invalid confidence at {segment.key} frame {frame_index}: {row[4]!r}"
                    )
                per_class[class_id] += 1
        if frames != segment.frame_count:
            raise ValueError(
                f"Decoded cache frames mismatch for {segment.key}: expected {segment.frame_count}, got {frames}"
            )

        file_sha256, payload_sha256 = cache_hashes(path)
        shard = shard_rows.get(segment.key)
        if shard is None:
            raise ValueError(f"Missing shard-manifest row for {segment.key}")
        expected_values = {
            "decoded_frames": str(frames),
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "cache_version": str(CACHE_VERSION),
            "cache_id": str(header["cache_id"]),
            "numeric_payload_sha256": payload_sha256,
            "cache_file_sha256": file_sha256,
            "cache_bytes": str(path.stat().st_size),
        }
        mismatches = {
            field: (shard.get(field), value)
            for field, value in expected_values.items()
            if shard.get(field) != value
        }
        if mismatches:
            raise ValueError(f"Shard-manifest mismatch for {segment.key}: {mismatches}")

        total_frames += frames
        total_detections += detections
        total_empty_frames += empty_frames
        ordered_rows.append(shard)
        verified.append(
            {
                "ordinal": ordinal,
                "video_id": segment.video_id,
                "segment_index": segment.segment_index,
                "relative_path": str(segment.relative_path),
                "frames": frames,
                "detections": detections,
                "empty_frames": empty_frames,
                "class_0_detections": per_class[0],
                "class_1_detections": per_class[1],
                "cache_bytes": path.stat().st_size,
                "cache_id": header["cache_id"],
                "numeric_payload_sha256": payload_sha256,
                "cache_file_sha256": file_sha256,
            }
        )
        print(
            f"verified {ordinal}/{FORMAL_SEGMENTS} {segment.key} "
            f"frames={frames} detections={detections} empty={empty_frames}",
            flush=True,
        )

    if total_frames != FORMAL_FRAMES:
        raise ValueError(f"Formal decoded-frame total must be {FORMAL_FRAMES}, got {total_frames}")
    collection_fingerprint = {
        f"{row['video_id']}:{row['segment_index']}": row["numeric_payload_sha256"]
        for row in verified
    }
    from detection_model_benchmark.rtdetr_r18_backend import stable_sha256

    report = {
        "passed": True,
        "segments": len(verified),
        "logical_videos": len({row["video_id"] for row in verified}),
        "total_frames": total_frames,
        "total_detections": total_detections,
        "total_empty_frames": total_empty_frames,
        "total_cache_bytes": sum(int(row["cache_bytes"]) for row in verified),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "cache_version": CACHE_VERSION,
        "field_decimals": expected_decimals,
        "collection_numeric_payload_sha256": stable_sha256(collection_fingerprint),
        "segments_in_manifest_order": verified,
    }
    atomic_csv(cache_root / "detection_cache_manifest.csv", ordered_rows)
    atomic_json(cache_root / "verification_report.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "segments_in_manifest_order"}, indent=2))


if __name__ == "__main__":
    main()

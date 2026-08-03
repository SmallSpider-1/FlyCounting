#!/usr/bin/env python3
"""Build a non-destructive, standard MOT view of the 15 laboratory sequences.

The numbered MP4/JPG/JSON files already present in ``datasets/fruitfly_mot_v1``
are treated as immutable import sources. The generated dataset is written to the
``standardized`` child directory. Source frame IDs are zero-based; exported MOT
frame IDs and image filenames are one-based.
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
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo


TIMEZONE = ZoneInfo("Asia/Shanghai")
EXPECTED_SEQUENCES = tuple(range(1, 16))
LABEL_TO_CLASS = {"Hendel": 0, "Bezzi": 1}
CLASS_NAMES = {
    0: "Bactrocera dorsalis",
    1: "Bactrocera correcta",
}
GENERATED_MARKER = ".fruitfly_mot_v1_generated.json"
CSV_DIALECT = {"encoding": "utf-8", "newline": ""}


class BuildError(RuntimeError):
    """Raised when the source import violates a frozen dataset invariant."""


def now_iso() -> str:
    return datetime.now(TIMEZONE).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", **CSV_DIALECT) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def format_number(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-9):
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def numeric_stems(paths: Iterable[Path], suffix: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in paths:
        if not path.stem.isdigit():
            raise BuildError(f"non-numeric {suffix} filename: {path}")
        frame = int(path.stem)
        if frame in result:
            raise BuildError(f"duplicate frame {frame} for {suffix}: {path}")
        result[frame] = path
    return result


def contiguous_intervals(values: Iterable[int]) -> list[tuple[int, int]]:
    ordered = sorted(set(values))
    if not ordered:
        return []
    intervals: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value != previous + 1:
            intervals.append((start, previous))
            start = value
        previous = value
    intervals.append((start, previous))
    return intervals


def interval_string(intervals: Iterable[tuple[int, int]]) -> str:
    return ";".join(str(start) if start == end else f"{start}-{end}" for start, end in intervals)


def load_decisions(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BuildError(f"missing source decision file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0":
        raise BuildError(f"unsupported source decision schema: {payload.get('schema_version')}")
    return payload


def video_metadata(path: Path) -> dict[str, Any]:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise BuildError(
            "OpenCV is required to verify MP4 metadata. Run this script with the "
            "bench_tracking environment."
        ) from exc

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise BuildError(f"unable to open video: {path}")
    try:
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        capture.release()
    if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
        raise BuildError(
            f"invalid decoded video metadata for {path}: "
            f"{width}x{height}, fps={fps}, frames={frame_count}"
        )
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": frame_count / fps,
    }


def link_image(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        relative_source = os.path.relpath(source, start=target.parent)
        target.symlink_to(relative_source)
    elif mode == "hardlink":
        os.link(source, target)
    elif mode == "copy":
        shutil.copy2(source, target)
    else:  # pragma: no cover - guarded by argparse
        raise BuildError(f"unsupported image mode: {mode}")


def prepare_destination(dataset_root: Path, output_name: str, replace: bool) -> tuple[Path, Path]:
    final_root = dataset_root / output_name
    staging_root = dataset_root / f".{output_name}.building-{os.getpid()}"
    if staging_root.exists():
        raise BuildError(f"staging directory already exists: {staging_root}")
    if final_root.exists():
        marker = final_root / GENERATED_MARKER
        if not replace:
            raise BuildError(f"output already exists; pass --replace to rebuild: {final_root}")
        if not marker.is_file():
            raise BuildError(f"refusing to replace unmarked directory: {final_root}")
    staging_root.mkdir(parents=False)
    return staging_root, final_root


def promote_destination(staging_root: Path, final_root: Path, replace: bool) -> Path | None:
    backup: Path | None = None
    if final_root.exists():
        if not replace:
            raise BuildError(f"output appeared during build: {final_root}")
        stamp = datetime.now(TIMEZONE).strftime("%Y%m%dT%H%M%S")
        backup = final_root.with_name(f"{final_root.name}.backup-{stamp}")
        if backup.exists():
            raise BuildError(f"backup destination already exists: {backup}")
        final_root.rename(backup)
    staging_root.rename(final_root)
    return backup


def relative_to_dataset(path: Path, dataset_root: Path) -> str:
    return path.resolve().relative_to(dataset_root.resolve()).as_posix()


def build(dataset_root: Path, output_name: str, image_mode: str, replace: bool) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    if not dataset_root.is_dir():
        raise BuildError(f"dataset root does not exist: {dataset_root}")

    decisions_path = dataset_root / "source_decisions.json"
    decisions = load_decisions(decisions_path)
    override_lookup: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in decisions.get("class_overrides", []):
        key = (int(row["sequence"]), int(row["source_frame"]), int(row["track_id"]))
        if key in override_lookup:
            raise BuildError(f"duplicate class override: {key}")
        override_lookup[key] = row

    declared_occlusions: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    occlusion_rows: list[dict[str, Any]] = []
    for row in decisions.get("fully_occluded_intervals", []):
        sequence = int(row["sequence"])
        track_id = int(row["track_id"])
        start = int(row["source_start_frame"])
        end = int(row["source_end_frame"])
        if start < 0 or end < start:
            raise BuildError(f"invalid occlusion interval: {row}")
        declared_occlusions[(sequence, track_id)].append((start, end))
        occlusion_rows.append(
            {
                "sequence_id": f"FFMOT-{sequence:02d}",
                "track_id": track_id,
                "source_start_frame": start,
                "source_end_frame": end,
                "mot_start_frame": start + 1,
                "mot_end_frame": end + 1,
                "reason": row["reason"],
            }
        )

    staging_root, final_root = prepare_destination(dataset_root, output_name, replace)
    build_time = now_iso()

    annotation_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    track_rows: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []
    checksum_rows: list[tuple[str, str]] = []
    checksum_rows.append(
        (relative_to_dataset(decisions_path, dataset_root), sha256_file(decisions_path))
    )
    global_shape_types: Counter[str] = Counter()
    global_labels: Counter[str] = Counter()
    total_clipped = 0
    total_source_difficult = 0
    used_overrides: set[tuple[int, int, int]] = set()

    try:
        for sequence in EXPECTED_SEQUENCES:
            sequence_id = f"FFMOT-{sequence:02d}"
            source_dir = dataset_root / str(sequence)
            source_video = dataset_root / f"{sequence}.mp4"
            if not source_dir.is_dir() or not source_video.is_file():
                raise BuildError(f"missing source directory or video for sequence {sequence}")

            images = numeric_stems(source_dir.glob("*.jpg"), "JPG")
            annotations = numeric_stems(source_dir.glob("*.json"), "JSON")
            if set(images) != set(annotations):
                raise BuildError(
                    f"sequence {sequence}: JPG/JSON frame mismatch; "
                    f"jpg_only={sorted(set(images) - set(annotations))}, "
                    f"json_only={sorted(set(annotations) - set(images))}"
                )
            if not images:
                raise BuildError(f"sequence {sequence}: no frames")
            source_frames = sorted(images)
            if source_frames != list(range(len(source_frames))):
                raise BuildError(f"sequence {sequence}: source frames are not contiguous from zero")

            video = video_metadata(source_video)
            if video["frame_count"] != len(source_frames):
                raise BuildError(
                    f"sequence {sequence}: video has {video['frame_count']} frames but import has {len(source_frames)}"
                )

            video_sha256 = sha256_file(source_video)
            checksum_rows.append((relative_to_dataset(source_video, dataset_root), video_sha256))
            sequence_shape_types: Counter[str] = Counter()
            sequence_class_boxes: Counter[int] = Counter()
            sequence_annotations: list[dict[str, Any]] = []
            observed_frames_by_track: dict[int, set[int]] = defaultdict(set)
            class_ids_by_track: dict[int, set[int]] = defaultdict(set)
            image_width: int | None = None
            image_height: int | None = None

            mot_sequence_root = staging_root / "MOTChallenge" / "all" / sequence_id
            gt_lines: list[str] = []

            for source_frame in source_frames:
                image_path = images[source_frame]
                json_path = annotations[source_frame]
                image_sha256 = sha256_file(image_path)
                json_sha256 = sha256_file(json_path)
                checksum_rows.append((relative_to_dataset(image_path, dataset_root), image_sha256))
                checksum_rows.append((relative_to_dataset(json_path, dataset_root), json_sha256))

                payload = json.loads(json_path.read_text(encoding="utf-8"))
                width = int(payload.get("imageWidth", 0))
                height = int(payload.get("imageHeight", 0))
                if width <= 0 or height <= 0:
                    raise BuildError(f"invalid image size in {json_path}: {width}x{height}")
                if image_width is None:
                    image_width, image_height = width, height
                elif (width, height) != (image_width, image_height):
                    raise BuildError(f"sequence {sequence}: inconsistent JSON image dimensions")
                if (width, height) != (video["width"], video["height"]):
                    raise BuildError(
                        f"sequence {sequence}: JSON image size {width}x{height} differs from video "
                        f"{video['width']}x{video['height']}"
                    )
                if Path(str(payload.get("imagePath", ""))).stem != image_path.stem:
                    raise BuildError(f"imagePath mismatch in {json_path}")

                mot_frame = source_frame + 1
                mot_image_name = f"{mot_frame:06d}.jpg"
                target_image = mot_sequence_root / "img1" / mot_image_name
                link_image(image_path, target_image, image_mode)
                frame_objects: list[dict[str, Any]] = []
                frame_track_ids: set[int] = set()

                for shape_index, shape in enumerate(payload.get("shapes", [])):
                    source_label = shape.get("label")
                    if source_label not in LABEL_TO_CLASS:
                        raise BuildError(f"unsupported label in {json_path}: {source_label!r}")
                    group_id = shape.get("group_id")
                    if isinstance(group_id, bool) or not isinstance(group_id, int) or group_id <= 0:
                        raise BuildError(
                            f"invalid group_id in {json_path}, shape {shape_index}: {group_id!r}"
                        )
                    if group_id in frame_track_ids:
                        raise BuildError(f"duplicate track_id {group_id} in {json_path}")
                    frame_track_ids.add(group_id)

                    canonical_label = source_label
                    override_key = (sequence, source_frame, group_id)
                    if override_key in override_lookup:
                        decision = override_lookup[override_key]
                        if source_label != decision["expected_source_label"]:
                            raise BuildError(
                                f"stale class override {override_key}: expected source label "
                                f"{decision['expected_source_label']}, found {source_label}"
                            )
                        canonical_label = decision["canonical_label"]
                        if canonical_label not in LABEL_TO_CLASS:
                            raise BuildError(f"invalid canonical override label: {canonical_label}")
                        used_overrides.add(override_key)
                        repair_rows.append(
                            {
                                "sequence_id": sequence_id,
                                "source_frame_id": source_frame,
                                "track_id": group_id,
                                "issue_type": "class_override",
                                "old_value": source_label,
                                "new_value": canonical_label,
                                "reason": decision["reason"],
                            }
                        )

                    points = shape.get("points")
                    if not isinstance(points, list) or len(points) < 2:
                        raise BuildError(f"invalid points in {json_path}, shape {shape_index}")
                    try:
                        xs = [float(point[0]) for point in points]
                        ys = [float(point[1]) for point in points]
                    except (TypeError, ValueError, IndexError) as exc:
                        raise BuildError(f"invalid point in {json_path}, shape {shape_index}") from exc
                    if not all(math.isfinite(value) for value in xs + ys):
                        raise BuildError(f"non-finite point in {json_path}, shape {shape_index}")

                    raw_x1, raw_y1 = min(xs), min(ys)
                    raw_x2, raw_y2 = max(xs), max(ys)
                    x1 = min(max(raw_x1, 0.0), float(width))
                    y1 = min(max(raw_y1, 0.0), float(height))
                    x2 = min(max(raw_x2, 0.0), float(width))
                    y2 = min(max(raw_y2, 0.0), float(height))
                    box_width = x2 - x1
                    box_height = y2 - y1
                    if box_width <= 0 or box_height <= 0:
                        raise BuildError(f"empty box after clipping in {json_path}, shape {shape_index}")
                    clipped = not all(
                        math.isclose(left, right, abs_tol=1e-9)
                        for left, right in zip((raw_x1, raw_y1, raw_x2, raw_y2), (x1, y1, x2, y2))
                    )
                    if clipped:
                        total_clipped += 1
                        repair_rows.append(
                            {
                                "sequence_id": sequence_id,
                                "source_frame_id": source_frame,
                                "track_id": group_id,
                                "issue_type": "clip_to_image_boundary",
                                "old_value": ",".join(format_number(v) for v in (raw_x1, raw_y1, raw_x2, raw_y2)),
                                "new_value": ",".join(format_number(v) for v in (x1, y1, x2, y2)),
                                "reason": "User instructed that out-of-bounds boxes be clipped to the image boundary.",
                            }
                        )

                    shape_type = str(shape.get("shape_type", ""))
                    if shape_type not in {"rectangle", "rotation"}:
                        raise BuildError(f"unsupported shape_type in {json_path}: {shape_type!r}")
                    class_id = LABEL_TO_CLASS[canonical_label]
                    source_difficult = int(bool(shape.get("difficult", False)))
                    total_source_difficult += source_difficult
                    sequence_shape_types[shape_type] += 1
                    global_shape_types[shape_type] += 1
                    global_labels[canonical_label] += 1
                    sequence_class_boxes[class_id] += 1
                    observed_frames_by_track[group_id].add(source_frame)
                    class_ids_by_track[group_id].add(class_id)

                    row = {
                        "sequence_id": sequence_id,
                        "source_frame_id": source_frame,
                        "mot_frame_id": mot_frame,
                        "track_id": group_id,
                        "class_id": class_id,
                        "class_name": CLASS_NAMES[class_id],
                        "source_label": source_label,
                        "x": format_number(x1),
                        "y": format_number(y1),
                        "width": format_number(box_width),
                        "height": format_number(box_height),
                        "mark": 1,
                        "visibility": "1.0",
                        "ignore": 0,
                        "source_shape_type": shape_type,
                        "source_difficult": source_difficult,
                        "source_json": relative_to_dataset(json_path, dataset_root),
                        "shape_index": shape_index,
                        "clipped_to_image": int(clipped),
                    }
                    frame_objects.append(row)

                frame_objects.sort(key=lambda row: int(row["track_id"]))
                sequence_annotations.extend(frame_objects)
                annotation_rows.extend(frame_objects)
                for row in frame_objects:
                    gt_lines.append(
                        ",".join(
                            [
                                str(row["mot_frame_id"]),
                                str(row["track_id"]),
                                str(row["x"]),
                                str(row["y"]),
                                str(row["width"]),
                                str(row["height"]),
                                "1",
                                "1",
                                "1.0",
                            ]
                        )
                    )

                frame_rows.append(
                    {
                        "sequence_id": sequence_id,
                        "source_frame_id": source_frame,
                        "mot_frame_id": mot_frame,
                        "source_image": relative_to_dataset(image_path, dataset_root),
                        "source_json": relative_to_dataset(json_path, dataset_root),
                        "mot_image": f"MOTChallenge/all/{sequence_id}/img1/{mot_image_name}",
                        "width": width,
                        "height": height,
                        "num_objects": len(frame_objects),
                        "image_bytes": image_path.stat().st_size,
                        "image_sha256": image_sha256,
                        "json_sha256": json_sha256,
                    }
                )

            if image_width is None or image_height is None:
                raise BuildError(f"sequence {sequence}: failed to read image dimensions")
            if len(class_ids_by_track) != 6:
                raise BuildError(f"sequence {sequence}: expected six tracks, found {len(class_ids_by_track)}")
            for track_id, class_ids in sorted(class_ids_by_track.items()):
                if len(class_ids) != 1:
                    raise BuildError(
                        f"sequence {sequence} track {track_id}: unstable canonical classes {sorted(class_ids)}"
                    )
                observed = sorted(observed_frames_by_track[track_id])
                missing = sorted(set(range(observed[0], observed[-1] + 1)) - set(observed))
                actual_intervals = contiguous_intervals(missing)
                expected_intervals = sorted(declared_occlusions.get((sequence, track_id), []))
                if actual_intervals != expected_intervals:
                    raise BuildError(
                        f"sequence {sequence} track {track_id}: unexplained missing frames; "
                        f"actual={actual_intervals}, declared={expected_intervals}"
                    )
                class_id = next(iter(class_ids))
                track_rows.append(
                    {
                        "sequence_id": sequence_id,
                        "track_id": track_id,
                        "class_id": class_id,
                        "class_name": CLASS_NAMES[class_id],
                        "first_source_frame": observed[0],
                        "last_source_frame": observed[-1],
                        "first_mot_frame": observed[0] + 1,
                        "last_mot_frame": observed[-1] + 1,
                        "observed_frames": len(observed),
                        "missing_frames": len(missing),
                        "missing_source_intervals": interval_string(actual_intervals),
                        "missing_reason": "fully_occluded" if actual_intervals else "",
                    }
                )

            gt_path = mot_sequence_root / "gt" / "gt.txt"
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            gt_path.write_text("\n".join(gt_lines) + "\n", encoding="utf-8")
            seqinfo = (
                "[Sequence]\n"
                f"name={sequence_id}\n"
                "imDir=img1\n"
                f"frameRate={format_number(float(video['fps']))}\n"
                f"seqLength={len(source_frames)}\n"
                f"imWidth={image_width}\n"
                f"imHeight={image_height}\n"
                "imExt=.jpg\n"
            )
            (mot_sequence_root / "seqinfo.ini").write_text(seqinfo, encoding="utf-8")

            tracks_for_sequence = [row for row in track_rows if row["sequence_id"] == sequence_id]
            sequence_rows.append(
                {
                    "sequence_id": sequence_id,
                    "source_sequence": sequence,
                    "source_video": relative_to_dataset(source_video, dataset_root),
                    "source_frame_dir": str(sequence),
                    "num_frames": len(source_frames),
                    "fps": format_number(float(video["fps"])),
                    "width": image_width,
                    "height": image_height,
                    "duration_seconds": f"{float(video['duration_seconds']):.6f}",
                    "num_tracks": len(tracks_for_sequence),
                    "num_boxes": len(sequence_annotations),
                    "dorsalis_tracks": sum(int(row["class_id"]) == 0 for row in tracks_for_sequence),
                    "correcta_tracks": sum(int(row["class_id"]) == 1 for row in tracks_for_sequence),
                    "dorsalis_boxes": sequence_class_boxes[0],
                    "correcta_boxes": sequence_class_boxes[1],
                    "source_shape_types": ";".join(
                        f"{key}:{value}" for key, value in sorted(sequence_shape_types.items())
                    ),
                    "video_bytes": source_video.stat().st_size,
                    "video_sha256": video_sha256,
                    "split": "unassigned",
                }
            )

        unused_overrides = set(override_lookup) - used_overrides
        if unused_overrides:
            raise BuildError(f"unused or stale class overrides: {sorted(unused_overrides)}")

        annotation_fields = [
            "sequence_id",
            "source_frame_id",
            "mot_frame_id",
            "track_id",
            "class_id",
            "class_name",
            "source_label",
            "x",
            "y",
            "width",
            "height",
            "mark",
            "visibility",
            "ignore",
            "source_shape_type",
            "source_difficult",
            "source_json",
            "shape_index",
            "clipped_to_image",
        ]
        write_csv(staging_root / "annotations" / "mot_annotations.csv", annotation_fields, annotation_rows)
        write_csv(
            staging_root / "manifests" / "frames.csv",
            [
                "sequence_id",
                "source_frame_id",
                "mot_frame_id",
                "source_image",
                "source_json",
                "mot_image",
                "width",
                "height",
                "num_objects",
                "image_bytes",
                "image_sha256",
                "json_sha256",
            ],
            frame_rows,
        )
        write_csv(
            staging_root / "manifests" / "sequences.csv",
            [
                "sequence_id",
                "source_sequence",
                "source_video",
                "source_frame_dir",
                "num_frames",
                "fps",
                "width",
                "height",
                "duration_seconds",
                "num_tracks",
                "num_boxes",
                "dorsalis_tracks",
                "correcta_tracks",
                "dorsalis_boxes",
                "correcta_boxes",
                "source_shape_types",
                "video_bytes",
                "video_sha256",
                "split",
            ],
            sequence_rows,
        )
        write_csv(
            staging_root / "manifests" / "tracks.csv",
            [
                "sequence_id",
                "track_id",
                "class_id",
                "class_name",
                "first_source_frame",
                "last_source_frame",
                "first_mot_frame",
                "last_mot_frame",
                "observed_frames",
                "missing_frames",
                "missing_source_intervals",
                "missing_reason",
            ],
            track_rows,
        )
        write_csv(
            staging_root / "manifests" / "splits.csv",
            ["sequence_id", "split", "status", "basis"],
            [
                {
                    "sequence_id": row["sequence_id"],
                    "split": "unassigned",
                    "status": "not_frozen",
                    "basis": "Awaiting capture-batch and motion-difficulty stratification.",
                }
                for row in sequence_rows
            ],
        )
        write_csv(
            staging_root / "manifests" / "fully_occluded_intervals.csv",
            [
                "sequence_id",
                "track_id",
                "source_start_frame",
                "source_end_frame",
                "mot_start_frame",
                "mot_end_frame",
                "reason",
            ],
            sorted(occlusion_rows, key=lambda row: (row["sequence_id"], int(row["track_id"]), int(row["source_start_frame"]))),
        )
        write_csv(
            staging_root / "manifests" / "motion_intervals.csv",
            ["sequence_id", "mot_start_frame", "mot_end_frame", "motion_type", "review_status", "notes"],
            [],
        )
        write_csv(
            staging_root / "quality_reports" / "repairs.csv",
            ["sequence_id", "source_frame_id", "track_id", "issue_type", "old_value", "new_value", "reason"],
            repair_rows,
        )

        seqmap_lines = ["name"] + [row["sequence_id"] for row in sequence_rows]
        seqmap_path = staging_root / "seqmaps" / "fruitfly_mot_v1-all.txt"
        seqmap_path.parent.mkdir(parents=True, exist_ok=True)
        seqmap_path.write_text("\n".join(seqmap_lines) + "\n", encoding="utf-8")

        metadata_dir = staging_root / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        (metadata_dir / "classes.yaml").write_text(
            "names:\n"
            "  0: Bactrocera dorsalis\n"
            "  1: Bactrocera correcta\n"
            "aliases:\n"
            "  Hendel: 0\n"
            "  Bezzi: 1\n"
            "trackeval_main_class_id: 1\n",
            encoding="utf-8",
        )
        (metadata_dir / "dataset.yaml").write_text(
            "dataset_name: fruitfly_mot_v1\n"
            "dataset_version: 1.0.0-build1\n"
            f"built_at: '{build_time}'\n"
            "source_frame_indexing: zero_based\n"
            "mot_frame_indexing: one_based\n"
            "primary_evaluation: class_agnostic\n"
            "trackeval_do_preproc: false\n"
            "split_status: unassigned\n"
            f"image_materialization: {image_mode}\n",
            encoding="utf-8",
        )
        (metadata_dir / "annotation_policy.md").write_text(
            "# fruitfly_mot_v1 annotation policy\n\n"
            "- The current imported AABB is accepted as the canonical object extent.\n"
            "- `group_id` is the per-sequence identity. IDs reset independently for every sequence.\n"
            "- Canonical classes are `0 = Bactrocera dorsalis (Hendel)` and "
            "`1 = Bactrocera correcta (Bezzi)`.\n"
            "- Exported MOT frames are one-based; source JPG/JSON frames are zero-based.\n"
            "- Fully occluded targets with no evaluable box have no GT row. Their confirmed gaps are "
            "listed in `manifests/fully_occluded_intervals.csv`; the same ID resumes after occlusion.\n"
            "- In the stock TrackEval CLEAR implementation these three GT gaps create an irreducible raw "
            "`Frag=3` even for a perfect tracker; this baseline is recorded by the smoke test.\n"
            "- Out-of-image coordinates are clipped to the image boundary.\n"
            "- Source `score` is not GT confidence. Every evaluable GT row uses mark/confidence 1.\n"
            "- Source `difficult` flags are preserved for audit but are not used as ignore labels in v1.\n"
            "- The main TrackEval view is class-agnostic and therefore writes class 1 for every GT row. "
            "Species remains authoritative in `annotations/mot_annotations.csv`.\n",
            encoding="utf-8",
        )
        (metadata_dir / "README.md").write_text(
            "# fruitfly_mot_v1 standardized view\n\n"
            "This directory is generated from the numbered MP4/JPG/JSON sources in its parent directory. "
            "Do not edit generated MOT files manually; update the source or `source_decisions.json` and rebuild.\n\n"
            "`MOTChallenge/all/` is the complete 15-sequence engineering view. The experiment split is not "
            "yet frozen, so results over `all` must not be presented as an independent test-set result.\n\n"
            "Run:\n\n"
            "```bash\n"
            "/home/admin1/.conda/envs/bench_tracking/bin/python scripts/validate_fruitfly_mot_v1.py "
            "--dataset-root datasets/fruitfly_mot_v1\n"
            "```\n",
            encoding="utf-8",
        )

        checksum_rows.sort(key=lambda row: row[0])
        checksum_text = "".join(f"{digest}  {path}\n" for path, digest in checksum_rows)
        reports_dir = staging_root / "quality_reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "checksums.sha256").write_text(checksum_text, encoding="utf-8")
        canonical_payload = (staging_root / "annotations" / "mot_annotations.csv").read_bytes()
        fingerprint = hashlib.sha256(checksum_text.encode("utf-8") + canonical_payload).hexdigest()
        (reports_dir / "dataset_fingerprint.txt").write_text(fingerprint + "\n", encoding="utf-8")

        audit = {
            "status": "source_imported",
            "built_at": build_time,
            "dataset_root": str(dataset_root),
            "output": str(final_root),
            "source_sequences": len(sequence_rows),
            "source_frames": len(frame_rows),
            "canonical_boxes": len(annotation_rows),
            "tracks": len(track_rows),
            "class_box_counts": {
                "Bactrocera dorsalis": global_labels["Hendel"],
                "Bactrocera correcta": global_labels["Bezzi"],
            },
            "source_shape_types": dict(sorted(global_shape_types.items())),
            "class_overrides": len(used_overrides),
            "clipped_boxes": total_clipped,
            "source_difficult_flags": total_source_difficult,
            "fully_occluded_intervals": len(occlusion_rows),
            "total_video_duration_seconds": sum(float(row["duration_seconds"]) for row in sequence_rows),
            "split_status": "unassigned",
            "dataset_fingerprint": fingerprint,
        }
        (reports_dir / "source_audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (staging_root / GENERATED_MARKER).write_text(
            json.dumps(
                {
                    "dataset_name": "fruitfly_mot_v1",
                    "generated_by": "scripts/build_fruitfly_mot_v1.py",
                    "built_at": build_time,
                    "dataset_fingerprint": fingerprint,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        backup = promote_destination(staging_root, final_root, replace)
        audit["backup"] = str(backup) if backup else None
        return audit
    except Exception:
        # The staging directory is deliberately retained for forensic inspection.
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/fruitfly_mot_v1"),
        help="Directory containing 1.mp4..15.mp4 and 1/..15/ frame annotations.",
    )
    parser.add_argument("--output-name", default="standardized")
    parser.add_argument("--image-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    parser.add_argument("--replace", action="store_true", help="Replace a prior marked build after backing it up.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build(args.dataset_root, args.output_name, args.image_mode, args.replace)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

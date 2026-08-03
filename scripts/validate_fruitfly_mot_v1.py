#!/usr/bin/env python3
"""Strictly validate the generated fruitfly_mot_v1 standard MOT view."""

from __future__ import annotations

import argparse
import configparser
import csv
import hashlib
import json
import math
import sys
import tarfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


TIMEZONE = ZoneInfo("Asia/Shanghai")
EXPECTED_SEQUENCE_IDS = tuple(f"FFMOT-{index:02d}" for index in range(1, 16))
EXPECTED_CLASSES = {
    0: "Bactrocera dorsalis",
    1: "Bactrocera correcta",
}


class ValidationError(RuntimeError):
    """Raised when the standardized dataset violates a required invariant."""


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValidationError(f"missing CSV: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_gt(path: Path) -> list[tuple[int, int, float, float, float, float, int, int, float]]:
    rows = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        fields = raw_line.split(",")
        if len(fields) != 9:
            raise ValidationError(f"{path}:{line_number}: expected 9 columns, found {len(fields)}")
        try:
            frame = int(fields[0])
            track_id = int(fields[1])
            x, y, width, height = map(float, fields[2:6])
            mark = int(fields[6])
            class_id = int(fields[7])
            visibility = float(fields[8])
        except ValueError as exc:
            raise ValidationError(f"{path}:{line_number}: invalid numeric value") from exc
        rows.append((frame, track_id, x, y, width, height, mark, class_id, visibility))
    return rows


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-6)


def video_metadata(path: Path) -> dict[str, Any]:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ValidationError("OpenCV is required for MP4 metadata validation") from exc
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValidationError(f"unable to open video: {path}")
    try:
        return {
            "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
            "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "frame_count": int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
        }
    finally:
        capture.release()


def validate(dataset_root: Path, verify_checksums: bool, write_report: bool) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    nested_root = dataset_root / "standardized"
    if (dataset_root / ".fruitfly_mot_v1_generated.json").is_file():
        standard_root = dataset_root
    elif (nested_root / ".fruitfly_mot_v1_generated.json").is_file():
        standard_root = nested_root
    else:
        raise ValidationError(f"missing generated dataset marker: {dataset_root}")

    sequences = read_csv(standard_root / "manifests" / "sequences.csv")
    frames = read_csv(standard_root / "manifests" / "frames.csv")
    tracks = read_csv(standard_root / "manifests" / "tracks.csv")
    annotations = read_csv(standard_root / "annotations" / "mot_annotations.csv")
    occlusions = read_csv(standard_root / "manifests" / "fully_occluded_intervals.csv")
    splits = read_csv(standard_root / "manifests" / "splits.csv")

    sequence_ids = tuple(row["sequence_id"] for row in sequences)
    if sequence_ids != EXPECTED_SEQUENCE_IDS:
        raise ValidationError(f"unexpected sequence order or coverage: {sequence_ids}")
    if len(frames) != 5567:
        raise ValidationError(f"expected 5567 frames, found {len(frames)}")
    if len(annotations) != 33341:
        raise ValidationError(f"expected 33341 canonical boxes after source repairs, found {len(annotations)}")
    if len(tracks) != 90:
        raise ValidationError(f"expected 90 tracks, found {len(tracks)}")

    frames_by_sequence: dict[str, list[dict[str, str]]] = defaultdict(list)
    annotations_by_sequence: dict[str, list[dict[str, str]]] = defaultdict(list)
    annotations_by_key: dict[tuple[str, int, int], dict[str, str]] = {}
    classes_by_track: dict[tuple[str, int], set[int]] = defaultdict(set)
    observed_by_track: dict[tuple[str, int], set[int]] = defaultdict(set)
    class_counts: Counter[int] = Counter()
    clipped_count = 0
    archived_json_members: dict[Path, set[str]] = defaultdict(set)

    for row in frames:
        sequence_id = row["sequence_id"]
        source_frame = int(row["source_frame_id"])
        mot_frame = int(row["mot_frame_id"])
        if mot_frame != source_frame + 1:
            raise ValidationError(f"invalid frame mapping: {row}")
        source_image = dataset_root / row["source_image"]
        mot_image = standard_root / row["mot_image"]
        source_json_ref = row.get("source_json", "").strip()
        if not source_json_ref:
            source_json_exists = True
        elif "::" in source_json_ref:
            archive_path_raw, member = source_json_ref.split("::", 1)
            archived_json_members[dataset_root / archive_path_raw].add(member)
            source_json_exists = (dataset_root / archive_path_raw).is_file()
        else:
            source_json_exists = (dataset_root / source_json_ref).is_file()
        if not source_image.is_file() or not source_json_exists or not mot_image.is_file():
            raise ValidationError(f"missing source or MOT frame for {sequence_id}/{mot_frame}")
        if mot_image.resolve() != source_image.resolve():
            raise ValidationError(f"MOT image does not resolve to source image: {mot_image}")
        frames_by_sequence[sequence_id].append(row)

    for archive_path, expected_members in archived_json_members.items():
        with tarfile.open(archive_path, "r:gz") as archive:
            actual_members = {member.name for member in archive.getmembers() if member.isfile()}
        missing_members = expected_members - actual_members
        if missing_members:
            raise ValidationError(
                f"source annotation archive is missing {len(missing_members)} members: "
                f"{sorted(missing_members)[:5]}"
            )

    for row in annotations:
        sequence_id = row["sequence_id"]
        mot_frame = int(row["mot_frame_id"])
        track_id = int(row["track_id"])
        key = (sequence_id, mot_frame, track_id)
        if key in annotations_by_key:
            raise ValidationError(f"duplicate sequence/frame/track annotation: {key}")
        annotations_by_key[key] = row
        source_frame = int(row["source_frame_id"])
        if mot_frame != source_frame + 1:
            raise ValidationError(f"invalid annotation frame mapping: {key}")
        class_id = int(row["class_id"])
        if class_id not in EXPECTED_CLASSES or row["class_name"] != EXPECTED_CLASSES[class_id]:
            raise ValidationError(f"invalid canonical class: {row}")
        x, y, width, height = map(float, (row["x"], row["y"], row["width"], row["height"]))
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            raise ValidationError(f"non-finite box: {key}")
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 2160 + 1e-6 or y + height > 3840 + 1e-6:
            raise ValidationError(f"out-of-bounds canonical box: {key}")
        if int(row["mark"]) != 1 or int(row["ignore"]) != 0 or not close(float(row["visibility"]), 1.0):
            raise ValidationError(f"unexpected mark/visibility/ignore state: {key}")
        classes_by_track[(sequence_id, track_id)].add(class_id)
        observed_by_track[(sequence_id, track_id)].add(source_frame)
        annotations_by_sequence[sequence_id].append(row)
        class_counts[class_id] += 1
        clipped_count += int(row["clipped_to_image"])

    for key, values in classes_by_track.items():
        if len(values) != 1:
            raise ValidationError(f"class changes within canonical track {key}: {values}")
    if class_counts != Counter({0: 16701, 1: 16640}):
        raise ValidationError(f"unexpected class box counts: {dict(class_counts)}")
    if clipped_count != 274:
        raise ValidationError(f"expected 274 clipped boxes, found {clipped_count}")

    expected_occlusions: dict[tuple[str, int], set[int]] = defaultdict(set)
    for row in occlusions:
        key = (row["sequence_id"], int(row["track_id"]))
        start = int(row["source_start_frame"])
        end = int(row["source_end_frame"])
        expected_occlusions[key].update(range(start, end + 1))
        if int(row["mot_start_frame"]) != start + 1 or int(row["mot_end_frame"]) != end + 1:
            raise ValidationError(f"invalid occlusion frame mapping: {row}")
    for row in tracks:
        key = (row["sequence_id"], int(row["track_id"]))
        observed = observed_by_track[key]
        first, last = min(observed), max(observed)
        missing = set(range(first, last + 1)) - observed
        if missing != expected_occlusions.get(key, set()):
            raise ValidationError(f"unexplained or stale occlusion gap for {key}: {sorted(missing)}")
        if int(row["observed_frames"]) != len(observed) or int(row["missing_frames"]) != len(missing):
            raise ValidationError(f"track summary mismatch for {key}")

    sequence_by_id = {row["sequence_id"]: row for row in sequences}
    for sequence_id in EXPECTED_SEQUENCE_IDS:
        seq = sequence_by_id[sequence_id]
        sequence_frames = sorted(frames_by_sequence[sequence_id], key=lambda row: int(row["mot_frame_id"]))
        expected_length = int(seq["num_frames"])
        if [int(row["mot_frame_id"]) for row in sequence_frames] != list(range(1, expected_length + 1)):
            raise ValidationError(f"non-contiguous MOT frames in {sequence_id}")
        if sum(int(row["num_objects"]) for row in sequence_frames) != len(annotations_by_sequence[sequence_id]):
            raise ValidationError(f"frame object counts differ in {sequence_id}")

        source_video = dataset_root / seq["source_video"]
        video = video_metadata(source_video)
        if video["frame_count"] != expected_length:
            raise ValidationError(f"video/frame-count mismatch in {sequence_id}")
        if video["width"] != int(seq["width"]) or video["height"] != int(seq["height"]):
            raise ValidationError(f"video dimensions mismatch in {sequence_id}")
        if not close(video["fps"], float(seq["fps"])):
            raise ValidationError(f"video FPS mismatch in {sequence_id}")

        seq_root = standard_root / "MOTChallenge" / "all" / sequence_id
        parser = configparser.ConfigParser()
        parser.read(seq_root / "seqinfo.ini", encoding="utf-8")
        info = parser["Sequence"]
        required = {
            "name": sequence_id,
            "imdir": "img1",
            "framerate": seq["fps"],
            "seqlength": seq["num_frames"],
            "imwidth": seq["width"],
            "imheight": seq["height"],
            "imext": ".jpg",
        }
        for key, expected in required.items():
            if info.get(key) != expected:
                raise ValidationError(f"{sequence_id} seqinfo {key}={info.get(key)!r}, expected {expected!r}")

        gt_rows = parse_gt(seq_root / "gt" / "gt.txt")
        canonical_rows = sorted(
            annotations_by_sequence[sequence_id],
            key=lambda row: (int(row["mot_frame_id"]), int(row["track_id"])),
        )
        if len(gt_rows) != len(canonical_rows):
            raise ValidationError(f"GT/canonical row count mismatch in {sequence_id}")
        for gt, canonical in zip(gt_rows, canonical_rows):
            expected = (
                int(canonical["mot_frame_id"]),
                int(canonical["track_id"]),
                float(canonical["x"]),
                float(canonical["y"]),
                float(canonical["width"]),
                float(canonical["height"]),
                1,
                1,
                1.0,
            )
            if gt[:2] != expected[:2] or gt[6:8] != expected[6:8] or not all(
                close(left, right) for left, right in zip(gt[2:6] + (gt[8],), expected[2:6] + (expected[8],))
            ):
                raise ValidationError(f"GT differs from canonical row in {sequence_id}: {gt} != {expected}")

    split_by_sequence = {row["sequence_id"]: row["split"] for row in splits}
    if split_by_sequence != {sequence_id: "unassigned" for sequence_id in EXPECTED_SEQUENCE_IDS}:
        raise ValidationError(f"unexpected split state: {split_by_sequence}")

    seqmap = (standard_root / "seqmaps" / "fruitfly_mot_v1-all.txt").read_text(encoding="utf-8").splitlines()
    if seqmap != ["name", *EXPECTED_SEQUENCE_IDS]:
        raise ValidationError(f"invalid seqmap: {seqmap}")

    verified_checksum_files = 0
    if verify_checksums:
        checksum_path = standard_root / "quality_reports" / "checksums.sha256"
        for line_number, raw_line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), 1):
            if not raw_line.strip():
                continue
            try:
                expected_digest, relative_path = raw_line.split("  ", 1)
            except ValueError as exc:
                raise ValidationError(f"invalid checksum line {line_number}") from exc
            actual_digest = sha256_file(dataset_root / relative_path)
            if actual_digest != expected_digest:
                raise ValidationError(f"checksum mismatch: {relative_path}")
            verified_checksum_files += 1

    report = {
        "status": "passed",
        "validated_at": datetime.now(TIMEZONE).isoformat(timespec="seconds"),
        "sequences": len(sequences),
        "frames": len(frames),
        "boxes": len(annotations),
        "tracks": len(tracks),
        "class_box_counts": {
            "Bactrocera dorsalis": class_counts[0],
            "Bactrocera correcta": class_counts[1],
        },
        "clipped_boxes": clipped_count,
        "fully_occluded_intervals": len(occlusions),
        "split_status": "unassigned",
        "checksums_verified": verify_checksums,
        "checksum_files_verified": verified_checksum_files,
    }
    if write_report:
        report_path = standard_root / "quality_reports" / "validation_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/fruitfly_mot_v1"))
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument("--no-write-report", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = validate(args.dataset_root, args.verify_checksums, not args.no_write_report)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

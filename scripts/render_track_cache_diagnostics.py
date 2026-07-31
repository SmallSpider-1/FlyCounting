#!/usr/bin/env python3
"""Render diagnostic videos from validated numeric track caches and formal count events.

This utility is deliberately downstream-only: it reads source videos, validated
``*.tracks.jsonl`` caches, and the CSV files produced by formal counting. It does
not import or run a detector, tracker, or counting implementation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_common.numeric_cache import CACHE_FIELDS, iter_cache_frames, read_cache_header


DEFAULT_VIDEO_ROOT = PROJECT_ROOT / "video_data"
DEFAULT_SEGMENTS_MANIFEST = DEFAULT_VIDEO_ROOT / "manifests" / "segments.csv"
DEFAULT_LONG_VIDEOS_MANIFEST = DEFAULT_VIDEO_ROOT / "manifests" / "long_videos.csv"
DEFAULT_TRACK_ROOT = (
    PROJECT_ROOT / "project_results" / "tracking_counting_baseline_v1" / "botsort" / "tracks"
)
DEFAULT_COUNT_ROOT = (
    PROJECT_ROOT / "project_results" / "tracking_counting_baseline_v1" / "botsort" / "counts"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "video_test" / "botsort_lv006"
DEFAULT_FFMPEG = Path("/home/admin1/.conda/envs/proxyformer/bin/ffmpeg")
DEFAULT_FFPROBE = Path("/home/admin1/.conda/envs/proxyformer/bin/ffprobe")

CLASS_NAMES = {0: "Bactrocera dorsalis", 1: "Bactrocera correcta"}
CLASS_SHORT = {0: "D", 1: "C"}
# OpenCV uses BGR.
CLASS_COLORS = {0: (0, 150, 255), 1: (255, 220, 0)}
ENTER_COLOR = (40, 220, 40)
EXIT_COLOR = (30, 30, 240)
CIRCLE_COLOR = (0, 230, 255)
FIRST_INSIDE_COLOR = (0, 130, 255)
CLASS_CHANGED_COLOR = (255, 0, 255)

EVENT_REQUIRED_COLUMNS = [
    "cache",
    "video",
    "frame",
    "track_id",
    "class_id",
    "class_name",
    "direction",
    "delta",
    "count_after",
    "center_x",
    "center_y",
    "radial_distance",
    "confidence",
]
MANIFEST_COLUMNS = [
    "source_video",
    "track_cache",
    "output_video",
    "source_frames",
    "cache_frames",
    "output_frames",
    "source_fps",
    "output_fps",
    "source_width",
    "source_height",
    "output_width",
    "output_height",
    "formal_event_count",
    "output_size_bytes",
    "validation_status",
    "render_seconds",
]


@dataclass(frozen=True)
class Segment:
    index: int
    source: Path
    cache: Path
    output: Path
    manifest_frames: int
    manifest_width: int
    manifest_height: int
    manifest_fps: str

    @property
    def cache_name(self) -> str:
        return self.cache.name

    @property
    def video_name(self) -> str:
        return self.source.name


@dataclass
class TrackInfo:
    first_frame: int
    last_frame: int
    observations: int
    first_center_x: float
    first_center_y: float
    first_seen_inside: bool
    classes: set[int]


@dataclass
class SegmentAnalysis:
    header: dict[str, Any]
    cache_frames: int
    track_info: dict[int, TrackInfo]
    cache_class_changed_ids: set[int]
    event_class_changed_ids: set[int]
    event_counts_by_track: dict[int, int]


class TaskLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message: str) -> None:
        stamp = datetime.now().astimezone().isoformat(timespec="seconds")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full", "verify"), required=True)
    parser.add_argument("--video-id", default="LV006")
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--segments-manifest", type=Path, default=DEFAULT_SEGMENTS_MANIFEST)
    parser.add_argument("--long-videos-manifest", type=Path, default=DEFAULT_LONG_VIDEOS_MANIFEST)
    parser.add_argument("--track-root", type=Path, default=DEFAULT_TRACK_ROOT)
    parser.add_argument("--count-root", type=Path, default=DEFAULT_COUNT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-width", type=int, default=1152)
    parser.add_argument("--output-height", type=int, default=648)
    parser.add_argument("--pilot-frames", type=int, default=3600)
    parser.add_argument("--trail-frames", type=int, default=30)
    parser.add_argument("--first-inside-label-frames", type=int, default=30)
    parser.add_argument("--event-hold-seconds", type=float, default=0.8)
    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    parser.add_argument("--ffprobe", type=Path, default=DEFAULT_FFPROBE)
    parser.add_argument("--encoder", choices=("h264_nvenc", "libx264"), default="libx264")
    parser.add_argument("--nvenc-cq", type=int, default=24)
    parser.add_argument("--x264-crf", type=int, default=23)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def atomic_write_csv(path: Path, columns: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_executables(args: argparse.Namespace) -> None:
    for name, path in (("ffmpeg", args.ffmpeg), ("ffprobe", args.ffprobe)):
        if not path.is_file() or not os.access(path, os.X_OK):
            raise FileNotFoundError(f"{name} executable is unavailable: {path}")


def load_segments(args: argparse.Namespace) -> tuple[list[Segment], dict[str, int], str]:
    manifest_rows = [row for row in read_csv(args.segments_manifest) if row["video_id"] == args.video_id]
    if not manifest_rows:
        raise ValueError(f"No segment rows found for {args.video_id}: {args.segments_manifest}")
    manifest_rows.sort(key=lambda row: int(row["segment_index"]))
    expected_indices = list(range(1, len(manifest_rows) + 1))
    actual_indices = [int(row["segment_index"]) for row in manifest_rows]
    if actual_indices != expected_indices:
        raise ValueError(f"Segment indices are not contiguous: {actual_indices}")

    long_video_rows = [row for row in read_csv(args.long_videos_manifest) if row["video_id"] == args.video_id]
    if len(long_video_rows) != 1:
        raise ValueError(f"Expected one long-video row for {args.video_id}, got {len(long_video_rows)}")
    logical = long_video_rows[0]
    directory_name = logical["directory_name"]
    ground_truth = {
        0: int(logical["dorsalis_gt"]),
        1: int(logical["correcta_gt"]),
    }
    if int(logical["segment_count"]) != len(manifest_rows):
        raise ValueError("Segment count differs between long_videos.csv and segments.csv")

    count_dir = args.count_root / directory_name
    track_dir = args.track_root / directory_name
    segments: list[Segment] = []
    for row in manifest_rows:
        relative = Path(row["relative_path"])
        source = (args.video_root / relative).resolve()
        stem = source.stem
        cache = (track_dir / f"{stem}.tracks.jsonl").resolve()
        output = (args.output / f"{stem}_botsort_overlay.mp4").resolve()
        segments.append(
            Segment(
                index=int(row["segment_index"]),
                source=source,
                cache=cache,
                output=output,
                manifest_frames=int(row["frame_count"]),
                manifest_width=int(row["width"]),
                manifest_height=int(row["height"]),
                manifest_fps=row["avg_fps"],
            )
        )
    return segments, ground_truth, directory_name


def load_formal_outputs(
    args: argparse.Namespace,
    directory_name: str,
    segments: list[Segment],
) -> tuple[
    list[dict[str, str]],
    dict[str, list[dict[str, str]]],
    dict[tuple[str, int], dict[str, str]],
    dict[int, int],
    list[Path],
]:
    count_dir = (args.count_root / directory_name).resolve()
    event_path = count_dir / "count_events.csv"
    summary_path = count_dir / "segment_summary.csv"
    final_path = count_dir / "final_counts.csv"
    replay_path = count_dir / "replay_config.csv"
    for path in (event_path, summary_path, final_path, replay_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    events = read_csv(event_path)
    if events:
        missing = [column for column in EVENT_REQUIRED_COLUMNS if column not in events[0]]
        if missing:
            raise ValueError(f"Formal count_events.csv lacks required columns: {missing}")
    events_by_cache: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row_index, event in enumerate(events, 1):
        event["_formal_row_index"] = str(row_index)
        events_by_cache[event["cache"]].append(event)

    summary_rows = read_csv(summary_path)
    summary: dict[tuple[str, int], dict[str, str]] = {}
    for row in summary_rows:
        key = (row["cache"], int(row["class_id"]))
        if key in summary:
            raise ValueError(f"Duplicate segment summary row: {key}")
        summary[key] = row

    final_counts = {int(row["class_id"]): int(row["final_region_count"]) for row in read_csv(final_path)}
    replay_rows = read_csv(replay_path)
    if len(replay_rows) != 1:
        raise ValueError("Expected exactly one replay_config.csv row")
    replay = replay_rows[0]
    replay_cache_paths = [Path(value).resolve() for value in replay["track_caches"].split("|")]
    expected_cache_paths = [segment.cache for segment in segments]
    if replay_cache_paths != expected_cache_paths:
        raise ValueError(
            "replay_config.csv track cache order differs from manifest-authoritative segment order"
        )
    if float(replay["reference_center_x"]) != 1185.0 or float(replay["reference_center_y"]) != 732.0:
        raise ValueError("Formal replay center is not (1185, 732)")
    if float(replay["reference_radius"]) != 215.0:
        raise ValueError("Formal replay radius is not 215 px")
    if replay["initial_count_policy"] != "fixed_zero_per_class" or replay["initial_counts"] != "0=0|1=0":
        raise ValueError("Formal replay initial-count policy is not fixed zero for both classes")

    known_caches = {segment.cache_name for segment in segments}
    if set(events_by_cache) - known_caches:
        raise ValueError(f"Formal events reference unknown caches: {set(events_by_cache) - known_caches}")
    return events, events_by_cache, summary, final_counts, [event_path, summary_path, final_path, replay_path]


def ffprobe_video(path: Path, ffprobe: Path, count_frames: bool = False) -> dict[str, Any]:
    command = [
        str(ffprobe),
        "-v",
        "error",
        "-select_streams",
        "v:0",
    ]
    if count_frames:
        command.append("-count_frames")
    command += [
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Expected exactly one video stream: {path}")
    return streams[0]


def fps_float(value: str) -> float:
    return float(Fraction(value))


def validate_source_metadata(segment: Segment, header: dict[str, Any], ffprobe: Path) -> dict[str, Any]:
    if not segment.source.is_file():
        raise FileNotFoundError(segment.source)
    if not segment.cache.is_file():
        raise FileNotFoundError(segment.cache)
    if header["fields"] != CACHE_FIELDS["tracks"]:
        raise ValueError(f"Unexpected track fields: {segment.cache}")
    if header.get("coordinate_space") != "full_frame_xyxy":
        raise ValueError(f"Track cache is not full-frame xyxy: {segment.cache}")
    if header.get("class_names") != {"0": CLASS_NAMES[0], "1": CLASS_NAMES[1]}:
        raise ValueError(f"Class mapping mismatch: {segment.cache}")
    if Path(header["video"]["path"]).resolve() != segment.source:
        raise ValueError(f"Cache header video path mismatch: {segment.cache}")
    stat = segment.source.stat()
    if int(header["video"]["size"]) != stat.st_size or int(header["video"]["mtime_ns"]) != stat.st_mtime_ns:
        raise ValueError(f"Source size/mtime differs from cache header: {segment.source}")
    if list(header["frame_size"]) != [segment.manifest_width, segment.manifest_height]:
        raise ValueError(f"Cache frame size differs from authoritative manifest: {segment.cache}")
    if int(header["reported_frame_count"]) != segment.manifest_frames:
        raise ValueError(f"Cache reported frame count differs from authoritative manifest: {segment.cache}")
    probe = ffprobe_video(segment.source, ffprobe)
    if int(probe["width"]) != segment.manifest_width or int(probe["height"]) != segment.manifest_height:
        raise ValueError(f"Source dimensions differ from authoritative manifest: {segment.source}")
    if int(probe["nb_frames"]) != segment.manifest_frames:
        raise ValueError(f"Source container frame count differs from authoritative manifest: {segment.source}")
    source_fps = fps_float(probe["avg_frame_rate"])
    if not math.isclose(source_fps, float(header["fps"]), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"Source FPS differs from track header: {segment.source}")
    if not math.isclose(source_fps, fps_float(segment.manifest_fps), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"Source FPS differs from authoritative manifest: {segment.source}")
    return probe


def event_track_match(
    event: dict[str, str],
    rows: list[list[float]],
    segment: Segment,
    previous_classes: dict[int, int],
) -> None:
    track_id = int(event["track_id"])
    matching = [row for row in rows if int(row[4]) == track_id]
    if len(matching) != 1:
        raise ValueError(
            f"Formal event row {event['_formal_row_index']} maps to {len(matching)} cache tracks "
            f"at {segment.cache_name} frame {event['frame']}"
        )
    row = matching[0]
    center_x = (float(row[0]) + float(row[2])) / 2.0
    center_y = (float(row[1]) + float(row[3])) / 2.0
    if not math.isclose(center_x, float(event["center_x"]), abs_tol=0.011):
        raise ValueError(f"Formal event center_x differs from cache row {event['_formal_row_index']}")
    if not math.isclose(center_y, float(event["center_y"]), abs_tol=0.011):
        raise ValueError(f"Formal event center_y differs from cache row {event['_formal_row_index']}")
    if not math.isclose(float(row[5]), float(event["confidence"]), abs_tol=0.00011):
        raise ValueError(f"Formal event confidence differs from cache row {event['_formal_row_index']}")
    current_class = int(row[6])
    # The frozen formal replay intentionally assigns an exit to the raw track's
    # previous observed class, then updates its class state. An exit event can
    # therefore differ from the current cache row when the class changes exactly
    # at the crossing. Enter events use the current row class.
    expected_event_class = (
        previous_classes.get(track_id, current_class)
        if event["direction"] == "exit"
        else current_class
    )
    if expected_event_class != int(event["class_id"]):
        raise ValueError(
            f"Formal event class does not match frozen replay semantics at row "
            f"{event['_formal_row_index']}: expected {expected_event_class}"
        )


def analyze_caches(
    segments: list[Segment],
    events_by_cache: dict[str, list[dict[str, str]]],
    ffprobe: Path,
    logger: TaskLogger,
) -> tuple[dict[str, SegmentAnalysis], dict[str, dict[str, Any]]]:
    analyses: dict[str, SegmentAnalysis] = {}
    source_probes: dict[str, dict[str, Any]] = {}
    for segment in segments:
        logger.write(f"Preflight cache {segment.index}/{len(segments)}: {segment.cache}")
        header = read_cache_header(segment.cache, expected_kind="tracks")
        probe = validate_source_metadata(segment, header, ffprobe)
        source_probes[segment.cache_name] = probe
        events = events_by_cache.get(segment.cache_name, [])
        by_frame: dict[int, list[dict[str, str]]] = defaultdict(list)
        event_classes: dict[int, set[int]] = defaultdict(set)
        event_counts_by_track: dict[int, int] = defaultdict(int)
        for event in events:
            frame_index = int(event["frame"])
            if frame_index < 1 or frame_index > segment.manifest_frames:
                raise ValueError(f"Formal event frame out of range: {event}")
            if event["video"] != segment.video_name or event["cache"] != segment.cache_name:
                raise ValueError(f"Formal event video/cache mapping mismatch: {event}")
            by_frame[frame_index].append(event)
            event_classes[int(event["track_id"])].add(int(event["class_id"]))
            event_counts_by_track[int(event["track_id"])] += 1

        track_info: dict[int, TrackInfo] = {}
        previous_classes: dict[int, int] = {}
        cache_frames = 0
        for frame_index, rows in iter_cache_frames(segment.cache, expected_kind="tracks", require_complete=True):
            cache_frames = frame_index
            for row in rows:
                track_id = int(row[4])
                center_x = (float(row[0]) + float(row[2])) / 2.0
                center_y = (float(row[1]) + float(row[3])) / 2.0
                class_id = int(row[6])
                info = track_info.get(track_id)
                if info is None:
                    inside = math.hypot(center_x - 1185.0, center_y - 732.0) <= 215.0
                    track_info[track_id] = TrackInfo(
                        first_frame=frame_index,
                        last_frame=frame_index,
                        observations=1,
                        first_center_x=center_x,
                        first_center_y=center_y,
                        first_seen_inside=inside,
                        classes={class_id},
                    )
                else:
                    info.last_frame = frame_index
                    info.observations += 1
                    info.classes.add(class_id)
            for event in by_frame.get(frame_index, []):
                event_track_match(event, rows, segment, previous_classes)
            for row in rows:
                previous_classes[int(row[4])] = int(row[6])
        if cache_frames != segment.manifest_frames:
            raise ValueError(
                f"Validated cache frames {cache_frames} != source/manifest frames "
                f"{segment.manifest_frames}: {segment.cache}"
            )
        cache_class_changed_ids = {
            track_id for track_id, info in track_info.items() if len(info.classes) > 1
        }
        event_class_changed_ids = {
            track_id for track_id, classes in event_classes.items() if len(classes) > 1
        }
        analyses[segment.cache_name] = SegmentAnalysis(
            header=header,
            cache_frames=cache_frames,
            track_info=track_info,
            cache_class_changed_ids=cache_class_changed_ids,
            event_class_changed_ids=event_class_changed_ids,
            event_counts_by_track=dict(event_counts_by_track),
        )
        logger.write(
            f"Cache PASS {segment.cache_name}: frames={cache_frames}, tracks={len(track_info)}, "
            f"events={len(events)}, first_seen_inside="
            f"{sum(info.first_seen_inside for info in track_info.values())}"
        )
    return analyses, source_probes


def verify_formal_replay(
    segments: list[Segment],
    events_by_cache: dict[str, list[dict[str, str]]],
    summary: dict[tuple[str, int], dict[str, str]],
    final_counts: dict[int, int],
) -> None:
    running = {0: 0, 1: 0}
    seen_formal_rows: set[int] = set()
    for segment in segments:
        for class_id in CLASS_NAMES:
            row = summary.get((segment.cache_name, class_id))
            if row is None:
                raise ValueError(f"Missing segment summary: {segment.cache_name}, class {class_id}")
            if row["video"] != segment.video_name:
                raise ValueError(f"Segment summary video mismatch: {row}")
            if int(row["start_count"]) != running[class_id]:
                raise ValueError(f"Segment start count breaks continuity: {row}")
            if int(row["frames"]) != segment.manifest_frames:
                raise ValueError(f"Segment summary frame count mismatch: {row}")
            if (
                float(row["circle_center_x"]) != 1185.0
                or float(row["circle_center_y"]) != 732.0
                or float(row["circle_radius"]) != 215.0
            ):
                raise ValueError(f"Segment summary circle mismatch: {row}")

        for event in events_by_cache.get(segment.cache_name, []):
            row_index = int(event["_formal_row_index"])
            if row_index in seen_formal_rows:
                raise ValueError(f"Formal event row would be replayed twice: {row_index}")
            seen_formal_rows.add(row_index)
            class_id = int(event["class_id"])
            delta = int(event["delta"])
            direction = event["direction"]
            if (direction, delta) not in {("enter", 1), ("exit", -1)}:
                raise ValueError(f"Invalid formal direction/delta: {event}")
            expected_after = running[class_id] + delta
            if int(event["count_after"]) != expected_after:
                raise ValueError(
                    f"Formal count_after is not an exact replay of preceding rows at row {row_index}"
                )
            running[class_id] = expected_after

        for class_id in CLASS_NAMES:
            if int(summary[(segment.cache_name, class_id)]["end_count"]) != running[class_id]:
                raise ValueError(f"Segment end count differs from formal event replay: {segment.cache_name}")
    if len(seen_formal_rows) != sum(len(rows) for rows in events_by_cache.values()):
        raise ValueError("Not all formal event rows were replayed exactly once")
    if running != final_counts:
        raise ValueError(f"Formal event replay final {running} != final_counts.csv {final_counts}")


def input_paths(segments: list[Segment], formal_paths: list[Path]) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    rows.extend(("source_video", segment.source) for segment in segments)
    rows.extend(("track_cache", segment.cache) for segment in segments)
    rows.extend(("formal_count_csv", path) for path in formal_paths)
    return rows


def current_snapshot(paths: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    rows = []
    for role, path in paths:
        stat = path.stat()
        rows.append(
            {
                "role": role,
                "path": str(path.resolve()),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256_file(path) if role == "formal_count_csv" else "",
            }
        )
    return rows


def establish_or_verify_snapshot(
    output_dir: Path,
    paths: list[tuple[str, Path]],
    logger: TaskLogger,
) -> list[dict[str, Any]]:
    before_path = output_dir / "input_snapshot_before.csv"
    columns = ["role", "path", "size_bytes", "mtime_ns", "sha256"]
    current = current_snapshot(paths)
    if not before_path.exists():
        atomic_write_csv(before_path, columns, current)
        logger.write(f"Created read-only input snapshot: {before_path}")
        return current
    before = read_csv(before_path)
    normalized_current = [{key: str(value) for key, value in row.items()} for row in current]
    if before != normalized_current:
        raise ValueError("Current input size/mtime/hash differs from input_snapshot_before.csv")
    logger.write("Input snapshot matches the first run")
    return current


def write_event_index(
    output_dir: Path,
    segments: list[Segment],
    events: list[dict[str, str]],
    analyses: dict[str, SegmentAnalysis],
    source_probes: dict[str, dict[str, Any]],
) -> None:
    segment_by_cache = {segment.cache_name: segment for segment in segments}
    columns = [
        "segment_index",
        "source_video",
        "output_video",
        "frame",
        "timestamp_seconds",
        "track_id",
        "class_id",
        "class_name",
        "direction",
        "delta",
        "count_after",
        "center_x",
        "center_y",
        "confidence",
        "first_seen_inside",
        "class_changed_for_raw_track",
    ]
    rows = []
    for event in events:
        segment = segment_by_cache[event["cache"]]
        track_id = int(event["track_id"])
        info = analyses[segment.cache_name].track_info[track_id]
        fps = fps_float(source_probes[segment.cache_name]["avg_frame_rate"])
        rows.append(
            {
                "segment_index": segment.index,
                "source_video": event["video"],
                "output_video": segment.output.name,
                "frame": event["frame"],
                "timestamp_seconds": f"{(int(event['frame']) - 1) / fps:.6f}",
                "track_id": event["track_id"],
                "class_id": event["class_id"],
                "class_name": event["class_name"],
                "direction": event["direction"],
                "delta": event["delta"],
                "count_after": event["count_after"],
                "center_x": event["center_x"],
                "center_y": event["center_y"],
                "confidence": event["confidence"],
                "first_seen_inside": str(info.first_seen_inside).lower(),
                "class_changed_for_raw_track": str(
                    track_id in analyses[segment.cache_name].event_class_changed_ids
                ).lower(),
            }
        )
    atomic_write_csv(output_dir / "event_index.csv", columns, rows)


def write_track_diagnostics(
    output_dir: Path,
    segments: list[Segment],
    analyses: dict[str, SegmentAnalysis],
    events_by_cache: dict[str, list[dict[str, str]]],
) -> None:
    columns = [
        "segment_index",
        "source_video",
        "raw_tracks",
        "tracks_observed_le_3_frames",
        "tracks_observed_le_10_frames",
        "first_seen_inside_tracks",
        "cache_class_changed_tracks",
        "formal_event_class_changed_tracks",
        "formal_events",
        "formal_event_tracks",
        "tracks_with_at_least_4_events",
        "maximum_events_on_one_track",
    ]
    rows = []
    for segment in segments:
        analysis = analyses[segment.cache_name]
        info_values = list(analysis.track_info.values())
        event_counts = list(analysis.event_counts_by_track.values())
        rows.append(
            {
                "segment_index": segment.index,
                "source_video": segment.video_name,
                "raw_tracks": len(info_values),
                "tracks_observed_le_3_frames": sum(info.observations <= 3 for info in info_values),
                "tracks_observed_le_10_frames": sum(info.observations <= 10 for info in info_values),
                "first_seen_inside_tracks": sum(info.first_seen_inside for info in info_values),
                "cache_class_changed_tracks": len(analysis.cache_class_changed_ids),
                "formal_event_class_changed_tracks": len(analysis.event_class_changed_ids),
                "formal_events": len(events_by_cache.get(segment.cache_name, [])),
                "formal_event_tracks": len(event_counts),
                "tracks_with_at_least_4_events": sum(count >= 4 for count in event_counts),
                "maximum_events_on_one_track": max(event_counts, default=0),
            }
        )
    atomic_write_csv(output_dir / "track_diagnostics.csv", columns, rows)


def preflight(
    args: argparse.Namespace,
    logger: TaskLogger,
) -> tuple[
    list[Segment],
    dict[int, int],
    list[dict[str, str]],
    dict[str, list[dict[str, str]]],
    dict[tuple[str, int], dict[str, str]],
    dict[int, int],
    dict[str, SegmentAnalysis],
    dict[str, dict[str, Any]],
    list[tuple[str, Path]],
]:
    ensure_executables(args)
    args.output.mkdir(parents=True, exist_ok=True)
    segments, ground_truth, directory_name = load_segments(args)
    events, events_by_cache, summary, final_counts, formal_paths = load_formal_outputs(
        args, directory_name, segments
    )
    if ground_truth != {0: 79, 1: 27}:
        raise ValueError(f"LV006 ground truth is not 79/27: {ground_truth}")
    if final_counts != {0: 27, 1: 21}:
        raise ValueError(f"BoT-SORT formal final counts are not 27/21: {final_counts}")
    paths = input_paths(segments, formal_paths)
    establish_or_verify_snapshot(args.output, paths, logger)
    analyses, source_probes = analyze_caches(segments, events_by_cache, args.ffprobe, logger)
    verify_formal_replay(segments, events_by_cache, summary, final_counts)
    write_event_index(args.output, segments, events, analyses, source_probes)
    write_track_diagnostics(args.output, segments, analyses, events_by_cache)
    logger.write(
        f"Preflight PASS: segments={len(segments)}, cache_frames="
        f"{sum(value.cache_frames for value in analyses.values())}, formal_events={len(events)}, "
        f"formal_final=D{final_counts[0]}/C{final_counts[1]}"
    )
    return (
        segments,
        ground_truth,
        events,
        events_by_cache,
        summary,
        final_counts,
        analyses,
        source_probes,
        paths,
    )


def draw_text_box(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font_scale: float,
    foreground: tuple[int, int, int],
    background: tuple[int, int, int],
    thickness: int,
) -> tuple[int, int]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (width, height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    cv2.rectangle(
        frame,
        (x - 3, y - height - 4),
        (x + width + 4, y + baseline + 3),
        background,
        -1,
    )
    cv2.putText(frame, text, (x, y), font, font_scale, foreground, thickness, cv2.LINE_AA)
    return width, height + baseline + 7


def draw_overlay(
    frame: np.ndarray,
    rows: list[list[float]],
    frame_index: int,
    fps: float,
    segment: Segment,
    segment_count: int,
    ground_truth: dict[int, int],
    running_counts: dict[int, int],
    analysis: SegmentAnalysis,
    events_now: list[dict[str, str]],
    active_events: deque[tuple[int, dict[str, str]]],
    trails: dict[int, deque[tuple[int, int, int]]],
    args: argparse.Namespace,
) -> None:
    sx = args.output_width / segment.manifest_width
    sy = args.output_height / segment.manifest_height
    if not math.isclose(sx, sy, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Diagnostic output must preserve aspect ratio so the formal circle remains a circle")
    scale = sx
    line_width = max(2, round(args.output_width / 700))
    thin_width = max(1, line_width - 1)
    font_scale = max(0.48, args.output_width / 2100)
    small_font = max(0.42, font_scale * 0.82)
    current_ids = {int(row[4]) for row in rows}
    event_ids = {int(event["track_id"]) for event in events_now}

    for row in rows:
        x1, y1, x2, y2 = [int(round(float(value) * scale)) for value in row[:4]]
        track_id = int(row[4])
        confidence = float(row[5])
        class_id = int(row[6])
        color = CLASS_COLORS[class_id]
        thickness = line_width + 2 if track_id in event_ids else line_width
        if track_id in event_ids:
            event = next(item for item in events_now if int(item["track_id"]) == track_id)
            color = ENTER_COLOR if event["direction"] == "enter" else EXIT_COLOR
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        center_x = int(round((float(row[0]) + float(row[2])) * 0.5 * scale))
        center_y = int(round((float(row[1]) + float(row[3])) * 0.5 * scale))
        trail = trails.setdefault(track_id, deque(maxlen=args.trail_frames))
        trail.append((frame_index, center_x, center_y))
        while trail and trail[0][0] < frame_index - args.trail_frames:
            trail.popleft()
        if len(trail) >= 2:
            points = np.asarray([(x, y) for _, x, y in trail], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], False, color, thin_width, cv2.LINE_AA)
        cv2.circle(frame, (center_x, center_y), max(2, line_width), color, -1, cv2.LINE_AA)

        label = f"ID {track_id} {CLASS_SHORT[class_id]} {confidence:.2f}"
        label_y = max(18, y1 - 5)
        _, label_height = draw_text_box(
            frame, label, (max(2, x1), label_y), small_font, (255, 255, 255), color, thin_width
        )
        info = analysis.track_info[track_id]
        diagnostic_y = min(args.output_height - 8, label_y + label_height)
        if (
            info.first_seen_inside
            and frame_index <= info.first_frame + args.first_inside_label_frames - 1
        ):
            _, height = draw_text_box(
                frame,
                "FIRST-SEEN-INSIDE",
                (max(2, x1), diagnostic_y),
                small_font,
                (255, 255, 255),
                FIRST_INSIDE_COLOR,
                thin_width,
            )
            diagnostic_y += height
        if track_id in analysis.event_class_changed_ids:
            draw_text_box(
                frame,
                "CLASS-CHANGED",
                (max(2, x1), diagnostic_y),
                small_font,
                (255, 255, 255),
                CLASS_CHANGED_COLOR,
                thin_width,
            )

    # Remove old trails so raw IDs that disappeared do not leave permanent state in memory.
    for track_id in list(trails):
        trail = trails[track_id]
        while trail and trail[0][0] < frame_index - args.trail_frames:
            trail.popleft()
        if not trail and track_id not in current_ids:
            del trails[track_id]

    circle_center = (int(round(1185 * scale)), int(round(732 * scale)))
    circle_radius = int(round(215 * scale))
    cv2.circle(frame, circle_center, circle_radius, CIRCLE_COLOR, line_width, cv2.LINE_AA)
    cv2.circle(frame, circle_center, max(3, line_width + 1), CIRCLE_COLOR, -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        "COUNT CIRCLE R=215",
        (circle_center[0] - circle_radius, max(18, circle_center[1] - circle_radius - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        small_font,
        CIRCLE_COLOR,
        thin_width,
        cv2.LINE_AA,
    )

    panel_width = min(args.output_width - 20, max(480, round(args.output_width * 0.54)))
    panel_height = max(122, round(args.output_height * 0.20))
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0.0, frame)
    text_x = 22
    line_y = 35
    line_step = max(24, round(args.output_height / 27))
    info_lines = [
        f"BoT-SORT cache replay | segment {segment.index}/{segment_count}: {segment.video_name}",
        f"Frame {frame_index}/{segment.manifest_frames} | time {(frame_index - 1) / fps:.3f} s",
        f"Formal cumulative  Dorsalis: {running_counts[0]}   Correcta: {running_counts[1]}",
        f"Final GT: D={ground_truth[0]}, C={ground_truth[1]}",
    ]
    for line in info_lines:
        cv2.putText(
            frame,
            line,
            (text_x, line_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thin_width,
            cv2.LINE_AA,
        )
        line_y += line_step

    while active_events and active_events[0][0] < frame_index:
        active_events.popleft()
    notice_y = args.output_height - 18
    for _, event in list(active_events)[-4:][::-1]:
        class_id = int(event["class_id"])
        direction = event["direction"]
        background = ENTER_COLOR if direction == "enter" else EXIT_COLOR
        notice = (
            f"FORMAL EVENT | ID {event['track_id']} {CLASS_SHORT[class_id]} "
            f"{direction} {int(event['delta']):+d} | count_after={event['count_after']}"
        )
        _, height = draw_text_box(
            frame,
            notice,
            (18, notice_y),
            font_scale,
            (255, 255, 255),
            background,
            thin_width,
        )
        notice_y -= height + 5


def encoder_command(
    args: argparse.Namespace,
    partial: Path,
    fps_expression: str,
) -> list[str]:
    command = [
        str(args.ffmpeg),
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-video_size",
        f"{args.output_width}x{args.output_height}",
        "-framerate",
        fps_expression,
        "-i",
        "-",
        "-an",
    ]
    if args.encoder == "h264_nvenc":
        command += [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            str(args.nvenc_cq),
            "-b:v",
            "0",
        ]
    else:
        command += [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(args.x264_crf),
        ]
    command += [
        "-pix_fmt",
        "yuv420p",
        "-fps_mode",
        "cfr",
        str(partial),
    ]
    return command


def quarantine(path: Path, reason: str, logger: TaskLogger) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    quarantined = path.with_name(f"{path.stem}.{reason}.{stamp}{path.suffix}")
    counter = 1
    while quarantined.exists():
        quarantined = path.with_name(f"{path.stem}.{reason}.{stamp}.{counter}{path.suffix}")
        counter += 1
    os.replace(path, quarantined)
    logger.write(f"Preserved non-final output as {quarantined}")
    return quarantined


def full_decode_check(path: Path, ffmpeg: Path, logger: TaskLogger) -> None:
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-f",
        "null",
        "-",
    ]
    with logger.path.open("a", encoding="utf-8") as log_handle:
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=log_handle)
    if result.returncode != 0:
        raise ValueError(f"ffmpeg full decode failed ({result.returncode}): {path}")


def validate_output_video(
    path: Path,
    expected_frames: int,
    expected_fps: float,
    args: argparse.Namespace,
    logger: TaskLogger,
    require_full_decode: bool = True,
) -> dict[str, Any]:
    probe = ffprobe_video(path, args.ffprobe, count_frames=True)
    output_frames = int(probe.get("nb_read_frames") or probe.get("nb_frames") or -1)
    output_fps = fps_float(probe["avg_frame_rate"])
    failures = []
    if probe.get("codec_name") != "h264":
        failures.append(f"codec={probe.get('codec_name')}")
    if probe.get("pix_fmt") != "yuv420p":
        failures.append(f"pix_fmt={probe.get('pix_fmt')}")
    if output_frames != expected_frames:
        failures.append(f"frames={output_frames}, expected={expected_frames}")
    if not math.isclose(output_fps, expected_fps, rel_tol=0.0, abs_tol=1e-6):
        failures.append(f"fps={output_fps:.12f}, expected={expected_fps:.12f}")
    if int(probe["width"]) != args.output_width or int(probe["height"]) != args.output_height:
        failures.append(f"size={probe['width']}x{probe['height']}")
    if failures:
        raise ValueError(f"Output validation failed for {path}: {'; '.join(failures)}")
    if require_full_decode:
        full_decode_check(path, args.ffmpeg, logger)
    return {
        **probe,
        "output_frames": output_frames,
        "output_fps_float": output_fps,
    }


def render_one(
    args: argparse.Namespace,
    segment: Segment,
    segment_count: int,
    ground_truth: dict[int, int],
    analysis: SegmentAnalysis,
    probe: dict[str, Any],
    events: list[dict[str, str]],
    summary: dict[tuple[str, int], dict[str, str]],
    logger: TaskLogger,
    output_path: Path,
    max_frames: int | None = None,
) -> tuple[dict[str, Any], float, int]:
    expected_frames = min(segment.manifest_frames, max_frames) if max_frames else segment.manifest_frames
    source_fps = fps_float(probe["avg_frame_rate"])
    if output_path.exists():
        try:
            validated = validate_output_video(
                output_path, expected_frames, source_fps, args, logger, require_full_decode=True
            )
            logger.write(f"Resume validation PASS; skipping existing output: {output_path}")
            return validated, 0.0, expected_frames
        except Exception as exc:
            logger.write(f"Existing output is invalid and will be preserved: {output_path}: {exc}")
            quarantine(output_path, "invalid", logger)

    partial = output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}")
    if partial.exists():
        quarantine(partial, "stale", logger)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    by_frame: dict[int, list[dict[str, str]]] = defaultdict(list)
    for event in events:
        if int(event["frame"]) <= expected_frames:
            by_frame[int(event["frame"])].append(event)
    running_counts = {
        class_id: int(summary[(segment.cache_name, class_id)]["start_count"])
        for class_id in CLASS_NAMES
    }
    applied_rows: set[int] = set()
    trails: dict[int, deque[tuple[int, int, int]]] = {}
    active_events: deque[tuple[int, dict[str, str]]] = deque()
    hold_frames = max(1, round(source_fps * args.event_hold_seconds))

    command = encoder_command(args, partial, probe["avg_frame_rate"])
    logger.write(
        f"Render start: {segment.video_name} -> {partial.name}; frames={expected_frames}; "
        f"fps={source_fps:.9f}; encoder={args.encoder}"
    )
    logger.write("Encoder command: " + " ".join(command))
    cap = cv2.VideoCapture(str(segment.source))
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open source video: {segment.source}")
    start = time.monotonic()
    decoded_frames = 0
    ffmpeg_log = logger.path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=ffmpeg_log,
    )
    try:
        cache_iterator = iter_cache_frames(segment.cache, expected_kind="tracks", require_complete=True)
        for frame_index, rows in cache_iterator:
            if frame_index > expected_frames:
                break
            ok, source_frame = cap.read()
            if not ok or source_frame is None:
                raise ValueError(
                    f"Source decode ended at frame {decoded_frames}; expected at least {expected_frames}: "
                    f"{segment.source}"
                )
            decoded_frames += 1
            frame = cv2.resize(
                source_frame,
                (args.output_width, args.output_height),
                interpolation=cv2.INTER_AREA,
            )
            events_now = by_frame.get(frame_index, [])
            for event in events_now:
                row_index = int(event["_formal_row_index"])
                if row_index in applied_rows:
                    raise ValueError(f"Formal event applied twice while rendering: row {row_index}")
                class_id = int(event["class_id"])
                expected_after = running_counts[class_id] + int(event["delta"])
                if expected_after != int(event["count_after"]):
                    raise ValueError(f"Formal event replay drift while rendering: row {row_index}")
                running_counts[class_id] = expected_after
                applied_rows.add(row_index)
                active_events.append((frame_index + hold_frames - 1, event))
            draw_overlay(
                frame,
                rows,
                frame_index,
                source_fps,
                segment,
                segment_count,
                ground_truth,
                running_counts,
                analysis,
                events_now,
                active_events,
                trails,
                args,
            )
            assert process.stdin is not None
            process.stdin.write(frame.tobytes())
            if frame_index % 5000 == 0 or frame_index == expected_frames:
                elapsed = time.monotonic() - start
                logger.write(
                    f"Render progress {segment.video_name}: {frame_index}/{expected_frames} "
                    f"({frame_index / max(elapsed, 1e-9):.2f} fps)"
                )
        if decoded_frames != expected_frames:
            raise ValueError(
                f"Source/cache loop produced {decoded_frames} frames, expected {expected_frames}: {segment.source}"
            )
        if max_frames is None:
            ok, _ = cap.read()
            if ok:
                raise ValueError(f"Source decodes beyond cache footer frame count: {segment.source}")
        expected_event_rows = {
            int(event["_formal_row_index"]) for event in events if int(event["frame"]) <= expected_frames
        }
        if applied_rows != expected_event_rows:
            missing = sorted(expected_event_rows - applied_rows)
            extra = sorted(applied_rows - expected_event_rows)
            raise ValueError(f"Formal event application mismatch; missing={missing[:10]}, extra={extra[:10]}")
        if max_frames is None:
            for class_id in CLASS_NAMES:
                expected_end = int(summary[(segment.cache_name, class_id)]["end_count"])
                if running_counts[class_id] != expected_end:
                    raise ValueError(
                        f"Rendered count ends at {running_counts}, expected summary end for class "
                        f"{class_id}={expected_end}"
                    )
        assert process.stdin is not None
        process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            raise ValueError(f"ffmpeg encoder exited with status {return_code}: {partial}")
    except Exception:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        cap.release()
        ffmpeg_log.close()
    render_seconds = time.monotonic() - start

    try:
        validated = validate_output_video(
            partial, expected_frames, source_fps, args, logger, require_full_decode=True
        )
    except Exception:
        if partial.exists():
            quarantine(partial, "invalid", logger)
        raise
    os.replace(partial, output_path)
    logger.write(
        f"Render PASS and atomic finalize: {output_path}; frames={expected_frames}; "
        f"seconds={render_seconds:.3f}"
    )
    return validated, render_seconds, decoded_frames


def extract_pilot_screenshots(
    pilot: Path,
    frames: list[int],
    args: argparse.Namespace,
    logger: TaskLogger,
) -> list[Path]:
    screenshot_dir = args.output / "pilot_screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for frame_index in frames:
        output = screenshot_dir / f"frame_{frame_index:06d}.png"
        if output.exists():
            image = cv2.imread(str(output))
            if image is not None and image.shape[:2] == (args.output_height, args.output_width):
                outputs.append(output)
                continue
            quarantine(output, "invalid", logger)
        command = [
            str(args.ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(pilot),
            "-vf",
            f"select=eq(n\\,{frame_index - 1})",
            "-frames:v",
            "1",
            str(output),
        ]
        subprocess.run(command, check=True)
        image = cv2.imread(str(output))
        if image is None or image.shape[:2] != (args.output_height, args.output_width):
            raise ValueError(f"Extracted pilot screenshot is invalid: {output}")
        outputs.append(output)
    logger.write("Pilot screenshots: " + " | ".join(str(path) for path in outputs))
    return outputs


def load_manifest_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    return {row["source_video"]: row for row in read_csv(path)}


def update_render_manifest(path: Path, row: dict[str, Any]) -> None:
    rows = load_manifest_rows(path)
    rows[str(row["source_video"])] = {key: str(value) for key, value in row.items()}
    ordered = sorted(rows.values(), key=lambda item: item["source_video"])
    atomic_write_csv(path, MANIFEST_COLUMNS, ordered)


def render_manifest_row(
    segment: Segment,
    analysis: SegmentAnalysis,
    source_probe: dict[str, Any],
    validated: dict[str, Any],
    event_count: int,
    render_seconds: float,
) -> dict[str, Any]:
    return {
        "source_video": str(segment.source),
        "track_cache": str(segment.cache),
        "output_video": str(segment.output),
        "source_frames": segment.manifest_frames,
        "cache_frames": analysis.cache_frames,
        "output_frames": validated["output_frames"],
        "source_fps": f"{fps_float(source_probe['avg_frame_rate']):.12f}",
        "output_fps": f"{validated['output_fps_float']:.12f}",
        "source_width": int(source_probe["width"]),
        "source_height": int(source_probe["height"]),
        "output_width": int(validated["width"]),
        "output_height": int(validated["height"]),
        "formal_event_count": event_count,
        "output_size_bytes": segment.output.stat().st_size,
        "validation_status": "PASS",
        "render_seconds": f"{render_seconds:.3f}",
    }


def finalize_input_integrity(
    output_dir: Path,
    paths: list[tuple[str, Path]],
    logger: TaskLogger,
) -> None:
    before = read_csv(output_dir / "input_snapshot_before.csv")
    current = current_snapshot(paths)
    current_map = {row["path"]: row for row in current}
    columns = [
        "role",
        "path",
        "before_size_bytes",
        "after_size_bytes",
        "before_mtime_ns",
        "after_mtime_ns",
        "before_sha256",
        "after_sha256",
        "unchanged",
    ]
    rows = []
    for old in before:
        new = current_map.get(old["path"])
        if new is None:
            raise ValueError(f"Input disappeared after rendering: {old['path']}")
        unchanged = (
            old["size_bytes"] == str(new["size_bytes"])
            and old["mtime_ns"] == str(new["mtime_ns"])
            and old["sha256"] == str(new["sha256"])
        )
        rows.append(
            {
                "role": old["role"],
                "path": old["path"],
                "before_size_bytes": old["size_bytes"],
                "after_size_bytes": new["size_bytes"],
                "before_mtime_ns": old["mtime_ns"],
                "after_mtime_ns": new["mtime_ns"],
                "before_sha256": old["sha256"],
                "after_sha256": new["sha256"],
                "unchanged": str(unchanged).lower(),
            }
        )
    if not all(row["unchanged"] == "true" for row in rows):
        raise ValueError("At least one read-only input changed during rendering")
    atomic_write_csv(output_dir / "input_integrity.csv", columns, rows)
    logger.write("Read-only input integrity PASS: all size/mtime/hash values unchanged")


def write_readme(
    args: argparse.Namespace,
    segments: list[Segment],
    ground_truth: dict[int, int],
    final_counts: dict[int, int],
    analyses: dict[str, SegmentAnalysis],
    events_by_cache: dict[str, list[dict[str, str]]],
    tmux_name: str,
) -> None:
    total_tracks = sum(len(value.track_info) for value in analyses.values())
    first_inside = sum(
        info.first_seen_inside for value in analyses.values() for info in value.track_info.values()
    )
    short_3 = sum(
        info.observations <= 3 for value in analyses.values() for info in value.track_info.values()
    )
    short_10 = sum(
        info.observations <= 10 for value in analyses.values() for info in value.track_info.values()
    )
    cache_class_changed = sum(len(value.cache_class_changed_ids) for value in analyses.values())
    formal_class_changed = sum(len(value.event_class_changed_ids) for value in analyses.values())
    repeated_event_tracks = sum(
        sum(count >= 4 for count in value.event_counts_by_track.values()) for value in analyses.values()
    )
    max_events = max(
        (
            count
            for value in analyses.values()
            for count in value.event_counts_by_track.values()
        ),
        default=0,
    )
    total_events = sum(len(rows) for rows in events_by_cache.values())
    output_lines = "\n".join(f"- `{segment.output}`" for segment in segments)
    text = f"""# BoT-SORT LV006 diagnostic overlays

This directory contains downstream-only visualizations of the already-frozen BoT-SORT
track caches and the already-written formal counting CSVs for `{args.video_id}`. No
detector, tracker, or formal counting logic was run by this renderer.

## Frozen inputs and replay result

- Authoritative segment order: `video_data/manifests/segments.csv`
- Formal circle: center `(1185, 732)`, radius `215 px` in `2304×1296`
- Final GT: D={ground_truth[0]}, C={ground_truth[1]}
- Formal BoT-SORT replay: D={final_counts[0]}, C={final_counts[1]}
- Formal event rows replayed: {total_events}
- Track cache fields: `Nx8 [x1,y1,x2,y2,track_id,confidence,class_id,detection_index]`

The displayed counts come only from `count_events.csv`; the renderer does not infer
or replace crossing events. Every physical segment resets raw tracker identity state,
while formal cumulative counts continue from the previous segment summary.

## Outputs

{output_lines}

- `event_index.csv`: formal event index with timestamps and diagnostic flags
- `render_manifest.csv`: source/cache/output validation results
- `track_diagnostics.csv`: segment-level raw-track diagnostic counts
- `input_integrity.csv`: before/after read-only input size, mtime, and CSV hash checks
- `pilot_segment_001_botsort_overlay.mp4` and `pilot_screenshots/`: pilot evidence
- `render.log`: persistent execution and validation log

## Preliminary diagnostic signals

- Raw tracks: {total_tracks}
- Raw tracks observed for at most 3 frames: {short_3}
- Raw tracks observed for at most 10 frames: {short_10}
- Raw tracks first observed inside the formal circle: {first_inside}
- Raw tracks whose cached class changes over their observed rows: {cache_class_changed}
- Raw tracks whose formal events contain more than one class ID: {formal_class_changed}
- Event-producing tracks with at least 4 formal crossings: {repeated_event_tracks}
- Maximum formal events on one raw track: {max_events}

These are preliminary localization clues only. `FIRST-SEEN-INSIDE` means that a raw
track's first cached box center is inside the circle. Repeated crossings, short tracks,
class changes, and nearby ID starts/ends can motivate manual inspection, but they do
not establish an ID switch, trajectory fragmentation cause, or counting-error cause
without frame-level identity ground truth.

## Reproduction

Python: `{sys.executable}`

Pilot:

```bash
{sys.executable} {Path(__file__).resolve()} --mode pilot
```

Full render and verification:

```bash
{sys.executable} {Path(__file__).resolve()} --mode full
```

Persistent tmux session used by this run: `{tmux_name}`.
"""
    atomic_write_text(args.output / "README.md", text)


def mode_pilot(
    args: argparse.Namespace,
    logger: TaskLogger,
    context: tuple[Any, ...],
) -> None:
    (
        segments,
        ground_truth,
        _events,
        events_by_cache,
        summary,
        _final_counts,
        analyses,
        source_probes,
        _paths,
    ) = context
    segment = segments[0]
    if args.pilot_frames < 3501:
        raise ValueError("Pilot must include frame 3501 so at least two formal event moments can be inspected")
    pilot = args.output / "pilot_segment_001_botsort_overlay.mp4"
    validated, render_seconds, decoded_frames = render_one(
        args,
        segment,
        len(segments),
        ground_truth,
        analyses[segment.cache_name],
        source_probes[segment.cache_name],
        events_by_cache.get(segment.cache_name, []),
        summary,
        logger,
        pilot,
        max_frames=args.pilot_frames,
    )
    screenshots = extract_pilot_screenshots(pilot, [100, 3061, 3501], args, logger)
    pilot_row = {
        "pilot_video": str(pilot),
        "frames": decoded_frames,
        "fps": f"{validated['output_fps_float']:.12f}",
        "width": validated["width"],
        "height": validated["height"],
        "size_bytes": pilot.stat().st_size,
        "render_seconds": f"{render_seconds:.3f}",
        "validation_status": "PASS",
        "screenshots": "|".join(str(path) for path in screenshots),
    }
    atomic_write_csv(args.output / "pilot_validation.csv", list(pilot_row), [pilot_row])
    logger.write("PILOT PASS: automated integrity checks complete; visual inspection is still required")


def mode_full(
    args: argparse.Namespace,
    logger: TaskLogger,
    context: tuple[Any, ...],
) -> None:
    (
        segments,
        ground_truth,
        _events,
        events_by_cache,
        summary,
        final_counts,
        analyses,
        source_probes,
        paths,
    ) = context
    pilot_validation = args.output / "pilot_validation.csv"
    if not pilot_validation.is_file():
        raise ValueError("Pilot validation is missing; run --mode pilot and visually inspect screenshots first")
    manifest_path = args.output / "render_manifest.csv"
    existing_manifest = load_manifest_rows(manifest_path)
    for segment in segments:
        validated, render_seconds, decoded_frames = render_one(
            args,
            segment,
            len(segments),
            ground_truth,
            analyses[segment.cache_name],
            source_probes[segment.cache_name],
            events_by_cache.get(segment.cache_name, []),
            summary,
            logger,
            segment.output,
            max_frames=None,
        )
        if decoded_frames != segment.manifest_frames:
            raise ValueError(f"Decoded source frame count mismatch after rendering: {segment.source}")
        if render_seconds == 0.0 and str(segment.source) in existing_manifest:
            render_seconds = float(existing_manifest[str(segment.source)]["render_seconds"])
        row = render_manifest_row(
            segment,
            analyses[segment.cache_name],
            source_probes[segment.cache_name],
            validated,
            len(events_by_cache.get(segment.cache_name, [])),
            render_seconds,
        )
        update_render_manifest(manifest_path, row)
        logger.write(f"Manifest updated immediately after segment {segment.index}: {manifest_path}")

    manifest = read_csv(manifest_path)
    if len(manifest) != len(segments) or any(row["validation_status"] != "PASS" for row in manifest):
        raise ValueError("Formal render manifest is incomplete or contains a non-PASS row")
    partials = sorted(args.output.glob("*.partial.mp4"))
    if partials:
        raise ValueError(f"Residual partial videos remain: {partials}")
    if final_counts != {0: 27, 1: 21}:
        raise ValueError(f"Final formal count changed unexpectedly: {final_counts}")
    finalize_input_integrity(args.output, paths, logger)
    tmux_name = os.environ.get("TMUX_RENDER_SESSION", "botsort_lv006_render")
    write_readme(
        args,
        segments,
        ground_truth,
        final_counts,
        analyses,
        events_by_cache,
        tmux_name,
    )
    atomic_write_text(
        args.output / "FULL_RENDER_COMPLETE",
        (
            f"completed_at={datetime.now().astimezone().isoformat(timespec='seconds')}\n"
            f"segments={len(segments)}\n"
            f"frames={sum(segment.manifest_frames for segment in segments)}\n"
            f"formal_final_dorsalis={final_counts[0]}\n"
            f"formal_final_correcta={final_counts[1]}\n"
        ),
    )
    logger.write("FULL RENDER PASS: 7/7 videos validated, formal final count D=27/C=21")


def mode_verify(
    args: argparse.Namespace,
    logger: TaskLogger,
    context: tuple[Any, ...],
) -> None:
    (
        segments,
        ground_truth,
        _events,
        events_by_cache,
        _summary,
        final_counts,
        analyses,
        source_probes,
        paths,
    ) = context
    manifest_path = args.output / "render_manifest.csv"
    rows = []
    for segment in segments:
        probe = source_probes[segment.cache_name]
        validated = validate_output_video(
            segment.output,
            segment.manifest_frames,
            fps_float(probe["avg_frame_rate"]),
            args,
            logger,
            require_full_decode=True,
        )
        rows.append(
            render_manifest_row(
                segment,
                analyses[segment.cache_name],
                probe,
                validated,
                len(events_by_cache.get(segment.cache_name, [])),
                float(load_manifest_rows(manifest_path).get(str(segment.source), {}).get("render_seconds", 0)),
            )
        )
    atomic_write_csv(manifest_path, MANIFEST_COLUMNS, rows)
    finalize_input_integrity(args.output, paths, logger)
    write_readme(
        args,
        segments,
        ground_truth,
        final_counts,
        analyses,
        events_by_cache,
        os.environ.get("TMUX_RENDER_SESSION", "botsort_lv006_render"),
    )
    logger.write("VERIFY PASS: all seven formal outputs passed full validation")


def main() -> int:
    args = parse_args()
    args.video_root = args.video_root.resolve()
    args.segments_manifest = args.segments_manifest.resolve()
    args.long_videos_manifest = args.long_videos_manifest.resolve()
    args.track_root = args.track_root.resolve()
    args.count_root = args.count_root.resolve()
    args.output = args.output.resolve()
    args.ffmpeg = args.ffmpeg.resolve()
    args.ffprobe = args.ffprobe.resolve()
    if args.output_width <= 0 or args.output_height <= 0:
        raise ValueError("Output dimensions must be positive")
    if args.trail_frames <= 1:
        raise ValueError("--trail-frames must exceed one")
    cv2.setNumThreads(max(1, min(8, os.cpu_count() or 1)))
    logger = TaskLogger(args.output / "render.log")
    logger.write(
        f"Invocation mode={args.mode}; pid={os.getpid()}; python={sys.executable}; cwd={Path.cwd()}"
    )
    logger.write(
        "Downstream-only guarantee: no detector, tracker, or formal counting module is imported or run"
    )
    context = preflight(args, logger)
    if args.mode == "pilot":
        mode_pilot(args, logger, context)
    elif args.mode == "full":
        mode_full(args, logger, context)
    else:
        mode_verify(args, logger, context)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)

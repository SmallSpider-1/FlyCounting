#!/usr/bin/env python3
"""Build the LV006 BoT-SORT short-case review workbench.

The program is deliberately downstream-only.  It reads frozen source videos,
numeric detection/track caches, and formal counting CSVs; it never runs a model,
tracker, or formal counter and never mutates those inputs.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import html
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_common.numeric_cache import CACHE_FIELDS, iter_cache_frames, read_cache_header


VIDEO_ID = "LV006"
DIRECTORY_NAME = "dorsalis_79-correcta_27"
RENDERER_VERSION = "lv006_error_audit_renderer_v4_pts_indexed"
FRAME_SEEK_METHOD = "ffprobe_packet_pts+opencv_pos_msec_v1"
FRAME_PTS_TOLERANCE_SECONDS = 0.002
CENTER = (1185.0, 732.0)
RADIUS = 215.0
RING_HALF_WIDTH = 80.0
LINK_RING_HALF_WIDTH = 140.0
CLASS_NAMES = {0: "Bactrocera dorsalis", 1: "Bactrocera correcta"}
CLASS_SHORT = {0: "D", 1: "C"}
CLASS_SPECIES = {0: "dorsalis", 1: "correcta"}
SPECIES_CLASS = {value: key for key, value in CLASS_SPECIES.items()}
CLASS_COLORS = {0: (0, 155, 255), 1: (255, 220, 0)}

DEFAULT_VIDEO_ROOT = PROJECT_ROOT / "video_data"
DEFAULT_SEGMENTS_MANIFEST = DEFAULT_VIDEO_ROOT / "manifests" / "segments.csv"
DEFAULT_LONG_VIDEOS_MANIFEST = DEFAULT_VIDEO_ROOT / "manifests" / "long_videos.csv"
DEFAULT_TRACK_DIR = (
    PROJECT_ROOT
    / "project_results"
    / "tracking_counting_baseline_v1"
    / "botsort"
    / "tracks"
    / DIRECTORY_NAME
)
DEFAULT_COUNT_DIR = (
    PROJECT_ROOT
    / "project_results"
    / "tracking_counting_baseline_v1"
    / "botsort"
    / "counts"
    / DIRECTORY_NAME
)
DEFAULT_DETECTION_DIR = (
    PROJECT_ROOT
    / "project_results"
    / "detection_cache"
    / "rtdetr_r18_official_v1"
    / DIRECTORY_NAME
)
DEFAULT_OUTPUT = PROJECT_ROOT / "video_test" / "botsort_lv006" / "error_audit"
DEFAULT_FFMPEG = Path("/home/admin1/.conda/envs/proxyformer/bin/ffmpeg")
DEFAULT_FFPROBE = Path("/home/admin1/.conda/envs/proxyformer/bin/ffprobe")

OUTCOME_VALUES = {
    "formal_correct",
    "true_crossing_missed",
    "formal_event_not_real",
    "wrong_direction",
    "wrong_species",
    "uncertain",
}
MECHANISM_VALUES = {
    "first_seen_inside",
    "suspected_id_break",
    "repeated_crossing",
    "box_center_jitter",
    "class_change",
    "detection_gap",
    "occlusion",
    "other",
}
REVIEW_STATUS_VALUES = {"unreviewed", "reviewed", "needs_second_review"}
CONFIDENCE_VALUES = {"high", "medium", "low"}

TARGET_BIASES = {
    "S007": (-23, -9),
    "S005": (-12, 5),
    "S003": (-7, 1),
    "S006": (-2, -6),
    "S004": (-2, 1),
    "S001-002": (-6, 2),
    "LV006": (-52, -6),
}
GROUP_ORDER = ["S007", "S005", "S003", "S006", "S004", "S001-002"]
GROUP_RANK = {name: index for index, name in enumerate(GROUP_ORDER)}
SEGMENT_PRIORITY_BONUS = {7: 12.0, 5: 10.0, 3: 8.0, 6: 6.0, 4: 4.0, 1: 2.0, 2: 2.0}
FLAG_WEIGHTS = {
    "first_seen_inside": 34.0,
    "suspected_id_break": 29.0,
    "repeated_crossing": 23.0,
    "box_center_jitter": 28.0,
    "class_change": 24.0,
    "detection_gap": 22.0,
    "occlusion": 18.0,
    "near_ring_no_event": 30.0,
    "unmatched_detection": 26.0,
}

CASE_MANIFEST_COLUMNS = [
    "case_id",
    "segment_group",
    "segment_index",
    "source_video",
    "track_cache",
    "detection_cache",
    "clip_path",
    "thumbnail_path",
    "start_frame",
    "anchor_frame",
    "end_frame",
    "start_timestamp",
    "anchor_timestamp",
    "end_timestamp",
    "track_ids",
    "candidate_flags",
    "priority_score",
    "priority_rank_global",
    "priority_rank_segment",
    "priority_reason",
    "formal_event_rows",
    "window_formal_event_rows",
    "formal_delta_D",
    "formal_delta_C",
    "track_history",
    "render_selected",
    "render_batch",
    "render_status",
]

HUMAN_COLUMNS = [
    "review_status",
    "review_outcome",
    "expected_species",
    "expected_direction",
    "expected_delta_D_override",
    "expected_delta_C_override",
    "mechanism_flags",
    "reviewer",
    "confidence",
    "notes",
    "second_reviewer",
    "second_review_outcome",
    "second_confidence",
    "adjudication_notes",
]

REVIEW_AUTO_COLUMNS = [
    "case_id",
    "segment_group",
    "segment_index",
    "clip_path",
    "thumbnail_path",
    "anchor_timestamp",
    "track_ids",
    "candidate_flags",
    "priority_score",
    "priority_reason",
    "formal_event_rows",
    "formal_delta_D",
    "formal_delta_C",
]
REVIEW_COLUMNS = REVIEW_AUTO_COLUMNS + HUMAN_COLUMNS

LEDGER_COLUMNS = [
    "case_id",
    "segment_group",
    "start_frame",
    "anchor_frame",
    "end_frame",
    "track_ids",
    "candidate_flags",
    "formal_event_rows",
    "expected_delta_D",
    "expected_delta_C",
    "formal_delta_D",
    "formal_delta_C",
    "bias_D",
    "bias_C",
    "correction_D",
    "correction_C",
    "delta_source",
    "confirmation_state",
    "review_status",
    "review_outcome",
    "mechanism_flags",
    "reviewer",
    "confidence",
    "notes",
    "validation_error",
]

RESIDUAL_COLUMNS = [
    "segment_group",
    "target_bias_D",
    "target_bias_C",
    "confirmed_bias_D",
    "confirmed_bias_C",
    "residual_D",
    "residual_C",
    "reviewed_cases",
    "uncertain_cases",
    "needs_second_review_cases",
    "unreviewed_cases",
    "invalid_cases",
    "next_recommendation",
]

GENERATION_COLUMNS = [
    "record_type",
    "case_id",
    "segment_index",
    "path",
    "size_bytes",
    "mtime_ns",
    "sha256",
    "cache_id",
    "upstream_cache_id",
    "cache_complete",
    "cache_frames",
    "codec",
    "pixel_format",
    "width",
    "height",
    "fps",
    "frame_count",
    "duration_seconds",
    "validation_status",
    "detail",
]


@dataclass(frozen=True)
class Segment:
    index: int
    source: Path
    track_cache: Path
    detection_cache: Path
    frame_count: int
    width: int
    height: int
    fps: float

    @property
    def group(self) -> str:
        return "S001-002" if self.index in (1, 2) else f"S{self.index:03d}"


@dataclass(frozen=True)
class Obs:
    frame: int
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: int
    confidence: float
    class_id: int
    detection_index: int

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def radial(self) -> float:
        return math.hypot(self.cx - CENTER[0], self.cy - CENTER[1])

    @property
    def width(self) -> float:
        return max(1.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(1.0, self.y2 - self.y1)


@dataclass
class TrackState:
    track_id: int
    first: Obs
    last: Obs
    previous: Obs | None = None
    second: Obs | None = None
    observations: int = 1
    classes: list[int] = field(default_factory=list)
    class_changes: list[tuple[int, int, int, float, float]] = field(default_factory=list)
    gaps: list[tuple[Obs, Obs]] = field(default_factory=list)
    near_count: int = 0
    near_min_radial: float = math.inf
    near_max_radial: float = -math.inf
    closest: Obs | None = None

    def __post_init__(self) -> None:
        self.classes = [self.first.class_id]
        self._observe_near(self.first)

    def _observe_near(self, obs: Obs) -> None:
        if abs(obs.radial - RADIUS) <= LINK_RING_HALF_WIDTH:
            self.near_count += 1
            self.near_min_radial = min(self.near_min_radial, obs.radial)
            self.near_max_radial = max(self.near_max_radial, obs.radial)
            if self.closest is None or abs(obs.radial - RADIUS) < abs(self.closest.radial - RADIUS):
                self.closest = obs

    def update(self, obs: Obs, fps: float) -> None:
        if self.second is None:
            self.second = obs
        if obs.class_id != self.last.class_id:
            self.class_changes.append((obs.frame, self.last.class_id, obs.class_id, obs.cx, obs.cy))
            if obs.class_id not in self.classes:
                self.classes.append(obs.class_id)
        gap = obs.frame - self.last.frame
        if 3 <= gap <= max(3, round(fps)):
            if abs(obs.radial - RADIUS) <= LINK_RING_HALF_WIDTH or abs(self.last.radial - RADIUS) <= LINK_RING_HALF_WIDTH:
                self.gaps.append((self.last, obs))
        self.previous = self.last
        self.last = obs
        self.observations += 1
        self._observe_near(obs)


@dataclass
class Seed:
    segment_index: int
    anchor_frame: int
    track_ids: set[int]
    flags: set[str]
    score: float
    reason: list[str]
    x: float
    y: float
    related_frames: list[int] = field(default_factory=list)


@dataclass
class Case:
    case_id: str
    segment_index: int
    segment_group: str
    start_frame: int
    anchor_frame: int
    end_frame: int
    track_ids: list[int]
    candidate_flags: list[str]
    priority_score: float
    priority_reason: str
    formal_events: list[dict[str, Any]] = field(default_factory=list)
    display_events: list[dict[str, Any]] = field(default_factory=list)
    formal_delta_D: int = 0
    formal_delta_C: int = 0
    track_history: str = ""
    priority_rank_global: int = 0
    priority_rank_segment: int = 0
    render_selected: bool = False
    render_batch: str = "not_selected"
    render_status: str = "not_selected"


class AuditLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message: str) -> None:
        line = f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--segments-manifest", type=Path, default=DEFAULT_SEGMENTS_MANIFEST)
    parser.add_argument("--long-videos-manifest", type=Path, default=DEFAULT_LONG_VIDEOS_MANIFEST)
    parser.add_argument("--track-dir", type=Path, default=DEFAULT_TRACK_DIR)
    parser.add_argument("--detection-dir", type=Path, default=DEFAULT_DETECTION_DIR)
    parser.add_argument("--count-dir", type=Path, default=DEFAULT_COUNT_DIR)
    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    parser.add_argument("--ffprobe", type=Path, default=DEFAULT_FFPROBE)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="discover candidates and render a review batch")
    generate.add_argument("--pilot", action="store_true", help="render only 10-15 segment-7 pilot cases")
    generate.add_argument("--discover-only", action="store_true", help="write manifests without rendering clips")
    generate.add_argument("--pilot-cases", type=int, default=12)
    generate.add_argument("--max-cases", type=int, default=100)
    generate.add_argument("--output-width", type=int, default=1440)
    generate.add_argument("--output-height", type=int, default=720)
    generate.add_argument("--output-fps", type=float, default=30.0)
    generate.add_argument("--context-before", type=float, default=3.0)
    generate.add_argument("--context-after", type=float, default=4.0)
    generate.add_argument("--anchor-hold", type=float, default=1.0)
    generate.add_argument("--slow-source-seconds", type=float, default=1.5)
    generate.add_argument("--slow-rate", type=float, default=0.25)
    generate.add_argument("--x264-crf", type=int, default=25)
    generate.add_argument("--x264-preset", default="veryfast")
    generate.add_argument(
        "--invalidate-existing-reviews",
        action="store_true",
        help="archive and clear reviews made from invalid clips before regenerating",
    )
    subparsers.add_parser("summarize", help="refresh ledger and residuals from review_sheet.csv")
    subparsers.add_parser("verify", help="validate outputs without re-encoding")
    return parser.parse_args(argv)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def atomic_write_csv(path: Path, columns: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fps_float(value: str) -> float:
    return float(Fraction(value))


def timestamp(frame: int, fps: float) -> str:
    total = max(0.0, (frame - 1) / fps)
    hours = int(total // 3600)
    minutes = int((total % 3600) // 60)
    seconds = total % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def radial(cx: float, cy: float) -> float:
    return math.hypot(cx - CENTER[0], cy - CENTER[1])


def obs_from_row(frame: int, row: Sequence[float]) -> Obs:
    return Obs(
        frame=frame,
        x1=float(row[0]),
        y1=float(row[1]),
        x2=float(row[2]),
        y2=float(row[3]),
        track_id=int(row[4]),
        confidence=float(row[5]),
        class_id=int(row[6]),
        detection_index=int(row[7]),
    )


def load_segments(args: argparse.Namespace) -> list[Segment]:
    long_rows = [row for row in read_csv(args.long_videos_manifest) if row["video_id"] == VIDEO_ID]
    if len(long_rows) != 1 or long_rows[0]["directory_name"] != DIRECTORY_NAME:
        raise ValueError("LV006 long-video manifest row is missing or inconsistent")
    if int(long_rows[0]["dorsalis_gt"]) != 79 or int(long_rows[0]["correcta_gt"]) != 27:
        raise ValueError("LV006 final ground truth differs from D=79, C=27")
    rows = [row for row in read_csv(args.segments_manifest) if row["video_id"] == VIDEO_ID]
    rows.sort(key=lambda row: int(row["segment_index"]))
    if [int(row["segment_index"]) for row in rows] != list(range(1, 8)):
        raise ValueError("LV006 must contain authoritative segment indices 1..7")
    segments = []
    for row in rows:
        index = int(row["segment_index"])
        source = (args.video_root / row["relative_path"]).resolve()
        stem = source.stem
        segments.append(
            Segment(
                index=index,
                source=source,
                track_cache=(args.track_dir / f"{stem}.tracks.jsonl").resolve(),
                detection_cache=(args.detection_dir / f"{stem}.detections.jsonl").resolve(),
                frame_count=int(row["frame_count"]),
                width=int(row["width"]),
                height=int(row["height"]),
                fps=fps_float(row["avg_fps"]),
            )
        )
    return segments


def load_formal_events(args: argparse.Namespace, segments: Sequence[Segment]) -> list[dict[str, Any]]:
    required = ["count_events.csv", "final_counts.csv", "segment_summary.csv", "replay_config.csv"]
    for name in required:
        if not (args.count_dir / name).is_file():
            raise FileNotFoundError(args.count_dir / name)
    final_counts = {int(row["class_id"]): int(row["final_region_count"]) for row in read_csv(args.count_dir / "final_counts.csv")}
    if final_counts != {0: 27, 1: 21}:
        raise ValueError(f"Formal LV006 result differs from D=27, C=21: {final_counts}")
    replay_rows = read_csv(args.count_dir / "replay_config.csv")
    if len(replay_rows) != 1:
        raise ValueError("Expected exactly one formal replay config row")
    replay = replay_rows[0]
    expected_caches = [str(segment.track_cache) for segment in segments]
    actual_caches = [str(Path(value).resolve()) for value in replay["track_caches"].split("|")]
    if actual_caches != expected_caches:
        raise ValueError("Formal replay cache order differs from the authoritative segment order")
    if (float(replay["reference_center_x"]), float(replay["reference_center_y"]), float(replay["reference_radius"])) != (1185.0, 732.0, 215.0):
        raise ValueError("Formal circle differs from center=(1185,732), radius=215")
    cache_to_segment = {segment.track_cache.name: segment.index for segment in segments}
    events: list[dict[str, Any]] = []
    for row_number, raw in enumerate(read_csv(args.count_dir / "count_events.csv"), 1):
        if raw["cache"] not in cache_to_segment:
            raise ValueError(f"Formal event references unknown cache: {raw['cache']}")
        event = dict(raw)
        event["event_key"] = f"E{row_number:06d}"
        event["formal_row_index"] = row_number
        event["segment_index"] = cache_to_segment[raw["cache"]]
        event["frame"] = int(raw["frame"])
        event["track_id"] = int(raw["track_id"])
        event["class_id"] = int(raw["class_id"])
        event["delta"] = int(raw["delta"])
        event["center_x"] = float(raw["center_x"])
        event["center_y"] = float(raw["center_y"])
        events.append(event)
    if len(events) != 2944:
        raise ValueError(f"Expected 2944 formal events, got {len(events)}")
    return events


def read_cache_footer(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        buffer = bytearray()
        while position > 0 and len(buffer) < 65536:
            step = min(4096, position)
            position -= step
            handle.seek(position)
            buffer[:0] = handle.read(step)
            lines = buffer.splitlines()
            if len(lines) >= 2:
                break
    for line in reversed(buffer.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "footer" in payload:
            return payload["footer"]
    raise ValueError(f"Missing numeric-cache footer: {path}")


def validate_cache_pair(segment: Segment) -> tuple[dict[str, Any], dict[str, Any]]:
    track_header = read_cache_header(segment.track_cache, expected_kind="tracks")
    detection_header = read_cache_header(segment.detection_cache, expected_kind="detections")
    if track_header["fields"] != CACHE_FIELDS["tracks"] or detection_header["fields"] != CACHE_FIELDS["detections"]:
        raise ValueError(f"Unexpected cache fields for segment {segment.index}")
    if track_header.get("upstream_detection_cache_id") != detection_header.get("cache_id"):
        raise ValueError(f"Track/detection cache ID mismatch for segment {segment.index}")
    for header, path in ((track_header, segment.track_cache), (detection_header, segment.detection_cache)):
        if Path(header["video"]["path"]).resolve() != segment.source:
            raise ValueError(f"Cache source path mismatch: {path}")
        stat = segment.source.stat()
        if int(header["video"]["size"]) != stat.st_size or int(header["video"]["mtime_ns"]) != stat.st_mtime_ns:
            raise ValueError(f"Source size/mtime differs from cache header: {segment.source}")
        if list(header["frame_size"]) != [segment.width, segment.height]:
            raise ValueError(f"Cache frame dimensions mismatch: {path}")
        if int(header["reported_frame_count"]) != segment.frame_count:
            raise ValueError(f"Cache frame count mismatch: {path}")
        footer = read_cache_footer(path)
        if not footer.get("complete") or footer.get("cache_id") != header.get("cache_id"):
            raise ValueError(f"Cache footer is incomplete or mismatched: {path}")
        if int(footer.get("frames_written", -1)) != segment.frame_count:
            raise ValueError(f"Cache footer frame count mismatch: {path}")
    return track_header, detection_header


def compute_formal_delta(events: Iterable[dict[str, Any]]) -> tuple[int, int]:
    totals = {0: 0, 1: 0}
    for event in events:
        class_id = int(event["class_id"])
        if class_id not in totals:
            raise ValueError(f"Illegal formal class ID: {class_id}")
        totals[class_id] += int(event["delta"])
    return totals[0], totals[1]


def split_bursts(events: Sequence[dict[str, Any]], fps: float, gap_seconds: float = 2.0, max_span_seconds: float = 4.0) -> list[list[dict[str, Any]]]:
    if not events:
        return []
    ordered = sorted(events, key=lambda event: int(event["frame"]))
    groups: list[list[dict[str, Any]]] = [[ordered[0]]]
    gap_frames = round(gap_seconds * fps)
    max_span = round(max_span_seconds * fps)
    for event in ordered[1:]:
        current = groups[-1]
        if int(event["frame"]) - int(current[-1]["frame"]) <= gap_frames and int(event["frame"]) - int(current[0]["frame"]) <= max_span:
            current.append(event)
        else:
            groups.append([event])
    return groups


def seed_score(segment_index: int, flags: Iterable[str], extra: float = 0.0) -> float:
    return SEGMENT_PRIORITY_BONUS.get(segment_index, 0.0) + sum(FLAG_WEIGHTS.get(flag, 5.0) for flag in set(flags)) + extra


def _finish_detection_tracklet(segment: Segment, tracklet: dict[str, Any], seeds: list[Seed]) -> None:
    observations: list[dict[str, Any]] = tracklet["observations"]
    if len(observations) < 3:
        return
    first, last = observations[0], observations[-1]
    duration = last["frame"] - first["frame"]
    displacement = math.hypot(last["cx"] - first["cx"], last["cy"] - first["cy"])
    closest = min(observations, key=lambda item: abs(item["radial"] - RADIUS))
    if duration < 2 or displacement < 5.0 or abs(closest["radial"] - RADIUS) > RING_HALF_WIDTH:
        return
    flags = {"unmatched_detection", "detection_gap", "near_ring_no_event"}
    seeds.append(
        Seed(
            segment_index=segment.index,
            anchor_frame=int(closest["frame"]),
            track_ids=set(),
            flags=flags,
            score=seed_score(segment.index, flags, min(12.0, len(observations) / 2.0)),
            reason=[
                f"{len(observations)} high-confidence detections near the ring were not represented by tracker rows",
                f"unmatched activity spans {duration} frames and moves {displacement:.1f}px",
            ],
            x=float(closest["cx"]),
            y=float(closest["cy"]),
            related_frames=[int(item["frame"]) for item in observations],
        )
    )


def analyze_segment_candidates(
    segment: Segment,
    segment_events: Sequence[dict[str, Any]],
    logger: AuditLogger,
) -> tuple[list[Seed], dict[int, TrackState], dict[str, int]]:
    """Scan one paired cache and return diagnostic seeds, never conclusions."""
    validate_cache_pair(segment)
    events_by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    event_frames: set[int] = set()
    for event in segment_events:
        events_by_track[int(event["track_id"])].append(event)
        event_frames.add(int(event["frame"]))

    states: dict[int, TrackState] = {}
    seeds: list[Seed] = []
    active_detection_tracklets: dict[int, dict[str, Any]] = {}
    next_detection_tracklet = 1
    track_frames = iter_cache_frames(segment.track_cache, expected_kind="tracks")
    detection_frames = iter_cache_frames(segment.detection_cache, expected_kind="detections")
    frames_seen = 0
    unmatched_rows = 0
    event_near_pairs = 0

    for (track_frame, track_rows), (detection_frame, detection_rows) in zip(track_frames, detection_frames):
        if track_frame != detection_frame:
            raise ValueError(f"Paired cache frame mismatch in segment {segment.index}")
        frame = track_frame
        frames_seen = frame
        observations = [obs_from_row(frame, row) for row in track_rows]
        for obs in observations:
            state = states.get(obs.track_id)
            if state is None:
                states[obs.track_id] = TrackState(track_id=obs.track_id, first=obs, last=obs)
            else:
                state.update(obs, segment.fps)

        # A multi-target interaction seed is emitted only at a formal-event frame,
        # which avoids flooding the review pool with ordinary simultaneous activity.
        if frame in event_frames:
            near = [obs for obs in observations if abs(obs.radial - RADIUS) <= LINK_RING_HALF_WIDTH]
            best_pair: tuple[Obs, Obs, float] | None = None
            for left_index, left in enumerate(near):
                for right in near[left_index + 1 :]:
                    distance = math.hypot(left.cx - right.cx, left.cy - right.cy)
                    overlap_x = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
                    overlap_y = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
                    if distance <= 110.0 or (overlap_x > 0 and overlap_y > 0):
                        if best_pair is None or distance < best_pair[2]:
                            best_pair = (left, right, distance)
            if best_pair is not None:
                left, right, distance = best_pair
                flags = {"occlusion"}
                seeds.append(
                    Seed(
                        segment_index=segment.index,
                        anchor_frame=frame,
                        track_ids={left.track_id, right.track_id},
                        flags=flags,
                        score=seed_score(segment.index, flags, max(0.0, 10.0 - distance / 20.0)),
                        reason=[f"two tracked targets are {distance:.1f}px apart near a formal crossing"],
                        x=(left.cx + right.cx) / 2.0,
                        y=(left.cy + right.cy) / 2.0,
                        related_frames=[frame],
                    )
                )
                event_near_pairs += 1

        matched_detection_indices = {obs.detection_index for obs in observations if obs.detection_index >= 0}
        unmatched: list[dict[str, Any]] = []
        for detection_index, row in enumerate(detection_rows):
            if detection_index in matched_detection_indices or float(row[4]) < 0.40:
                continue
            cx = (float(row[0]) + float(row[2])) / 2.0
            cy = (float(row[1]) + float(row[3])) / 2.0
            distance_to_center = radial(cx, cy)
            if abs(distance_to_center - RADIUS) > RING_HALF_WIDTH:
                continue
            unmatched.append(
                {
                    "frame": frame,
                    "cx": cx,
                    "cy": cy,
                    "radial": distance_to_center,
                    "width": max(1.0, float(row[2]) - float(row[0])),
                    "height": max(1.0, float(row[3]) - float(row[1])),
                    "confidence": float(row[4]),
                    "class_id": int(row[5]),
                }
            )
        unmatched_rows += len(unmatched)

        for tracklet_id in list(active_detection_tracklets):
            if frame - int(active_detection_tracklets[tracklet_id]["last_frame"]) > 3:
                _finish_detection_tracklet(segment, active_detection_tracklets.pop(tracklet_id), seeds)

        used_tracklets: set[int] = set()
        for detection in sorted(unmatched, key=lambda item: -float(item["confidence"])):
            best_id = None
            best_distance = math.inf
            for tracklet_id, tracklet in active_detection_tracklets.items():
                if tracklet_id in used_tracklets:
                    continue
                gap = frame - int(tracklet["last_frame"])
                if gap < 1 or gap > 3:
                    continue
                last = tracklet["observations"][-1]
                distance = math.hypot(float(detection["cx"]) - float(last["cx"]), float(detection["cy"]) - float(last["cy"]))
                size_ratio = (float(detection["width"]) * float(detection["height"])) / max(1.0, float(last["width"]) * float(last["height"]))
                if distance <= 65.0 and 0.25 <= size_ratio <= 4.0 and distance < best_distance:
                    best_id, best_distance = tracklet_id, distance
            if best_id is None:
                best_id = next_detection_tracklet
                next_detection_tracklet += 1
                active_detection_tracklets[best_id] = {"last_frame": frame, "observations": []}
            active_detection_tracklets[best_id]["observations"].append(detection)
            active_detection_tracklets[best_id]["last_frame"] = frame
            used_tracklets.add(best_id)

    for tracklet in active_detection_tracklets.values():
        _finish_detection_tracklet(segment, tracklet, seeds)
    if frames_seen != segment.frame_count:
        raise ValueError(f"Cache scan ended at frame {frames_seen}, expected {segment.frame_count}")

    # Track-local structural candidates.
    for track_id, state in states.items():
        events = sorted(events_by_track.get(track_id, []), key=lambda event: int(event["frame"]))
        exits = [event for event in events if event["direction"] == "exit"]
        first_inside = state.first.radial < RADIUS
        if first_inside and exits:
            event = exits[0]
            flags = {"first_seen_inside"}
            seeds.append(
                Seed(
                    segment_index=segment.index,
                    anchor_frame=int(event["frame"]),
                    track_ids={track_id},
                    flags=flags,
                    score=seed_score(segment.index, flags, 18.0),
                    reason=["raw track is first observed inside the circle and later has a formal exit"],
                    x=float(event["center_x"]),
                    y=float(event["center_y"]),
                    related_frames=[state.first.frame, int(event["frame"])],
                )
            )

        for burst in split_bursts(events, segment.fps):
            if len(burst) < 2:
                continue
            flags = {"repeated_crossing"}
            directions = [event["direction"] for event in burst]
            minimum_gap = min(int(right["frame"]) - int(left["frame"]) for left, right in zip(burst, burst[1:]))
            if any(left != right for left, right in zip(directions, directions[1:])) and minimum_gap <= round(0.6 * segment.fps):
                flags.add("box_center_jitter")
            anchor_event = min(burst, key=lambda event: abs(float(event["radial_distance"]) - RADIUS))
            seeds.append(
                Seed(
                    segment_index=segment.index,
                    anchor_frame=int(anchor_event["frame"]),
                    track_ids={track_id},
                    flags=flags,
                    score=seed_score(segment.index, flags, min(18.0, len(burst) * 2.0)),
                    reason=[f"raw ID has {len(burst)} formal crossings within {((int(burst[-1]['frame']) - int(burst[0]['frame'])) / segment.fps):.2f}s"],
                    x=float(anchor_event["center_x"]),
                    y=float(anchor_event["center_y"]),
                    related_frames=[int(event["frame"]) for event in burst],
                )
            )

        for change_frame, old_class, new_class, cx, cy in state.class_changes:
            nearby = [event for event in events if abs(int(event["frame"]) - change_frame) <= round(segment.fps)]
            if not nearby or abs(radial(cx, cy) - RADIUS) > LINK_RING_HALF_WIDTH:
                continue
            closest_event = min(nearby, key=lambda event: abs(int(event["frame"]) - change_frame))
            flags = {"class_change"}
            seeds.append(
                Seed(
                    segment_index=segment.index,
                    anchor_frame=int(closest_event["frame"]),
                    track_ids={track_id},
                    flags=flags,
                    score=seed_score(segment.index, flags, 10.0),
                    reason=[f"class changes {CLASS_SHORT[old_class]}->{CLASS_SHORT[new_class]} within 1s of a formal crossing"],
                    x=float(closest_event["center_x"]),
                    y=float(closest_event["center_y"]),
                    related_frames=[change_frame, int(closest_event["frame"])],
                )
            )

        for before, after in state.gaps:
            flags = {"detection_gap"}
            seeds.append(
                Seed(
                    segment_index=segment.index,
                    anchor_frame=after.frame,
                    track_ids={track_id},
                    flags=flags,
                    score=seed_score(segment.index, flags, min(10.0, (after.frame - before.frame) / max(1.0, segment.fps) * 10.0)),
                    reason=[f"same raw ID disappears for {after.frame - before.frame - 1} cached frames near the ring"],
                    x=after.cx,
                    y=after.cy,
                    related_frames=[before.frame, after.frame],
                )
            )

        if not events and state.near_count >= 3 and state.closest is not None and abs(state.closest.radial - RADIUS) <= RING_HALF_WIDTH:
            flags = {"near_ring_no_event"}
            reason = [f"raw ID has {state.near_count} ring-near observations but no formal event"]
            extra = min(12.0, state.near_count / 5.0)
            if first_inside:
                flags.add("first_seen_inside")
                reason.append("raw ID is first observed inside the circle")
                extra += 8.0
            seeds.append(
                Seed(
                    segment_index=segment.index,
                    anchor_frame=state.closest.frame,
                    track_ids={track_id},
                    flags=flags,
                    score=seed_score(segment.index, flags, extra),
                    reason=reason,
                    x=state.closest.cx,
                    y=state.closest.cy,
                    related_frames=[state.first.frame, state.closest.frame, state.last.frame],
                )
            )

    # Endpoint-to-start matching is only a suspicion flag.  Each ending track is
    # linked to at most two plausible starts to control combinatorial noise.
    starts = sorted(
        [state for state in states.values() if abs(state.first.radial - RADIUS) <= LINK_RING_HALF_WIDTH],
        key=lambda state: state.first.frame,
    )
    start_frames = [state.first.frame for state in starts]
    for old in states.values():
        if abs(old.last.radial - RADIUS) > LINK_RING_HALF_WIDTH:
            continue
        begin = bisect.bisect_right(start_frames, old.last.frame)
        finish = bisect.bisect_right(start_frames, old.last.frame + round(segment.fps))
        plausible: list[tuple[float, TrackState, float, float]] = []
        for new in starts[begin:finish]:
            if new.track_id == old.track_id:
                continue
            gap = new.first.frame - old.last.frame
            distance = math.hypot(new.first.cx - old.last.cx, new.first.cy - old.last.cy)
            area_ratio = (new.first.width * new.first.height) / max(1.0, old.last.width * old.last.height)
            if distance > 120.0 or not 0.30 <= area_ratio <= 3.3:
                continue
            class_penalty = 0.0 if new.first.class_id == old.last.class_id else 15.0
            cost = distance + gap / max(1.0, segment.fps) * 40.0 + class_penalty
            plausible.append((cost, new, distance, area_ratio))
        for cost, new, distance, area_ratio in sorted(plausible, key=lambda item: item[0])[:1]:
            flags = {"suspected_id_break"}
            opposite_sides = (old.last.radial < RADIUS) != (new.first.radial < RADIUS)
            if opposite_sides:
                flags.add("near_ring_no_event")
            seeds.append(
                Seed(
                    segment_index=segment.index,
                    anchor_frame=new.first.frame,
                    track_ids={old.track_id, new.track_id},
                    flags=flags,
                    score=seed_score(segment.index, flags, max(0.0, 18.0 - cost / 10.0) + (8.0 if opposite_sides else 0.0)),
                    reason=[
                        f"old ID {old.track_id} ends and new ID {new.track_id} begins after {new.first.frame - old.last.frame} frames",
                        f"endpoint distance={distance:.1f}px, box-area ratio={area_ratio:.2f}; identity continuity is unconfirmed",
                    ],
                    x=(old.last.cx + new.first.cx) / 2.0,
                    y=(old.last.cy + new.first.cy) / 2.0,
                    related_frames=[old.last.frame, new.first.frame],
                )
            )

    stats = {
        "frames": frames_seen,
        "raw_tracks": len(states),
        "raw_seeds": len(seeds),
        "unmatched_detection_rows": unmatched_rows,
        "event_near_pairs": event_near_pairs,
    }
    logger.write(
        f"segment {segment.index}: frames={frames_seen}, tracks={len(states)}, "
        f"raw_candidates={len(seeds)}, unmatched_ring_detections={unmatched_rows}"
    )
    return seeds, states, stats


def _seed_can_merge(seed: Seed, cluster: dict[str, Any], fps: float) -> bool:
    if seed.anchor_frame - int(cluster["last_anchor"]) > round(2.0 * fps):
        return False
    if seed.anchor_frame - int(cluster["first_anchor"]) > round(4.0 * fps):
        return False
    shared_track = bool(seed.track_ids & cluster["track_ids"])
    distance = math.hypot(seed.x - float(cluster["x"]), seed.y - float(cluster["y"]))
    # Cross-ID relationships are already explicit in suspected-rebuild and
    # occlusion seeds.  Do not merge two unrelated tracked targets merely because
    # they are nearby; only an ID-less unmatched-detection seed may join spatially.
    spatially_linkable = (
        distance <= 80.0
        and (not seed.track_ids or not cluster["track_ids"])
        and ("unmatched_detection" in seed.flags or "unmatched_detection" in cluster["flags"])
    )
    return shared_track or spatially_linkable


def deduplicate_seeds(
    segment: Segment,
    seeds: Sequence[Seed],
    states: dict[int, TrackState],
    segment_events: Sequence[dict[str, Any]],
    context_before: float = 3.0,
    context_after: float = 4.0,
) -> list[Case]:
    clusters: list[dict[str, Any]] = []
    active: deque[int] = deque()
    for seed in sorted(seeds, key=lambda item: (item.anchor_frame, -item.score, sorted(item.track_ids), sorted(item.flags))):
        while active and seed.anchor_frame - int(clusters[active[0]]["last_anchor"]) > round(2.0 * segment.fps):
            active.popleft()
        matches = [index for index in active if _seed_can_merge(seed, clusters[index], segment.fps)]
        if matches:
            cluster_index = max(matches, key=lambda index: clusters[index]["score"])
            cluster = clusters[cluster_index]
            total_weight = float(cluster["weight"]) + max(1.0, seed.score)
            cluster["x"] = (float(cluster["x"]) * float(cluster["weight"]) + seed.x * max(1.0, seed.score)) / total_weight
            cluster["y"] = (float(cluster["y"]) * float(cluster["weight"]) + seed.y * max(1.0, seed.score)) / total_weight
            cluster["weight"] = total_weight
            cluster["last_anchor"] = max(int(cluster["last_anchor"]), seed.anchor_frame)
            cluster["track_ids"].update(seed.track_ids)
            cluster["flags"].update(seed.flags)
            cluster["reasons"].extend(seed.reason)
            cluster["related_frames"].extend(seed.related_frames or [seed.anchor_frame])
            cluster["seed_count"] += 1
            if seed.score > float(cluster["max_seed_score"]):
                cluster["max_seed_score"] = seed.score
                cluster["anchor"] = seed.anchor_frame
            cluster["score"] = float(cluster["max_seed_score"]) + min(10.0, 1.5 * (len(cluster["flags"]) - 1)) + min(8.0, math.sqrt(int(cluster["seed_count"])))
        else:
            cluster_index = len(clusters)
            clusters.append(
                {
                    "first_anchor": seed.anchor_frame,
                    "last_anchor": seed.anchor_frame,
                    "anchor": seed.anchor_frame,
                    "track_ids": set(seed.track_ids),
                    "flags": set(seed.flags),
                    "reasons": list(seed.reason),
                    "related_frames": list(seed.related_frames or [seed.anchor_frame]),
                    "score": seed.score,
                    "max_seed_score": seed.score,
                    "seed_count": 1,
                    "weight": max(1.0, seed.score),
                    "x": seed.x,
                    "y": seed.y,
                }
            )
            active.append(cluster_index)

    events_by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in segment_events:
        events_by_track[int(event["track_id"])].append(event)
    cases: list[Case] = []
    ordered_clusters = sorted(clusters, key=lambda cluster: (int(cluster["anchor"]), sorted(cluster["track_ids"])))
    for case_number, cluster in enumerate(ordered_clusters, 1):
        related = cluster["related_frames"] or [cluster["anchor"]]
        process_start = min(related)
        process_end = max(related)
        anchor = int(cluster["anchor"])
        # The rendered context remains compact even if a diagnostic link spans a
        # long time; the anchor is the locally reviewable point.
        if process_end - process_start > round(4.0 * segment.fps):
            process_start = anchor
            process_end = anchor
        start_frame = max(1, min(process_start, anchor) - round(context_before * segment.fps))
        end_frame = min(segment.frame_count, max(process_end, anchor) + round(context_after * segment.fps))
        history_parts = []
        for track_id in sorted(cluster["track_ids"]):
            state = states.get(track_id)
            if state is None:
                continue
            event_summary = ",".join(
                f"{event['event_key']}:{CLASS_SHORT[int(event['class_id'])]}:{event['direction']}"
                for event in events_by_track.get(track_id, [])
            ) or "none"
            class_sequence = ">".join(CLASS_SHORT[class_id] for class_id in state.classes)
            history_parts.append(
                f"ID {track_id} born={state.first.frame} last={state.last.frame} "
                f"obs={state.observations} classes={class_sequence} events={event_summary}"
            )
        cases.append(
            Case(
                case_id=f"LV006_S{segment.index:03d}_C{case_number:04d}",
                segment_index=segment.index,
                segment_group=segment.group,
                start_frame=start_frame,
                anchor_frame=anchor,
                end_frame=end_frame,
                track_ids=sorted(cluster["track_ids"]),
                candidate_flags=sorted(cluster["flags"]),
                priority_score=round(float(cluster["score"]), 3),
                priority_reason=" | ".join(dict.fromkeys(cluster["reasons"])),
                track_history=" || ".join(history_parts) or "unmatched detections; no raw track ID assigned",
            )
        )
    return cases


def assign_formal_events(cases: Sequence[Case], events: Sequence[dict[str, Any]]) -> None:
    cases_by_segment: dict[int, list[Case]] = defaultdict(list)
    for case in cases:
        case.formal_events = []
        case.display_events = []
        cases_by_segment[case.segment_index].append(case)
    for case in cases:
        case.display_events = [
            event
            for event in events
            if int(event["segment_index"]) == case.segment_index
            and case.start_frame <= int(event["frame"]) <= case.end_frame
        ]
    for event in events:
        candidates = [
            case
            for case in cases_by_segment[int(event["segment_index"])]
            if int(event["track_id"]) in case.track_ids and case.start_frame <= int(event["frame"]) <= case.end_frame
        ]
        if not candidates:
            continue
        owner = min(
            candidates,
            key=lambda case: (
                abs(case.anchor_frame - int(event["frame"])),
                -case.priority_score,
                case.case_id,
            ),
        )
        owner.formal_events.append(event)
    seen: set[str] = set()
    for case in cases:
        case.formal_events.sort(key=lambda event: int(event["formal_row_index"]))
        keys = [str(event["event_key"]) for event in case.formal_events]
        duplicate = seen.intersection(keys)
        if duplicate:
            raise ValueError(f"Formal events assigned to more than one case: {sorted(duplicate)}")
        seen.update(keys)
        case.formal_delta_D, case.formal_delta_C = compute_formal_delta(case.formal_events)


def rank_and_select_cases(cases: Sequence[Case], pilot: bool, pilot_cases: int, max_cases: int) -> list[Case]:
    for case in cases:
        case.render_selected = False
        case.render_batch = "not_selected"
        case.render_status = "not_selected"
    global_order = sorted(cases, key=lambda case: (-case.priority_score, GROUP_RANK[case.segment_group], case.anchor_frame, case.case_id))
    segment_counts: dict[int, int] = defaultdict(int)
    for rank, case in enumerate(global_order, 1):
        case.priority_rank_global = rank
        segment_counts[case.segment_index] += 1
        case.priority_rank_segment = segment_counts[case.segment_index]
    if pilot:
        if not 10 <= pilot_cases <= 15:
            raise ValueError("Pilot size must remain between 10 and 15 cases")
        selected = [case for case in global_order if case.segment_index == 7][:pilot_cases]
        for case in selected:
            case.render_selected = True
            case.render_batch = "pilot"
        return selected

    if max_cases > 100:
        raise ValueError("The first review round is capped at 100 cases")
    # Allocation is target-weighted and favors the two largest observed gaps;
    # it is intentionally not an equal per-segment split.
    desired = {"S007": 36, "S005": 23, "S003": 15, "S006": 14, "S004": 6, "S001-002": 6}
    scale = min(1.0, max_cases / 100.0)
    quotas = {group: max(1, round(count * scale)) for group, count in desired.items()}
    selected: list[Case] = []
    selected_ids: set[str] = set()
    for group in GROUP_ORDER:
        group_cases = [case for case in global_order if case.segment_group == group]
        group_selected: list[Case] = []
        if group == "S001-002":
            # The diagnostic target is combined, but both physical source
            # segments still need first-round representation.
            minimum_each = min(2, quotas[group] // 2)
            for segment_index in (1, 2):
                group_selected.extend(
                    [case for case in group_cases if case.segment_index == segment_index][:minimum_each]
                )
        for case in group_cases:
            if len(group_selected) >= quotas[group]:
                break
            if case not in group_selected:
                group_selected.append(case)
        for case in group_selected:
            if len(selected) >= max_cases:
                break
            selected.append(case)
            selected_ids.add(case.case_id)
    if len(selected) < max_cases:
        for case in global_order:
            if case.case_id in selected_ids:
                continue
            selected.append(case)
            selected_ids.add(case.case_id)
            if len(selected) >= max_cases:
                break
    for case in selected:
        case.render_selected = True
        case.render_batch = "round1"
    return selected


def discover_cases(
    args: argparse.Namespace,
    segments: Sequence[Segment],
    events: Sequence[dict[str, Any]],
    logger: AuditLogger,
) -> tuple[list[Case], list[dict[str, Any]]]:
    all_cases: list[Case] = []
    stats_rows: list[dict[str, Any]] = []
    events_by_segment: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        events_by_segment[int(event["segment_index"])].append(event)
    for segment in segments:
        seeds, states, stats = analyze_segment_candidates(segment, events_by_segment[segment.index], logger)
        cases = deduplicate_seeds(
            segment,
            seeds,
            states,
            events_by_segment[segment.index],
            context_before=float(getattr(args, "context_before", 3.0)),
            context_after=float(getattr(args, "context_after", 4.0)),
        )
        stats["deduplicated_cases"] = len(cases)
        stats["segment_index"] = segment.index
        stats_rows.append(stats)
        logger.write(f"segment {segment.index}: deduplicated_cases={len(cases)}")
        all_cases.extend(cases)
    assign_formal_events(all_cases, events)
    return all_cases, stats_rows


def case_output_paths(output: Path, case: Case) -> tuple[Path, Path]:
    clip = output / "cases" / f"segment_{case.segment_index:03d}" / f"{case.case_id}.mp4"
    thumbnail = output / "thumbnails" / f"segment_{case.segment_index:03d}" / f"{case.case_id}.jpg"
    return clip.resolve(), thumbnail.resolve()


def case_to_manifest_row(case: Case, segment: Segment, output: Path) -> dict[str, Any]:
    clip, thumbnail = case_output_paths(output, case)
    return {
        "case_id": case.case_id,
        "segment_group": case.segment_group,
        "segment_index": case.segment_index,
        "source_video": str(segment.source),
        "track_cache": str(segment.track_cache),
        "detection_cache": str(segment.detection_cache),
        "clip_path": str(clip) if case.render_selected else "",
        "thumbnail_path": str(thumbnail) if case.render_selected else "",
        "start_frame": case.start_frame,
        "anchor_frame": case.anchor_frame,
        "end_frame": case.end_frame,
        "start_timestamp": timestamp(case.start_frame, segment.fps),
        "anchor_timestamp": timestamp(case.anchor_frame, segment.fps),
        "end_timestamp": timestamp(case.end_frame, segment.fps),
        "track_ids": ";".join(str(track_id) for track_id in case.track_ids),
        "candidate_flags": ";".join(case.candidate_flags),
        "priority_score": f"{case.priority_score:.3f}",
        "priority_rank_global": case.priority_rank_global,
        "priority_rank_segment": case.priority_rank_segment,
        "priority_reason": case.priority_reason,
        "formal_event_rows": ";".join(str(event["event_key"]) for event in case.formal_events),
        "window_formal_event_rows": ";".join(str(event["event_key"]) for event in case.display_events),
        "formal_delta_D": case.formal_delta_D,
        "formal_delta_C": case.formal_delta_C,
        "track_history": case.track_history,
        "render_selected": "yes" if case.render_selected else "no",
        "render_batch": case.render_batch,
        "render_status": case.render_status,
    }


def write_case_manifest(output: Path, cases: Sequence[Case], segments: Sequence[Segment]) -> list[dict[str, Any]]:
    by_index = {segment.index: segment for segment in segments}
    rows = [case_to_manifest_row(case, by_index[case.segment_index], output) for case in cases]
    rows.sort(key=lambda row: (int(row["priority_rank_global"]), row["case_id"]))
    atomic_write_csv(output / "case_manifest.csv", CASE_MANIFEST_COLUMNS, rows)
    return rows


def load_preserved_human_fields(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    rows = read_csv(path)
    preserved: dict[str, dict[str, str]] = {}
    for row in rows:
        case_id = row.get("case_id", "")
        if not case_id or case_id in preserved:
            raise ValueError(f"Duplicate or blank case_id in existing review sheet: {case_id!r}")
        preserved[case_id] = {column: row.get(column, "") for column in HUMAN_COLUMNS}
    return preserved


def invalidate_existing_reviews(path: Path, archive: Path) -> int:
    """Archive non-empty human fields and clear them after a proven rendering failure."""
    if not path.is_file():
        return 0
    rows = read_csv(path)
    invalidated = []
    for row in rows:
        fields = {column: row.get(column, "") for column in HUMAN_COLUMNS}
        if any(value.strip() for value in fields.values()):
            invalidated.append({"case_id": row["case_id"], **fields})
        for column in HUMAN_COLUMNS:
            row[column] = ""
    if invalidated:
        atomic_write_csv(archive, ["case_id", *HUMAN_COLUMNS], invalidated)
    atomic_write_csv(path, REVIEW_COLUMNS, rows)
    return len(invalidated)


def write_review_sheet(output: Path, manifest_rows: Sequence[dict[str, Any]]) -> None:
    path = output / "review_sheet.csv"
    preserved = load_preserved_human_fields(path)
    selected_ids = {
        str(manifest["case_id"]) for manifest in manifest_rows if manifest["render_selected"] == "yes"
    }
    removed_reviewed = [
        case_id for case_id, fields in preserved.items() if case_id not in selected_ids and any(fields.values())
    ]
    if removed_reviewed:
        raise ValueError(
            "Regeneration would remove cases with human review fields; keep the same batch or archive the review first: "
            f"{removed_reviewed}"
        )
    rows = []
    for manifest in manifest_rows:
        if manifest["render_selected"] != "yes":
            continue
        row = {column: manifest.get(column, "") for column in REVIEW_AUTO_COLUMNS}
        row.update(preserved.get(str(manifest["case_id"]), {column: "" for column in HUMAN_COLUMNS}))
        rows.append(row)
    atomic_write_csv(path, REVIEW_COLUMNS, rows)


def _parse_optional_int(value: str, column: str) -> int | None:
    stripped = value.strip()
    if stripped == "":
        return None
    try:
        number = int(stripped)
    except ValueError as exc:
        raise ValueError(f"{column} must be an integer") from exc
    return number


def derive_expected_delta(review: dict[str, str], formal_d: int, formal_c: int) -> dict[str, Any]:
    """Validate one review row and derive expected/bias/correction.

    Pending and uncertain rows deliberately return blank numerical fields.
    """
    result = {
        "expected_delta_D": None,
        "expected_delta_C": None,
        "bias_D": None,
        "bias_C": None,
        "correction_D": None,
        "correction_C": None,
        "delta_source": "",
        "confirmation_state": "unreviewed",
        "validation_error": "",
    }
    try:
        status = review.get("review_status", "").strip()
        outcome = review.get("review_outcome", "").strip()
        if status == "":
            status = "unreviewed"
        if status not in REVIEW_STATUS_VALUES:
            raise ValueError(f"illegal review_status={status!r}")
        mechanisms = [value for value in review.get("mechanism_flags", "").split(";") if value]
        illegal_mechanisms = sorted(set(mechanisms) - MECHANISM_VALUES)
        if illegal_mechanisms:
            raise ValueError(f"illegal mechanism_flags={';'.join(illegal_mechanisms)}")
        second_outcome = review.get("second_review_outcome", "").strip()
        if second_outcome and second_outcome not in OUTCOME_VALUES:
            raise ValueError(f"illegal second_review_outcome={second_outcome!r}")
        second_confidence = review.get("second_confidence", "").strip()
        if second_confidence and second_confidence not in CONFIDENCE_VALUES:
            raise ValueError(f"illegal second_confidence={second_confidence!r}")

        override_d = _parse_optional_int(review.get("expected_delta_D_override", ""), "expected_delta_D_override")
        override_c = _parse_optional_int(review.get("expected_delta_C_override", ""), "expected_delta_C_override")
        if (override_d is None) != (override_c is None):
            raise ValueError("both delta overrides must be filled together")

        if status == "unreviewed":
            if outcome or override_d is not None:
                raise ValueError("unreviewed row contains an outcome or delta override")
            return result
        if outcome not in OUTCOME_VALUES:
            raise ValueError(f"reviewed row has illegal or missing review_outcome={outcome!r}")
        if not review.get("reviewer", "").strip():
            raise ValueError("reviewer is required for a reviewed row")
        confidence = review.get("confidence", "").strip()
        if confidence not in CONFIDENCE_VALUES:
            raise ValueError("confidence must be high, medium, or low")

        if outcome == "uncertain":
            if override_d is not None:
                raise ValueError("uncertain row must not contain delta overrides")
            result["confirmation_state"] = "uncertain"
            return result

        species = review.get("expected_species", "").strip()
        direction = review.get("expected_direction", "").strip()
        if species and species not in {"dorsalis", "correcta", "none"}:
            raise ValueError(f"illegal expected_species={species!r}")
        if direction and direction not in {"enter", "exit", "none"}:
            raise ValueError(f"illegal expected_direction={direction!r}")

        if override_d is not None:
            expected_d, expected_c = override_d, override_c
            source = "manual_override"
        elif outcome == "formal_correct":
            expected_d, expected_c = formal_d, formal_c
            source = "formal_delta"
        elif outcome == "formal_event_not_real":
            expected_d, expected_c = 0, 0
            source = "outcome_rule"
        else:
            if species not in SPECIES_CLASS or direction not in {"enter", "exit"}:
                raise ValueError(f"{outcome} requires expected_species and expected_direction")
            delta = 1 if direction == "enter" else -1
            expected_d = delta if species == "dorsalis" else 0
            expected_c = delta if species == "correcta" else 0
            source = "species_direction"

        result.update(
            {
                "expected_delta_D": expected_d,
                "expected_delta_C": expected_c,
                "bias_D": formal_d - expected_d,
                "bias_C": formal_c - expected_c,
                "correction_D": expected_d - formal_d,
                "correction_C": expected_c - formal_c,
                "delta_source": source,
                "confirmation_state": "pending_second_review" if status == "needs_second_review" else "confirmed",
            }
        )
        return result
    except ValueError as exc:
        result["confirmation_state"] = "invalid"
        result["validation_error"] = str(exc)
        return result


def build_ledger_rows(manifest_rows: Sequence[dict[str, str]], review_rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    manifest_by_id = {row["case_id"]: row for row in manifest_rows}
    if len(manifest_by_id) != len(manifest_rows):
        raise ValueError("case_manifest.csv contains duplicate case IDs")
    ledger = []
    seen: set[str] = set()
    for review in review_rows:
        case_id = review.get("case_id", "")
        if case_id in seen:
            raise ValueError(f"review_sheet.csv contains duplicate case ID: {case_id}")
        seen.add(case_id)
        if case_id not in manifest_by_id:
            raise ValueError(f"review_sheet.csv references an unknown case: {case_id}")
        manifest = manifest_by_id[case_id]
        formal_d, formal_c = int(manifest["formal_delta_D"]), int(manifest["formal_delta_C"])
        derived = derive_expected_delta(review, formal_d, formal_c)
        ledger.append(
            {
                "case_id": case_id,
                "segment_group": manifest["segment_group"],
                "start_frame": manifest["start_frame"],
                "anchor_frame": manifest["anchor_frame"],
                "end_frame": manifest["end_frame"],
                "track_ids": manifest["track_ids"],
                "candidate_flags": manifest["candidate_flags"],
                "formal_event_rows": manifest["formal_event_rows"],
                "formal_delta_D": formal_d,
                "formal_delta_C": formal_c,
                "review_status": review.get("review_status", ""),
                "review_outcome": review.get("review_outcome", ""),
                "mechanism_flags": review.get("mechanism_flags", ""),
                "reviewer": review.get("reviewer", ""),
                "confidence": review.get("confidence", ""),
                "notes": review.get("notes", ""),
                **{key: "" if value is None else value for key, value in derived.items()},
            }
        )
    return ledger


def _next_recommendation(residual_d: int, residual_c: int, unreviewed: int, uncertain: int) -> str:
    if residual_d == 0 and residual_c == 0:
        return "numerically closed; retain causal caution and verify diagnostic-truth semantics"
    if unreviewed:
        return "review remaining selected high-priority cases before expanding the batch"
    if uncertain:
        return "second-review uncertain cases, then add non-event ring-activity windows if residual remains"
    if abs(residual_d) >= 5 or abs(residual_c) >= 5:
        return "expand non-event ring activity and suspected ID-rebuild/detection-gap search; verify diagnostic-truth semantics"
    return "inspect next-ranked unrendered candidates, emphasizing class changes and boundary jitter"


def build_residual_rows(ledger_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for group in GROUP_ORDER + ["LV006"]:
        rows = list(ledger_rows) if group == "LV006" else [row for row in ledger_rows if row["segment_group"] == group]
        confirmed = [row for row in rows if row["confirmation_state"] == "confirmed"]
        confirmed_d = sum(int(row["bias_D"]) for row in confirmed)
        confirmed_c = sum(int(row["bias_C"]) for row in confirmed)
        target_d, target_c = TARGET_BIASES[group]
        uncertain = sum(row["confirmation_state"] == "uncertain" for row in rows)
        needs_second = sum(row["confirmation_state"] == "pending_second_review" for row in rows)
        unreviewed = sum(row["confirmation_state"] == "unreviewed" for row in rows)
        invalid = sum(row["confirmation_state"] == "invalid" for row in rows)
        residual_d, residual_c = target_d - confirmed_d, target_c - confirmed_c
        result.append(
            {
                "segment_group": group,
                "target_bias_D": target_d,
                "target_bias_C": target_c,
                "confirmed_bias_D": confirmed_d,
                "confirmed_bias_C": confirmed_c,
                "residual_D": residual_d,
                "residual_C": residual_c,
                "reviewed_cases": len(confirmed),
                "uncertain_cases": uncertain,
                "needs_second_review_cases": needs_second,
                "unreviewed_cases": unreviewed,
                "invalid_cases": invalid,
                "next_recommendation": _next_recommendation(residual_d, residual_c, unreviewed, uncertain),
            }
        )
    return result


def summarize_outputs(output: Path, logger: AuditLogger, fail_on_invalid: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_path = output / "case_manifest.csv"
    review_path = output / "review_sheet.csv"
    if not manifest_path.is_file() or not review_path.is_file():
        raise FileNotFoundError("case_manifest.csv and review_sheet.csv must exist before summarize")
    ledger = build_ledger_rows(read_csv(manifest_path), read_csv(review_path))
    residuals = build_residual_rows(ledger)
    atomic_write_csv(output / "error_ledger.csv", LEDGER_COLUMNS, ledger)
    atomic_write_csv(output / "residual_summary.csv", RESIDUAL_COLUMNS, residuals)
    invalid = [row for row in ledger if row["confirmation_state"] == "invalid"]
    report_rows = [
        {"severity": "error", "case_id": row["case_id"], "message": row["validation_error"]}
        for row in invalid
    ]
    atomic_write_csv(output / "validation_report.csv", ["severity", "case_id", "message"], report_rows)
    logger.write(
        f"summarize: ledger_cases={len(ledger)}, confirmed={sum(row['confirmation_state'] == 'confirmed' for row in ledger)}, "
        f"uncertain={sum(row['confirmation_state'] == 'uncertain' for row in ledger)}, invalid={len(invalid)}"
    )
    if invalid and fail_on_invalid:
        raise ValueError(f"{len(invalid)} review rows are invalid; see validation_report.csv")
    return ledger, residuals


def ffprobe_video(path: Path, ffprobe: Path, count_frames: bool = True) -> dict[str, Any]:
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
        "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames,nb_read_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Expected exactly one video stream: {path}")
    return streams[0]


def full_decode_check(path: Path, ffmpeg: Path, logger: AuditLogger) -> None:
    with logger.path.open("a", encoding="utf-8") as log_handle:
        result = subprocess.run(
            [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
            stdout=subprocess.DEVNULL,
            stderr=log_handle,
        )
    if result.returncode:
        raise ValueError(f"ffmpeg full decode failed for {path}")


def opencv_decode_check(path: Path, expected_frames: int) -> None:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"OpenCV cannot open {path}")
    try:
        for frame_zero in sorted({0, max(0, expected_frames // 2), max(0, expected_frames - 1)}):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_zero)
            ok, image = capture.read()
            if not ok or image is None or image.size == 0:
                raise ValueError(f"OpenCV cannot decode frame {frame_zero + 1} of {path}")
    finally:
        capture.release()


def clip_schedule(case: Case, segment: Segment, config: dict[str, Any]) -> tuple[list[int], list[int], int]:
    output_fps = float(config["output_fps"])
    normal_seconds = (case.end_frame - case.start_frame + 1) / segment.fps
    normal_count = max(1, round(normal_seconds * output_fps))
    normal = np.rint(np.linspace(case.start_frame, case.end_frame, normal_count)).astype(int).tolist()
    half_slow = float(config["slow_source_seconds"]) / 2.0
    slow_start = max(case.start_frame, case.anchor_frame - round(half_slow * segment.fps))
    slow_end = min(case.end_frame, case.anchor_frame + round(half_slow * segment.fps))
    slow_seconds = (slow_end - slow_start + 1) / segment.fps
    slow_count = max(1, round(slow_seconds / float(config["slow_rate"]) * output_fps))
    slow = np.rint(np.linspace(slow_start, slow_end, slow_count)).astype(int).tolist()
    hold_count = max(1, round(float(config["anchor_hold"]) * output_fps))
    return normal, slow, hold_count


def expected_clip_frames(case: Case, segment: Segment, config: dict[str, Any]) -> int:
    normal, slow, hold = clip_schedule(case, segment, config)
    return len(normal) + hold + len(slow)


def validate_clip(
    path: Path,
    expected_frames: int,
    config: dict[str, Any],
    ffprobe: Path,
    ffmpeg: Path,
    logger: AuditLogger,
    require_full_decode: bool = True,
) -> dict[str, Any]:
    probe = ffprobe_video(path, ffprobe, count_frames=True)
    actual_frames = int(probe.get("nb_read_frames") or probe.get("nb_frames") or -1)
    actual_fps = fps_float(probe["avg_frame_rate"])
    failures = []
    if probe.get("codec_name") != "h264":
        failures.append(f"codec={probe.get('codec_name')}")
    if probe.get("pix_fmt") != "yuv420p":
        failures.append(f"pixel_format={probe.get('pix_fmt')}")
    if int(probe["width"]) != int(config["output_width"]) or int(probe["height"]) != int(config["output_height"]):
        failures.append(f"dimensions={probe.get('width')}x{probe.get('height')}")
    if not math.isclose(actual_fps, float(config["output_fps"]), abs_tol=1e-6, rel_tol=0.0):
        failures.append(f"fps={actual_fps}")
    if actual_frames != expected_frames:
        failures.append(f"frames={actual_frames}, expected={expected_frames}")
    if failures:
        raise ValueError(f"Clip validation failed for {path}: {'; '.join(failures)}")
    if require_full_decode:
        full_decode_check(path, ffmpeg, logger)
        opencv_decode_check(path, expected_frames)
    return {
        "codec": probe.get("codec_name", ""),
        "pixel_format": probe.get("pix_fmt", ""),
        "width": int(probe["width"]),
        "height": int(probe["height"]),
        "fps": actual_fps,
        "frame_count": actual_frames,
        "duration_seconds": float(probe.get("duration") or actual_frames / actual_fps),
    }


def merge_intervals(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def collect_render_rows(
    segment: Segment,
    cases: Sequence[Case],
    logger: AuditLogger,
) -> tuple[dict[int, list[list[float]]], dict[int, list[list[float]]]]:
    intervals = merge_intervals([(case.start_frame, case.end_frame) for case in cases])
    track_output: dict[int, list[list[float]]] = {}
    detection_output: dict[int, list[list[float]]] = {}
    interval_index = 0
    track_frames = iter_cache_frames(segment.track_cache, expected_kind="tracks")
    detection_frames = iter_cache_frames(segment.detection_cache, expected_kind="detections")
    for (track_frame, track_rows), (detection_frame, detection_rows) in zip(track_frames, detection_frames):
        if track_frame != detection_frame:
            raise ValueError(f"Paired cache frame mismatch during render collection: segment {segment.index}")
        while interval_index < len(intervals) and track_frame > intervals[interval_index][1]:
            interval_index += 1
        if interval_index >= len(intervals):
            continue
        start, end = intervals[interval_index]
        if start <= track_frame <= end:
            track_output[track_frame] = track_rows
            detection_output[track_frame] = detection_rows
    expected_union_frames = sum(end - start + 1 for start, end in intervals)
    if len(track_output) != expected_union_frames or len(detection_output) != expected_union_frames:
        raise ValueError(
            f"Render cache collection incomplete for segment {segment.index}: "
            f"tracks={len(track_output)}, detections={len(detection_output)}, expected={expected_union_frames}"
        )
    logger.write(f"segment {segment.index}: collected {expected_union_frames} unique cached frames for rendering")
    return track_output, detection_output


def _put_text(
    image: np.ndarray,
    text_value: str,
    origin: tuple[int, int],
    scale: float = 0.55,
    color: tuple[int, int, int] = (245, 245, 245),
    thickness: int = 1,
) -> None:
    cv2.putText(image, text_value, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text_value, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_panel_tracks(
    panel: np.ndarray,
    track_rows: Sequence[Sequence[float]],
    detection_rows: Sequence[Sequence[float]],
    case: Case,
    source_width: int,
    source_height: int,
    roi: tuple[int, int, int, int] | None,
) -> None:
    panel_height, panel_width = panel.shape[:2]
    if roi is None:
        offset_x, offset_y, domain_width, domain_height = 0, 0, source_width, source_height
    else:
        offset_x, offset_y, right, bottom = roi
        domain_width, domain_height = right - offset_x, bottom - offset_y
    scale_x, scale_y = panel_width / domain_width, panel_height / domain_height

    def transform(x: float, y: float) -> tuple[int, int]:
        return round((x - offset_x) * scale_x), round((y - offset_y) * scale_y)

    circle_center = transform(CENTER[0], CENTER[1])
    circle_radius = round(RADIUS * (scale_x + scale_y) / 2.0)
    cv2.circle(panel, circle_center, circle_radius, (0, 230, 255), 2, cv2.LINE_AA)
    related = set(case.track_ids)
    for row in track_rows:
        track_id, class_id = int(row[4]), int(row[6])
        relevant = track_id in related
        x1, y1 = transform(float(row[0]), float(row[1]))
        x2, y2 = transform(float(row[2]), float(row[3]))
        if x2 < 0 or y2 < 0 or x1 >= panel_width or y1 >= panel_height:
            continue
        color = CLASS_COLORS.get(class_id, (180, 180, 180)) if relevant else (95, 95, 95)
        thickness = 3 if relevant else 1
        cv2.rectangle(panel, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        if relevant:
            label_y = max(18, y1 - 5)
            _put_text(panel, f"ID {track_id} {CLASS_SHORT.get(class_id, '?')} {float(row[5]):.2f}", (max(0, x1), label_y), 0.48, color, 1)

    if "unmatched_detection" in case.candidate_flags:
        matched = {int(row[7]) for row in track_rows if int(row[7]) >= 0}
        for index, row in enumerate(detection_rows):
            if index in matched or float(row[4]) < 0.40:
                continue
            cx, cy = (float(row[0]) + float(row[2])) / 2.0, (float(row[1]) + float(row[3])) / 2.0
            if abs(radial(cx, cy) - RADIUS) > RING_HALF_WIDTH:
                continue
            x1, y1 = transform(float(row[0]), float(row[1]))
            x2, y2 = transform(float(row[2]), float(row[3]))
            cv2.rectangle(panel, (x1, y1), (x2, y2), (255, 0, 255), 2, cv2.LINE_AA)
            _put_text(panel, f"UNMATCHED {CLASS_SHORT.get(int(row[5]), '?')} {float(row[4]):.2f}", (max(0, x1), max(18, y1 - 4)), 0.42, (255, 0, 255), 1)


def render_canvas(
    source_frame: np.ndarray,
    frame_number: int,
    phase: str,
    case: Case,
    segment: Segment,
    track_rows: Sequence[Sequence[float]],
    detection_rows: Sequence[Sequence[float]],
    config: dict[str, Any],
) -> np.ndarray:
    width, height = int(config["output_width"]), int(config["output_height"])
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    top = round(height * 0.15)
    bottom = round(height * 0.15)
    content_height = height - top - bottom
    full_width = round(width * 2 / 3)
    roi_width = width - full_width
    full_panel = cv2.resize(source_frame, (full_width, content_height), interpolation=cv2.INTER_AREA)
    roi_half = 350
    roi = (
        max(0, round(CENTER[0]) - roi_half),
        max(0, round(CENTER[1]) - roi_half),
        min(segment.width, round(CENTER[0]) + roi_half),
        min(segment.height, round(CENTER[1]) + roi_half),
    )
    crop = source_frame[roi[1] : roi[3], roi[0] : roi[2]]
    roi_panel = cv2.resize(crop, (roi_width, content_height), interpolation=cv2.INTER_AREA)
    _draw_panel_tracks(full_panel, track_rows, detection_rows, case, segment.width, segment.height, None)
    _draw_panel_tracks(roi_panel, track_rows, detection_rows, case, segment.width, segment.height, roi)
    canvas[top : top + content_height, :full_width] = full_panel
    canvas[top : top + content_height, full_width:] = roi_panel
    cv2.line(canvas, (full_width, top), (full_width, top + content_height), (210, 210, 210), 2)
    phase_color = {"CONTEXT 1.0x": (80, 230, 80), "ANCHOR HOLD": (0, 190, 255), "REPLAY 0.25x": (255, 180, 40)}.get(phase, (240, 240, 240))
    _put_text(canvas, f"{case.case_id}  SEG {segment.index}  {phase}", (18, 30), 0.72, phase_color, 2)
    _put_text(canvas, f"source frame {frame_number}/{segment.frame_count}  t={timestamp(frame_number, segment.fps)}  anchor={case.anchor_frame}", (18, 60), 0.58)
    _put_text(canvas, "FULL FRAME", (18, top - 6), 0.48, (210, 210, 210))
    _put_text(canvas, "COUNTING-CIRCLE ROI", (full_width + 15, top - 6), 0.48, (210, 210, 210))
    nearby_events = [
        event for event in case.display_events if abs(int(event["frame"]) - frame_number) <= round(0.75 * segment.fps)
    ]
    events_to_show = nearby_events if nearby_events else case.formal_events
    event_summary = "; ".join(
        f"{event['event_key']}@{int(event['frame'])} {CLASS_SHORT[int(event['class_id'])]} {event['direction']} {int(event['delta']):+d}"
        for event in events_to_show
    ) or "none"
    event_label = "window formal events near frame" if nearby_events else "assigned case events"
    _put_text(canvas, f"track history: {case.track_history[:140]}", (18, height - 82), 0.46, (205, 205, 205))
    _put_text(canvas, f"flags: {';'.join(case.candidate_flags)}", (18, height - 54), 0.50, (230, 230, 230))
    _put_text(canvas, f"{event_label}: {event_summary[:165]}", (18, height - 25), 0.50, (230, 230, 230))
    if abs(frame_number - case.anchor_frame) <= 1:
        cv2.rectangle(canvas, (4, 4), (width - 5, height - 5), (0, 170, 255), 5, cv2.LINE_AA)
        _put_text(canvas, "ANCHOR", (width - 125, 33), 0.68, (0, 190, 255), 2)
    return canvas


def parse_packet_pts_lines(lines: Iterable[str], expected_frames: int) -> list[float]:
    """Return one strictly increasing presentation timestamp per decoded video frame."""
    values: list[float] = []
    invalid: list[str] = []
    for raw_line in lines:
        token = raw_line.strip().split(",", 1)[0]
        if not token:
            continue
        if token == "N/A":
            invalid.append(token)
            continue
        try:
            value = float(token)
        except ValueError:
            invalid.append(token)
            continue
        if not math.isfinite(value):
            invalid.append(token)
            continue
        values.append(value)
    if invalid:
        raise ValueError(f"Video packet PTS contains invalid values: {invalid[:5]}")
    values.sort()
    if len(values) != expected_frames:
        raise ValueError(f"Video packet PTS count {len(values)} != expected frames {expected_frames}")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("Video packet presentation timestamps are not strictly increasing")
    return values


def build_source_frame_pts(source: Path, expected_frames: int, ffprobe: Path) -> list[float]:
    """Build the ordinal-frame to PTS map without decoding pixels."""
    result = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=pts_time",
            "-of",
            "csv=p=0",
            str(source),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise ValueError(f"ffprobe packet PTS scan failed for {source}: {result.stderr[-2000:]}")
    return parse_packet_pts_lines(result.stdout.splitlines(), expected_frames)


def source_frame_sha256(image: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(image)
    return hashlib.sha256(memoryview(contiguous)).hexdigest()


def scheduled_source_frames(
    source: Path,
    indices: Sequence[int],
    frame_pts: Sequence[float],
) -> Iterator[tuple[int, np.ndarray]]:
    """Read ordinal frames through their exact presentation timestamps.

    CAP_PROP_POS_FRAMES is intentionally forbidden here: on the LV006 long VFR
    sources it silently returned a different frame while reporting the requested
    ordinal position.
    """
    if not indices:
        return
    if not frame_pts:
        raise ValueError(f"Missing frame PTS index for {source}")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"OpenCV cannot open source video: {source}")
    try:
        current: int | None = None
        image: np.ndarray | None = None

        def ordinal_for_actual_pts(actual_pts: float) -> int:
            insertion = bisect.bisect_left(frame_pts, actual_pts)
            candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(frame_pts)]
            nearest = min(candidates, key=lambda index: abs(float(frame_pts[index]) - actual_pts))
            expected_pts = float(frame_pts[nearest])
            if not math.isclose(
                actual_pts,
                expected_pts,
                abs_tol=FRAME_PTS_TOLERANCE_SECONDS,
                rel_tol=0.0,
            ):
                raise ValueError(
                    f"Decoded source PTS is absent from the frame index: "
                    f"actual={actual_pts:.6f}, nearest={expected_pts:.6f}, source={source}"
                )
            return nearest + 1

        for target in indices:
            target = int(target)
            if not 1 <= target <= len(frame_pts):
                raise ValueError(f"Source frame {target} is outside 1..{len(frame_pts)}: {source}")
            if current is None or target < current:
                # Seek before the target, identify the actual returned ordinal by
                # PTS, then decode sequentially. A backend can land several frames
                # after the requested timestamp, so expand the lookback until the
                # decoded frame is at or before the target.
                lookback = 3
                while True:
                    seek_frame = max(1, target - lookback)
                    seek_pts = float(frame_pts[seek_frame - 1])
                    if not capture.set(cv2.CAP_PROP_POS_MSEC, seek_pts * 1000.0):
                        raise ValueError(f"OpenCV rejected PTS seek for frame {target}: {source}")
                    ok, image = capture.read()
                    if not ok or image is None:
                        raise ValueError(f"Source decode failed while seeking frame {target}: {source}")
                    actual_pts = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                    current = ordinal_for_actual_pts(actual_pts)
                    if current <= target:
                        break
                    if seek_frame == 1:
                        raise ValueError(f"PTS seek from the start overshot frame {target}: {source}")
                    lookback *= 4
            while current is None or current < target:
                ok, image = capture.read()
                if not ok or image is None:
                    next_frame = 1 if current is None else current + 1
                    raise ValueError(f"Source decode failed before frame {next_frame}: {source}")
                actual_pts = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                decoded_ordinal = ordinal_for_actual_pts(actual_pts)
                if current is not None and decoded_ordinal != current + 1:
                    raise ValueError(
                        f"Non-sequential source decode: previous={current}, actual={decoded_ordinal}, source={source}"
                    )
                current = decoded_ordinal
                if current > target:
                    raise ValueError(f"PTS seek overshot frame {target} and returned {current}: {source}")
            if image is None:
                raise ValueError(f"No decoded source frame for {target}: {source}")
            yield target, image
    finally:
        capture.release()


def encoder_command(partial: Path, config: dict[str, Any], ffmpeg: Path) -> list[str]:
    return [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-video_size",
        f"{config['output_width']}x{config['output_height']}",
        "-framerate",
        str(config["output_fps"]),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        str(config["x264_preset"]),
        "-crf",
        str(config["x264_crf"]),
        "-pix_fmt",
        "yuv420p",
        "-fps_mode",
        "cfr",
        "-movflags",
        "+faststart",
        str(partial),
    ]


def quarantine_invalid(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    quarantine_dir = path.parents[2] / "_superseded"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    target = quarantine_dir / f"{path.stem}.invalid.{stamp}{path.suffix}"
    counter = 1
    while target.exists():
        target = quarantine_dir / f"{path.stem}.invalid.{stamp}.{counter}{path.suffix}"
        counter += 1
    os.replace(path, target)
    return target


def render_signature(case: Case, segment: Segment, config: dict[str, Any]) -> str:
    payload = {
        "renderer_version": RENDERER_VERSION,
        "case_id": case.case_id,
        "segment_index": segment.index,
        "start_frame": case.start_frame,
        "anchor_frame": case.anchor_frame,
        "end_frame": case.end_frame,
        "track_ids": case.track_ids,
        "candidate_flags": case.candidate_flags,
        "formal_event_rows": [event["event_key"] for event in case.formal_events],
        "window_formal_event_rows": [event["event_key"] for event in case.display_events],
        "track_history": case.track_history,
        "render_config": config,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def acceptable_render_signatures(case: Case, segment: Segment, config: dict[str, Any]) -> set[str]:
    # Batch membership does not affect pixels.  A pilot clip can therefore be
    # reused in round 1 when every actual rendering parameter is unchanged.
    signatures = {render_signature(case, segment, config)}
    for batch in ("pilot", "round1"):
        variant = dict(config)
        variant["batch"] = batch
        signatures.add(render_signature(case, segment, variant))
    return signatures


def render_case(
    case: Case,
    segment: Segment,
    track_frames: dict[int, list[list[float]]],
    detection_frames: dict[int, list[list[float]]],
    output: Path,
    config: dict[str, Any],
    frame_pts: Sequence[float],
    ffmpeg: Path,
    ffprobe: Path,
    logger: AuditLogger,
) -> dict[str, Any]:
    clip, thumbnail = case_output_paths(output, case)
    sidecar = clip.with_suffix(".render.json")
    clip.parent.mkdir(parents=True, exist_ok=True)
    thumbnail.parent.mkdir(parents=True, exist_ok=True)
    normal, slow, hold_count = clip_schedule(case, segment, config)
    expected_frames = len(normal) + hold_count + len(slow)
    resumed = False
    if clip.is_file():
        try:
            if not sidecar.is_file():
                raise ValueError("render signature sidecar is missing")
            sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
            if sidecar_payload.get("render_signature") not in acceptable_render_signatures(case, segment, config):
                raise ValueError("render signature differs from current case/configuration")
            metadata = validate_clip(clip, expected_frames, config, ffprobe, ffmpeg, logger, require_full_decode=True)
            resumed = True
            case.render_status = "resumed_valid"
            logger.write(f"{case.case_id}: existing valid clip reused")
        except Exception as exc:
            preserved = quarantine_invalid(clip)
            logger.write(f"{case.case_id}: invalid prior clip preserved as {preserved}: {exc}")
    if not resumed:
        partial = clip.with_name(f".{clip.stem}.partial.mp4")
        partial.unlink(missing_ok=True)
        process = subprocess.Popen(encoder_command(partial, config, ffmpeg), stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        assert process.stdin is not None
        written = 0
        thumbnail_written = False
        anchor_source_hash = ""
        try:
            phases = [("CONTEXT 1.0x", normal), ("ANCHOR HOLD", [case.anchor_frame] * hold_count), ("REPLAY 0.25x", slow)]
            for phase, schedule in phases:
                for frame_number, source_frame in scheduled_source_frames(segment.source, schedule, frame_pts):
                    if frame_number == case.anchor_frame and not anchor_source_hash:
                        anchor_source_hash = source_frame_sha256(source_frame)
                    canvas = render_canvas(
                        source_frame,
                        frame_number,
                        phase,
                        case,
                        segment,
                        track_frames[frame_number],
                        detection_frames[frame_number],
                        config,
                    )
                    if not thumbnail_written and frame_number == case.anchor_frame:
                        if not cv2.imwrite(str(thumbnail), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                            raise ValueError(f"Failed to write thumbnail: {thumbnail}")
                        thumbnail_written = True
                    process.stdin.write(canvas.tobytes())
                    written += 1
            process.stdin.close()
            stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
            return_code = process.wait()
            if return_code:
                raise RuntimeError(f"ffmpeg exited {return_code}: {stderr[-2000:]}")
            if written != expected_frames:
                raise ValueError(f"Rendered frame count {written} != expected {expected_frames}")
            if not anchor_source_hash:
                raise ValueError(f"Anchor source frame was not decoded for {case.case_id}")
            os.replace(partial, clip)
        except Exception:
            if process.poll() is None:
                process.kill()
                process.wait()
            partial.unlink(missing_ok=True)
            raise
        metadata = validate_clip(clip, expected_frames, config, ffprobe, ffmpeg, logger, require_full_decode=True)
        atomic_write_text(
            sidecar,
            json.dumps(
                {
                    "renderer_version": RENDERER_VERSION,
                    "frame_seek_method": FRAME_SEEK_METHOD,
                    "render_signature": render_signature(case, segment, config),
                    "case_id": case.case_id,
                    "clip": str(clip),
                    "anchor_frame": case.anchor_frame,
                    "anchor_source_pts": float(frame_pts[case.anchor_frame - 1]),
                    "anchor_source_sha256": anchor_source_hash,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        case.render_status = "rendered_valid"
        logger.write(f"{case.case_id}: rendered and validated ({expected_frames} frames)")
    if not thumbnail.is_file():
        anchor_frame = next(scheduled_source_frames(segment.source, [case.anchor_frame], frame_pts))[1]
        canvas = render_canvas(
            anchor_frame,
            case.anchor_frame,
            "ANCHOR HOLD",
            case,
            segment,
            track_frames[case.anchor_frame],
            detection_frames[case.anchor_frame],
            config,
        )
        if not cv2.imwrite(str(thumbnail), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise ValueError(f"Failed to write thumbnail: {thumbnail}")
    stat = clip.stat()
    return {
        "record_type": "clip",
        "case_id": case.case_id,
        "segment_index": segment.index,
        "path": str(clip),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(clip),
        "cache_id": "",
        "upstream_cache_id": "",
        "cache_complete": "",
        "cache_frames": "",
        **metadata,
        "validation_status": case.render_status,
        "detail": (
            f"source={segment.source}; anchor={case.anchor_frame}; "
            f"anchor_pts={frame_pts[case.anchor_frame - 1]:.6f}; "
            f"seek_method={FRAME_SEEK_METHOD}; thumbnail={thumbnail}"
        ),
    }


def render_selected_cases(
    args: argparse.Namespace,
    segments: Sequence[Segment],
    selected: Sequence[Case],
    config: dict[str, Any],
    logger: AuditLogger,
) -> list[dict[str, Any]]:
    output_rows = []
    by_segment: dict[int, list[Case]] = defaultdict(list)
    for case in selected:
        by_segment[case.segment_index].append(case)
    for segment in segments:
        segment_cases = sorted(by_segment.get(segment.index, []), key=lambda case: (-case.priority_score, case.case_id))
        if not segment_cases:
            continue
        logger.write(f"segment {segment.index}: rendering {len(segment_cases)} selected cases")
        pts_started = time.monotonic()
        frame_pts = build_source_frame_pts(segment.source, segment.frame_count, args.ffprobe)
        logger.write(
            f"segment {segment.index}: built exact frame PTS index ({len(frame_pts)} frames) "
            f"in {time.monotonic() - pts_started:.1f}s"
        )
        track_frames, detection_frames = collect_render_rows(segment, segment_cases, logger)
        for index, case in enumerate(segment_cases, 1):
            output_rows.append(
                render_case(
                    case,
                    segment,
                    track_frames,
                    detection_frames,
                    args.output,
                    config,
                    frame_pts,
                    args.ffmpeg,
                    args.ffprobe,
                    logger,
                )
            )
            if index % 5 == 0 or index == len(segment_cases):
                logger.write(f"segment {segment.index}: case progress {index}/{len(segment_cases)}")
        del track_frames, detection_frames, frame_pts
    return output_rows


def authoritative_inputs(args: argparse.Namespace, segments: Sequence[Segment]) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = [
        ("manifest", args.segments_manifest.resolve()),
        ("manifest", args.long_videos_manifest.resolve()),
    ]
    for name in ("count_events.csv", "final_counts.csv", "segment_summary.csv", "replay_config.csv"):
        paths.append(("formal_csv", (args.count_dir / name).resolve()))
    for segment in segments:
        paths.extend(
            [
                ("source_video", segment.source),
                ("track_cache", segment.track_cache),
                ("detection_cache", segment.detection_cache),
            ]
        )
    return paths


def capture_input_snapshot(args: argparse.Namespace, segments: Sequence[Segment]) -> dict[str, dict[str, Any]]:
    snapshot = {}
    for kind, path in authoritative_inputs(args, segments):
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        row: dict[str, Any] = {"kind": kind, "path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if kind in {"manifest", "formal_csv"}:
            row["sha256"] = sha256_file(path)
        if kind in {"track_cache", "detection_cache"}:
            header = read_cache_header(path)
            footer = read_cache_footer(path)
            row.update(
                {
                    "cache_id": header["cache_id"],
                    "upstream_cache_id": header.get("upstream_detection_cache_id", ""),
                    "cache_complete": bool(footer.get("complete")),
                    "cache_frames": int(footer.get("frames_written", -1)),
                }
            )
        if kind == "source_video":
            segment = next(segment for segment in segments if segment.source == path)
            track_header = read_cache_header(segment.track_cache, expected_kind="tracks")
            row["sha256"] = track_header["video"].get("sha256", "")
            row["sha256_source"] = "reused_track_cache_header"
        snapshot[str(path)] = row
    return snapshot


def establish_input_snapshot(output: Path, current: dict[str, dict[str, Any]]) -> None:
    path = output / "input_snapshot.json"
    if path.is_file():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        if recorded != current:
            differing = sorted(set(recorded) | set(current))
            differing = [key for key in differing if recorded.get(key) != current.get(key)]
            raise ValueError(f"Authoritative inputs differ from the established snapshot: {differing}")
    else:
        atomic_write_text(path, json.dumps(current, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def input_generation_rows(snapshot: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in snapshot.values():
        rows.append(
            {
                "record_type": item["kind"],
                "case_id": "",
                "segment_index": "",
                "path": item["path"],
                "size_bytes": item["size_bytes"],
                "mtime_ns": item["mtime_ns"],
                "sha256": item.get("sha256", ""),
                "cache_id": item.get("cache_id", ""),
                "upstream_cache_id": item.get("upstream_cache_id", ""),
                "cache_complete": item.get("cache_complete", ""),
                "cache_frames": item.get("cache_frames", ""),
                "codec": "",
                "pixel_format": "",
                "width": "",
                "height": "",
                "fps": "",
                "frame_count": "",
                "duration_seconds": "",
                "validation_status": "unchanged",
                "detail": item.get("sha256_source", ""),
            }
        )
    return rows


def write_contact_sheet(output: Path, manifest_rows: Sequence[dict[str, Any]]) -> None:
    cards = []
    output_root = output.resolve()
    for row in sorted(
        [row for row in manifest_rows if row["render_selected"] == "yes"],
        key=lambda item: int(item["priority_rank_global"]),
    ):
        try:
            thumbnail_uri = Path(str(row["thumbnail_path"])).resolve().relative_to(output_root).as_posix()
            clip_uri = Path(str(row["clip_path"])).resolve().relative_to(output_root).as_posix()
        except ValueError as exc:
            raise ValueError(f"contact-sheet asset is outside output directory: {row['case_id']}") from exc
        cards.append(
            "<article><a href='{clip}'><img src='{thumb}' alt='{case}'></a>"
            "<h3>{case}</h3><p>S{segment:03d} · {time} · score {score}</p>"
            "<p><code>{flags}</code></p><p>{reason}</p></article>".format(
                clip=html.escape(clip_uri),
                thumb=html.escape(thumbnail_uri),
                case=html.escape(str(row["case_id"])),
                segment=int(row["segment_index"]),
                time=html.escape(str(row["anchor_timestamp"])),
                score=html.escape(str(row["priority_score"])),
                flags=html.escape(str(row["candidate_flags"])),
                reason=html.escape(str(row["priority_reason"])),
            )
        )
    content = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>LV006 error-audit contact sheet</title><style>
body{font-family:sans-serif;background:#171717;color:#eee;margin:24px}main{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}
article{background:#252525;padding:12px;border-radius:8px}img{width:100%;height:auto;border-radius:5px}h3{margin:.6em 0 .2em}p{font-size:13px;line-height:1.35;color:#ccc}code{color:#ffd16b}
</style></head><body><h1>LV006 BoT-SORT error-audit cases</h1><p>Click a thumbnail to open the source-frame case clip.</p><main>""" + "\n".join(cards) + "</main></body></html>\n"
    atomic_write_text(output / "contact_sheet.html", content)


def case_from_manifest(row: dict[str, str]) -> Case:
    return Case(
        case_id=row["case_id"],
        segment_index=int(row["segment_index"]),
        segment_group=row["segment_group"],
        start_frame=int(row["start_frame"]),
        anchor_frame=int(row["anchor_frame"]),
        end_frame=int(row["end_frame"]),
        track_ids=[int(value) for value in row.get("track_ids", "").split(";") if value],
        candidate_flags=[value for value in row.get("candidate_flags", "").split(";") if value],
        priority_score=float(row["priority_score"]),
        priority_reason=row.get("priority_reason", ""),
        formal_delta_D=int(row["formal_delta_D"]),
        formal_delta_C=int(row["formal_delta_C"]),
        track_history=row.get("track_history", ""),
        priority_rank_global=int(row.get("priority_rank_global", 0) or 0),
        priority_rank_segment=int(row.get("priority_rank_segment", 0) or 0),
        render_selected=row.get("render_selected") == "yes",
        render_batch=row.get("render_batch", ""),
        render_status=row.get("render_status", ""),
    )


def verify_workspace(args: argparse.Namespace, logger: AuditLogger, require_clips: bool = True) -> list[dict[str, Any]]:
    config_path = args.output / "generation_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    segments = load_segments(args)
    events = load_formal_events(args, segments)
    current_snapshot = capture_input_snapshot(args, segments)
    establish_input_snapshot(args.output, current_snapshot)
    manifest_path = args.output / "case_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    rows = read_csv(manifest_path)
    ids = [row["case_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("case_id values are not unique")
    event_map = {event["event_key"]: event for event in events}
    assigned_events: set[str] = set()
    segments_by_index = {segment.index: segment for segment in segments}
    selected_rows = [row for row in rows if row["render_selected"] == "yes"]
    if len(selected_rows) > 100:
        raise ValueError(f"Selected first-round case count exceeds 100: {len(selected_rows)}")
    output_records = []
    frame_pts_by_segment: dict[int, list[float]] = {}
    for row in rows:
        case = case_from_manifest(row)
        segment = segments_by_index[case.segment_index]
        if case.segment_group != segment.group:
            raise ValueError(f"Incorrect diagnostic group for {case.case_id}")
        if not (1 <= case.start_frame <= case.anchor_frame <= case.end_frame <= segment.frame_count):
            raise ValueError(f"Illegal frame range for {case.case_id}")
        keys = [value for value in row.get("formal_event_rows", "").split(";") if value]
        window_keys = [value for value in row.get("window_formal_event_rows", "").split(";") if value]
        if assigned_events.intersection(keys):
            raise ValueError(f"Formal event ownership is duplicated at {case.case_id}")
        assigned_events.update(keys)
        try:
            owned = [event_map[key] for key in keys]
            window_events = [event_map[key] for key in window_keys]
        except KeyError as exc:
            raise ValueError(f"Unknown formal event key in {case.case_id}: {exc}") from exc
        delta_d, delta_c = compute_formal_delta(owned)
        case.formal_events = owned
        case.display_events = window_events
        expected_window_keys = {
            event["event_key"]
            for event in events
            if int(event["segment_index"]) == case.segment_index
            and case.start_frame <= int(event["frame"]) <= case.end_frame
        }
        if set(window_keys) != expected_window_keys:
            raise ValueError(f"Window formal-event list is incomplete for {case.case_id}")
        if not set(keys).issubset(set(window_keys)):
            raise ValueError(f"Assigned event is not present in the case window for {case.case_id}")
        if (delta_d, delta_c) != (case.formal_delta_D, case.formal_delta_C):
            raise ValueError(f"Formal delta does not match assigned event rows for {case.case_id}")
        if not case.render_selected:
            continue
        clip = Path(row["clip_path"])
        thumbnail = Path(row["thumbnail_path"])
        if not clip.is_absolute() or not thumbnail.is_absolute():
            raise ValueError(f"Selected output paths must be absolute: {case.case_id}")
        if not require_clips:
            continue
        if not clip.is_file() or not thumbnail.is_file():
            raise FileNotFoundError(f"Missing clip or thumbnail for {case.case_id}")
        sidecar = clip.with_suffix(".render.json")
        if not sidecar.is_file():
            raise FileNotFoundError(f"Missing render signature sidecar for {case.case_id}")
        sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if sidecar_payload.get("render_signature") not in acceptable_render_signatures(case, segment, config):
            raise ValueError(f"Render signature mismatch for {case.case_id}")
        if sidecar_payload.get("frame_seek_method") != FRAME_SEEK_METHOD:
            raise ValueError(f"Frame seek method mismatch for {case.case_id}")
        if segment.index not in frame_pts_by_segment:
            frame_pts_by_segment[segment.index] = build_source_frame_pts(
                segment.source, segment.frame_count, args.ffprobe
            )
        frame_pts = frame_pts_by_segment[segment.index]
        expected_anchor_pts = float(frame_pts[case.anchor_frame - 1])
        if not math.isclose(
            float(sidecar_payload.get("anchor_source_pts", float("nan"))),
            expected_anchor_pts,
            abs_tol=1e-9,
            rel_tol=0.0,
        ):
            raise ValueError(f"Anchor source PTS mismatch for {case.case_id}")
        anchor_source = next(scheduled_source_frames(segment.source, [case.anchor_frame], frame_pts))[1]
        if sidecar_payload.get("anchor_source_sha256") != source_frame_sha256(anchor_source):
            raise ValueError(f"Anchor source frame hash mismatch for {case.case_id}")
        expected_frames = expected_clip_frames(case, segment, config)
        metadata = validate_clip(clip, expected_frames, config, args.ffprobe, args.ffmpeg, logger, require_full_decode=True)
        stat = clip.stat()
        output_records.append(
            {
                "record_type": "clip",
                "case_id": case.case_id,
                "segment_index": case.segment_index,
                "path": str(clip),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256_file(clip),
                "cache_id": "",
                "upstream_cache_id": "",
                "cache_complete": "",
                "cache_frames": "",
                **metadata,
                "validation_status": "verified",
                "detail": (
                    f"thumbnail={thumbnail}; anchor_pts={expected_anchor_pts:.6f}; "
                    f"seek_method={FRAME_SEEK_METHOD}; anchor_source_hash=verified"
                ),
            }
        )
    if require_clips:
        generation_path = args.output / "generation_manifest.csv"
        if not generation_path.is_file():
            raise FileNotFoundError(generation_path)
        recorded_clips = {
            row["case_id"] for row in read_csv(generation_path) if row["record_type"] == "clip"
        }
        selected_ids = {row["case_id"] for row in selected_rows}
        if recorded_clips != selected_ids:
            raise ValueError("generation_manifest clip rows are not one-to-one with selected cases")
        case_directory_clips = {str(path.resolve()) for path in (args.output / "cases").rglob("*.mp4")}
        selected_clip_paths = {str(Path(row["clip_path"]).resolve()) for row in selected_rows}
        if case_directory_clips != selected_clip_paths:
            raise ValueError("cases/ MP4 files are not one-to-one with selected manifest rows")
    review_rows = read_csv(args.output / "review_sheet.csv")
    if {row["case_id"] for row in review_rows} != {row["case_id"] for row in selected_rows}:
        raise ValueError("review_sheet rows are not one-to-one with selected cases")
    temporary = [
        path
        for path in args.output.rglob("*")
        if path.is_file() and (path.name.startswith(".") and (".tmp" in path.name or ".partial" in path.name))
    ]
    if temporary:
        raise ValueError(f"Temporary output files remain: {temporary}")
    logger.write(
        f"verify: cases={len(rows)}, selected={len(selected_rows)}, assigned_formal_events={len(assigned_events)}, "
        f"clips_checked={len(output_records)}, inputs_unchanged=yes"
    )
    return output_records


def generation_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "renderer_version": RENDERER_VERSION,
        "frame_seek_method": FRAME_SEEK_METHOD,
        "video_id": VIDEO_ID,
        "directory_name": DIRECTORY_NAME,
        "circle_center": list(CENTER),
        "circle_radius": RADIUS,
        "class_mapping": {str(key): value for key, value in CLASS_NAMES.items()},
        "batch": "pilot" if args.pilot else "round1",
        "output_width": args.output_width,
        "output_height": args.output_height,
        "output_fps": args.output_fps,
        "context_before": args.context_before,
        "context_after": args.context_after,
        "anchor_hold": args.anchor_hold,
        "slow_source_seconds": args.slow_source_seconds,
        "slow_rate": args.slow_rate,
        "x264_crf": args.x264_crf,
        "x264_preset": args.x264_preset,
        "candidate_thresholds": {
            "ring_half_width_px": RING_HALF_WIDTH,
            "id_link_ring_half_width_px": LINK_RING_HALF_WIDTH,
            "id_link_max_gap_seconds": 1.0,
            "id_link_max_distance_px": 120.0,
            "event_cluster_gap_seconds": 2.0,
            "event_cluster_max_span_seconds": 4.0,
            "unmatched_detection_confidence": 0.40,
        },
    }


def generate(args: argparse.Namespace) -> None:
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    logger = AuditLogger(args.output / "generation.log")
    if not (args.output / "README.md").is_file():
        raise FileNotFoundError("The review protocol README.md must exist and is not generated by this script")
    for executable in (args.ffmpeg, args.ffprobe):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise FileNotFoundError(executable)
    started = time.monotonic()
    logger.write(f"generate started: batch={'pilot' if args.pilot else 'round1'}, discover_only={args.discover_only}")
    segments = load_segments(args)
    events = load_formal_events(args, segments)
    before = capture_input_snapshot(args, segments)
    establish_input_snapshot(args.output, before)
    config = generation_config(args)
    atomic_write_text(args.output / "generation_config.json", json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    cases, stats_rows = discover_cases(args, segments, events, logger)
    selected = rank_and_select_cases(cases, args.pilot, args.pilot_cases, args.max_cases)
    stats_columns = ["segment_index", "frames", "raw_tracks", "raw_seeds", "deduplicated_cases", "unmatched_detection_rows", "event_near_pairs"]
    atomic_write_csv(args.output / "candidate_summary.csv", stats_columns, stats_rows)
    logger.write(f"candidate discovery complete: raw={sum(int(row['raw_seeds']) for row in stats_rows)}, deduplicated={len(cases)}, selected={len(selected)}")

    if args.invalidate_existing_reviews:
        invalidated = invalidate_existing_reviews(
            args.output / "review_sheet.csv",
            args.output / "invalidated_reviews_before_renderer_v4.csv",
        )
        logger.write(f"archived and cleared {invalidated} reviews made from invalid pre-v4 clips")
    preserved_before = load_preserved_human_fields(args.output / "review_sheet.csv")
    manifest_rows = write_case_manifest(args.output, cases, segments)
    write_review_sheet(args.output, manifest_rows)
    preserved_after = load_preserved_human_fields(args.output / "review_sheet.csv")
    for case_id, fields in preserved_before.items():
        if case_id in preserved_after and preserved_after[case_id] != fields:
            raise ValueError(f"Human review fields changed during regeneration: {case_id}")

    output_rows: list[dict[str, Any]] = []
    if not args.discover_only:
        output_rows = render_selected_cases(args, segments, selected, config, logger)
        manifest_rows = write_case_manifest(args.output, cases, segments)
        write_review_sheet(args.output, manifest_rows)
        write_contact_sheet(args.output, manifest_rows)
    summarize_outputs(args.output, logger, fail_on_invalid=False)
    after = capture_input_snapshot(args, segments)
    if before != after:
        raise ValueError("Authoritative inputs changed during generation")
    atomic_write_csv(args.output / "generation_manifest.csv", GENERATION_COLUMNS, input_generation_rows(after) + output_rows)

    if not args.discover_only:
        verified_output_rows = verify_workspace(args, logger, require_clips=True)
        # Rebuild output rows from the independent verification pass.
        atomic_write_csv(args.output / "generation_manifest.csv", GENERATION_COLUMNS, input_generation_rows(after) + verified_output_rows)
        if args.pilot:
            pilot_rows = [
                {
                    "case_id": row["case_id"],
                    "source_frames_present": "pass",
                    "circle_and_labels_present": "pass",
                    "anchor_hold_encoded": "pass",
                    "slow_replay_encoded": "pass",
                    "formal_events_backfilled": "pass",
                    "ffmpeg_full_decode": "pass",
                    "opencv_seek_decode": "pass",
                    "duplicate_formal_event": "none",
                }
                for row in verified_output_rows
            ]
            atomic_write_csv(
                args.output / "pilot_validation.csv",
                [
                    "case_id",
                    "source_frames_present",
                    "circle_and_labels_present",
                    "anchor_hold_encoded",
                    "slow_replay_encoded",
                    "formal_events_backfilled",
                    "ffmpeg_full_decode",
                    "opencv_seek_decode",
                    "duplicate_formal_event",
                ],
                pilot_rows,
            )
    logger.write(f"generate complete in {time.monotonic() - started:.1f}s")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output = args.output.resolve()
    logger = AuditLogger(args.output / "generation.log")
    try:
        if args.command == "generate":
            generate(args)
        elif args.command == "summarize":
            summarize_outputs(args.output, logger, fail_on_invalid=True)
        elif args.command == "verify":
            verify_workspace(args, logger, require_clips=True)
        else:
            raise ValueError(f"Unknown command: {args.command}")
        return 0
    except Exception as exc:
        logger.write(f"FAILED: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())

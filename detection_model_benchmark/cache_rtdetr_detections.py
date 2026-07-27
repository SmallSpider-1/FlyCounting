#!/usr/bin/env python3
"""Generate manifest-ordered full-frame caches with the frozen RT-DETR-R18."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
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
    EXPECTED_CHECKPOINT_SHA256,
    RTDETRR18Backend,
    require_benchmark_environment,
    sha256_file,
)


DEFAULT_MANIFEST = ROOT / "video_data" / "manifests" / "segments.csv"
DEFAULT_VIDEO_ROOT = ROOT / "video_data"
DEFAULT_OUTPUT = ROOT / "project_results" / "detection_cache" / "rtdetr_r18_official_v1"
FORMAL_SEGMENTS = 25
FORMAL_LOGICAL_VIDEOS = 6
FORMAL_FRAMES = 10_007_059
FORMAL_WIDTH = 2304
FORMAL_HEIGHT = 1296
CACHE_CONFIDENCE = 0.10
DEFAULT_BATCH_SIZE = 16


@dataclass(frozen=True)
class Segment:
    video_id: str
    segment_index: int
    relative_path: Path
    frame_count: int
    width: int
    height: int
    fps_text: str
    size_bytes: int

    @property
    def key(self) -> str:
        return f"{self.video_id}:{self.segment_index}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen RT-DETR-R18 NMS-free FP32 cache generator. "
            "--iou is intentionally unsupported because RT-DETR does not run NMS."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--summary-name",
        default="detection_cache_manifest.csv",
        help="Summary filename inside --output; use distinct names for disjoint multi-GPU shards.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="VIDEO_ID:SEGMENT_INDEX",
        help="Engineering pilot only; repeat to select manifest entries. Omit for the formal 25-segment run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep only existing caches that pass full v2/frame/hash validation; otherwise refuse overwrite.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print manifest order without loading torch.")
    return parser


def parse_fps(value: str) -> float:
    numerator, denominator = value.split("/", 1)
    denominator_value = float(denominator)
    if denominator_value == 0:
        raise ValueError(f"Invalid manifest FPS: {value}")
    return float(numerator) / denominator_value


def load_manifest(path: Path, video_root: Path) -> list[Segment]:
    path = path.resolve()
    video_root = video_root.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as stream:
        raw_rows = list(csv.DictReader(stream))
    segments = [
        Segment(
            video_id=row["video_id"].strip(),
            segment_index=int(row["segment_index"]),
            relative_path=Path(row["relative_path"]),
            frame_count=int(row["frame_count"]),
            width=int(row["width"]),
            height=int(row["height"]),
            fps_text=row["avg_fps"].strip(),
            size_bytes=int(row["size_bytes"]),
        )
        for row in raw_rows
    ]
    segments.sort(key=lambda item: (item.video_id, item.segment_index))
    if len(segments) != FORMAL_SEGMENTS:
        raise ValueError(f"Formal manifest must contain {FORMAL_SEGMENTS} segments, got {len(segments)}")
    if len({item.video_id for item in segments}) != FORMAL_LOGICAL_VIDEOS:
        raise ValueError(f"Formal manifest must contain {FORMAL_LOGICAL_VIDEOS} logical videos")
    if len({item.key for item in segments}) != len(segments):
        raise ValueError("Manifest contains duplicate video_id + segment_index")
    if sum(item.frame_count for item in segments) != FORMAL_FRAMES:
        raise ValueError(
            f"Manifest total frames must be {FORMAL_FRAMES}, got {sum(item.frame_count for item in segments)}"
        )
    for video_id in sorted({item.video_id for item in segments}):
        indexes = [item.segment_index for item in segments if item.video_id == video_id]
        if indexes != list(range(1, len(indexes) + 1)):
            raise ValueError(f"Manifest segment indexes are not consecutive for {video_id}: {indexes}")
    for item in segments:
        if (item.width, item.height) != (FORMAL_WIDTH, FORMAL_HEIGHT):
            raise ValueError(f"Manifest resolution mismatch for {item.key}: {(item.width, item.height)}")
        source = (video_root / item.relative_path).resolve()
        try:
            source.relative_to(video_root)
        except ValueError as exc:
            raise ValueError(f"Manifest path escapes video root: {item.relative_path}") from exc
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.stat().st_size != item.size_bytes:
            raise ValueError(
                f"Manifest byte-size mismatch for {item.key}: expected {item.size_bytes}, "
                f"got {source.stat().st_size}"
            )
        parse_fps(item.fps_text)
    return segments


def select_segments(segments: list[Segment], only: list[str]) -> list[Segment]:
    if not only:
        return segments
    requested = set(only)
    known = {item.key for item in segments}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"Unknown --only segment keys: {unknown}")
    return [item for item in segments if item.key in requested]


def cache_path(output_root: Path, segment: Segment) -> Path:
    return output_root / segment.relative_path.parent / f"{segment.relative_path.stem}.detections.jsonl"


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    fieldnames = list(rows[0]) if rows else []
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def cache_hashes(path: Path) -> tuple[str, str]:
    full_digest = hashlib.sha256()
    payload_digest = hashlib.sha256()
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream):
            full_digest.update(line)
            if line_number > 0 and line.startswith(b'{"f":'):
                payload_digest.update(line)
    return full_digest.hexdigest(), payload_digest.hexdigest()


def video_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(path),
    }


def inspect_capture(path: Path) -> tuple[int, int, float, int]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    values = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        float(capture.get(cv2.CAP_PROP_FPS)),
        int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    capture.release()
    return values


def validate_existing(
    path: Path,
    segment: Segment,
    detector_metadata: dict[str, Any],
) -> dict[str, Any]:
    header = read_cache_header(path, "detections")
    if header.get("manifest", {}).get("video_id") != segment.video_id:
        raise ValueError(f"Existing cache video_id mismatch: {path}")
    if int(header.get("manifest", {}).get("segment_index", -1)) != segment.segment_index:
        raise ValueError(f"Existing cache segment_index mismatch: {path}")
    if header.get("producer", {}).get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"Existing cache checkpoint mismatch: {path}")
    if header.get("producer", {}).get("detector_config_sha256") != detector_metadata["detector_config_sha256"]:
        raise ValueError(f"Existing cache detector config mismatch: {path}")
    frames = sum(1 for _ in iter_cache_frames(path, "detections", require_complete=True))
    if frames != segment.frame_count:
        raise ValueError(f"Existing cache decoded-frame mismatch: expected {segment.frame_count}, got {frames}")
    file_hash, payload_hash = cache_hashes(path)
    return {
        "video_id": segment.video_id,
        "segment_index": segment.segment_index,
        "relative_path": str(segment.relative_path),
        "manifest_frames": segment.frame_count,
        "decoded_frames": frames,
        "total_detections": "",
        "class_0_detections": "",
        "class_1_detections": "",
        "video_sha256": header.get("video", {}).get("sha256", ""),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "cache_version": header["cache_version"],
        "cache_id": header["cache_id"],
        "numeric_payload_sha256": payload_hash,
        "cache_file_sha256": file_hash,
        "cache_bytes": path.stat().st_size,
        "bytes_per_frame": path.stat().st_size / frames,
        "inference_seconds": "",
        "wall_seconds": "",
        "decoded_fps": "",
        "cache_path": str(path),
        "status": "validated_existing",
    }


def process_segment(
    backend: RTDETRR18Backend,
    segment: Segment,
    video_root: Path,
    output_root: Path,
    batch_size: int,
    detector_metadata: dict[str, Any],
) -> dict[str, Any]:
    import cv2
    from PIL import Image

    source = (video_root / segment.relative_path).resolve()
    destination = cache_path(output_root, segment)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing cache: {destination}")

    width, height, container_fps, container_frame_count = inspect_capture(source)
    if (width, height) != (segment.width, segment.height):
        raise ValueError(
            f"Decoded metadata resolution mismatch for {segment.key}: "
            f"manifest={(segment.width, segment.height)}, capture={(width, height)}"
        )
    signature = video_signature(source)
    writer = NumericCacheWriter(
        destination,
        "detections",
        {
            "video": signature,
            "manifest": {
                "video_id": segment.video_id,
                "segment_index": segment.segment_index,
                "relative_path": str(segment.relative_path),
                "frame_count": segment.frame_count,
                "size_bytes": segment.size_bytes,
            },
            "frame_size": [segment.width, segment.height],
            "fps": round(parse_fps(segment.fps_text), 9),
            "reported_frame_count": segment.frame_count,
            "container_reported_frame_count_diagnostic_only": container_frame_count,
            "coordinate_space": "full_frame_xyxy",
            "class_names": {str(key): value for key, value in EXPECTED_CLASS_NAMES.items()},
            "producer": {"stage": "rtdetr_r18_video_detection_cache", **detector_metadata},
        },
    )

    batch_queue: queue.Queue[Any] = queue.Queue(maxsize=3)
    sentinel = object()
    stop_decode = threading.Event()
    decode_state: dict[str, Any] = {"frames": 0, "error": None}
    total_detections = 0
    per_class = {0: 0, 1: 0}
    inference_seconds = 0.0
    wall_start = time.perf_counter()

    def queue_item(value: Any) -> bool:
        while not stop_decode.is_set():
            try:
                batch_queue.put(value, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def decode_worker() -> None:
        capture = cv2.VideoCapture(str(source))
        pending: list[Any] = []
        first_pending_frame = 1
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open video: {source}")
            while not stop_decode.is_set():
                ok, frame = capture.read()
                if not ok:
                    break
                decode_state["frames"] += 1
                frame_index = int(decode_state["frames"])
                if frame.shape[:2] != (segment.height, segment.width):
                    raise ValueError(
                        f"Decoded frame size changed at {segment.key} frame {frame_index}: {frame.shape[:2]}"
                    )
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if not pending:
                    first_pending_frame = frame_index
                pending.append(Image.fromarray(rgb, mode="RGB"))
                if len(pending) == batch_size:
                    if not queue_item((first_pending_frame, pending)):
                        return
                    pending = []
            if pending and not stop_decode.is_set():
                if not queue_item((first_pending_frame, pending)):
                    return
        except BaseException as error:
            decode_state["error"] = error
        finally:
            capture.release()
            queue_item(sentinel)

    decode_thread = threading.Thread(
        target=decode_worker,
        name=f"decode-{segment.video_id}-{segment.segment_index}",
        daemon=True,
    )

    print(
        f"[{segment.key}] start {segment.relative_path} "
        f"manifest_frames={segment.frame_count} capture_count_diagnostic={container_frame_count}",
        flush=True,
    )
    decode_thread.start()
    try:
        while True:
            item = batch_queue.get()
            if item is sentinel:
                break
            first_frame, pending_images = item
            if first_frame != writer.frames_written + 1:
                raise RuntimeError(
                    f"Decode queue frame order mismatch: expected {writer.frames_written + 1}, got {first_frame}"
                )
            infer_start = time.perf_counter()
            batch_results = backend.predict_pil(pending_images, CACHE_CONFIDENCE)
            backend.torch.cuda.synchronize() if backend.device.type == "cuda" else None
            inference_seconds += time.perf_counter() - infer_start
            for offset, detections in enumerate(batch_results):
                frame_index = first_frame + offset
                writer.write_frame(frame_index, detections)
                total_detections += len(detections)
                for row in detections:
                    class_id = int(row[5])
                    if class_id not in per_class:
                        raise ValueError(f"Unexpected RT-DETR class ID {class_id} at frame {frame_index}")
                    per_class[class_id] += 1
            if writer.frames_written % 10_000 < len(batch_results):
                elapsed = time.perf_counter() - wall_start
                print(
                    f"[{segment.key}] written={writer.frames_written}/{segment.frame_count} "
                    f"wall_fps={writer.frames_written / elapsed:.2f}",
                    flush=True,
                )
        decode_thread.join()
        if decode_state["error"] is not None:
            raise decode_state["error"]
        decoded_frames = int(decode_state["frames"])
        if decoded_frames != segment.frame_count:
            raise ValueError(
                f"Actual decoded-frame count mismatch for {segment.key}: "
                f"manifest={segment.frame_count}, decoded={decoded_frames}; "
                "CAP_PROP_FRAME_COUNT is diagnostic only"
            )
        if writer.frames_written != decoded_frames:
            raise RuntimeError(
                f"Cache frame count mismatch before close: decoded={decoded_frames}, written={writer.frames_written}"
            )
        writer.close()
    except Exception:
        stop_decode.set()
        decode_thread.join()
        writer.abort()
        raise
    finally:
        stop_decode.set()

    wall_seconds = time.perf_counter() - wall_start
    file_hash, payload_hash = cache_hashes(destination)
    cache_bytes = destination.stat().st_size
    print(
        f"[{segment.key}] complete frames={decoded_frames} detections={total_detections} "
        f"wall_fps={decoded_frames / wall_seconds:.2f} bytes_per_frame={cache_bytes / decoded_frames:.2f}",
        flush=True,
    )
    return {
        "video_id": segment.video_id,
        "segment_index": segment.segment_index,
        "relative_path": str(segment.relative_path),
        "manifest_frames": segment.frame_count,
        "decoded_frames": decoded_frames,
        "total_detections": total_detections,
        "class_0_detections": per_class[0],
        "class_1_detections": per_class[1],
        "video_sha256": signature["sha256"],
        "checkpoint_sha256": backend.checkpoint_sha256,
        "cache_version": writer.header["cache_version"],
        "cache_id": writer.cache_id,
        "numeric_payload_sha256": payload_hash,
        "cache_file_sha256": file_hash,
        "cache_bytes": cache_bytes,
        "bytes_per_frame": cache_bytes / decoded_frames,
        "inference_seconds": inference_seconds,
        "wall_seconds": wall_seconds,
        "decoded_fps": decoded_frames / wall_seconds,
        "cache_path": str(destination),
        "status": "generated",
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.batch_size != DEFAULT_BATCH_SIZE:
        raise ValueError(
            f"--batch-size is frozen at {DEFAULT_BATCH_SIZE} to reproduce val_mixed numerics; got {args.batch_size}"
        )
    segments = load_manifest(args.manifest, args.video_root)
    selected = select_segments(segments, args.only)
    print("Manifest execution order:", flush=True)
    for item in selected:
        print(f"  {item.key} {item.relative_path} frames={item.frame_count}", flush=True)
    if args.dry_run:
        return

    require_benchmark_environment()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if Path(args.summary_name).name != args.summary_name or not args.summary_name.endswith(".csv"):
        raise ValueError("--summary-name must be a plain .csv filename")
    backend = RTDETRR18Backend(args.weights, args.runtime_config, args.device)
    detector_metadata = backend.metadata(args.batch_size, CACHE_CONFIDENCE)
    rows: list[dict[str, Any]] = []
    summary_path = output_root / args.summary_name
    for segment in selected:
        destination = cache_path(output_root, segment)
        if destination.exists() and args.resume:
            row = validate_existing(destination, segment, detector_metadata)
            print(f"[{segment.key}] validated existing cache {destination}", flush=True)
        else:
            row = process_segment(
                backend,
                segment,
                args.video_root.resolve(),
                output_root,
                args.batch_size,
                detector_metadata,
            )
        rows.append(row)
        atomic_write_csv(summary_path, rows)
    print(
        f"Completed caches={len(rows)} decoded_frames={sum(int(row['decoded_frames']) for row in rows)} "
        f"manifest={summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()

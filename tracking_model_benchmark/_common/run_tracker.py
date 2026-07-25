#!/usr/bin/env python3
"""Run one official tracker over one or more unified detection caches."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_common.numeric_cache import NumericCacheWriter, iter_cache_frames, read_cache_header
from tracking_model_benchmark._common.tracker_adapters import (
    TRACKER_PROJECTS,
    create_tracker_adapter,
    resolved_config,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified cached-detection runner for the eight official trackers.")
    parser.add_argument("--tracker", required=True, choices=TRACKER_PROJECTS)
    parser.add_argument(
        "--detections",
        type=Path,
        required=True,
        help="单个 .detections.jsonl，或递归包含检测缓存的目录。",
    )
    parser.add_argument("--output", type=Path, required=True, help="统一轨迹缓存输出目录。")
    parser.add_argument("--config", type=Path, default=None, help="可选 JSON 参数覆盖文件。")
    return parser.parse_args()


def resolve_detection_caches(source: Path) -> tuple[list[Path], Path]:
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(f"检测缓存来源不存在: {source}")
    if source.is_file():
        if not source.name.endswith(".detections.jsonl"):
            raise ValueError(f"检测缓存文件名必须以 .detections.jsonl 结尾: {source}")
        return [source], source.parent
    caches = sorted(source.rglob("*.detections.jsonl"))
    if not caches:
        raise FileNotFoundError(f"目录中没有检测缓存: {source}")
    return caches, source


def load_overrides(path: Path | None) -> dict:
    if path is None:
        return {}
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"跟踪器配置必须是 JSON 对象: {path}")
    return value


def output_cache_path(cache_path: Path, source_root: Path, output_root: Path) -> Path:
    relative_parent = cache_path.parent.relative_to(source_root)
    stem = cache_path.name[: -len(".detections.jsonl")]
    return output_root / relative_parent / f"{stem}.tracks.jsonl"


def run_cache(cache_path: Path, output_path: Path, tracker_name: str, config: dict) -> dict:
    header = read_cache_header(cache_path, "detections")
    if header.get("coordinate_space") != "full_frame_xyxy":
        raise ValueError(f"只接受 full_frame_xyxy 检测缓存: {cache_path}")
    detector_confidence = header.get("producer", {}).get("confidence", "")
    frame_size = tuple(map(int, header["frame_size"]))
    fps = float(header["fps"])
    adapter = create_tracker_adapter(tracker_name, frame_size, fps, config)
    writer = NumericCacheWriter(
        output_path,
        "tracks",
        {
            "video": header["video"],
            "frame_size": list(frame_size),
            "fps": fps,
            "reported_frame_count": int(header.get("reported_frame_count", 0)),
            "coordinate_space": "full_frame_xyxy",
            "class_names": header.get("class_names", {}),
            "upstream_detection_cache_id": header["cache_id"],
            "producer": {"stage": "unified_tracker_benchmark", **adapter.cache_metadata()},
        },
    )
    frame_count = 0
    output_track_rows = 0
    tracking_seconds = 0.0
    wall_start = time.perf_counter()
    try:
        for frame_index, rows in iter_cache_frames(cache_path, "detections"):
            detections = np.asarray(rows, dtype=np.float32)
            if detections.size == 0:
                detections = np.empty((0, 6), dtype=np.float32)
            update_start = time.perf_counter()
            tracks = adapter.update(detections, frame_index)
            tracking_seconds += time.perf_counter() - update_start
            writer.write_frame(frame_index, tracks.tolist())
            frame_count += 1
            output_track_rows += len(tracks)
        writer.close()
    except Exception:
        writer.abort()
        raise
    wall_seconds = time.perf_counter() - wall_start
    return {
        "tracker": tracker_name,
        "detection_cache": str(cache_path),
        "detection_cache_id": header["cache_id"],
        "track_cache": str(output_path),
        "track_cache_id": writer.cache_id,
        "frames": frame_count,
        "track_rows": output_track_rows,
        "tracking_seconds": round(tracking_seconds, 6),
        "tracking_fps": round(frame_count / tracking_seconds, 6) if tracking_seconds > 0 else "",
        "wall_seconds": round(wall_seconds, 6),
        "detector_confidence": detector_confidence,
        "tracking_space": "full_frame",
    }


def write_summary(path: Path, rows: list[dict]):
    fieldnames = list(rows[0]) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    caches, source_root = resolve_detection_caches(args.detections)
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = resolved_config(args.tracker, load_overrides(args.config))
    rows = []
    for cache_path in caches:
        destination = output_cache_path(cache_path, source_root, output_root)
        result = run_cache(cache_path, destination, args.tracker, config)
        rows.append(result)
        print(
            f"[{args.tracker}] {cache_path.name}: frames={result['frames']}, "
            f"tracking_fps={result['tracking_fps']}, output={destination}"
        )
    write_summary(output_root / "tracking_run.csv", rows)
    with open(output_root / "resolved_tracker_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"完成 {args.tracker}: caches={len(caches)}, summary={output_root / 'tracking_run.csv'}")


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_common.numeric_cache import iter_cache_frames, read_cache_header, stable_hash
from counting_model_benchmark.counting_core import (
    REFERENCE_FRAME_SIZE,
    apply_signed_count_delta,
    circle_transition,
    point_circle_state,
    scale_count_region,
)


COUNTING_ROOT = Path(__file__).resolve().parent
DEFAULT_COUNTING_CONFIG = COUNTING_ROOT / "configs" / "baseline_circle_v1.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Replay unified counting directly from numeric track caches.")
    parser.add_argument("--tracks", type=Path, required=True, help="单个 .tracks.jsonl 文件或包含这些文件的目录。")
    parser.add_argument("--output", type=Path, required=True, help="计数重放结果目录。")
    parser.add_argument("--config", type=Path, default=DEFAULT_COUNTING_CONFIG, help="版本化计数配置 JSON。")
    parser.add_argument("--center-x", type=float, default=None, help="覆盖配置中的参考圆心 x。")
    parser.add_argument("--center-y", type=float, default=None, help="覆盖配置中的参考圆心 y。")
    parser.add_argument("--radius", type=float, default=None, help="覆盖配置中的参考半径。")
    parser.add_argument(
        "--initial-count",
        action="append",
        default=[],
        metavar="CLASS_ID=COUNT",
        help="可重复指定初始瓶内数量，例如 --initial-count 0=2。",
    )
    return parser.parse_args()


def load_counting_config(path):
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "schema_version",
        "name",
        "reference_frame_size",
        "reference_center",
        "reference_radius",
        "count_point",
        "enter_transition",
        "enter_delta",
        "exit_transition",
        "exit_delta",
        "first_observation_creates_event",
        "clamp_negative_counts",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"计数配置缺少字段: {sorted(missing)}")
    supported = {
        "schema_version": 1,
        "reference_frame_size": list(REFERENCE_FRAME_SIZE),
        "count_point": "bbox_center",
        "enter_transition": ["outside", "inside"],
        "enter_delta": 1,
        "exit_transition": ["inside", "outside"],
        "exit_delta": -1,
        "first_observation_creates_event": False,
        "clamp_negative_counts": False,
    }
    for key, expected in supported.items():
        if config[key] != expected:
            raise ValueError(f"当前计数器尚不支持 {key}={config[key]!r}，期望 {expected!r}")
    center = config["reference_center"]
    if not isinstance(center, list) or len(center) != 2:
        raise ValueError("reference_center 必须是 [x,y]。")
    if float(config["reference_radius"]) <= 0:
        raise ValueError("reference_radius 必须大于 0。")
    return config


def resolve_track_caches(source):
    if not source.exists():
        raise FileNotFoundError(f"轨迹缓存来源不存在: {source}")
    if source.is_file():
        return [source]
    caches = sorted(source.glob("*.tracks.jsonl"))
    if not caches:
        raise FileNotFoundError(f"目录中没有 .tracks.jsonl: {source}")
    return caches


def parse_initial_counts(values):
    counts = defaultdict(int)
    for value in values:
        try:
            class_text, count_text = value.split("=", 1)
            counts[int(class_text)] = int(count_text)
        except ValueError as exc:
            raise ValueError(f"无效 --initial-count: {value}，应为 CLASS_ID=COUNT") from exc
    return counts


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def replay_cache(cache_path, reference_center, reference_radius, region_counts, event_rows, segment_rows):
    header = read_cache_header(cache_path, "tracks")
    if header.get("coordinate_space") != "full_frame_xyxy":
        raise ValueError(f"计数重放只接受 full_frame_xyxy 坐标: {cache_path}")
    frame_size = tuple(header["frame_size"])
    center, radius = scale_count_region(frame_size, reference_center, reference_radius)
    class_names = {int(key): str(value) for key, value in header.get("class_names", {}).items()}
    track_states = {}
    track_classes = {}
    start_counts = dict(region_counts)
    frames = 0
    enters = 0
    exits = 0

    for frame_index, rows in iter_cache_frames(cache_path, "tracks"):
        frames += 1
        for row in rows:
            if len(row) < 7:
                raise ValueError(f"轨迹缓存行列数不足: {cache_path}, frame={frame_index}")
            x1, y1, x2, y2 = map(float, row[:4])
            track_id = int(row[4])
            confidence = float(row[5])
            class_id = int(row[6])
            center_point = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            distance = float(np.hypot(center_point[0] - center[0], center_point[1] - center[1]))
            current_state = point_circle_state(center_point, center, radius)
            direction, delta = circle_transition(track_states.get(track_id), current_state)
            if delta:
                count_class_id = track_classes.get(track_id, class_id) if delta < 0 else class_id
                count_after = apply_signed_count_delta(region_counts, count_class_id, delta)
                enters += int(delta > 0)
                exits += int(delta < 0)
                event_rows.append(
                    {
                        "cache": cache_path.name,
                        "video": Path(header["video"]["path"]).name,
                        "frame": frame_index,
                        "track_id": track_id,
                        "class_id": count_class_id,
                        "class_name": class_names.get(count_class_id, str(count_class_id)),
                        "direction": direction,
                        "delta": delta,
                        "count_after": count_after,
                        "center_x": round(center_point[0], 3),
                        "center_y": round(center_point[1], 3),
                        "radial_distance": round(distance, 3),
                        "confidence": round(confidence, 6),
                    }
                )
            track_states[track_id] = current_state
            track_classes[track_id] = class_id

    for class_id in sorted(set(class_names) | set(region_counts) | set(start_counts)):
        segment_rows.append(
            {
                "cache": cache_path.name,
                "video": Path(header["video"]["path"]).name,
                "class_id": class_id,
                "class_name": class_names.get(class_id, str(class_id)),
                "start_count": start_counts.get(class_id, 0),
                "end_count": region_counts[class_id],
                "frames": frames,
                "circle_center_x": center[0],
                "circle_center_y": center[1],
                "circle_radius": radius,
                "enter_events": enters,
                "exit_events": exits,
                "track_cache_id": header["cache_id"],
                "upstream_detection_cache_id": header.get("upstream_detection_cache_id", ""),
            }
        )
    return class_names


def main():
    args = parse_args()
    config = load_counting_config(args.config)
    center_x = float(config["reference_center"][0]) if args.center_x is None else args.center_x
    center_y = float(config["reference_center"][1]) if args.center_y is None else args.center_y
    radius = float(config["reference_radius"]) if args.radius is None else args.radius
    if radius <= 0:
        raise ValueError("--radius 必须大于 0。")
    caches = resolve_track_caches(args.tracks)
    args.output.mkdir(parents=True, exist_ok=True)
    region_counts = parse_initial_counts(args.initial_count)
    initial_counts = dict(region_counts)
    event_rows = []
    segment_rows = []
    all_class_names = {}
    reference_center = (center_x, center_y)
    resolved_config = {
        **config,
        "reference_center": [center_x, center_y],
        "reference_radius": radius,
    }
    counting_config_id = stable_hash(resolved_config)

    for cache_path in caches:
        cache_class_names = replay_cache(
            cache_path,
            reference_center,
            radius,
            region_counts,
            event_rows,
            segment_rows,
        )
        for class_id, class_name in cache_class_names.items():
            previous_name = all_class_names.get(class_id)
            if previous_name is not None and previous_name != class_name:
                raise ValueError(
                    f"轨迹缓存类别映射不一致: class_id={class_id}, {previous_name!r} != {class_name!r}"
                )
            all_class_names[class_id] = class_name

    write_csv(
        args.output / "count_events.csv",
        [
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
        ],
        event_rows,
    )
    write_csv(
        args.output / "segment_summary.csv",
        [
            "cache",
            "video",
            "class_id",
            "class_name",
            "start_count",
            "end_count",
            "frames",
            "circle_center_x",
            "circle_center_y",
            "circle_radius",
            "enter_events",
            "exit_events",
            "track_cache_id",
            "upstream_detection_cache_id",
        ],
        segment_rows,
    )
    class_ids = sorted(set(all_class_names) | set(region_counts) | set(initial_counts))
    write_csv(
        args.output / "final_counts.csv",
        ["class_id", "class_name", "final_region_count"],
        [
            {
                "class_id": class_id,
                "class_name": all_class_names.get(class_id, str(class_id)),
                "final_region_count": region_counts[class_id],
            }
            for class_id in class_ids
        ],
    )
    write_csv(
        args.output / "replay_config.csv",
        [
            "track_caches",
            "counting_config",
            "counting_config_id",
            "reference_center_x",
            "reference_center_y",
            "reference_radius",
            "initial_counts",
        ],
        [
            {
                "track_caches": "|".join(str(path.resolve()) for path in caches),
                "counting_config": str(args.config.resolve()),
                "counting_config_id": counting_config_id,
                "reference_center_x": center_x,
                "reference_center_y": center_y,
                "reference_radius": radius,
                "initial_counts": "|".join(f"{key}={value}" for key, value in sorted(initial_counts.items())),
            }
        ],
    )
    print(f"计数重放完成：{len(caches)} 个轨迹缓存，{len(event_rows)} 个跨界事件，输出 {args.output}")


if __name__ == "__main__":
    main()

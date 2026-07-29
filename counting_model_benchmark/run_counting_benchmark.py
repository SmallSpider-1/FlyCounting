#!/usr/bin/env python3
"""Run unified tracking and counting for any or all of the eight tracker candidates."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import shutil
import subprocess
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BENCHMARK_ROOT = ROOT / "tracking_model_benchmark"
DEFAULT_COUNTING_CONFIG = ROOT / "counting_model_benchmark" / "configs" / "baseline_circle_v1.json"

from benchmark_common.numeric_cache import iter_cache_frames, read_cache_header, stable_hash
from counting_model_benchmark.replay_counting import OUTPUT_FILENAMES, load_counting_config
from tracking_model_benchmark._common.run_tracker import (
    load_overrides,
    output_cache_path,
    resolve_detection_caches,
)
from tracking_model_benchmark._common.tracker_adapters import (
    TRACKER_COMMITS,
    resolved_config,
    resolve_sfsort_timeout,
)


TRACKERS = (
    "sort",
    "bytetrack",
    "ocsort",
    "sfsort",
    "fasttracker",
    "boosttrack",
    "hybridsort",
    "botsort",
)
PROJECTS = dict(
    zip(
        TRACKERS,
        (
            "01_sort",
            "02_bytetrack",
            "03_ocsort",
            "04_sfsort",
            "05_fasttracker",
            "06_boosttrack",
            "07_hybridsort",
            "08_botsort",
        ),
    )
)
MANIFEST_NAME = "benchmark_run_manifest.csv"
MANIFEST_FIELDS = [
    "run_id",
    "run_started_at",
    "tracker",
    "config",
    "detection_source",
    "track_root",
    "count_root",
    "logical_groups",
    "counting_enabled",
    "counting_config",
    "counting_config_id",
    "reference_center_x",
    "reference_center_y",
    "reference_radius",
    "center_x_override",
    "center_y_override",
    "radius_override",
    "initial_count_policy",
    "initial_counts",
    "tracks_reused",
    "track_run_id",
]
DETECTION_CONTEXT_FIELDS = ("detection_source",)
COUNTING_CONTEXT_FIELDS = (
    "counting_config",
    "counting_config_id",
    "reference_center_x",
    "reference_center_y",
    "reference_radius",
    "initial_count_policy",
    "initial_counts",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Eight-tracker unified cached counting benchmark.")
    parser.add_argument("--detections", type=Path, required=True, help="统一检测缓存文件或根目录。")
    parser.add_argument("--output", type=Path, required=True, help="八算法轨迹和计数结果根目录。")
    parser.add_argument(
        "--trackers",
        default="all",
        help="all 或逗号分隔名称：sort,bytetrack,ocsort,sfsort,fasttracker,boosttrack,hybridsort,botsort。",
    )
    parser.add_argument("--skip-counting", action="store_true", help="只生成统一轨迹缓存。")
    parser.add_argument("--counting-config", type=Path, default=DEFAULT_COUNTING_CONFIG)
    parser.add_argument("--center-x", type=float, default=None, help="可选的配置覆盖。")
    parser.add_argument("--center-y", type=float, default=None, help="可选的配置覆盖。")
    parser.add_argument("--radius", type=float, default=None, help="可选的配置覆盖。")
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--overwrite",
        action="store_true",
        help="复用严格校验通过的轨迹缓存，只重写本次选中 tracker 的计数 CSV。",
    )
    action_group.add_argument(
        "--purge",
        action="store_true",
        help="显式删除本次选中的 tracker 整个目录，并从跟踪阶段全量重跑。",
    )
    return parser.parse_args()


def selected_trackers(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(TRACKERS)
    selected = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = set(selected) - set(TRACKERS)
    if unknown:
        raise ValueError(f"未知跟踪器: {sorted(unknown)}")
    if not selected:
        raise ValueError("--trackers 不能为空。")
    duplicates = sorted({tracker for tracker in selected if selected.count(tracker) > 1})
    if duplicates:
        raise ValueError(f"--trackers 包含重复项，会导致同一次运行内覆盖: {duplicates}")
    return selected


def count_groups(track_root: Path) -> list[Path]:
    return sorted({path.parent for path in track_root.rglob("*.tracks.jsonl")})


def run_checked(command: list[str]):
    print("执行:", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def manifest_value(value) -> str:
    return "" if value is None else str(value)


def make_batch_identity() -> tuple[str, str]:
    now = datetime.now().astimezone()
    run_id = f"{now.strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}"
    return run_id, now.isoformat(timespec="microseconds")


def resolve_counting_context(
    detection_source: Path,
    counting_config: Path,
    center_x: float | None,
    center_y: float | None,
    radius: float | None,
) -> dict:
    config = load_counting_config(counting_config)
    resolved_center_x = float(config["reference_center"][0]) if center_x is None else float(center_x)
    resolved_center_y = float(config["reference_center"][1]) if center_y is None else float(center_y)
    resolved_radius = float(config["reference_radius"]) if radius is None else float(radius)
    if resolved_radius <= 0:
        raise ValueError("--radius 必须大于 0。")
    resolved_counting_config = {
        **config,
        "reference_center": [resolved_center_x, resolved_center_y],
        "reference_radius": resolved_radius,
    }
    return {
        "detection_source": str(detection_source.resolve()),
        "counting_config": str(counting_config.resolve()),
        "counting_config_id": stable_hash(resolved_counting_config),
        "reference_center_x": resolved_center_x,
        "reference_center_y": resolved_center_y,
        "reference_radius": resolved_radius,
        "initial_count_policy": "fixed_zero_per_class",
        "initial_counts": "all_classes=0",
    }


@contextmanager
def manifest_lock(path: Path):
    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_manifest_unlocked(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != MANIFEST_FIELDS:
            raise ValueError(
                f"运行清单字段结构不兼容，拒绝猜测或迁移: {path}\n"
                f"期望={MANIFEST_FIELDS}\n实际={reader.fieldnames}\n"
                "请使用新的 --output 目录。"
            )
        rows = list(reader)
    seen_trackers = {}
    seen_count_roots = {}
    for line_number, row in enumerate(rows, 2):
        tracker = row["tracker"]
        count_root = row["count_root"]
        if tracker in seen_trackers:
            raise ValueError(
                f"运行清单已有重复 tracker={tracker!r}: "
                f"第 {seen_trackers[tracker]} 行和第 {line_number} 行"
            )
        if count_root in seen_count_roots:
            raise ValueError(
                f"运行清单已有重复 count_root={count_root!r}: "
                f"{seen_count_roots[count_root]!r}, {tracker!r}"
            )
        seen_trackers[tracker] = line_number
        seen_count_roots[count_root] = tracker
    return rows


def validate_manifest_context(rows: list[dict], context: dict, fields: tuple[str, ...]) -> None:
    mismatches = []
    for row in rows:
        for field in fields:
            expected = manifest_value(context[field])
            actual = row.get(field, "")
            if actual != expected:
                mismatches.append(
                    f"tracker={row['tracker']}, field={field}, manifest={actual!r}, current={expected!r}"
                )
    if mismatches:
        details = "\n".join(f"  - {item}" for item in mismatches)
        raise ValueError(
            "本次运行条件与现有 benchmark_run_manifest.csv 不一致，结果不可放在同一评价名单:\n"
            f"{details}\n请使用新的 --output 目录。"
        )


def atomic_write_manifest(path: Path, rows: list[dict]) -> None:
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temp_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def inspect_manifest(path: Path, context: dict, include_counting_context: bool) -> list[dict]:
    with manifest_lock(path):
        rows = read_manifest_unlocked(path)
        validate_manifest_context(rows, context, DETECTION_CONTEXT_FIELDS)
        if include_counting_context:
            validate_manifest_context(rows, context, COUNTING_CONTEXT_FIELDS)
        return rows


def remove_manifest_tracker(
    path: Path,
    tracker: str,
    context: dict,
    include_counting_context: bool,
) -> dict | None:
    with manifest_lock(path):
        rows = read_manifest_unlocked(path)
        fields = DETECTION_CONTEXT_FIELDS
        if include_counting_context:
            fields += COUNTING_CONTEXT_FIELDS
        validate_manifest_context(rows, context, fields)
        previous = next((row for row in rows if row["tracker"] == tracker), None)
        if previous is not None:
            atomic_write_manifest(path, [row for row in rows if row["tracker"] != tracker])
        return previous


def append_manifest_row(path: Path, row: dict, context: dict) -> None:
    with manifest_lock(path):
        rows = read_manifest_unlocked(path)
        validate_manifest_context(rows, context, DETECTION_CONTEXT_FIELDS + COUNTING_CONTEXT_FIELDS)
        if any(existing["tracker"] == row["tracker"] for existing in rows):
            raise ValueError(
                f"运行清单已存在 tracker={row['tracker']!r}；"
                "默认不允许重复，重跑请显式使用 --overwrite 或 --purge。"
            )
        rows.append({field: manifest_value(row.get(field)) for field in MANIFEST_FIELDS})
        atomic_write_manifest(path, rows)


def plan_tracker_actions(
    output_root: Path,
    trackers: list[str],
    manifest_rows: list[dict],
    overwrite: bool,
    purge: bool,
) -> dict[str, str]:
    manifest_trackers = {row["tracker"] for row in manifest_rows}
    actions = {}
    conflicts = []
    for tracker in trackers:
        tracker_root = output_root / tracker
        root_exists = tracker_root.exists() or tracker_root.is_symlink()
        listed = tracker in manifest_trackers
        if root_exists:
            if purge:
                actions[tracker] = "purge"
            elif overwrite:
                actions[tracker] = "reuse"
            else:
                conflicts.append(f"{tracker}: 目标目录已存在 {tracker_root}")
        elif listed:
            if purge:
                actions[tracker] = "purge"
            elif overwrite:
                conflicts.append(
                    f"{tracker}: manifest 有记录但轨迹目录不存在，--overwrite 无法复用；"
                    "请改用 --purge 或新的输出目录"
                )
            else:
                conflicts.append(f"{tracker}: manifest 已有同名算法，但目标目录不存在")
        else:
            actions[tracker] = "run"
    if conflicts:
        details = "\n".join(f"  - {item}" for item in conflicts)
        raise FileExistsError(
            f"本次批处理输出冲突，拒绝混写或覆盖:\n{details}\n"
            "请选择新的 --output；复用轨迹重算计数用 --overwrite；全清重跑用 --purge。"
        )
    return actions


def expected_runtime_tracker_config(tracker: str, config: dict, detection_header: dict) -> dict:
    runtime = dict(config)
    runtime.pop("metadata_match_iou", None)
    if tracker == "sfsort":
        width, height = map(int, detection_header["frame_size"])
        fps = float(detection_header["fps"])
        horizontal_ratio = float(runtime.pop("horizontal_margin_ratio"))
        vertical_ratio = float(runtime.pop("vertical_margin_ratio"))
        runtime.update(
            frame_width=width,
            frame_height=height,
            marginal_timeout=resolve_sfsort_timeout(runtime["marginal_timeout"], fps, "marginal_timeout"),
            central_timeout=resolve_sfsort_timeout(runtime["central_timeout"], fps, "central_timeout"),
            horizontal_margin=int(width * horizontal_ratio),
            vertical_margin=int(height * vertical_ratio),
        )
    return runtime


def reuse_error(tracker: str, message: str) -> ValueError:
    return ValueError(
        f"{tracker} 现有轨迹缓存不能安全复用: {message}\n"
        "未改动现有结果；请使用新的 --output，或显式使用 --purge 全量重跑。"
    )


def validate_reusable_tracks(
    tracker: str,
    detection_source: Path,
    track_root: Path,
    config_path: Path,
) -> int:
    if not track_root.is_dir():
        raise reuse_error(tracker, f"轨迹目录不存在: {track_root}")
    resolved_config_path = track_root / "resolved_tracker_config.json"
    tracking_run_path = track_root / "tracking_run.csv"
    if not resolved_config_path.is_file() or not tracking_run_path.is_file():
        raise reuse_error(
            tracker,
            f"缺少 resolved_tracker_config.json 或 tracking_run.csv: {track_root}",
        )

    expected_config = resolved_config(tracker, load_overrides(config_path))
    with open(resolved_config_path, encoding="utf-8") as handle:
        saved_config = json.load(handle)
    if saved_config != expected_config:
        raise reuse_error(
            tracker,
            f"解析配置不一致: saved={saved_config!r}, current={expected_config!r}",
        )

    detection_caches, source_root = resolve_detection_caches(detection_source)
    expected_pairs = {
        output_cache_path(path, source_root, track_root).resolve(): path.resolve()
        for path in detection_caches
    }
    actual_tracks = {path.resolve() for path in track_root.rglob("*.tracks.jsonl")}
    expected_tracks = set(expected_pairs)
    if actual_tracks != expected_tracks:
        missing = sorted(str(path) for path in expected_tracks - actual_tracks)
        extra = sorted(str(path) for path in actual_tracks - expected_tracks)
        raise reuse_error(tracker, f"轨迹缓存集合不一致: missing={missing}, extra={extra}")

    with open(tracking_run_path, newline="", encoding="utf-8-sig") as handle:
        summary_rows = list(csv.DictReader(handle))
    summary_tracks = {Path(row["track_cache"]).resolve() for row in summary_rows}
    if len(summary_rows) != len(expected_pairs) or summary_tracks != expected_tracks:
        raise reuse_error(tracker, "tracking_run.csv 与预期轨迹缓存集合不一致")

    total_frames = 0
    for track_path, detection_path in sorted(expected_pairs.items()):
        detection_header = read_cache_header(detection_path, "detections")
        track_header = read_cache_header(track_path, "tracks")
        producer = track_header.get("producer", {})
        checks = {
            "coordinate_space": (
                track_header.get("coordinate_space"),
                "full_frame_xyxy",
            ),
            "upstream_detection_cache_id": (
                track_header.get("upstream_detection_cache_id"),
                detection_header["cache_id"],
            ),
            "frame_size": (
                track_header.get("frame_size"),
                detection_header.get("frame_size"),
            ),
            "fps": (
                track_header.get("fps"),
                detection_header.get("fps"),
            ),
            "class_names": (
                track_header.get("class_names"),
                detection_header.get("class_names"),
            ),
            "tracker_name": (
                producer.get("tracker_name"),
                tracker,
            ),
            "implementation_commit": (
                producer.get("implementation_commit"),
                TRACKER_COMMITS[tracker],
            ),
            "metadata_match_iou": (
                producer.get("metadata_match_iou"),
                expected_config["metadata_match_iou"],
            ),
            "runtime_config": (
                producer.get("config"),
                expected_runtime_tracker_config(tracker, expected_config, detection_header),
            ),
        }
        failed = {
            name: {"saved": saved, "expected": expected}
            for name, (saved, expected) in checks.items()
            if saved != expected
        }
        if failed:
            raise reuse_error(tracker, f"{track_path} 头信息不一致: {failed}")

        detection_frames = sum(1 for _ in iter_cache_frames(detection_path, "detections"))
        track_frames = sum(1 for _ in iter_cache_frames(track_path, "tracks"))
        if detection_frames != track_frames:
            raise reuse_error(
                tracker,
                f"{track_path} 帧数与检测缓存不一致: tracks={track_frames}, detections={detection_frames}",
            )
        total_frames += track_frames
    print(
        f"[{tracker}] 复用轨迹严格校验通过: caches={len(expected_pairs)}, "
        f"frames={total_frames}, track_root={track_root}"
    )
    return total_frames


def clear_count_outputs(count_root: Path) -> list[Path]:
    removed = []
    if not count_root.exists():
        return removed
    for filename in OUTPUT_FILENAMES:
        for path in sorted(count_root.rglob(filename)):
            if not path.is_file() and not path.is_symlink():
                raise ValueError(f"计数输出目标不是普通文件，拒绝删除: {path}")
            path.unlink()
            removed.append(path)
    return removed


def purge_tracker_root(tracker_root: Path) -> None:
    if tracker_root.is_dir() and not tracker_root.is_symlink():
        shutil.rmtree(tracker_root)
    elif tracker_root.exists() or tracker_root.is_symlink():
        tracker_root.unlink()


def main():
    args = parse_args()
    if not args.skip_counting and not args.counting_config.is_file():
        raise FileNotFoundError(f"计数配置不存在: {args.counting_config}")
    trackers = selected_trackers(args.trackers)
    detection_source = args.detections.resolve()
    resolve_detection_caches(detection_source)
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / MANIFEST_NAME
    context = {"detection_source": str(detection_source)}
    if not args.skip_counting:
        context.update(
            resolve_counting_context(
                detection_source,
                args.counting_config.resolve(),
                args.center_x,
                args.center_y,
                args.radius,
            )
        )
    manifest_rows = inspect_manifest(
        manifest,
        context,
        include_counting_context=not args.skip_counting,
    )
    actions = plan_tracker_actions(
        output_root,
        trackers,
        manifest_rows,
        overwrite=args.overwrite,
        purge=args.purge,
    )
    previous_rows = {row["tracker"]: row for row in manifest_rows}
    run_id, run_started_at = make_batch_identity()

    for tracker in trackers:
        if actions[tracker] == "reuse":
            tracker_root = output_root / tracker
            validate_reusable_tracks(
                tracker,
                detection_source,
                tracker_root / "tracks",
                BENCHMARK_ROOT / PROJECTS[tracker] / "configs" / "counting_candidate_v1.json",
            )

    for tracker in trackers:
        action = actions[tracker]
        tracker_root = output_root / tracker
        track_root = tracker_root / "tracks"
        count_root = tracker_root / "counts"
        config = BENCHMARK_ROOT / PROJECTS[tracker] / "configs" / "counting_candidate_v1.json"
        previous_row = previous_rows.get(tracker)

        if action == "purge" or (action == "reuse" and not args.skip_counting):
            removed_row = remove_manifest_tracker(
                manifest,
                tracker,
                context,
                include_counting_context=not args.skip_counting,
            )
            if removed_row is not None:
                previous_row = removed_row
        if action == "purge":
            purge_tracker_root(tracker_root)
            print(f"[{tracker}] --purge 已删除并准备重建: {tracker_root}")
        elif action == "reuse" and not args.skip_counting:
            removed = clear_count_outputs(count_root)
            print(f"[{tracker}] --overwrite 仅删除计数 CSV: files={len(removed)}，轨迹目录保留")

        if action in {"run", "purge"}:
            run_checked(
                [
                    sys.executable,
                    str(BENCHMARK_ROOT / "_common" / "run_tracker.py"),
                    "--tracker",
                    tracker,
                    "--detections",
                    str(detection_source),
                    "--output",
                    str(track_root),
                    "--config",
                    str(config),
                ]
            )
        else:
            print(f"[{tracker}] 跳过跟踪计算，复用 {track_root}")

        groups = count_groups(track_root)
        if not args.skip_counting:
            for group in groups:
                relative = group.relative_to(track_root)
                destination = count_root / relative
                command = [
                    sys.executable,
                    str(ROOT / "counting_model_benchmark" / "replay_counting.py"),
                    "--tracks",
                    str(group),
                    "--output",
                    str(destination),
                    "--config",
                    str(args.counting_config.resolve()),
                ]
                for option, value in (
                    ("--center-x", args.center_x),
                    ("--center-y", args.center_y),
                    ("--radius", args.radius),
                ):
                    if value is not None:
                        command.extend((option, str(value)))
                run_checked(command)
            if action == "reuse":
                track_run_id = (
                    previous_row.get("track_run_id")
                    if previous_row
                    else ""
                ) or (
                    previous_row.get("run_id")
                    if previous_row
                    else ""
                ) or "unknown_existing"
            else:
                track_run_id = run_id
            row = {
                "run_id": run_id,
                "run_started_at": run_started_at,
                "tracker": tracker,
                "config": str(config),
                "detection_source": str(detection_source),
                "track_root": str(track_root),
                "count_root": str(count_root),
                "logical_groups": len(groups),
                "counting_enabled": True,
                "counting_config": str(args.counting_config.resolve()),
                "counting_config_id": context["counting_config_id"],
                "reference_center_x": context["reference_center_x"],
                "reference_center_y": context["reference_center_y"],
                "reference_radius": context["reference_radius"],
                "center_x_override": "" if args.center_x is None else args.center_x,
                "center_y_override": "" if args.center_y is None else args.center_y,
                "radius_override": "" if args.radius is None else args.radius,
                "initial_count_policy": context["initial_count_policy"],
                "initial_counts": context["initial_counts"],
                "tracks_reused": action == "reuse",
                "track_run_id": track_run_id,
            }
            append_manifest_row(manifest, row, context)
            print(f"[{tracker}] 已原子追加运行清单: run_id={run_id}, manifest={manifest}")
        else:
            print(f"[{tracker}] --skip-counting：不写入用于评价的 {MANIFEST_NAME}")

    if args.skip_counting:
        print(f"统一跟踪运行完成: trackers={len(trackers)}, evaluation_manifest=unchanged")
    else:
        print(f"统一跟踪计数运行完成: trackers={len(trackers)}, run_id={run_id}, manifest={manifest}")


if __name__ == "__main__":
    main()

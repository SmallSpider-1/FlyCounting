#!/usr/bin/env python3
"""Run unified tracking and counting for any or all of the eight tracker candidates."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT / "tracking_model_benchmark"
DEFAULT_COUNTING_CONFIG = ROOT / "counting_model_benchmark" / "configs" / "baseline_circle_v1.json"
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
    parser.add_argument("--initial-count", action="append", default=[], metavar="CLASS_ID=COUNT")
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
    return selected


def count_groups(track_root: Path) -> list[Path]:
    return sorted({path.parent for path in track_root.rglob("*.tracks.jsonl")})


def run_checked(command: list[str]):
    print("执行:", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def write_manifest(path: Path, rows: list[dict]):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if not args.skip_counting and not args.counting_config.is_file():
        raise FileNotFoundError(f"计数配置不存在: {args.counting_config}")
    trackers = selected_trackers(args.trackers)
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    detection_source = args.detections.resolve()
    rows = []

    for tracker in trackers:
        tracker_root = output_root / tracker
        track_root = tracker_root / "tracks"
        count_root = tracker_root / "counts"
        config = BENCHMARK_ROOT / PROJECTS[tracker] / "configs" / "counting_candidate_v1.json"
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
                for initial_count in args.initial_count:
                    command.extend(("--initial-count", initial_count))
                run_checked(command)

        rows.append(
            {
                "tracker": tracker,
                "config": str(config),
                "detection_source": str(detection_source),
                "track_root": str(track_root),
                "count_root": "" if args.skip_counting else str(count_root),
                "logical_groups": len(groups),
                "counting_enabled": not args.skip_counting,
                "counting_config": str(args.counting_config.resolve()),
                "center_x_override": "" if args.center_x is None else args.center_x,
                "center_y_override": "" if args.center_y is None else args.center_y,
                "radius_override": "" if args.radius is None else args.radius,
            }
        )

    manifest = output_root / "benchmark_run_manifest.csv"
    write_manifest(manifest, rows)
    print(f"统一跟踪计数运行完成: trackers={len(trackers)}, manifest={manifest}")


if __name__ == "__main__":
    main()

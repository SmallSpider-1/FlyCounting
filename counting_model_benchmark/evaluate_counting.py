#!/usr/bin/env python3
"""Evaluate eight-tracker counting outputs against logical-long-video ground truth."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LONG_VIDEOS = ROOT / "video_data" / "manifests" / "long_videos.csv"
DEFAULT_GROUND_TRUTH = ROOT / "video_data" / "manifests" / "ground_truth_v1.csv"
EXPECTED_TRACKERS = (
    "sort",
    "bytetrack",
    "ocsort",
    "sfsort",
    "fasttracker",
    "boosttrack",
    "hybridsort",
    "botsort",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate unified tracker counting baselines.")
    parser.add_argument("--results", type=Path, required=True, help="run_counting_benchmark.py 的输出根目录。")
    parser.add_argument("--output", type=Path, default=None, help="评价 CSV 目录；默认 <results>/evaluation。")
    parser.add_argument("--long-videos", type=Path, default=DEFAULT_LONG_VIDEOS)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--allow-incomplete", action="store_true", help="允许缺少部分逻辑长视频，仅作诊断。")
    parser.add_argument(
        "--allow-partial-trackers",
        action="store_true",
        help="允许 manifest 缺少协议规定的部分 tracker，仅生成分批诊断表；正式评价禁止使用。",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_truth(long_videos_path: Path, ground_truth_path: Path):
    directories = {row["directory_name"]: row["video_id"] for row in read_csv(long_videos_path)}
    truth = defaultdict(dict)
    for row in read_csv(ground_truth_path):
        truth[row["video_id"]][row["species"]] = int(row["count"])
    return directories, dict(truth)


def load_predictions(path: Path) -> dict[str, int]:
    predictions = {}
    for row in read_csv(path):
        predictions[row["class_name"]] = int(row["final_region_count"])
    return predictions


def load_manifest_tracker_roots(results_root: Path) -> list[tuple[str, Path]]:
    manifest = results_root / "benchmark_run_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"缺少本次运行清单: {manifest}。评价默认只接受 "
            "run_counting_benchmark.py 生成的 benchmark_run_manifest.csv，不再扫描结果目录。"
        )
    rows = read_csv(manifest)
    if not rows:
        raise ValueError(f"本次运行清单为空: {manifest}")
    required = {"tracker", "count_root"}
    missing_columns = required - set(rows[0])
    if missing_columns:
        raise ValueError(f"本次运行清单缺少字段 {sorted(missing_columns)}: {manifest}")

    tracker_roots = []
    seen_trackers = {}
    seen_roots = {}
    for line_number, row in enumerate(rows, 2):
        tracker = row["tracker"].strip()
        count_root_text = row["count_root"].strip()
        counting_enabled = row.get("counting_enabled", "True").strip().lower()
        if not tracker:
            raise ValueError(f"本次运行清单第 {line_number} 行 tracker 为空: {manifest}")
        if counting_enabled not in {"true", "1", "yes"} or not count_root_text:
            raise ValueError(f"本次运行清单中的 {tracker} 没有本次计数结果: {manifest}")
        if tracker in seen_trackers:
            raise ValueError(
                f"本次运行清单包含重复 tracker={tracker!r}: "
                f"第 {seen_trackers[tracker]} 行和第 {line_number} 行"
            )
        count_root = Path(count_root_text)
        if not count_root.is_absolute():
            count_root = results_root / count_root
        count_root = count_root.resolve()
        if count_root in seen_roots:
            raise ValueError(
                f"本次运行清单中多个 tracker 指向同一 count_root={count_root}: "
                f"{seen_roots[count_root]!r}, {tracker!r}"
            )
        if not count_root.is_dir():
            raise FileNotFoundError(f"manifest 中的 count_root 不存在或不是目录: {tracker} -> {count_root}")
        seen_trackers[tracker] = line_number
        seen_roots[count_root] = tracker
        tracker_roots.append((tracker, count_root))
    return tracker_roots


def validate_manifest_tracker_set(
    tracker_roots: list[tuple[str, Path]],
    allow_partial: bool,
) -> list[str]:
    expected = set(EXPECTED_TRACKERS)
    actual = {tracker for tracker, _ in tracker_roots}
    missing = [tracker for tracker in EXPECTED_TRACKERS if tracker not in actual]
    unexpected = sorted(actual - expected)
    if unexpected or (missing and not allow_partial):
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        hint = (
            "\n分批诊断可显式传入 --allow-partial-trackers；"
            "正式 baseline 不得使用该开关。"
            if missing and not unexpected
            else ""
        )
        raise ValueError(
            "评价名单中的 tracker 集合不符合协议规定的 8 个算法: "
            + ", ".join(details)
            + hint
        )
    return missing


def collect_result_files(
    tracker: str,
    count_root: Path,
    expected_directories: set[str],
) -> dict[str, Path]:
    grouped = defaultdict(list)
    invalid = []
    for path in sorted(count_root.rglob("final_counts.csv")):
        relative_parts = path.relative_to(count_root).parts[:-1]
        matches = sorted(expected_directories.intersection(relative_parts))
        if len(matches) != 1:
            invalid.append((path.resolve(), matches))
        else:
            grouped[matches[0]].append(path.resolve())
    if invalid:
        details = "\n".join(
            f"  - {path} (匹配目录名={matches})"
            for path, matches in invalid
        )
        raise ValueError(
            f"{tracker} 存在无法唯一归属逻辑长视频的 final_counts.csv；"
            f"每个文件必须匹配恰好一个目录名:\n{details}"
        )
    duplicates = {name: paths for name, paths in grouped.items() if len(paths) >= 2}
    if duplicates:
        details = "\n".join(
            f"  - {directory_name}:\n"
            + "\n".join(f"      {path}" for path in paths)
            for directory_name, paths in sorted(duplicates.items())
        )
        raise ValueError(
            f"{tracker} 的同一逻辑长视频匹配到多个 final_counts.csv，拒绝静默覆盖:\n{details}"
        )
    return {name: paths[0] for name, paths in grouped.items()}


def main():
    args = parse_args()
    results_root = args.results.resolve()
    tracker_roots = load_manifest_tracker_roots(results_root)
    missing_trackers = validate_manifest_tracker_set(
        tracker_roots,
        allow_partial=args.allow_partial_trackers,
    )
    if missing_trackers:
        print(
            "警告: --allow-partial-trackers 已启用，本次只生成分批诊断表，"
            f"缺少协议 tracker: {missing_trackers}"
        )

    output_root = (args.output or (results_root / "evaluation")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    directory_to_id, truth = load_truth(args.long_videos, args.ground_truth)
    expected_directories = set(directory_to_id)
    detail_rows = []
    summary_rows = []

    for tracker, count_root in tracker_roots:
        result_files = collect_result_files(tracker, count_root, expected_directories)
        missing = expected_directories - set(result_files)
        if missing and not args.allow_incomplete:
            raise ValueError(f"{tracker} 缺少逻辑长视频结果: {sorted(missing)}")

        evaluated_directories = sorted(expected_directories & set(result_files))
        species_errors = defaultdict(list)
        total_errors = []
        exact_videos = 0
        negative_videos = 0
        for directory_name in evaluated_directories:
            video_id = directory_to_id[directory_name]
            predictions = load_predictions(result_files[directory_name])
            species_names = sorted(truth[video_id])
            missing_species = set(species_names) - set(predictions)
            if missing_species and not args.allow_incomplete:
                raise ValueError(
                    f"{tracker}/{directory_name} 缺少物种输出: {sorted(missing_species)}"
                )
            video_exact = True
            video_negative = False
            predicted_total = 0
            truth_total = 0
            for species in species_names:
                predicted = int(predictions.get(species, 0))
                target = int(truth[video_id][species])
                error = predicted - target
                species_errors[species].append(abs(error))
                video_exact &= error == 0
                video_negative |= predicted < 0
                predicted_total += predicted
                truth_total += target
                detail_rows.append(
                    {
                        "tracker": tracker,
                        "video_id": video_id,
                        "directory_name": directory_name,
                        "species": species,
                        "ground_truth": target,
                        "prediction": predicted,
                        "error": error,
                        "absolute_error": abs(error),
                    }
                )
            total_errors.append(abs(predicted_total - truth_total))
            exact_videos += int(video_exact)
            negative_videos += int(video_negative)

        all_absolute_errors = [value for values in species_errors.values() for value in values]
        video_count = len(evaluated_directories)
        correcta = "Bactrocera correcta"
        dorsalis = "Bactrocera dorsalis"
        summary_rows.append(
            {
                "tracker": tracker,
                "evaluated_videos": video_count,
                "correcta_mae": round(sum(species_errors[correcta]) / len(species_errors[correcta]), 6)
                if species_errors[correcta]
                else "",
                "dorsalis_mae": round(sum(species_errors[dorsalis]) / len(species_errors[dorsalis]), 6)
                if species_errors[dorsalis]
                else "",
                "class_mae": round(sum(all_absolute_errors) / len(all_absolute_errors), 6)
                if all_absolute_errors
                else "",
                "total_mae": round(sum(total_errors) / len(total_errors), 6) if total_errors else "",
                "exact_video_rate": round(exact_videos / video_count, 6) if video_count else "",
                "negative_video_rate": round(negative_videos / video_count, 6) if video_count else "",
                "complete_six_video_set": not missing,
            }
        )

    detail_path = output_root / "per_video_species_errors.csv"
    summary_path = output_root / "counting_metrics.csv"
    write_csv(
        detail_path,
        detail_rows,
        [
            "tracker",
            "video_id",
            "directory_name",
            "species",
            "ground_truth",
            "prediction",
            "error",
            "absolute_error",
        ],
    )
    write_csv(
        summary_path,
        summary_rows,
        [
            "tracker",
            "evaluated_videos",
            "correcta_mae",
            "dorsalis_mae",
            "class_mae",
            "total_mae",
            "exact_video_rate",
            "negative_video_rate",
            "complete_six_video_set",
        ],
    )
    print(f"计数评价完成: trackers={len(summary_rows)}, summary={summary_path}")


if __name__ == "__main__":
    main()

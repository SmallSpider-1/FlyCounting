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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate unified tracker counting baselines.")
    parser.add_argument("--results", type=Path, required=True, help="run_counting_benchmark.py 的输出根目录。")
    parser.add_argument("--output", type=Path, default=None, help="评价 CSV 目录；默认 <results>/evaluation。")
    parser.add_argument("--long-videos", type=Path, default=DEFAULT_LONG_VIDEOS)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--allow-incomplete", action="store_true", help="允许缺少部分逻辑长视频，仅作诊断。")
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


def main():
    args = parse_args()
    results_root = args.results.resolve()
    output_root = (args.output or (results_root / "evaluation")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    directory_to_id, truth = load_truth(args.long_videos, args.ground_truth)
    expected_directories = set(directory_to_id)
    detail_rows = []
    summary_rows = []

    tracker_dirs = sorted(path for path in results_root.iterdir() if (path / "counts").is_dir())
    if not tracker_dirs:
        raise FileNotFoundError(f"没有找到 <tracker>/counts 结果: {results_root}")

    for tracker_dir in tracker_dirs:
        tracker = tracker_dir.name
        result_files = {}
        for path in (tracker_dir / "counts").rglob("final_counts.csv"):
            relative_parts = path.relative_to(tracker_dir / "counts").parts
            matches = [part for part in relative_parts if part in expected_directories]
            if len(matches) == 1:
                result_files[matches[0]] = path
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

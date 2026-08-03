#!/usr/bin/env python3
"""Run TrackEval with a perfect tracker copied from fruitfly_mot_v1 GT."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any


def serializable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def read_sequences(path: Path) -> dict[str, int]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["sequence_id"]: int(row["num_frames"]) for row in csv.DictReader(handle)}


def build_perfect_tracker(gt_root: Path, tracker_data: Path, sequences: dict[str, int]) -> None:
    tracker_data.mkdir(parents=True)
    for sequence in sequences:
        gt_path = gt_root / sequence / "gt" / "gt.txt"
        output_lines = []
        for raw_line in gt_path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            fields = raw_line.split(",")
            if len(fields) != 9:
                raise RuntimeError(f"invalid GT line in {gt_path}: {raw_line}")
            output_lines.append(",".join([*fields[:6], "1", "1", "-1", "-1"]))
        (tracker_data / f"{sequence}.txt").write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def assert_perfect(combined: dict[str, Any], expected_frag: int) -> dict[str, Any]:
    hota = combined["HOTA"]
    clear = combined["CLEAR"]
    identity = combined["Identity"]
    arrays = {
        "HOTA": hota["HOTA"],
        "DetA": hota["DetA"],
        "AssA": hota["AssA"],
        "LocA": hota["LocA"],
    }
    for name, values in arrays.items():
        flattened = values.tolist() if hasattr(values, "tolist") else list(values)
        if not flattened or not all(math.isclose(float(value), 1.0, abs_tol=1e-12) for value in flattened):
            raise RuntimeError(f"perfect tracker produced non-perfect {name}: {flattened}")
    scalar_one = {
        "MOTA": clear["MOTA"],
        "MOTP": clear["MOTP"],
        "IDF1": identity["IDF1"],
        "IDP": identity["IDP"],
        "IDR": identity["IDR"],
    }
    for name, value in scalar_one.items():
        if not math.isclose(float(value), 1.0, abs_tol=1e-12):
            raise RuntimeError(f"perfect tracker produced {name}={value}")
    scalar_zero = {"IDSW": clear["IDSW"]}
    for name, value in scalar_zero.items():
        if int(value) != 0:
            raise RuntimeError(f"perfect tracker produced {name}={value}")
    if int(clear["Frag"]) != expected_frag:
        raise RuntimeError(
            f"perfect tracker produced Frag={clear['Frag']}, expected GT-gap baseline {expected_frag}"
        )
    return {
        "HOTA": 1.0,
        "DetA": 1.0,
        "AssA": 1.0,
        "LocA": 1.0,
        "MOTA": 1.0,
        "MOTP": 1.0,
        "IDF1": 1.0,
        "IDP": 1.0,
        "IDR": 1.0,
        "IDSW": 0,
        "Frag": expected_frag,
        "Frag_gt_gap_baseline": expected_frag,
    }


def run(dataset_root: Path, trackeval_root: Path, write_report: bool) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    if (dataset_root / ".fruitfly_mot_v1_generated.json").is_file():
        standard_root = dataset_root
    else:
        standard_root = dataset_root / "standardized"
    gt_root = standard_root / "MOTChallenge" / "all"
    sequences = read_sequences(standard_root / "manifests" / "sequences.csv")
    with (standard_root / "manifests" / "fully_occluded_intervals.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        expected_frag = sum(1 for _ in csv.DictReader(handle))
    trackeval_root = trackeval_root.resolve()
    if not (trackeval_root / "trackeval" / "__init__.py").is_file():
        raise RuntimeError(f"invalid TrackEval root: {trackeval_root}")
    sys.path.insert(0, str(trackeval_root))
    import trackeval  # type: ignore

    with tempfile.TemporaryDirectory(prefix="fruitfly_mot_trackeval_") as temp_name:
        temp_root = Path(temp_name)
        tracker_root = temp_root / "trackers"
        build_perfect_tracker(gt_root, tracker_root / "perfect" / "data", sequences)
        output_root = temp_root / "output"

        eval_config = trackeval.Evaluator.get_default_eval_config()
        eval_config.update(
            {
                "USE_PARALLEL": False,
                "BREAK_ON_ERROR": True,
                "PRINT_RESULTS": False,
                "PRINT_CONFIG": False,
                "TIME_PROGRESS": False,
                "OUTPUT_SUMMARY": False,
                "OUTPUT_DETAILED": False,
                "PLOT_CURVES": False,
                "LOG_ON_ERROR": None,
            }
        )
        dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        dataset_config.update(
            {
                "GT_FOLDER": str(gt_root),
                "TRACKERS_FOLDER": str(tracker_root),
                "OUTPUT_FOLDER": str(output_root),
                "TRACKERS_TO_EVAL": ["perfect"],
                "CLASSES_TO_EVAL": ["pedestrian"],
                "BENCHMARK": "fruitfly_mot_v1",
                "SPLIT_TO_EVAL": "all",
                "DO_PREPROC": False,
                "TRACKER_SUB_FOLDER": "data",
                "OUTPUT_SUB_FOLDER": "",
                "SEQ_INFO": sequences,
                "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt",
                "SKIP_SPLIT_FOL": True,
                "PRINT_CONFIG": False,
            }
        )
        metric_config = {
            "METRICS": ["HOTA", "CLEAR", "Identity"],
            "THRESHOLD": 0.5,
            "PRINT_CONFIG": False,
        }
        metrics = [
            trackeval.metrics.HOTA(metric_config),
            trackeval.metrics.CLEAR(metric_config),
            trackeval.metrics.Identity(metric_config),
        ]
        evaluator = trackeval.Evaluator(eval_config)
        results, messages = evaluator.evaluate(
            [trackeval.datasets.MotChallenge2DBox(dataset_config)], metrics
        )
        dataset_name = next(iter(results))
        if messages[dataset_name]["perfect"] != "Success":
            raise RuntimeError(f"TrackEval did not report success: {messages}")
        combined = results[dataset_name]["perfect"]["COMBINED_SEQ"]["pedestrian"]
        metrics_report = assert_perfect(combined, expected_frag)

    report = {
        "status": "passed",
        "test": "perfect_tracker_equals_ground_truth",
        "trackeval_root": str(trackeval_root),
        "sequences": len(sequences),
        "metrics": metrics_report,
    }
    if write_report:
        path = standard_root / "quality_reports" / "trackeval_smoke_test.json"
        path.write_text(json.dumps(serializable(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/fruitfly_mot_v1"))
    parser.add_argument(
        "--trackeval-root",
        type=Path,
        default=Path("tracking_model_benchmark/06_boosttrack/src/external/TrackEval"),
    )
    parser.add_argument("--no-write-report", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run(args.dataset_root, args.trackeval_root, not args.no_write_report)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

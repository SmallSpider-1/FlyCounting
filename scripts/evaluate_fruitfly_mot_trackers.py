#!/usr/bin/env python3
"""Convert unified track caches and evaluate all eight trackers with TrackEval."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_common.numeric_cache import iter_cache_frames, read_cache_header


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
DEFAULT_DATASET = ROOT / "datasets" / "fruitfly_mot_v1"
DEFAULT_EXPERIMENT = (
    ROOT / "project_results" / "fruitfly_mot_v1" / "rtdetr_r18_official_defaults_all_v1"
)
DEFAULT_TRACKEVAL = ROOT / "tracking_model_benchmark" / "06_boosttrack" / "src" / "external" / "TrackEval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--trackeval-root", type=Path, default=DEFAULT_TRACKEVAL)
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=None,
        help="Optional IoU-baseline summary.csv used to write a tracker-aligned delta table.",
    )
    return parser.parse_args()


def load_sequences(dataset_root: Path) -> dict[str, int]:
    path = dataset_root / "manifests" / "sequences.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {row["sequence_id"]: int(row["num_frames"]) for row in rows}


def serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def cache_to_mot(
    cache_path: Path,
    output_path: Path,
    expected_frames: int,
    expected_detection_cache_id: str,
) -> dict[str, Any]:
    header = read_cache_header(cache_path, "tracks")
    if header.get("upstream_detection_cache_id") != expected_detection_cache_id:
        raise ValueError(f"Upstream detection cache mismatch: {cache_path}")
    lines: list[str] = []
    track_rows = 0
    ids: set[int] = set()
    mot_id_by_native: dict[int, int] = {}
    frames = 0
    for frame_index, rows in iter_cache_frames(cache_path, "tracks", require_complete=True):
        frames += 1
        frame_ids: set[int] = set()
        for row in rows:
            x1, y1, x2, y2, track_id_value, confidence, _class_id, _detection_index = row
            native_track_id = int(track_id_value)
            if native_track_id < 0:
                raise ValueError(f"Negative track ID in {cache_path}, frame {frame_index}: {native_track_id}")
            if native_track_id in frame_ids:
                raise ValueError(
                    f"Duplicate track ID in {cache_path}, frame {frame_index}: {native_track_id}"
                )
            if native_track_id not in mot_id_by_native:
                mot_id_by_native[native_track_id] = len(mot_id_by_native) + 1
            mot_track_id = mot_id_by_native[native_track_id]
            width, height = float(x2) - float(x1), float(y2) - float(y1)
            if not all(math.isfinite(float(value)) for value in (x1, y1, width, height, confidence)):
                raise ValueError(f"Non-finite track row in {cache_path}, frame {frame_index}")
            if width < 0 or height < 0:
                raise ValueError(f"Invalid track extent in {cache_path}, frame {frame_index}")
            frame_ids.add(native_track_id)
            ids.add(native_track_id)
            lines.append(
                f"{frame_index},{mot_track_id},{float(x1):.2f},{float(y1):.2f},"
                f"{width:.2f},{height:.2f},{float(confidence):.4f},1,-1,-1"
            )
            track_rows += 1
    if frames != expected_frames:
        raise ValueError(f"Track cache frame mismatch: {cache_path}: {frames} != {expected_frames}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return {
        "frames": frames,
        "track_rows": track_rows,
        "unique_ids": len(ids),
        "mot_id_export": "stable_first_appearance_remap_to_positive_1_based",
    }


def mean_metric(metric: dict[str, Any], key: str) -> float:
    values = np.asarray(metric[key], dtype=float)
    return float(values.mean())


def metric_row(tracker: str, sequence: str, metrics: dict[str, Any]) -> dict[str, Any]:
    hota = metrics["HOTA"]
    clear = metrics["CLEAR"]
    identity = metrics["Identity"]
    count = metrics.get("Count", {})
    percent = lambda value: round(float(value) * 100.0, 6)
    return {
        "tracker": tracker,
        "sequence": sequence,
        "HOTA": percent(mean_metric(hota, "HOTA")),
        "DetA": percent(mean_metric(hota, "DetA")),
        "AssA": percent(mean_metric(hota, "AssA")),
        "LocA": percent(mean_metric(hota, "LocA")),
        "MOTA": percent(clear["MOTA"]),
        "MOTP": percent(clear["MOTP"]),
        "IDF1": percent(identity["IDF1"]),
        "IDP": percent(identity["IDP"]),
        "IDR": percent(identity["IDR"]),
        "IDSW": int(clear["IDSW"]),
        "Frag": int(clear["Frag"]),
        "MT": int(clear["MT"]),
        "PT": int(clear["PT"]),
        "ML": int(clear["ML"]),
        "TP": int(clear["CLR_TP"]),
        "FP": int(clear["CLR_FP"]),
        "FN": int(clear["CLR_FN"]),
        "Pred_Dets": int(count.get("Dets", 0)),
        "GT_Dets": int(count.get("GT_Dets", 0)),
        "Pred_IDs": int(count.get("IDs", 0)),
        "GT_IDs": int(count.get("GT_IDs", 0)),
    }


def experiment_description(association_metrics: dict[str, str]) -> tuple[str, str]:
    values = set(association_metrics.values())
    if values == {"normalized_euclidean"}:
        return (
            "fruitfly_mot_v1：8 个跟踪器归一化欧氏中心距离实验",
            "8 个跟踪器的原生 ID 关联空间项统一替换为框中心欧氏距离/两框最小外接矩形对角线；"
            "相似度接口使用 `1-distance`，原数值阈值及其余参数不变",
        )
    if values == {"iou"}:
        return (
            "fruitfly_mot_v1：8 个跟踪器工程 baseline",
            "各跟踪器使用其登记的原生 IoU/IoU-like 关联空间项和官方默认检测型配置",
        )
    return (
        "fruitfly_mot_v1：8 个跟踪器混合关联指标实验",
        "各跟踪器关联指标见 evaluation_protocol.json",
    )


def markdown_summary(rows: list[dict[str, Any]], association_metrics: dict[str, str]) -> str:
    title, association_condition = experiment_description(association_metrics)
    columns = ("rank", "tracker", "HOTA", "DetA", "AssA", "MOTA", "MOTP", "IDF1", "IDSW", "Frag", "FP", "FN")
    lines = [
        f"# {title}",
        "",
        "> 全部 15 段（split=unassigned）仅作工程 baseline；不是独立测试集结论。百分比指标单位为 %。",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for rank, row in enumerate(rows, 1):
        values = {**row, "rank": rank}
        lines.append("| " + " | ".join(str(values[column]) for column in columns) + " |")
    lines.extend(
        [
            "",
            f"评测条件：RT-DETR-R18 EMA/FP32、640×640 直接拉伸、检测缓存下限 0.10；{association_condition}；在线、AABB、类无关、无 ReID、无 GMC/ECC、无离线插值；TrackEval 匹配阈值仍为标准 IoU=0.5。HOTA/DetA/AssA/LocA 为 0.05–0.95 阈值均值。",
            "",
            "数据本身存在 3 个已确认的完全遮挡 GT 间断，因此原版 CLEAR 即使预测完全等于 GT 也报告 Frag=3；解读算法 Frag 时应同时保留这一本底。",
        ]
    )
    return "\n".join(lines) + "\n"


def load_association_metrics(track_root: Path) -> dict[str, str]:
    metrics = {}
    for tracker in TRACKERS:
        path = track_root / tracker / "numeric_cache" / "resolved_tracker_config.json"
        with path.open(encoding="utf-8") as stream:
            config = json.load(stream)
        metric = str(config.get("association_metric", "iou"))
        if metric not in {"iou", "normalized_euclidean"}:
            raise ValueError(f"Unknown association metric for {tracker}: {metric}")
        metrics[tracker] = metric
    return metrics


def comparison_rows(current_rows: list[dict[str, Any]], baseline_summary: Path) -> list[dict[str, Any]]:
    baseline_rows = read_csv(baseline_summary)
    baseline_by_tracker = {row["tracker"]: row for row in baseline_rows}
    if set(baseline_by_tracker) != set(TRACKERS):
        raise ValueError(f"Baseline tracker set mismatch: {baseline_summary}")
    output = []
    for current in current_rows:
        tracker = str(current["tracker"])
        baseline = baseline_by_tracker[tracker]
        row: dict[str, Any] = {"tracker": tracker}
        for metric in ("HOTA", "IDF1", "MOTA", "IDSW", "Frag"):
            baseline_value = float(baseline[metric])
            current_value = float(current[metric])
            if metric in {"IDSW", "Frag"}:
                row[f"{metric}_iou"] = int(baseline_value)
                row[f"{metric}_euclidean"] = int(current_value)
                row[f"delta_{metric}"] = int(current_value - baseline_value)
            else:
                row[f"{metric}_iou"] = round(baseline_value, 6)
                row[f"{metric}_euclidean"] = round(current_value, 6)
                row[f"delta_{metric}"] = round(current_value - baseline_value, 6)
        output.append(row)
    return output


def comparison_markdown(rows: list[dict[str, Any]]) -> str:
    columns = (
        "tracker",
        "HOTA_iou",
        "HOTA_euclidean",
        "delta_HOTA",
        "IDF1_iou",
        "IDF1_euclidean",
        "delta_IDF1",
        "IDSW_iou",
        "IDSW_euclidean",
        "delta_IDSW",
    )
    lines = [
        "# 归一化欧氏关联与 IoU baseline 对照",
        "",
        "> HOTA/IDF1 单位为百分点；IDSW 越低越好。仅为全 15 段工程对照，不是独立测试集结论。",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def tracking_speed_rows(track_root: Path, expected_frames: int) -> list[dict[str, Any]]:
    output = []
    for tracker in TRACKERS:
        path = track_root / tracker / "numeric_cache" / "tracking_run.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        frames = sum(int(row["frames"]) for row in rows)
        if len(rows) != 15 or frames != expected_frames:
            raise ValueError(f"Tracking run summary is incomplete for {tracker}: rows={len(rows)}, frames={frames}")
        tracking_seconds = sum(float(row["tracking_seconds"]) for row in rows)
        wall_seconds = sum(float(row["wall_seconds"]) for row in rows)
        output.append(
            {
                "tracker": tracker,
                "sequences": len(rows),
                "frames": frames,
                "tracking_seconds": round(tracking_seconds, 6),
                "tracking_fps": round(frames / tracking_seconds, 6),
                "cache_replay_wall_seconds": round(wall_seconds, 6),
                "cache_replay_wall_fps": round(frames / wall_seconds, 6),
                "timing_scope": "tracker_update_only_without_detector_or_cache_IO",
            }
        )
    return sorted(output, key=lambda row: -float(row["tracking_fps"]))


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    experiment_root = args.experiment_root.resolve()
    trackeval_root = args.trackeval_root.resolve()
    sequences = load_sequences(dataset_root)
    detection_root = experiment_root / "detections" / "numeric_cache"
    track_root = experiment_root / "tracks"
    input_root = experiment_root / "trackeval_input"
    output_root = experiment_root / "trackeval_output"
    metrics_root = experiment_root / "metrics"
    association_metrics = load_association_metrics(track_root)

    conversion_rows: list[dict[str, Any]] = []
    for tracker in TRACKERS:
        for sequence, num_frames in sequences.items():
            detection_cache = detection_root / f"{sequence}.detections.jsonl"
            detection_header = read_cache_header(detection_cache, "detections")
            track_cache = track_root / tracker / "numeric_cache" / f"{sequence}.tracks.jsonl"
            stats = cache_to_mot(
                track_cache,
                input_root / tracker / "data" / f"{sequence}.txt",
                num_frames,
                detection_header["cache_id"],
            )
            conversion_rows.append({"tracker": tracker, "sequence": sequence, **stats})

    if not (trackeval_root / "trackeval" / "__init__.py").is_file():
        raise FileNotFoundError(f"Invalid TrackEval root: {trackeval_root}")
    sys.path.insert(0, str(trackeval_root))
    import trackeval  # type: ignore

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
            "GT_FOLDER": str(dataset_root / "MOTChallenge" / "all"),
            "TRACKERS_FOLDER": str(input_root),
            "OUTPUT_FOLDER": str(output_root),
            "TRACKERS_TO_EVAL": list(TRACKERS),
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
    metric_config = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5, "PRINT_CONFIG": False}
    metrics = [
        trackeval.metrics.HOTA(metric_config),
        trackeval.metrics.CLEAR(metric_config),
        trackeval.metrics.Identity(metric_config),
    ]
    evaluator = trackeval.Evaluator(eval_config)
    results, messages = evaluator.evaluate([trackeval.datasets.MotChallenge2DBox(dataset_config)], metrics)
    dataset_name = next(iter(results))

    combined_rows: list[dict[str, Any]] = []
    per_sequence_rows: list[dict[str, Any]] = []
    for tracker in TRACKERS:
        message = messages[dataset_name][tracker]
        if message != "Success":
            raise RuntimeError(f"TrackEval failed for {tracker}: {message}")
        tracker_results = results[dataset_name][tracker]
        combined_rows.append(metric_row(tracker, "COMBINED_SEQ", tracker_results["COMBINED_SEQ"]["pedestrian"]))
        for sequence in sequences:
            per_sequence_rows.append(metric_row(tracker, sequence, tracker_results[sequence]["pedestrian"]))

    combined_rows.sort(key=lambda row: (-float(row["HOTA"]), -float(row["IDF1"]), int(row["IDSW"])))
    metrics_root.mkdir(parents=True, exist_ok=True)
    write_csv(metrics_root / "summary.csv", combined_rows)
    write_csv(metrics_root / "per_sequence.csv", per_sequence_rows)
    write_csv(metrics_root / "cache_conversion_validation.csv", conversion_rows)
    speed_rows = tracking_speed_rows(track_root, sum(sequences.values()))
    write_csv(metrics_root / "tracking_speed.csv", speed_rows)
    (metrics_root / "summary.md").write_text(
        markdown_summary(combined_rows, association_metrics), encoding="utf-8"
    )
    (metrics_root / "trackeval_full_results.json").write_text(
        json.dumps(serializable(results), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.baseline_summary is not None:
        baseline_summary = args.baseline_summary.resolve()
        if set(association_metrics.values()) != {"normalized_euclidean"}:
            raise ValueError("--baseline-summary 仅用于全归一化欧氏关联与 IoU baseline 对照")
        comparison = comparison_rows(combined_rows, baseline_summary)
        write_csv(metrics_root / "comparison_vs_iou_baseline.csv", comparison)
        (metrics_root / "comparison_vs_iou_baseline.md").write_text(
            comparison_markdown(comparison), encoding="utf-8"
        )
    protocol = {
        "status": "complete",
        "scope": "engineering_all_sequences_not_independent_test",
        "dataset_root": str(dataset_root),
        "dataset_fingerprint": (dataset_root / "quality_reports" / "dataset_fingerprint.txt")
        .read_text(encoding="utf-8")
        .strip(),
        "split": "unassigned/all",
        "sequences": len(sequences),
        "frames": sum(sequences.values()),
        "trackers": list(TRACKERS),
        "tracking_association": {
            "metric_by_tracker": association_metrics,
            "normalized_euclidean_definition": (
                "bbox_center_euclidean_over_pairwise_enclosing_box_diagonal"
            ),
            "native_numeric_thresholds_unchanged": True,
            "trackeval_evaluation_metric_unchanged": True,
            "comparison_baseline_summary": (
                str(args.baseline_summary.resolve()) if args.baseline_summary is not None else None
            ),
        },
        "evaluation": {
            "implementation": str(trackeval_root),
            "class_agnostic": True,
            "do_preproc": False,
            "matching_iou_threshold": 0.5,
            "metrics": ["HOTA", "CLEAR", "Identity"],
            "hota_average_thresholds": "0.05:0.05:0.95",
            "known_perfect_tracker_frag_baseline": 3,
        },
        "integrity": {
            "detection_caches": len(list(detection_root.glob("*.detections.jsonl"))),
            "track_caches": len(list(track_root.glob("*/numeric_cache/*.tracks.jsonl"))),
            "validated_track_cache_frames": sum(int(row["frames"]) for row in conversion_rows),
            "expected_track_cache_frames": len(TRACKERS) * sum(sequences.values()),
            "temporary_files": len(list(experiment_root.rglob(".*.tmp"))),
            "upstream_detection_cache_ids_checked": True,
            "duplicate_track_ids_per_frame_checked": True,
            "trackeval_export_ids": "stable_first_appearance_remap_to_positive_1_based",
        },
        "messages": serializable(messages),
    }
    (metrics_root / "evaluation_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    title, association_condition = experiment_description(association_metrics)
    (experiment_root / "README.md").write_text(
        f"# {title}\n\n"
        "本目录保存统一 RT-DETR-R18 检测缓存、8 个算法的统一轨迹缓存、TrackEval 输入、合并/逐序列指标、"
        "解析后的跟踪器配置与速度汇总。\n\n"
        f"轨迹关联条件：{association_condition}。\n\n"
        "权威总表为 `metrics/summary.csv`，便于阅读的表为 `metrics/summary.md`；冻结运行条件见 "
        "`metrics/evaluation_protocol.json` 与 `detections/detection_protocol.json`。\n\n"
        "当前 15 段均为 `split=unassigned`，所以这是全数据工程 baseline，不是独立测试集结论。TrackEval 文本中的"
        "轨迹 ID 按序列无损映射为从 1 开始的正整数；各算法原生 ID 保留在 `tracks/*/numeric_cache/`。\n",
        encoding="utf-8",
    )
    print(markdown_summary(combined_rows, association_metrics))
    print(f"Results: {metrics_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

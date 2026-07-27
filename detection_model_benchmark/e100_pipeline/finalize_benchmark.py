#!/usr/bin/env python3
"""Freeze the e100 benchmark before test and build central final summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    DATASET_ROOT,
    EXPECTED_DATASET_FINGERPRINT,
    MODEL_SPECS,
    RESULT_ROOT,
    atomic_write_json,
    atomic_write_text,
    now_iso,
    read_json,
    run_dir,
    sha256_file,
)


SUMMARY = RESULT_ROOT / "summary"
DOMAINS = {
    "mixed": DATASET_ROOT / "coco/annotations/instances_test.json",
    "C": DATASET_ROOT / "coco/annotations/instances_test_C.json",
    "R1": DATASET_ROOT / "coco/annotations/instances_test_R1.json",
    "R2": DATASET_ROOT / "coco/annotations/instances_test_R2.json",
}
VAL_ANN = DATASET_ROOT / "coco/annotations/instances_val.json"


def current_evaluation_dataset_fingerprint() -> str:
    stats = read_json(DATASET_ROOT / "metadata/dataset_stats.json")
    fingerprint = str(stats.get("dataset_fingerprint_sha256", ""))
    if len(fingerprint) != 64:
        raise RuntimeError("Missing current evaluation dataset fingerprint")
    return fingerprint


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def checkpoint_path(model_key: str, name: str = "best") -> Path:
    suffix = ".pth" if MODEL_SPECS[model_key]["family"] in {"deim", "rt_detr", "faster_rcnn"} else ".pt"
    return run_dir(model_key) / f"checkpoints/{name}{suffix}"


def verify_configuration_manifest() -> dict[str, Any]:
    manifest = RESULT_ROOT / "configuration_manifest.sha256"
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = RESULT_ROOT / relative
        actual = sha256_file(path)
        rows.append({"path": relative, "expected": expected, "actual": actual, "ok": expected == actual})
    if not all(row["ok"] for row in rows):
        raise RuntimeError(f"Frozen configuration manifest mismatch: {[row for row in rows if not row['ok']]}")
    return {"manifest": str(manifest), "entries": len(rows), "checks": rows}


def convergence(model_key: str) -> dict[str, Any]:
    history = read_csv(run_dir(model_key) / "metrics_history.csv")
    values = {int(row["epoch"]): float(row["val_map50_95"]) for row in history}
    if sorted(values) != list(range(1, 101)):
        return {"model": model_key, "metric_scale": "0-1", "convergence_status": "insufficient_validation_history"}
    previous_epoch = max(range(71, 91), key=lambda epoch: (values[epoch], epoch))
    final_epoch = max(range(91, 101), key=lambda epoch: (values[epoch], epoch))
    global_epoch = max(range(1, 101), key=lambda epoch: (values[epoch], epoch))
    previous_best = values[previous_epoch]
    final_best = values[final_epoch]
    delta = final_best - previous_best
    if delta > 0.003:
        status = "still_improving_at_e100"
        interpretation = "The final ten epochs improve the preceding 20-epoch best by more than 0.3 percentage points."
    elif global_epoch >= 71:
        status = "basic_converged"
        interpretation = "The global best occurs in epoch 71-100 without a material final-window gain."
    else:
        status = "early_peak_or_overfit"
        interpretation = "The global best predates epoch 71; the late curve does not recover that peak."
    return {
        "model": model_key,
        "display": MODEL_SPECS[model_key]["display"],
        "metric_scale": "0-1",
        "previous_best": previous_best,
        "previous_best_epoch": previous_epoch,
        "final_best": final_best,
        "final_best_epoch": final_epoch,
        "delta": delta,
        "global_best": values[global_epoch],
        "global_best_epoch": global_epoch,
        "early_global_peak": global_epoch < 71,
        "convergence_status": status,
        "curve_interpretation": interpretation,
        "notes": "Computed from val only before independent test inference.",
    }


def verify_training_completion(model_key: str) -> dict[str, Any]:
    directory = run_dir(model_key)
    status = read_json(directory / "status.json")
    metrics = read_csv(directory / "metrics_history.csv")
    steps = read_csv(directory / "optimizer_steps.csv")
    runtime = read_json(directory / "runtime_config_validation.json")
    required = [checkpoint_path(model_key, "last"), checkpoint_path(model_key, "best")]
    suffix = required[0].suffix
    required.extend(directory / f"checkpoints/epoch_{epoch:03d}{suffix}" for epoch in range(10, 101, 10))
    checks = {
        "status_complete": status.get("status") == "complete",
        "metric_epochs_1_100": [int(row["epoch"]) for row in metrics] == list(range(1, 101)),
        "optimizer_epochs_1_100": [int(row["epoch"]) for row in steps] == list(range(1, 101)),
        "optimizer_steps_50900": sum(int(row["optimizer_steps"]) for row in steps) == 50900,
        "runtime_config_validation": all(runtime.get("checks", {}).values()),
        "twelve_checkpoints": all(path.exists() and path.stat().st_size > 0 for path in required),
    }
    if not all(checks.values()):
        raise RuntimeError(f"{model_key} is not safe to freeze: {checks}")
    return checks


def pretest_freeze() -> None:
    SUMMARY.mkdir(parents=True, exist_ok=True)
    manifest = verify_configuration_manifest()
    convergence_rows = []
    frozen_rows = []
    completion = {}
    frozen_at = now_iso()
    for model_key in MODEL_SPECS:
        completion[model_key] = verify_training_completion(model_key)
        result = convergence(model_key)
        atomic_write_json(run_dir(model_key) / "convergence_check.json", result)
        convergence_rows.append(result)
        best = checkpoint_path(model_key)
        selection = {
            "model": model_key,
            "selection_split": "mixed_val",
            "selection_metric": "val_mAP50_95",
            "selection_rule": "pure numeric maximum over epochs 1-100",
            "best_epoch": result["global_best_epoch"],
            "best_val_map50_95": result["global_best"],
            "checkpoint": str(best),
            "checkpoint_sha256": sha256_file(best),
            "checkpoint_bytes": best.stat().st_size,
            "frozen_at": frozen_at,
        }
        atomic_write_json(run_dir(model_key) / "checkpoint_selection.json", selection)
        frozen_rows.append(selection)

    convergence_fields = [
        "model", "display", "metric_scale", "previous_best", "previous_best_epoch", "final_best",
        "final_best_epoch", "delta", "global_best", "global_best_epoch", "early_global_peak",
        "convergence_status", "curve_interpretation", "notes",
    ]
    write_csv(SUMMARY / "convergence_status.csv", convergence_fields, convergence_rows)

    extension_fields = [
        "model", "display", "reason", "val_map50_95", "params_M", "unified_GFLOPs",
        "convergence_status", "selected_at",
    ]
    extension_rows = []
    for row in convergence_rows:
        if row["convergence_status"] == "still_improving_at_e100":
            profile = read_json(run_dir(row["model"]) / "speed/efficiency_profile.json")
            extension_rows.append(
                {
                    "model": row["model"],
                    "display": row["display"],
                    "reason": "still_improving_at_e100 under frozen val-only rule",
                    "val_map50_95": row["global_best"],
                    "params_M": profile["params_M"],
                    "unified_GFLOPs": profile["unified_GFLOPs"],
                    "convergence_status": row["convergence_status"],
                    "selected_at": frozen_at,
                }
            )
    extension_rows.sort(key=lambda row: (-float(row["val_map50_95"]), float(row["params_M"])))
    extension_rows = extension_rows[:3]
    extension_path = SUMMARY / "extension_candidates_pretest.csv"
    write_csv(extension_path, extension_fields, extension_rows)
    atomic_write_text(SUMMARY / "extension_candidates_pretest.sha256", f"{sha256_file(extension_path)}  {extension_path.name}\n")

    frozen_fields = [
        "model", "selection_split", "selection_metric", "selection_rule", "best_epoch",
        "best_val_map50_95", "checkpoint", "checkpoint_sha256", "checkpoint_bytes", "frozen_at",
    ]
    write_csv(SUMMARY / "frozen_checkpoints.csv", frozen_fields, frozen_rows)

    existing_test_predictions = [str(path) for path in RESULT_ROOT.glob("runs/*/predictions/test_mixed.json")]
    if existing_test_predictions:
        raise RuntimeError(f"Test predictions existed before access freeze: {existing_test_predictions}")
    access_log = {
        "schema_version": 1,
        "authorized_at": now_iso(),
        "authorization": "User explicitly requested val and independent test results on 2026-07-22.",
        "all_models_terminal_and_complete": True,
        "training_completion_checks": completion,
        "dataset_fingerprint": EXPECTED_DATASET_FINGERPRINT,
        "configuration_manifest": {"path": manifest["manifest"], "entries": manifest["entries"]},
        "convergence_status_sha256": sha256_file(SUMMARY / "convergence_status.csv"),
        "extension_candidates_pretest_sha256": sha256_file(extension_path),
        "frozen_checkpoints_sha256": sha256_file(SUMMARY / "frozen_checkpoints.csv"),
        "postfreeze_protocol_sha256": sha256_file(SUMMARY / "postfreeze_evaluation_protocol.md"),
        "prior_test_prediction_or_metric_files": [],
        "prior_test_metric_access": False,
        "metadata_audit_before_formal_freeze": {
            "occurred": True,
            "scope": "annotation counts and first image records only, after all training/checkpoints completed",
            "model_inference": False,
            "predictions_generated": False,
            "metrics_computed": False,
            "selection_influenced": False,
        },
        "test_inference_permitted_after_this_record": True,
    }
    atomic_write_json(RESULT_ROOT / "test_access_log.json", access_log)
    atomic_write_json(SUMMARY / "pretest_freeze_validation.json", {"created_at": frozen_at, "manifest": manifest, "completion": completion})
    print(json.dumps({"frozen_models": len(frozen_rows), "extension_candidates": len(extension_rows), "test_access_log": str(RESULT_ROOT / "test_access_log.json")}, ensure_ascii=False, indent=2))


def validate_prediction_ids(predictions: list[dict[str, Any]], annotation: dict[str, Any]) -> dict[str, Any]:
    ids = {int(image["id"]) for image in annotation["images"]}
    predicted = {int(row["image_id"]) for row in predictions}
    unknown = sorted(predicted - ids)
    categories = {int(row["category_id"]) for row in predictions}
    per_image = defaultdict(int)
    for row in predictions:
        per_image[int(row["image_id"])] += 1
    over_max = {key: value for key, value in per_image.items() if value > 300}
    checks = {
        "known_image_ids": not unknown,
        "canonical_categories": categories <= {0, 1},
        "max_det_300": not over_max,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Prediction validation failed: {checks}, unknown={unknown[:10]}, over_max={list(over_max.items())[:10]}")
    return {"checks": checks, "truth_images": len(ids), "predicted_images": len(predicted), "prediction_rows": len(predictions)}


def coco_metrics(annotation_path: Path, prediction_path: Path) -> dict[str, Any]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with redirect_stdout(io.StringIO()):
        truth = COCO(str(annotation_path))
        predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
        result = truth.loadRes(predictions)
        evaluator = COCOeval(truth, result, "bbox")
        evaluator.params.maxDets = [1, 10, 300]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]
    all_precision = precision[:, :, :, 0, 2]
    valid_precision = all_precision[all_precision > -1]
    ap50_values = precision[0, :, :, 0, 2]
    valid_ap50 = ap50_values[ap50_values > -1]
    iou75_index = int(np.argmin(np.abs(np.asarray(evaluator.params.iouThrs) - 0.75)))
    ap75_values = precision[iou75_index, :, :, 0, 2]
    valid_ap75 = ap75_values[ap75_values > -1]
    recall_values = recall[:, :, 0, 2]
    valid_recall = recall_values[recall_values > -1]
    class_ap: dict[str, Any] = {}
    categories = sorted(truth.loadCats(truth.getCatIds()), key=lambda row: int(row["id"]))
    for category_index, category in enumerate(categories):
        values = precision[:, :, category_index, 0, 2]
        valid = values[values > -1]
        class_ap[str(category["id"])] = float(np.mean(valid)) if valid.size else None
    return {
        # COCOeval's stock summary hard-codes maxDets=100 for its headline AP.
        # The benchmark freezes maxDet=300, so use the accumulated tensors at
        # maxDets index 2 directly for every reported statistic.
        "mAP50_95": float(np.mean(valid_precision)),
        "AP50": float(np.mean(valid_ap50)),
        "AP75": float(np.mean(valid_ap75)),
        "AR": float(np.mean(valid_recall)),
        "per_class_AP": class_ap,
        "max_det": 300,
    }


def xywh_to_xyxy(box: list[float]) -> np.ndarray:
    x, y, width, height = [float(value) for value in box]
    return np.array([x, y, x + width, y + height], dtype=np.float64)


def iou(box: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    if not len(candidates):
        return np.empty((0,), dtype=np.float64)
    top_left = np.maximum(box[:2], candidates[:, :2])
    bottom_right = np.minimum(box[2:], candidates[:, 2:])
    intersection = np.maximum(0.0, bottom_right - top_left)
    intersection_area = intersection[:, 0] * intersection[:, 1]
    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    candidate_area = np.maximum(0.0, candidates[:, 2] - candidates[:, 0]) * np.maximum(0.0, candidates[:, 3] - candidates[:, 1])
    union = box_area + candidate_area - intersection_area
    return np.divide(intersection_area, union, out=np.zeros_like(intersection_area), where=union > 0)


def prepare_precision_recall(annotation_path: Path, prediction_path: Path) -> dict[str, Any]:
    truth = json.loads(annotation_path.read_text(encoding="utf-8"))
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    ground_truth: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    for row in truth["annotations"]:
        ground_truth[(int(row["image_id"]), int(row["category_id"]))].append(xywh_to_xyxy(row["bbox"]))
    predicted: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        predicted[(int(row["image_id"]), int(row["category_id"]))].append(row)
    events: list[tuple[float, bool]] = []
    keys = set(ground_truth) | set(predicted)
    for key in keys:
        targets = np.array(ground_truth.get(key, []), dtype=np.float64).reshape(-1, 4)
        used = np.zeros(len(targets), dtype=bool)
        rows = sorted(predicted.get(key, []), key=lambda row: float(row["score"]), reverse=True)
        for row in rows:
            overlaps = iou(xywh_to_xyxy(row["bbox"]), targets)
            overlaps[used] = -1
            if len(overlaps) and float(overlaps.max()) >= 0.5:
                match = int(overlaps.argmax())
                used[match] = True
                events.append((float(row["score"]), True))
            else:
                events.append((float(row["score"]), False))
    return {
        "events": events,
        "ground_truth_count": sum(len(rows) for rows in ground_truth.values()),
    }


def precision_recall_from_prepared(prepared: dict[str, Any], threshold: float) -> dict[str, Any]:
    true_positive = sum(1 for score, matched in prepared["events"] if score >= threshold and matched)
    false_positive = sum(1 for score, matched in prepared["events"] if score >= threshold and not matched)
    false_negative = int(prepared["ground_truth_count"]) - true_positive
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "matching": "same-class score-descending greedy IoU>=0.5",
    }


def precision_recall(annotation_path: Path, prediction_path: Path, threshold: float) -> dict[str, Any]:
    return precision_recall_from_prepared(prepare_precision_recall(annotation_path, prediction_path), threshold)


def best_val_threshold(prediction_path: Path) -> dict[str, Any]:
    prepared = prepare_precision_recall(VAL_ANN, prediction_path)
    candidates = [precision_recall_from_prepared(prepared, round(index / 100, 2)) for index in range(1, 100)]
    best = max(candidates, key=lambda row: (row["f1"], row["threshold"]))
    return {**best, "scan_start": 0.01, "scan_end": 0.99, "scan_step": 0.01, "tie_break": "higher_threshold"}


def filter_domain_predictions(model_key: str, domain: str, full_predictions: list[dict[str, Any]]) -> Path:
    annotation = json.loads(DOMAINS[domain].read_text(encoding="utf-8"))
    ids = {int(image["id"]) for image in annotation["images"]}
    filtered = [row for row in full_predictions if int(row["image_id"]) in ids]
    destination = run_dir(model_key) / "predictions" / f"test_{domain}.json"
    destination.write_text(json.dumps(filtered, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    validate_prediction_ids(filtered, annotation)
    return destination


def markdown_percent(value: Any) -> str:
    return "N/A" if value is None else f"{100 * float(value):.2f}"


def create_plots(results: list[dict[str, Any]]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    created = []
    plt.figure(figsize=(12, 7))
    for model_key in MODEL_SPECS:
        history = read_csv(run_dir(model_key) / "metrics_history.csv")
        plt.plot([int(row["epoch"]) for row in history], [float(row["val_map50_95"]) for row in history], label=MODEL_SPECS[model_key]["display"], linewidth=1.2)
    plt.xlabel("Epoch")
    plt.ylabel("Val mAP50-95")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    path = SUMMARY / "training_curves.png"
    plt.savefig(path, dpi=180)
    plt.close()
    created.append(str(path))

    plt.figure(figsize=(10, 7))
    for row in results:
        plt.scatter(float(row["FP32_FPS"]), float(row["test_mAP50_95"]), s=max(25, float(row["params_M"]) * 3), alpha=0.8)
        plt.annotate(row["display"], (float(row["FP32_FPS"]), float(row["test_mAP50_95"])), fontsize=7, xytext=(3, 3), textcoords="offset points")
    plt.xlabel("GPU0 FP32 end-to-end FPS")
    plt.ylabel("Mixed test mAP50-95")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    path = SUMMARY / "accuracy_efficiency_tradeoff.png"
    plt.savefig(path, dpi=180)
    plt.close()
    created.append(str(path))

    plt.figure(figsize=(12, 7))
    for model_key in MODEL_SPECS:
        rows = read_csv(run_dir(model_key) / "lr_schedule.csv")
        fields = rows[0].keys() if rows else []
        lr_field = next((field for field in fields if "lr" in field.lower() and field.lower() not in {"model"}), None)
        epoch_field = next((field for field in fields if "epoch" in field.lower()), None)
        if lr_field and epoch_field:
            plt.plot([float(row[epoch_field]) for row in rows], [float(row[lr_field]) for row in rows], label=MODEL_SPECS[model_key]["display"], linewidth=1.1)
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    path = SUMMARY / "learning_rate_schedules.png"
    plt.savefig(path, dpi=180)
    plt.close()
    created.append(str(path))
    return created


def write_reproducibility_manifest(plot_paths: list[str] | None = None) -> None:
    manifest_path = SUMMARY / "reproducibility_manifest.json"
    manifest_files = sorted(
        path
        for path in SUMMARY.iterdir()
        if path.is_file() and path != manifest_path and path.suffix != ".log"
    )
    if plot_paths is None:
        plot_paths = [str(path) for path in manifest_files if path.suffix == ".png"]
    reproducibility = {
        "created_at": now_iso(),
        "dataset_fingerprint": current_evaluation_dataset_fingerprint(),
        "training_dataset_fingerprint": EXPECTED_DATASET_FINGERPRINT,
        "evaluation_dataset_fingerprint": current_evaluation_dataset_fingerprint(),
        "configuration_manifest_sha256": sha256_file(RESULT_ROOT / "configuration_manifest.sha256"),
        "test_access_log_sha256": sha256_file(RESULT_ROOT / "test_access_log.json"),
        "test_ground_truth_repair_sha256": (
            sha256_file(RESULT_ROOT / "test_ground_truth_repair_20260722.json")
            if (RESULT_ROOT / "test_ground_truth_repair_20260722.json").exists()
            else None
        ),
        "evaluation_code": [
            {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            {
                "path": str((Path(__file__).parent / "postprocess_worker.py").resolve()),
                "sha256": sha256_file((Path(__file__).parent / "postprocess_worker.py").resolve()),
            },
        ],
        "summary_files": [{"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in manifest_files],
        "plots": plot_paths,
        "excluded": ["reproducibility_manifest.json (self-reference)", "*.log (mutable execution logs)"],
    }
    atomic_write_json(manifest_path, reproducibility)


def summarize() -> None:
    access = read_json(RESULT_ROOT / "test_access_log.json")
    if not access or not access.get("test_inference_permitted_after_this_record"):
        raise RuntimeError("Missing valid test_access_log.json")
    all_results = []
    domain_rows = []
    class_rows = []
    efficiency_rows = []
    image_validation: dict[str, Any] = {"created_at": now_iso(), "models": {}}

    for model_key in MODEL_SPECS:
        directory = run_dir(model_key)
        val_prediction = directory / "predictions/val_mixed.json"
        test_prediction = directory / "predictions/test_mixed.json"
        if not val_prediction.exists() or not test_prediction.exists():
            raise FileNotFoundError(f"Missing central predictions for {model_key}")
        val_truth = json.loads(VAL_ANN.read_text(encoding="utf-8"))
        val_rows = json.loads(val_prediction.read_text(encoding="utf-8"))
        test_truth = json.loads(DOMAINS["mixed"].read_text(encoding="utf-8"))
        test_rows = json.loads(test_prediction.read_text(encoding="utf-8"))
        image_validation["models"][model_key] = {
            "val": validate_prediction_ids(val_rows, val_truth),
            "test_mixed": validate_prediction_ids(test_rows, test_truth),
        }

        val_ap = coco_metrics(VAL_ANN, val_prediction)
        threshold = best_val_threshold(val_prediction)
        val_pr = precision_recall(VAL_ANN, val_prediction, threshold["threshold"])
        model_domains = {}
        for domain, truth_path in DOMAINS.items():
            domain_prediction = test_prediction if domain == "mixed" else filter_domain_predictions(model_key, domain, test_rows)
            truth = json.loads(truth_path.read_text(encoding="utf-8"))
            image_validation["models"][model_key][f"test_{domain}"] = validate_prediction_ids(
                json.loads(domain_prediction.read_text(encoding="utf-8")), truth
            )
            metrics = coco_metrics(truth_path, domain_prediction)
            model_domains[domain] = metrics
            domain_rows.append({"model": model_key, "display": MODEL_SPECS[model_key]["display"], "domain": domain, **{key: metrics[key] for key in ("mAP50_95", "AP50", "AP75", "AR")}})
            for class_id, class_name in ((0, "Bactrocera dorsalis"), (1, "Bactrocera correcta")):
                value = metrics["per_class_AP"].get(str(class_id))
                if domain == "R1" and class_id == 1:
                    value = None
                class_rows.append({"model": model_key, "display": MODEL_SPECS[model_key]["display"], "domain": domain, "class_id": class_id, "class_name": class_name, "AP50_95": value})
        test_pr = precision_recall(DOMAINS["mixed"], test_prediction, threshold["threshold"])
        profile = read_json(directory / "speed/efficiency_profile.json")
        convergence_result = read_json(directory / "convergence_check.json")
        selection = read_json(directory / "checkpoint_selection.json")
        steps = read_csv(directory / "optimizer_steps.csv")
        status = read_json(directory / "status.json")
        assignment = read_json(directory / "gpu_assignment.json")
        fp32 = profile["speed"]["FP32"]
        fp16 = profile["speed"]["FP16"]
        result = {
            "model": model_key,
            "display": MODEL_SPECS[model_key]["display"],
            "status": status["status"],
            "best_epoch": selection["best_epoch"],
            "best_val_history_mAP50_95": selection["best_val_map50_95"],
            "convergence_status": convergence_result["convergence_status"],
            "convergence_delta": convergence_result["delta"],
            "val_threshold": threshold["threshold"],
            "val_precision": val_pr["precision"],
            "val_recall": val_pr["recall"],
            "val_f1": val_pr["f1"],
            "val_mAP50_95": val_ap["mAP50_95"],
            "val_AP50": val_ap["AP50"],
            "val_AP75": val_ap["AP75"],
            "test_precision": test_pr["precision"],
            "test_recall": test_pr["recall"],
            "test_f1": test_pr["f1"],
            "test_mAP50_95": model_domains["mixed"]["mAP50_95"],
            "test_AP50": model_domains["mixed"]["AP50"],
            "test_AP75": model_domains["mixed"]["AP75"],
            "C_mAP50_95": model_domains["C"]["mAP50_95"],
            "R1_mAP50_95": model_domains["R1"]["mAP50_95"],
            "R2_mAP50_95": model_domains["R2"]["mAP50_95"],
            "worst_domain_mAP": min(model_domains[name]["mAP50_95"] for name in ("C", "R1", "R2")),
            "dorsalis_AP": model_domains["mixed"]["per_class_AP"]["0"],
            "correcta_AP": model_domains["mixed"]["per_class_AP"]["1"],
            "params_M": profile["params_M"],
            "unified_GFLOPs": profile["unified_GFLOPs"],
            "FLOPs_coverage": profile["counted_ops_coverage"],
            "checkpoint_MB": checkpoint_path(model_key).stat().st_size / 1024**2,
            "FP32_latency_ms": fp32["end_to_end"]["mean_ms"],
            "FP32_FPS": fp32["fps_from_mean_end_to_end"],
            "FP16_method": fp16.get("implementation", "unsupported"),
            "FP16_latency_ms": fp16.get("end_to_end", {}).get("mean_ms"),
            "FP16_FPS": fp16.get("fps_from_mean_end_to_end"),
            "peak_VRAM_MB": read_json(directory / "adapter_summary.json")["peak_cuda_memory_mib"],
            "actual_optimizer_steps": sum(int(row["optimizer_steps"]) for row in steps),
            "train_gpu_id": assignment.get("physical_gpu_id"),
            "micro_batch": int(steps[0]["micro_batch"]),
            "gradient_accumulation": int(steps[0]["accumulation"]),
            "effective_batch": int(steps[0]["effective_batch"]),
            "seed": 42,
        }
        all_results.append(result)
        efficiency_rows.append({key: result[key] for key in ("model", "display", "params_M", "unified_GFLOPs", "FLOPs_coverage", "checkpoint_MB", "FP32_latency_ms", "FP32_FPS", "FP16_method", "FP16_latency_ms", "FP16_FPS", "peak_VRAM_MB")})
        atomic_write_json(directory / "evaluation/central_metrics.json", {"model": model_key, "val": val_ap, "val_f1_threshold": threshold, "val_precision_recall": val_pr, "test_precision_recall": test_pr, "test_domains": model_domains})

    all_results.sort(key=lambda row: float(row["test_mAP50_95"]), reverse=True)
    fields = list(all_results[0])
    write_csv(SUMMARY / "benchmark_results.csv", fields, all_results)
    write_csv(SUMMARY / "per_domain_metrics.csv", ["model", "display", "domain", "mAP50_95", "AP50", "AP75", "AR"], domain_rows)
    write_csv(SUMMARY / "per_class_metrics.csv", ["model", "display", "domain", "class_id", "class_name", "AP50_95"], class_rows)
    write_csv(SUMMARY / "efficiency_metrics.csv", list(efficiency_rows[0]), efficiency_rows)
    atomic_write_json(SUMMARY / "image_id_validation.json", image_validation)
    atomic_write_json(RESULT_ROOT / "image_id_validation.json", image_validation)

    md = [
        "# Benchmark results",
        "",
        "All P/R values use each model's mixed-val F1-optimal threshold, frozen before application to mixed test. AP is central canonical COCO evaluation. Values below are percentages except Params/GFLOPs/FPS.",
        "",
        "## Mixed validation",
        "",
        "| Model | P | R | mAP50 | mAP50-95 | Params(M) | FLOPs(G) | FP32 FPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in all_results:
        md.append(f"| {row['display']} | {markdown_percent(row['val_precision'])} | {markdown_percent(row['val_recall'])} | {markdown_percent(row['val_AP50'])} | {markdown_percent(row['val_mAP50_95'])} | {float(row['params_M']):.3f} | {float(row['unified_GFLOPs']):.3f} | {float(row['FP32_FPS']):.1f} |")
    md.extend(["", "## Mixed independent test", "", "| Model | P | R | mAP50 | mAP50-95 | Params(M) | FLOPs(G) | FP32 FPS |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in all_results:
        md.append(f"| {row['display']} | {markdown_percent(row['test_precision'])} | {markdown_percent(row['test_recall'])} | {markdown_percent(row['test_AP50'])} | {markdown_percent(row['test_mAP50_95'])} | {float(row['params_M']):.3f} | {float(row['unified_GFLOPs']):.3f} | {float(row['FP32_FPS']):.1f} |")
    md.extend(["", "FP32 FPS is batch-1 end-to-end tensor inference on physical GPU0 (GRID A100DX-80C), excluding disk decode. This is a server result, not Raspberry Pi performance.", ""])
    atomic_write_text(SUMMARY / "benchmark_results.md", "\n".join(md))

    accuracy_leader = all_results[0]
    worst_domain_leader = max(all_results, key=lambda row: float(row["worst_domain_mAP"]))
    median_params = float(np.median([float(row["params_M"]) for row in all_results]))
    compact_pool = [row for row in all_results if float(row["params_M"]) <= median_params]
    compact_leader = max(compact_pool, key=lambda row: (float(row["test_mAP50_95"]), float(row["FP32_FPS"])))
    candidates = []
    for row in (accuracy_leader, worst_domain_leader, compact_leader):
        if row["model"] not in {candidate["model"] for candidate in candidates}:
            candidates.append(row)
    recommendation = [
        "# Final recommendation",
        "",
        "This is a single-seed, fixed-100-epoch engineering screen, not a final statistically significant model conclusion.",
        "",
        "## Candidates for multi-seed and deployment follow-up",
        "",
    ]
    for row in candidates:
        recommendation.append(f"- {row['display']}: mixed test mAP50-95={100*float(row['test_mAP50_95']):.2f}%, worst-domain mAP={100*float(row['worst_domain_mAP']):.2f}%, {float(row['params_M']):.3f}M parameters, {float(row['FP32_FPS']):.1f} FP32 FPS.")
    recommendation.extend([
        "",
        "No model met `still_improving_at_e100`; no automatic e160 extension is recommended from the frozen convergence rule.",
        "",
        "## Required interpretation limits",
        "",
        "1. Only seed 42 was run under a common 100-epoch budget; 100 epochs does not guarantee every architecture reached its official-practice ceiling.",
        "2. Optimizer families were retained, but effective batch, LR and schedule stages were normalized; AdamW models also used the common linear LR scaling rule.",
        "3. Mosaic, mixup and other strong augmentation were disabled. This may penalize models whose official recipes rely on them, especially some YOLO families.",
        "4. Training time from different vGPU profiles is not an efficiency ranking. Inference speed was remeasured serially on physical GPU0.",
        "5. Server A100 FPS is not Raspberry Pi FPS. R1 has no correcta annotations, so R1 correcta AP is N/A.",
    ])
    atomic_write_text(SUMMARY / "final_recommendation.md", "\n".join(recommendation) + "\n")
    plot_paths = create_plots(all_results)
    write_reproducibility_manifest(plot_paths)
    print(json.dumps({"models": len(all_results), "accuracy_leader": accuracy_leader["display"], "benchmark_results": str(SUMMARY / "benchmark_results.csv")}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["pretest", "summarize", "manifest"])
    args = parser.parse_args()
    if args.mode == "pretest":
        pretest_freeze()
    elif args.mode == "summarize":
        summarize()
    else:
        write_reproducibility_manifest()


if __name__ == "__main__":
    main()

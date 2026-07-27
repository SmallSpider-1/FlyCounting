#!/usr/bin/env python3
"""Generate frozen-checkpoint predictions and unified efficiency profiles.

This worker is launched with the Conda interpreter assigned to each model.  It
never trains or changes a checkpoint.  Prediction JSON is normalized to the
canonical 0-based COCO image/category IDs before it leaves the worker.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import yaml

from common import (
    DATASET_ROOT,
    MODEL_SPECS,
    RESULT_ROOT,
    atomic_write_json,
    load_resolved_config,
    now_iso,
    run_dir,
    sha256_file,
)


CONF_THRESHOLD = 0.001
NMS_IOU = 0.7
MAX_DET = 300
IMAGE_SIZE = 640
WARMUP_ITERS = 50
MEASURE_ITERS = 300


def trusted_torch_load(torch: Any, path: Path, device: str = "cpu") -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def best_checkpoint(model_key: str) -> Path:
    suffix = ".pth" if MODEL_SPECS[model_key]["family"] in {"deim", "rt_detr", "faster_rcnn"} else ".pt"
    path = run_dir(model_key) / f"checkpoints/best{suffix}"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def annotation_path(split: str) -> Path:
    if split not in {"val", "test"}:
        raise ValueError(split)
    return DATASET_ROOT / f"coco/annotations/instances_{split}.json"


def image_root(split: str) -> Path:
    return DATASET_ROOT / f"coco/{split}"


def evaluation_dataset_config(model_key: str) -> Path:
    """Create a read-only data view whose generated label caches stay local."""
    source_config = Path(load_resolved_config(model_key)["dataset_config"])
    payload = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    view = run_dir(model_key) / "evaluation" / "dataset_view"
    for kind in ("images", "labels"):
        local_group = view / kind
        local_group.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            local_split = local_group / split
            target = DATASET_ROOT / "yolo" / kind / split
            if local_split.is_symlink():
                if local_split.resolve() != target.resolve():
                    raise RuntimeError(f"Unexpected dataset-view link: {local_split} -> {local_split.resolve()}")
            elif local_split.exists():
                raise RuntimeError(f"Dataset-view path exists but is not a symlink: {local_split}")
            else:
                local_split.symlink_to(target, target_is_directory=True)
    payload["path"] = str(view)
    # Older YOLOv7/YOLOR dataset parsing predates the top-level ``path`` key
    # and resolves relative splits against the source checkout.  Absolute
    # split paths work in both the legacy and current loaders.
    for split in ("train", "val", "test"):
        if MODEL_SPECS[model_key]["family"].startswith("legacy_yolo"):
            # YOLOv5/9 resolve directory symlinks before deriving labels.cache.
            # A model-local image list makes its cache path the adjacent
            # model-local ``<split>.cache`` while retaining read-only images.
            image_list = view / f"{split}.txt"
            images = sorted(path for path in (DATASET_ROOT / "yolo/images" / split).iterdir() if path.is_file())
            image_list.write_text("\n".join(str(path) for path in images) + "\n", encoding="utf-8")
            payload[split] = str(image_list)
        else:
            payload[split] = str(view / "images" / split)
    destination = run_dir(model_key) / "evaluation" / "dataset_config.yaml"
    destination.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return destination


def prediction_path(model_key: str, split: str) -> Path:
    return run_dir(model_key) / "predictions" / ("val_mixed.json" if split == "val" else "test_mixed.json")


def load_annotation(split: str) -> dict[str, Any]:
    return json.loads(annotation_path(split).read_text(encoding="utf-8"))


def validate_canonical_predictions(predictions: list[dict[str, Any]], split: str) -> dict[str, Any]:
    truth = load_annotation(split)
    valid_ids = {int(item["id"]) for item in truth["images"]}
    unknown_ids = sorted({int(item["image_id"]) for item in predictions} - valid_ids)
    invalid_categories = sorted({int(item["category_id"]) for item in predictions} - {0, 1})
    per_image = Counter(int(item["image_id"]) for item in predictions)
    over_max = {str(key): value for key, value in per_image.items() if value > MAX_DET}
    invalid_bbox = []
    for index, item in enumerate(predictions):
        bbox = item.get("bbox", [])
        if len(bbox) != 4 or not all(math.isfinite(float(value)) for value in bbox) or float(bbox[2]) < 0 or float(bbox[3]) < 0:
            invalid_bbox.append(index)
    checks = {
        "all_prediction_image_ids_in_truth": not unknown_ids,
        "canonical_categories_0_1": not invalid_categories,
        "max_det_300": not over_max,
        "valid_xywh_boxes": not invalid_bbox,
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"Canonical prediction validation failed for {split}: checks={checks}, "
            f"unknown_ids={unknown_ids[:10]}, invalid_categories={invalid_categories}, "
            f"over_max={list(over_max.items())[:10]}, invalid_bbox={invalid_bbox[:10]}"
        )
    return {
        "split": split,
        "truth_images": len(valid_ids),
        "prediction_rows": len(predictions),
        "images_with_predictions": len(per_image),
        "checks": checks,
    }


def canonicalize_native_json(native_path: Path, output_path: Path, split: str) -> None:
    truth = load_annotation(split)
    filename_to_id = {item["file_name"]: int(item["id"]) for item in truth["images"]}
    stem_to_id = {Path(name).stem: image_id for name, image_id in filename_to_id.items()}
    valid_ids = set(filename_to_id.values())
    raw = json.loads(native_path.read_text(encoding="utf-8"))
    raw_categories = {int(item["category_id"]) for item in raw}
    # Current Ultralytics writes custom-dataset COCO JSON with one-based
    # category IDs, while the legacy forks write their zero-based class IDs.
    category_offset = 1 if raw_categories <= {1, 2} and 2 in raw_categories else 0
    canonical: list[dict[str, Any]] = []
    for item in raw:
        raw_id = item["image_id"]
        if isinstance(raw_id, int) and raw_id in valid_ids:
            image_id = raw_id
        else:
            token = Path(str(raw_id)).stem
            if token not in stem_to_id:
                raise KeyError(f"Unknown native image_id {raw_id!r} in {native_path}")
            image_id = stem_to_id[token]
        category = int(item["category_id"]) - category_offset
        canonical.append(
            {
                "image_id": image_id,
                "category_id": category,
                "bbox": [round(float(value), 4) for value in item["bbox"]],
                "score": float(item["score"]),
            }
        )
    validate_canonical_predictions(canonical, split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(canonical, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)


def predict_legacy(model_key: str, split: str, output_path: Path) -> None:
    config = load_resolved_config(model_key)
    source_dir = Path(config["source_dir"])
    sys.path.insert(0, str(source_dir))
    os.chdir(source_dir)
    native_root = run_dir(model_key) / "evaluation" / "native_predictions"
    native_dir = native_root / split
    native_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = best_checkpoint(model_key)
    data = str(evaluation_dataset_config(model_key))

    if model_key == "yolov7_tiny":
        import test as native_eval

        native_eval.opt = SimpleNamespace(
            device="0",
            project=str(native_root),
            name=split,
            exist_ok=True,
            task=split,
            single_cls=False,
        )
        native_eval.test(
            data=data,
            weights=str(checkpoint),
            batch_size=16,
            imgsz=IMAGE_SIZE,
            conf_thres=CONF_THRESHOLD,
            iou_thres=NMS_IOU,
            save_json=True,
            plots=False,
            half_precision=False,
            trace=False,
            is_coco=False,
            v5_metric=False,
        )
    else:
        if model_key == "yolov9_t":
            import val_dual as native_eval
        else:
            import val as native_eval

        if model_key == "yolov9_t":
            native_eval.opt = SimpleNamespace(min_items=0)
        native_eval.run(
            data=data,
            weights=str(checkpoint),
            batch_size=16,
            imgsz=IMAGE_SIZE,
            conf_thres=CONF_THRESHOLD,
            iou_thres=NMS_IOU,
            max_det=MAX_DET,
            task=split,
            device="0",
            workers=8,
            augment=False,
            save_json=True,
            project=native_root,
            name=split,
            exist_ok=True,
            half=False,
            plots=False,
        )
    native_json = native_dir / f"{checkpoint.stem}_predictions.json"
    if not native_json.exists():
        candidates = sorted(native_dir.glob("*predictions.json"))
        if len(candidates) != 1:
            raise FileNotFoundError(f"Expected one native prediction JSON in {native_dir}; got {candidates}")
        native_json = candidates[0]
    canonicalize_native_json(native_json, output_path, split)


def predict_ultralytics(model_key: str, split: str, output_path: Path) -> None:
    config = load_resolved_config(model_key)
    source_dir = Path(config["source_dir"])
    sys.path.insert(0, str(source_dir))
    os.chdir(source_dir)
    from ultralytics import YOLO
    if model_key == "yolov10n":
        from ultralytics import YOLOv10
    import ultralytics.data.dataset as dataset_module

    # Shared Ultralytics label caches are NumPy pickles and are not portable
    # between the frozen NumPy 1.x and 2.x environments.  Redirect generated
    # caches for every fork, keeping the canonical shared dataset read-only.
    original_load = dataset_module.load_dataset_cache_file
    original_save = dataset_module.save_dataset_cache_file
    cache_root = run_dir(model_key) / "evaluation" / "dataset_cache"

    def local_cache_path(path: str | Path) -> Path:
        cache_path = Path(path)
        if cache_path.name in {"train.cache", "val.cache", "test.cache"}:
            return cache_root / cache_path.name
        return cache_path

    def load_local_cache(path: str | Path) -> dict[str, Any]:
        return original_load(local_cache_path(path))

    def save_local_cache(prefix: str, path: str | Path, payload: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        destination = local_cache_path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        original_save(prefix, destination, payload, *args, **kwargs)

    dataset_module.load_dataset_cache_file = load_local_cache
    dataset_module.save_dataset_cache_file = save_local_cache

    native_root = run_dir(model_key) / "evaluation" / "native_predictions"
    wrapper_class = YOLOv10 if model_key == "yolov10n" else YOLO
    model = wrapper_class(str(best_checkpoint(model_key)))
    model.val(
        data=str(evaluation_dataset_config(model_key)),
        split=split,
        batch=16,
        imgsz=IMAGE_SIZE,
        device=0,
        workers=8,
        conf=CONF_THRESHOLD,
        iou=NMS_IOU,
        max_det=MAX_DET,
        half=False,
        augment=False,
        rect=True,
        save_json=True,
        plots=False,
        project=str(native_root),
        name=split,
        exist_ok=True,
        verbose=False,
    )
    native_json = native_root / split / "predictions.json"
    if not native_json.exists():
        raise FileNotFoundError(native_json)
    canonicalize_native_json(native_json, output_path, split)


def transformer_eval_config(model_key: str, split: str) -> Path:
    source = run_dir(model_key) / "runtime_config.yaml"
    if split == "val":
        return source
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    dataset = payload.setdefault("val_dataloader", {}).setdefault("dataset", {})
    dataset["img_folder"] = str(image_root(split))
    dataset["ann_file"] = str(annotation_path(split))
    destination = run_dir(model_key) / "evaluation" / f"runtime_config_{split}.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return destination


def build_transformer_runner(model_key: str, split: str = "val") -> tuple[Any, Any, Any, Any, Callable[[], None]]:
    import torch

    # Avoid intermittent ancillary-file-descriptor transfer failures from the
    # high-worker-count validation loaders in long sequential evaluation runs.
    torch.multiprocessing.set_sharing_strategy("file_system")

    config = load_resolved_config(model_key)
    source_dir = Path(config["source_dir"])
    sys.path.insert(0, str(source_dir))
    os.chdir(source_dir)
    runtime = transformer_eval_config(model_key, split)
    checkpoint = best_checkpoint(model_key)

    if model_key == "deim_dfine_n":
        from engine.core import YAMLConfig
        from engine.misc import dist_utils
        from engine.solver import TASKS

        dist_utils.setup_distributed(0, "builtin", seed=42)
        cfg = YAMLConfig(
            str(runtime),
            resume=str(checkpoint),
            tuning=None,
            use_amp=False,
            output_dir=str(run_dir(model_key) / "evaluation" / "native_eval"),
            device="cuda",
            seed=42,
        )
        solver = TASKS[cfg.yaml_cfg["task"]](cfg)
        solver.eval()
        cleanup = dist_utils.cleanup
    else:
        from src.core import YAMLConfig
        from src.misc import dist
        from src.solver import TASKS

        dist.init_distributed()
        dist.set_seed(42)
        cfg = YAMLConfig(str(runtime), resume=str(checkpoint), tuning=None, use_amp=False)
        cfg.output_dir = str(run_dir(model_key) / "evaluation" / "native_eval")
        solver = TASKS[cfg.yaml_cfg["task"]](cfg)
        solver.eval()
        cleanup = lambda: None
    module = solver.ema.module if solver.ema is not None else solver.model
    module.eval()
    return solver, module, solver.postprocessor, solver.device, cleanup


def predict_transformer(model_key: str, split: str, output_path: Path) -> None:
    import torch

    solver, module, postprocessor, device, cleanup = build_transformer_runner(model_key, split)
    predictions: list[dict[str, Any]] = []
    try:
        with torch.inference_mode():
            for samples, targets in solver.val_dataloader:
                samples = samples.to(device)
                targets = [{key: value.to(device) if hasattr(value, "to") else value for key, value in target.items()} for target in targets]
                outputs = module(samples)
                original_sizes = torch.stack([target["orig_size"] for target in targets], dim=0)
                results = postprocessor(outputs, original_sizes)
                for target, result in zip(targets, results):
                    image_id = int(target["image_id"].item())
                    scores = result["scores"].detach().float().cpu()
                    boxes = result["boxes"].detach().float().cpu()
                    labels = result["labels"].detach().long().cpu()
                    order = scores.argsort(descending=True)[:MAX_DET]
                    for index in order.tolist():
                        score = float(scores[index])
                        if score < CONF_THRESHOLD:
                            continue
                        x1, y1, x2, y2 = [float(value) for value in boxes[index].tolist()]
                        predictions.append(
                            {
                                "image_id": image_id,
                                "category_id": int(labels[index]),
                                "bbox": [round(x1, 4), round(y1, 4), round(max(0.0, x2 - x1), 4), round(max(0.0, y2 - y1), 4)],
                                "score": score,
                            }
                        )
    finally:
        cleanup()
    validate_canonical_predictions(predictions, split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(predictions, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def build_faster_model(device: Any) -> Any:
    import torch
    import torchvision

    model = torchvision.models.get_model(
        "fasterrcnn_resnet50_fpn",
        weights=None,
        weights_backbone=None,
        num_classes=3,
        min_size=IMAGE_SIZE,
        max_size=IMAGE_SIZE,
        box_score_thresh=CONF_THRESHOLD,
        box_nms_thresh=NMS_IOU,
        box_detections_per_img=MAX_DET,
    )
    checkpoint = trusted_torch_load(torch, best_checkpoint("faster_rcnn_r50_fpn"))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def predict_faster(split: str, output_path: Path) -> None:
    import torch
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor

    device = torch.device("cuda:0")
    model = build_faster_model(device)
    truth = load_annotation(split)
    records = truth["images"]
    predictions: list[dict[str, Any]] = []
    batch_size = 16
    with torch.inference_mode():
        for offset in range(0, len(records), batch_size):
            batch = records[offset : offset + batch_size]
            images = []
            for record in batch:
                with Image.open(image_root(split) / record["file_name"]) as image:
                    tensor = pil_to_tensor(image.convert("RGB")).float().div_(255.0)
                images.append(tensor.to(device, non_blocking=True))
            outputs = model(images)
            for record, result in zip(batch, outputs):
                scores = result["scores"].detach().float().cpu()
                order = scores.argsort(descending=True)[:MAX_DET]
                boxes = result["boxes"].detach().float().cpu()
                labels = result["labels"].detach().long().cpu()
                for index in order.tolist():
                    score = float(scores[index])
                    if score < CONF_THRESHOLD:
                        continue
                    x1, y1, x2, y2 = [float(value) for value in boxes[index].tolist()]
                    predictions.append(
                        {
                            "image_id": int(record["id"]),
                            "category_id": int(labels[index]) - 1,
                            "bbox": [round(x1, 4), round(y1, 4), round(max(0.0, x2 - x1), 4), round(max(0.0, y2 - y1), 4)],
                            "score": score,
                        }
                    )
    validate_canonical_predictions(predictions, split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(predictions, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def predict(model_key: str, split: str, force: bool) -> None:
    output = prediction_path(model_key, split)
    if output.exists() and not force:
        existing = json.loads(output.read_text(encoding="utf-8"))
        validate_canonical_predictions(existing, split)
        print(f"Validated existing prediction file: {output}")
        return
    family = MODEL_SPECS[model_key]["family"]
    if family.startswith("legacy_yolo"):
        predict_legacy(model_key, split, output)
    elif family in {"ultralytics_current", "ultralytics_yolov10", "ultralytics_yolo12"}:
        predict_ultralytics(model_key, split, output)
    elif family in {"deim", "rt_detr"}:
        predict_transformer(model_key, split, output)
    elif family == "faster_rcnn":
        predict_faster(split, output)
    else:
        raise KeyError(family)
    metadata = {
        "model": model_key,
        "split": split,
        "checkpoint": str(best_checkpoint(model_key)),
        "checkpoint_sha256": sha256_file(best_checkpoint(model_key)),
        "prediction": str(output),
        "prediction_sha256": sha256_file(output),
        "created_at": now_iso(),
        "confidence_export_threshold": CONF_THRESHOLD,
        "nms_iou": "N/A" if MODEL_SPECS[model_key].get("postprocess") == "NMS_FREE" else NMS_IOU,
        "postprocess_type": MODEL_SPECS[model_key].get("postprocess", "NMS"),
        "max_det": MAX_DET,
        "precision": "FP32",
    }
    atomic_write_json(run_dir(model_key) / "evaluation" / f"prediction_{split}_metadata.json", metadata)
    print(f"Created {output} ({output.stat().st_size} bytes)")


class InferenceRunner:
    def __init__(self, model: Any, forward: Callable[[Any], Any], postprocess: Callable[[Any], Any], device: Any, cleanup: Callable[[], None] = lambda: None):
        self.model = model
        self.forward = forward
        self.postprocess = postprocess
        self.device = device
        self.cleanup = cleanup


def call_nms(function: Callable[..., Any], prediction: Any, end2end: bool = False) -> Any:
    parameters = inspect.signature(function).parameters
    kwargs: dict[str, Any] = {}
    candidates = {
        "conf_thres": CONF_THRESHOLD,
        "iou_thres": NMS_IOU,
        "max_det": MAX_DET,
        "nc": 2,
        "end2end": end2end,
        "multi_label": True,
    }
    for key, value in candidates.items():
        if key in parameters:
            kwargs[key] = value
    return function(prediction, **kwargs)


def build_inference_runner(model_key: str) -> InferenceRunner:
    import torch

    device = torch.device("cuda:0")
    family = MODEL_SPECS[model_key]["family"]
    config = load_resolved_config(model_key)
    source_dir = Path(config["source_dir"])
    if family != "faster_rcnn":
        sys.path.insert(0, str(source_dir))
        os.chdir(source_dir)
    else:
        # The checkout contains a pure-Python torchvision source tree but the
        # benchmark intentionally uses the matching installed binary build.
        # Do not shadow that binary package with the checkout during profiling.
        os.chdir(source_dir.parent)

    if family in {"ultralytics_current", "ultralytics_yolov10", "ultralytics_yolo12"}:
        from ultralytics import YOLO
        try:
            from ultralytics.utils.nms import non_max_suppression
        except ImportError:
            from ultralytics.utils.ops import non_max_suppression

        wrapper = YOLO(str(best_checkpoint(model_key)))
        model = wrapper.model.to(device).float().eval()
        end2end = bool(getattr(model.model[-1], "end2end", False)) if hasattr(model, "model") else False
        if model_key == "yolov10n":
            import torch
            from ultralytics.utils import ops

            def yolov10_postprocess(output: Any) -> Any:
                prediction = output["one2one"] if isinstance(output, dict) else output
                if isinstance(prediction, (list, tuple)):
                    prediction = prediction[0]
                if prediction.shape[-1] == 6:
                    return prediction
                prediction = prediction.transpose(-1, -2)
                boxes, scores, labels = ops.v10postprocess(prediction, MAX_DET, 2)
                boxes = ops.xywh2xyxy(boxes)
                return torch.cat((boxes, scores.unsqueeze(-1), labels.unsqueeze(-1)), dim=-1)

            return InferenceRunner(model, lambda tensor: model(tensor), yolov10_postprocess, device)
        return InferenceRunner(
            model,
            lambda tensor: model(tensor),
            lambda output: call_nms(non_max_suppression, output, end2end=end2end),
            device,
        )
    if family in {"legacy_yolo5", "legacy_yolo9"}:
        from models.common import DetectMultiBackend
        from utils.general import non_max_suppression

        backend = DetectMultiBackend(str(best_checkpoint(model_key)), device=device, data=str(config["dataset_config"]), fp16=False)
        backend.model.to(device).float().eval()
        backend.eval()
        # YOLOv9's bare model returns a list of inference heads, while its
        # official DetectMultiBackend wrapper selects the deploy prediction.
        select_prediction = (lambda output: output[0][1]) if model_key == "yolov9_t" else (lambda output: output)
        return InferenceRunner(
            backend,
            lambda tensor: backend(tensor),
            lambda output: call_nms(non_max_suppression, select_prediction(output)),
            device,
        )
    if family == "legacy_yolo7":
        from models.experimental import attempt_load
        from utils.general import non_max_suppression

        model = attempt_load(str(best_checkpoint(model_key)), map_location=device).float().eval()
        return InferenceRunner(model, lambda tensor: model(tensor)[0], lambda output: call_nms(non_max_suppression, output), device)
    if family in {"deim", "rt_detr"}:
        solver, model, postprocessor, _, cleanup = build_transformer_runner(model_key, "val")
        original_size = torch.tensor([[IMAGE_SIZE, IMAGE_SIZE]], device=device)
        return InferenceRunner(model, lambda tensor: model(tensor), lambda output: postprocessor(output, original_size), device, cleanup)
    if family == "faster_rcnn":
        model = build_faster_model(device)
        return InferenceRunner(model, lambda tensor: model([tensor[0]]), lambda output: output, device)
    raise KeyError(family)


class UnifiedFlopMode:
    """Versioned local TorchDispatch FLOP counter shared by all environments."""

    VERSION = "fruitfly_dispatch_flops_v1"

    def __init__(self) -> None:
        from torch.utils._python_dispatch import TorchDispatchMode

        outer = self

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                output = func(*args, **kwargs)
                outer.record(str(func), args, output)
                return output

        self.mode = Mode()
        self.total = 0
        self.counted = Counter()
        self.unhandled = Counter()

    @staticmethod
    def tensor_numel(value: Any) -> int:
        if hasattr(value, "numel") and hasattr(value, "shape"):
            return int(value.numel())
        if isinstance(value, (list, tuple)):
            return sum(UnifiedFlopMode.tensor_numel(item) for item in value)
        if isinstance(value, dict):
            return sum(UnifiedFlopMode.tensor_numel(item) for item in value.values())
        return 0

    def add(self, name: str, value: int) -> None:
        value = max(0, int(value))
        self.total += value
        self.counted[name] += value

    def record(self, name: str, args: tuple[Any, ...], output: Any) -> None:
        output_numel = self.tensor_numel(output)
        try:
            if ("convolution" in name or any(token in name for token in ("aten.conv1d", "aten.conv2d", "aten.conv3d"))) and len(args) >= 2 and hasattr(args[1], "shape"):
                weight = args[1]
                kernel = math.prod(int(value) for value in weight.shape[2:])
                in_per_group = int(weight.shape[1])
                bias_ops = 1 if len(args) > 2 and args[2] is not None else 0
                self.add("convolution", output_numel * (2 * kernel * in_per_group + bias_ops))
                return
            if "aten.linear" in name and len(args) >= 2:
                weight = args[1]
                bias_ops = 1 if len(args) > 2 and args[2] is not None else 0
                self.add("linear", output_numel * (2 * int(weight.shape[-1]) + bias_ops))
                return
            if "addmm" in name and len(args) >= 3:
                matrix1, matrix2 = args[-2], args[-1]
                self.add("addmm", 2 * int(matrix1.shape[-2]) * int(matrix2.shape[-1]) * int(matrix1.shape[-1]))
                return
            if ".mm." in name and len(args) >= 2:
                self.add("mm", 2 * int(args[0].shape[-2]) * int(args[1].shape[-1]) * int(args[0].shape[-1]))
                return
            if "baddbmm" in name or ".bmm." in name:
                left, right = args[-2], args[-1]
                self.add("bmm", 2 * int(left.shape[0]) * int(left.shape[-2]) * int(right.shape[-1]) * int(left.shape[-1]))
                return
            if "scaled_dot_product" in name and len(args) >= 3:
                query, key, value = args[:3]
                batches = math.prod(int(value_) for value_ in query.shape[:-2])
                q_len, dimension = int(query.shape[-2]), int(query.shape[-1])
                k_len = int(key.shape[-2])
                self.add("scaled_dot_product_attention", 4 * batches * q_len * k_len * dimension)
                return
            if "grid_sampler" in name:
                self.add("grid_sampler", output_numel * 8)
                return
            if "batch_norm" in name or "layer_norm" in name or "group_norm" in name:
                self.add("normalization", output_numel * 2)
                return
            if "max_pool" in name or "avg_pool" in name:
                kernel = args[1] if len(args) > 1 else 1
                if isinstance(kernel, int):
                    kernel_ops = kernel * kernel
                elif isinstance(kernel, (list, tuple)):
                    kernel_ops = math.prod(int(value) for value in kernel)
                else:
                    kernel_ops = 1
                self.add("pooling", output_numel * kernel_ops)
                return
            elementwise_weights = {
                "softmax": 5,
                "sigmoid": 4,
                "silu": 4,
                "gelu": 8,
                "relu": 1,
                "exp": 4,
                "sqrt": 4,
                "mul": 1,
                "div": 1,
                "sub": 1,
                "add": 1,
            }
            for token, weight in elementwise_weights.items():
                if f"aten.{token}" in name or f".{token}." in name:
                    self.add(token, output_numel * weight)
                    return
        except (IndexError, TypeError, AttributeError, ValueError):
            pass
        ignored = ("view", "reshape", "permute", "transpose", "detach", "clone", "copy", "slice", "select", "cat", "stack", "empty", "zeros", "ones", "to.", "_to_copy", "lift_fresh", "alias", "contiguous")
        if output_numel and not any(token in name for token in ignored):
            self.unhandled[name] += output_numel

    def __enter__(self) -> "UnifiedFlopMode":
        self.mode.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> Any:
        return self.mode.__exit__(exc_type, exc_value, traceback)


def distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(q: float) -> float:
        position = (len(ordered) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)

    return {
        "mean_ms": statistics.fmean(ordered),
        "median_ms": statistics.median(ordered),
        "p90_ms": percentile(0.90),
        "p95_ms": percentile(0.95),
        "std_ms": statistics.pstdev(ordered),
    }


def speed_precision(runner: InferenceRunner, precision: str) -> dict[str, Any]:
    import torch
    import numpy as np
    from PIL import Image

    truth = load_annotation("val")
    sample_path = image_root("val") / truth["images"][0]["file_name"]
    with Image.open(sample_path) as source_image:
        source_image = source_image.convert("RGB")
        source_image.thumbnail((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (114, 114, 114))
        canvas.paste(source_image, ((IMAGE_SIZE - source_image.width) // 2, (IMAGE_SIZE - source_image.height) // 2))
        array = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1).copy() / 255.0
    cpu_tensor = torch.from_numpy(array).unsqueeze(0)
    use_amp = precision == "FP16"
    autocast = (lambda: torch.autocast(device_type="cuda", dtype=torch.float16)) if use_amp else nullcontext
    preprocess_times: list[float] = []
    forward_times: list[float] = []
    postprocess_times: list[float] = []
    total_times: list[float] = []

    def iteration(record: bool) -> None:
        start = time.perf_counter()
        tensor = cpu_tensor.to(runner.device)
        torch.cuda.synchronize()
        after_preprocess = time.perf_counter()
        with autocast():
            raw = runner.forward(tensor)
        torch.cuda.synchronize()
        after_forward = time.perf_counter()
        with autocast():
            result = runner.postprocess(raw)
        torch.cuda.synchronize()
        end = time.perf_counter()
        if record:
            preprocess_times.append((after_preprocess - start) * 1000)
            forward_times.append((after_forward - after_preprocess) * 1000)
            postprocess_times.append((end - after_forward) * 1000)
            total_times.append((end - start) * 1000)
        del tensor, raw, result

    with torch.inference_mode():
        for _ in range(WARMUP_ITERS):
            iteration(False)
        for _ in range(MEASURE_ITERS):
            iteration(True)
    total = distribution(total_times)
    return {
        "status": "ok",
        "precision": precision,
        "implementation": "torch.autocast_fp16" if use_amp else "model_fp32",
        "preprocess": distribution(preprocess_times),
        "forward": distribution(forward_times),
        "postprocess": distribution(postprocess_times),
        "end_to_end": total,
        "fps_from_mean_end_to_end": 1000.0 / total["mean_ms"],
    }


def profile(model_key: str, force: bool) -> None:
    import torch

    output = run_dir(model_key) / "speed" / "efficiency_profile.json"
    if output.exists() and not force:
        print(f"Keeping existing efficiency profile: {output}")
        return
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Efficiency worker must see exactly physical GPU0; visible CUDA devices={torch.cuda.device_count()}")
    runner = build_inference_runner(model_key)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sample = torch.rand((1, 3, IMAGE_SIZE, IMAGE_SIZE), device=runner.device)
    counter = UnifiedFlopMode()
    with torch.inference_mode(), counter:
        complexity_output = runner.forward(sample)
    del complexity_output, sample
    torch.cuda.synchronize()
    parameters = sum(int(parameter.numel()) for parameter in runner.model.parameters())
    speed: dict[str, Any] = {}
    for precision in ("FP32", "FP16"):
        try:
            speed[precision] = speed_precision(runner, precision)
        except Exception as error:
            speed[precision] = {"status": "unsupported", "precision": precision, "error": repr(error)}
            if precision == "FP32":
                runner.cleanup()
                raise
    payload = {
        "model": model_key,
        "display": MODEL_SPECS[model_key]["display"],
        "created_at": now_iso(),
        "device": torch.cuda.get_device_name(0),
        "physical_gpu_id": 0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "batch": 1,
        "input_shape": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
        "parameters": parameters,
        "params_M": parameters / 1e6,
        "flops": counter.total,
        "unified_GFLOPs": counter.total / 1e9,
        "MACs_equivalent": counter.total / 2,
        "multiply_add_convention": "2 FLOPs per multiply-add",
        "tool_name": "repository-local TorchDispatch counter",
        "tool_version": UnifiedFlopMode.VERSION,
        "counted_ops": dict(counter.counted),
        "unsupported_ops_top20": counter.unhandled.most_common(20),
        "counted_ops_coverage": "major arithmetic dispatch ops; unsupported operations listed explicitly",
        "complexity_scope": "model forward only; image decode, file I/O and external NMS excluded",
        "speed_protocol": {
            "warmup": WARMUP_ITERS,
            "measurements": MEASURE_ITERS,
            "preprocess": "preloaded FP32 tensor host-to-device transfer; image decode excluded",
            "postprocess": "framework-native NMS/top-300; Faster R-CNN postprocess is internal to model forward",
            "cuda_synchronize": True,
        },
        "speed": speed,
        "peak_cuda_memory_MiB": torch.cuda.max_memory_allocated() / 1024**2,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, payload)
    runner.cleanup()
    print(json.dumps({"model": model_key, "params_M": payload["params_M"], "GFLOPs": payload["unified_GFLOPs"], "FP32_FPS": speed["FP32"].get("fps_from_mean_end_to_end"), "FP16_FPS": speed["FP16"].get("fps_from_mean_end_to_end")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_SPECS))
    parser.add_argument("--mode", required=True, choices=["predict", "profile"])
    parser.add_argument("--split", choices=["val", "test"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    os.environ.setdefault("YOLO_OFFLINE", "true")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("COMET_MODE", "DISABLED")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if args.mode == "predict":
        if not args.split:
            parser.error("--split is required for predict")
        predict(args.model, args.split, args.force)
    else:
        profile(args.model, args.force)


if __name__ == "__main__":
    main()

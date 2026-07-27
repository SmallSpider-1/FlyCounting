"""Frozen RT-DETR-R18 FP32 inference backend shared by cache and validation tools."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "detection_model_benchmark" / "10_rt_detr_r18" / "src" / "rtdetr_pytorch"
RUN_DIR = (
    ROOT
    / "detection_model_benchmark"
    / "results"
    / "fruitfly_detection_v1"
    / "benchmark_640_e100_s42"
    / "runs"
    / "10_rt_detr_r18"
)
DEFAULT_WEIGHTS = RUN_DIR / "checkpoints" / "best.pth"
DEFAULT_RUNTIME_CONFIG = RUN_DIR / "runtime_config.yaml"
DEFAULT_DATASET_CONFIG = ROOT / "detection_model_benchmark" / "10_rt_detr_r18" / "dataset_config.yaml"
CANONICAL_CLASSES = ROOT / "datasets" / "fruitfly_detection_v1" / "metadata" / "classes.yaml"

EXPECTED_CHECKPOINT_SHA256 = "558b7fd2c7b6488649d64cada5d6c860841301d00b2afec7907f29eeef44cc6f"
EXPECTED_SOURCE_COMMIT = "1c8ac3f7ba84f14bd5651ab7b1b70d69a5f55f47"
EXPECTED_IMAGE_SIZE = (640, 640)
PREPROCESS_WORKERS = 8
TORCH_CPU_THREADS = 16
EXPECTED_CLASS_NAMES = {
    0: "Bactrocera dorsalis",
    1: "Bactrocera correcta",
}
EXPECTED_VAL_TRANSFORMS = (
    {"type": "Resize", "size": [640, 640]},
    {"type": "ToImageTensor"},
    {"type": "ConvertDtype"},
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_commit(source_dir: Path = SOURCE_DIR) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _runtime_payload(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"RT-DETR runtime config must be a mapping: {path}")
    return value


def validate_frozen_runtime(runtime_config: Path) -> dict[str, Any]:
    payload = _runtime_payload(runtime_config)
    val_ops = (
        payload.get("val_dataloader", {})
        .get("dataset", {})
        .get("transforms", {})
        .get("ops")
    )
    if tuple(val_ops or ()) != EXPECTED_VAL_TRANSFORMS:
        raise ValueError(
            "Frozen RT-DETR val preprocessing changed; expected exactly "
            "Resize([640,640]) -> ToImageTensor -> ConvertDtype, "
            f"got {val_ops!r}"
        )
    if payload.get("use_ema") is not True:
        raise ValueError("Frozen RT-DETR runtime config must have use_ema=true")

    included = {Path(item).resolve() for item in payload.get("__include__", [])}
    if DEFAULT_DATASET_CONFIG.resolve() not in included:
        raise ValueError(
            f"Frozen runtime config must include {DEFAULT_DATASET_CONFIG}, got {sorted(map(str, included))}"
        )
    dataset_payload = _runtime_payload(DEFAULT_DATASET_CONFIG)
    if int(dataset_payload.get("num_classes", -1)) != 2:
        raise ValueError(
            f"Frozen RT-DETR num_classes must be 2, got {dataset_payload.get('num_classes')!r}"
        )
    if dataset_payload.get("remap_mscoco_category") is not False:
        raise ValueError("Frozen RT-DETR remap_mscoco_category must be false")

    classes_payload = _runtime_payload(CANONICAL_CLASSES)
    actual_names = {
        int(class_id): str(name)
        for class_id, name in classes_payload.get("canonical_yolo_and_deim_rtdetr_ids", {}).items()
    }
    conflicts = [
        (class_id, actual_names.get(class_id), expected_name)
        for class_id, expected_name in EXPECTED_CLASS_NAMES.items()
        if actual_names.get(class_id) != expected_name
    ]
    if conflicts:
        raise ValueError(
            "Canonical class order conflicts with the frozen detector mapping: "
            + "; ".join(
                f"class {class_id}: actual={actual!r}, expected={expected!r}"
                for class_id, actual, expected in conflicts
            )
        )
    return payload


class RTDETRR18Backend:
    """Load the original R18 source and its EMA state for NMS-free FP32 inference."""

    def __init__(
        self,
        weights: Path = DEFAULT_WEIGHTS,
        runtime_config: Path = DEFAULT_RUNTIME_CONFIG,
        device: str = "cuda:0",
    ) -> None:
        self.weights = Path(weights).resolve()
        self.runtime_config = Path(runtime_config).resolve()
        self.device_text = str(device)
        if not self.weights.is_file():
            raise FileNotFoundError(self.weights)
        if not self.runtime_config.is_file():
            raise FileNotFoundError(self.runtime_config)

        self.checkpoint_sha256 = sha256_file(self.weights)
        if self.checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
            raise ValueError(
                "Frozen RT-DETR checkpoint SHA-256 mismatch: "
                f"expected {EXPECTED_CHECKPOINT_SHA256}, got {self.checkpoint_sha256}"
            )
        self.runtime_config_sha256 = sha256_file(self.runtime_config)
        self.runtime_payload = validate_frozen_runtime(self.runtime_config)
        self.source_commit = source_commit()
        if self.source_commit != EXPECTED_SOURCE_COMMIT:
            raise ValueError(
                f"RT-DETR source commit mismatch: expected {EXPECTED_SOURCE_COMMIT}, got {self.source_commit}"
            )

        if str(SOURCE_DIR) not in sys.path:
            sys.path.insert(0, str(SOURCE_DIR))

        import torch
        import torchvision

        torchvision.disable_beta_transforms_warning()
        from torchvision.transforms import v2 as transforms
        from src.core import YAMLConfig

        self.torch = torch
        torch.set_num_threads(TORCH_CPU_THREADS)
        self.device = torch.device(self.device_text)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested but unavailable: {self.device_text}")

        try:
            checkpoint = torch.load(self.weights, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.weights, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise TypeError(f"RT-DETR checkpoint must be a mapping, got {type(checkpoint).__name__}")
        ema = checkpoint.get("ema")
        if not isinstance(ema, dict) or not isinstance(ema.get("module"), dict):
            raise KeyError("Frozen checkpoint must contain EMA weights at checkpoint['ema']['module']")
        self.checkpoint_top_keys = list(checkpoint)
        self.ema_updates = int(ema.get("updates", -1))
        self.ema_warmups = int(ema.get("warmups", -1))
        if self.ema_updates != 50900 or self.ema_warmups != 191:
            raise ValueError(
                "Unexpected frozen EMA metadata: "
                f"updates={self.ema_updates}, warmups={self.ema_warmups}"
            )

        cfg = YAMLConfig(str(self.runtime_config), use_amp=False)
        model = cfg.model
        model.load_state_dict(ema["module"], strict=True)
        self.model = model.to(self.device).float().eval()
        self.postprocessor = cfg.postprocessor.to(self.device).float().eval()
        if int(self.postprocessor.num_top_queries) != 300:
            raise ValueError("Frozen RT-DETR postprocessor must use num_top_queries=300")
        if not bool(self.postprocessor.use_focal_loss):
            raise ValueError("Frozen RT-DETR postprocessor must use focal-loss top-k")
        if int(self.postprocessor.num_classes) != 2:
            raise ValueError("Frozen RT-DETR postprocessor must use num_classes=2")
        if bool(self.postprocessor.remap_mscoco_category):
            raise ValueError("Frozen RT-DETR postprocessor must not remap COCO categories")

        # These are the exact torchvision v2 operations instantiated by the
        # frozen validation loader. PIL input is intentional: tensor resize has
        # different interpolation numerics and does not reproduce val_mixed.
        self.transform = transforms.Compose(
            [
                transforms.Resize(EXPECTED_IMAGE_SIZE),
                transforms.ToImageTensor(),
                transforms.ConvertDtype(),
            ]
        )
        self.preprocess_workers = PREPROCESS_WORKERS
        self.preprocess_executor = ThreadPoolExecutor(
            max_workers=self.preprocess_workers,
            thread_name_prefix="rtdetr-preprocess",
        )
        self.checkpoint_state = "ema.module"

    def metadata(self, batch_size: int, confidence: float) -> dict[str, Any]:
        value = {
            "architecture": "RT-DETR-R18",
            "source_tree": str(SOURCE_DIR),
            "source_commit": self.source_commit,
            "weights": str(self.weights),
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_state": self.checkpoint_state,
            "checkpoint_top_keys": self.checkpoint_top_keys,
            "ema_updates": self.ema_updates,
            "ema_warmups": self.ema_warmups,
            "runtime_config": str(self.runtime_config),
            "runtime_config_sha256": self.runtime_config_sha256,
            "input_size": list(EXPECTED_IMAGE_SIZE),
            "preprocess": [
                "Resize([640,640], direct_stretch)",
                "ToImageTensor",
                "ConvertDtype(float32, scale=True)",
            ],
            "color_input": "RGB",
            "precision": "FP32",
            "amp": False,
            "tta": False,
            "postprocess": "RTDETRPostProcessor",
            "use_focal_loss": True,
            "top_queries": 300,
            "nms": False,
            "nms_iou": "not_applicable",
            "confidence": float(confidence),
            "class_filter": "none",
            "class_names": {str(key): value for key, value in EXPECTED_CLASS_NAMES.items()},
            "orig_target_sizes_order": ["width", "height"],
            "batch_size": int(batch_size),
            "preprocess_workers": self.preprocess_workers,
            "torch_cpu_threads": TORCH_CPU_THREADS,
        }
        return {**value, "detector_config_sha256": stable_sha256(value)}

    def predict_pil(
        self,
        images: Sequence[Any],
        confidence: float,
    ) -> list[list[list[float | int]]]:
        """Return one sorted Nx6 xyxy list per PIL RGB image."""
        if not images:
            return []
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError(f"confidence must be in [0,1], got {confidence}")
        sizes: list[tuple[int, int]] = []
        rgb_images = []
        for image in images:
            if getattr(image, "mode", None) != "RGB":
                raise ValueError("RT-DETR backend requires PIL RGB images")
            width, height = image.size
            sizes.append((int(width), int(height)))
            rgb_images.append(image)
        tensors = list(self.preprocess_executor.map(self.transform, rgb_images))
        for tensor in tensors:
            if tensor.dtype != self.torch.float32 or tuple(tensor.shape) != (3, 640, 640):
                raise RuntimeError(
                    f"Frozen preprocessing produced unexpected tensor {tensor.dtype} {tuple(tensor.shape)}"
                )

        batch = self.torch.stack(tensors).to(self.device, non_blocking=True)
        original_sizes = self.torch.tensor(sizes, dtype=self.torch.float32, device=self.device)
        with self.torch.inference_mode():
            outputs = self.model(batch)
            results = self.postprocessor(outputs, original_sizes)

        batches: list[list[list[float | int]]] = []
        for result in results:
            scores = result["scores"].detach().float().cpu()
            labels = result["labels"].detach().long().cpu()
            boxes = result["boxes"].detach().float().cpu()
            order = scores.argsort(descending=True)[:300]
            rows: list[list[float | int]] = []
            for index in order.tolist():
                score = float(scores[index])
                if score < confidence:
                    continue
                x1, y1, x2, y2 = (float(value) for value in boxes[index].tolist())
                rows.append([x1, y1, x2, y2, score, int(labels[index])])
            batches.append(rows)
        return batches


def require_benchmark_environment() -> None:
    conda_environment = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_environment != "bench_detection_torch201":
        raise RuntimeError(
            "Activate the frozen environment first: conda activate bench_detection_torch201 "
            f"(current CONDA_DEFAULT_ENV={conda_environment!r})"
        )

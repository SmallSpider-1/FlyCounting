#!/usr/bin/env python3
"""Generate one full-frame detection-cache tree shared by all tracker candidates."""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_common.numeric_cache import NumericCacheWriter, file_signature


warnings.filterwarnings("ignore")

DEFAULT_VIDEO_DIR = ROOT / "video_data"
DEFAULT_OUTPUT_DIR = ROOT / "project_results" / "detection_cache"
# 与 datasets/fruitfly_detection_v1/metadata/classes.yaml 的
# canonical_yolo_and_deim_rtdetr_ids 保持一致。
DEFAULT_NAMES = {
    0: "Bactrocera dorsalis",
    1: "Bactrocera correcta",
}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate full-frame YOLO detection caches shared by all tracker candidates."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_VIDEO_DIR, help="单个视频或视频目录。")
    parser.add_argument("--weights", type=Path, default=None, help="冻结检测器的权重路径。")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="检测缓存和汇总结果目录。")
    parser.add_argument("--imgsz", type=int, default=640, help="全帧推理尺寸。")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.1,
        help="共享跟踪缓存的最低置信度；默认 0.10，0.10~0.25 仅供支持低分框的跟踪器补轨迹。",
    )
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值。")
    parser.add_argument("--device", default="0", help="推理设备，例如 0、cpu。")
    parser.add_argument("--half", action="store_true", help="使用 FP16 半精度推理。")
    parser.add_argument("--classes", default="", help="只检测指定类别，逗号分隔；默认不过滤。")
    parser.add_argument(
        "--numeric-cache-dir",
        type=Path,
        default=None,
        help="检测数值缓存目录；默认 <output>/numeric_cache。",
    )
    return parser.parse_args()


def validate_args(args):
    if args.imgsz <= 0:
        raise ValueError("--imgsz 必须大于 0。")
    if not 0 <= args.conf <= 1:
        raise ValueError("--conf 必须位于 [0,1]。")
    if not 0 <= args.iou <= 1:
        raise ValueError("--iou 必须位于 [0,1]。")


def resolve_weights(weights):
    if weights is not None:
        if not weights.exists():
            raise FileNotFoundError(f"模型权重不存在: {weights}")
        return weights

    # 不再隐式回落到 legacy/weights/ 下的历史权重：那些权重的类别顺序与现行
    # 定义相反，静默选中会产出物种对调但不报错的缓存。详见 legacy/README.md。
    raise FileNotFoundError(
        "必须使用 --weights 显式指定冻结后的 .pt 文件；"
        "不要使用 legacy/weights/ 下的历史权重（类别顺序与现行定义相反）。"
    )


def iter_videos(source):
    if not source.exists():
        raise FileNotFoundError(f"视频来源不存在: {source}")
    if source.is_file():
        if source.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f"不支持的视频格式: {source}")
        return [source]

    videos = [
        path
        for path in sorted(source.rglob("*"))
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    ]
    if not videos:
        raise FileNotFoundError(f"目录下没有找到视频文件: {source}")
    return videos


def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return (width, height), fps, frame_count


def parse_classes(classes_text):
    if classes_text is None or not classes_text.strip():
        return None
    return [int(item.strip()) for item in classes_text.split(",") if item.strip()]


def get_class_names(model):
    names = getattr(model, "names", None) or DEFAULT_NAMES
    if isinstance(names, dict):
        resolved = {int(key): str(value) for key, value in names.items()}
    else:
        resolved = {index: str(name) for index, name in enumerate(names)}
    check_class_order(resolved)
    return resolved


def check_class_order(class_names):
    """拒绝类别顺序与现行统一定义冲突的权重。

    下游按物种名字符串匹配真值，若权重的 class ID 与 canonical 定义对调，
    整条链路不会报错，只会得到两个物种数值互换的结果。详见 legacy/README.md。
    """
    conflicts = [
        (class_id, name, DEFAULT_NAMES[class_id])
        for class_id, name in sorted(class_names.items())
        if class_id in DEFAULT_NAMES and name != DEFAULT_NAMES[class_id]
    ]
    if conflicts:
        detail = "；".join(
            f"class {class_id} 权重为 {actual}，应为 {expected}"
            for class_id, actual, expected in conflicts
        )
        raise ValueError(
            f"权重的类别顺序与现行统一定义冲突：{detail}。"
            "现行定义见 datasets/fruitfly_detection_v1/metadata/classes.yaml；"
            "legacy/weights/ 下的历史权重顺序相反，不能直接用于生成缓存。"
        )


def detections_from_results(results):
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy()
        for box, confidence, class_id in zip(xyxy, confidences, class_ids):
            detections.append([*box.tolist(), float(confidence), float(class_id)])
    if not detections:
        return np.empty((0, 6), dtype=np.float32)
    return np.asarray(detections, dtype=np.float32)


def create_detection_cache_writer(
    video_path,
    source_root,
    args,
    class_names,
    frame_size,
    fps,
    reported_frame_count,
):
    cache_dir = args.numeric_cache_dir if args.numeric_cache_dir is not None else args.output / "numeric_cache"
    relative_parent = video_path.parent.resolve().relative_to(source_root)
    cache_path = cache_dir / relative_parent / f"{video_path.stem}.detections.jsonl"
    writer = NumericCacheWriter(
        cache_path,
        "detections",
        {
            "video": file_signature(video_path),
            "frame_size": list(frame_size),
            "fps": round(float(fps), 6),
            "reported_frame_count": int(reported_frame_count),
            "coordinate_space": "full_frame_xyxy",
            "class_names": {str(key): str(value) for key, value in class_names.items()},
            "producer": {
                "stage": "detector",
                "weights": file_signature(args.resolved_weights),
                "imgsz": int(args.imgsz),
                "confidence": float(args.conf),
                "iou": float(args.iou),
                "half": bool(args.half),
                "classes": args.classes or "",
                "input_region": "full_frame",
            },
        },
    )
    return writer, cache_path


def process_video(model, video_path, source_root, args, class_names, selected_classes):
    frame_size, fps, reported_frame_count = get_video_info(video_path)
    writer, cache_path = create_detection_cache_writer(
        video_path,
        source_root,
        args,
        class_names,
        frame_size,
        fps,
        reported_frame_count,
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        writer.abort()
        raise RuntimeError(f"无法打开视频: {video_path}")

    frame_index = 0
    detections_by_class = defaultdict(int)
    print(f"开始生成全帧检测缓存: {video_path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            results = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                half=args.half,
                classes=selected_classes,
                verbose=False,
            )
            detections = detections_from_results(results)
            writer.write_frame(frame_index, detections.tolist())
            for class_id in detections[:, 5].astype(int) if len(detections) else []:
                detections_by_class[int(class_id)] += 1
        writer.close()
    except Exception:
        writer.abort()
        raise
    finally:
        cap.release()

    total_detections = sum(detections_by_class.values())
    print(f"完成检测缓存: {video_path.name}，帧数={frame_index}，检测数={total_detections}，缓存={cache_path}")
    return {
        "video": str(video_path.resolve().relative_to(source_root)),
        "frames": frame_index,
        "fps": round(float(fps), 3),
        "input_width": frame_size[0],
        "input_height": frame_size[1],
        "total_detections": total_detections,
        "detections_by_class": "|".join(
            f"{class_id}={detections_by_class[class_id]}" for class_id in sorted(detections_by_class)
        ),
        "detection_cache": str(cache_path),
        "detection_cache_id": writer.cache_id,
    }


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    validate_args(args)
    args.output.mkdir(parents=True, exist_ok=True)
    args.resolved_weights = resolve_weights(args.weights)
    videos = iter_videos(args.source)
    source_root = args.source.resolve() if args.source.is_dir() else args.source.resolve().parent

    print(f"模型权重: {args.resolved_weights}")
    print(f"视频来源: {args.source}")
    print(f"输出目录: {args.output}")
    print(f"视频数量: {len(videos)}")
    print("当前入口只生成全帧检测缓存，不运行跟踪器、计数器或 ROI 优化。")

    model = YOLO(str(args.resolved_weights))
    try:
        model.fuse()
    except Exception:
        pass
    class_names = get_class_names(model)
    selected_classes = parse_classes(args.classes)
    summary_rows = [
        process_video(model, video_path, source_root, args, class_names, selected_classes)
        for video_path in videos
    ]

    summary_path = args.output / "detection_cache_summary.csv"
    config_path = args.output / "run_config.csv"
    write_csv(
        summary_path,
        [
            "video",
            "frames",
            "fps",
            "input_width",
            "input_height",
            "total_detections",
            "detections_by_class",
            "detection_cache",
            "detection_cache_id",
        ],
        summary_rows,
    )
    cache_dir = args.numeric_cache_dir if args.numeric_cache_dir is not None else args.output / "numeric_cache"
    write_csv(
        config_path,
        [
            "source",
            "source_videos",
            "weights",
            "output",
            "imgsz",
            "conf",
            "iou",
            "device",
            "half",
            "classes",
            "detector_input_region",
            "numeric_cache_dir",
            "downstream_trackers",
        ],
        [
            {
                "source": str(args.source.resolve()),
                "source_videos": "|".join(str(path.resolve()) for path in videos),
                "weights": str(args.resolved_weights.resolve()),
                "output": str(args.output.resolve()),
                "imgsz": args.imgsz,
                "conf": args.conf,
                "iou": args.iou,
                "device": args.device,
                "half": args.half,
                "classes": args.classes,
                "detector_input_region": "full_frame",
                "numeric_cache_dir": str(cache_dir.resolve()),
                "downstream_trackers": "tracking_model_benchmark/01_sort..08_botsort",
            }
        ],
    )
    print(f"检测缓存汇总: {summary_path}")
    print(f"运行配置: {config_path}")


if __name__ == "__main__":
    main()

import argparse
import csv
import importlib
import queue
import sys
import threading
import types
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from track_counting import RegionOccupancyCounter

warnings.filterwarnings("ignore")


ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO_DIR = ROOT / "two_class_61_118"
DEFAULT_OUTPUT_DIR = ROOT / "result_boxmot"


def load_boxmot_bytetrack():
    class BoxMotLogger:
        def debug(self, *args, **kwargs):
            pass

        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            message = args[0] if args else ""
            if message:
                print(f"BoxMOT warning: {message}")

        def success(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            message = args[0] if args else ""
            if message:
                print(f"BoxMOT error: {message}")

        def opt(self, *args, **kwargs):
            return self

    boxmot_dir = ROOT / "boxmot"
    if boxmot_dir.exists():
        package_paths = {
            "boxmot": boxmot_dir,
            "boxmot.trackers": boxmot_dir / "trackers",
            "boxmot.trackers.bytetrack": boxmot_dir / "trackers/bytetrack",
            "boxmot.utils": boxmot_dir / "utils",
        }
        for package_name, package_path in package_paths.items():
            package = sys.modules.get(package_name)
            if package is None:
                package = types.ModuleType(package_name)
                package.__path__ = [str(package_path)]
                if package_name == "boxmot.utils":
                    package.logger = BoxMotLogger()
                sys.modules[package_name] = package
    return importlib.import_module("boxmot.trackers.bytetrack.bytetrack").ByteTrack


ByteTrack = load_boxmot_bytetrack()

DEFAULT_NAMES = {
    0: "Bactrocera correcta",
    1: "Bactrocera dorsalis",
}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v"}


def parse_args():
    parser = argparse.ArgumentParser(description="Batch YOLO predict + BoxMOT ByteTrack counting for videos.")
    parser.add_argument("--source", type=Path, default=DEFAULT_VIDEO_DIR, help="需要处理的视频目录。")
    parser.add_argument("--weights", type=Path, default=None, help="模型权重路径，例如 runs/.../weights/best.pt。")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="结果保存目录。")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸。")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值。")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值。")
    parser.add_argument("--device", default="0", help="推理设备，例如 0、cpu。")
    parser.add_argument("--half", action="store_true", help="使用 FP16 半精度推理，CUDA 上通常更快。")
    parser.add_argument("--classes", default="", help="只跟踪指定类别，逗号分隔；默认不过滤。")
    parser.add_argument("--max-frames", type=int, default=0, help="最多处理多少帧，0 表示不限制；主要用于快速测试。")
    parser.add_argument("--max-seconds", type=float, default=0, help="最多处理合并视频的多少秒，0 表示不限制。")
    parser.add_argument("--sync-write", action="store_true", help="同步写视频/图片；默认异步写出以提高速度。")
    parser.add_argument("--save-raw-event", action="store_true", help="同时保存未标注的事件帧。")
    parser.add_argument("--no-video", action="store_true", help="不写合并视频，用于快速调参。")
    parser.add_argument("--no-event-frames", action="store_true", help="不保存事件帧图片，用于快速调参。")
    parser.add_argument("--count-band", type=float, default=15.0, help="计数圆内外稳定区之间的环带半宽，基于 2304x1296。")
    parser.add_argument("--stitch-max-gap-seconds", type=float, default=0.5, help="跨 ID 轨迹片段允许的最大时间间隔。")
    parser.add_argument("--stitch-max-distance", type=float, default=80.0, help="跨 ID 拼接的最大预测中心距离，基于 2304x1296。")
    parser.add_argument("--stitch-max-size-ratio", type=float, default=2.5, help="跨 ID 拼接允许的最大宽高比例变化。")
    parser.add_argument("--stitch-min-score", type=float, default=0.6, help="跨 ID 拼接的最低综合匹配分数。")
    parser.add_argument("--disable-track-stitching", action="store_true", help="禁用跨 ByteTrack ID 的轨迹片段拼接。")
    parser.add_argument("--recovery-min-gap-seconds", type=float, default=0.1, help="原轨迹不可见多久后才允许断轨离开恢复。")
    parser.add_argument("--recovery-max-size-ratio", type=float, default=2.5, help="断轨离开恢复允许的最大框尺寸比例变化。")
    parser.add_argument("--recovery-min-score", type=float, default=0.4, help="断轨离开恢复的最低综合分数。")
    parser.add_argument("--recovery-class-min-scores", default="", help="按库存类别设置恢复阈值，例如 0:0.3,1:0.9。")
    parser.add_argument("--recovery-require-class-match", action="store_true", help="断轨离开恢复时要求离开轨迹投票类别与库存类别一致。")
    parser.add_argument("--recovery-class-match-required", default="", help="只对指定库存类别要求恢复类别一致，例如 1 或 0,1；为空则使用全局设置。")
    parser.add_argument("--enable-band-start-recovery", action="store_true", help="新轨迹从环带内侧开始并离开外区时，也允许按库存令牌恢复离开。")
    parser.add_argument("--band-start-max-offset", type=float, default=0.0, help="允许环带起始恢复的最大半径偏移，0 表示只允许圆内半侧环带。")
    parser.add_argument("--enable-outer-start-recovery", action="store_true", help="新轨迹从外侧近边界出现并继续向外运动时，也允许按库存令牌恢复离开。")
    parser.add_argument("--outer-start-max-offset", type=float, default=35.0, help="外侧起始恢复允许的最大起始半径偏移，基于 2304x1296。")
    parser.add_argument("--outer-start-min-outward", type=float, default=10.0, help="外侧起始恢复要求的最小向外位移，基于 2304x1296。")
    parser.add_argument("--outer-start-min-frames", type=int, default=2, help="外侧起始恢复至少观察多少帧后才允许扣减。")
    parser.add_argument("--outer-start-max-frames", type=int, default=30, help="外侧起始恢复最多在起始后多少帧内触发。")
    parser.add_argument("--reentry-cooldown-seconds", type=float, default=0.0, help="同一逻辑轨迹离开后，多少秒内再次进入不计数；0 表示关闭。")
    parser.add_argument("--disable-exit-recovery", action="store_true", help="禁用库存令牌的断轨离开恢复。")
    parser.add_argument("--class-vote-weights", default="", help="类别投票权重，例如 0:0.35,1:1.0。")
    parser.add_argument("--entry-detected-class-override", default="", help="指定类别在进入瞬间可用当前检测类别覆盖历史投票，例如 1。")
    return parser.parse_args()


def resolve_weights(weights):
    if weights is not None:
        if not weights.exists():
            raise FileNotFoundError(f"模型权重不存在: {weights}")
        return weights

    candidates = [
        ROOT / "best.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    best_weights = sorted((ROOT / "runs/detect").glob("**/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if best_weights:
        return best_weights[0]

    fallback_weights = [
        ROOT / "yolo8n_two-class.pt",
        ROOT / "yolo26n.pt",
    ]
    for candidate in fallback_weights:
        if candidate.exists():
            return candidate

    raise FileNotFoundError("没有找到模型权重，请使用 --weights 指定 .pt 文件。")


def iter_videos(video_dir):
    if not video_dir.exists():
        raise FileNotFoundError(f"视频目录不存在: {video_dir}")

    videos = [p for p in sorted(video_dir.iterdir()) if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES]
    if not videos:
        raise FileNotFoundError(f"目录下没有找到视频文件: {video_dir}")
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


def create_video_writer(output_path, size, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频: {output_path}")
    return writer


class AsyncVideoWriter:
    def __init__(self, writer, max_queue=256):
        self.writer = writer
        self.items = queue.Queue(maxsize=max_queue)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            frame = self.items.get()
            try:
                if frame is None:
                    return
                self.writer.write(frame)
            finally:
                self.items.task_done()

    def write(self, frame):
        self.items.put(frame)

    def release(self):
        self.items.put(None)
        self.items.join()
        self.thread.join()
        self.writer.release()


class AsyncImageSaver:
    def __init__(self, max_queue=128):
        self.items = queue.Queue(maxsize=max_queue)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            item = self.items.get()
            try:
                if item is None:
                    return
                path, image = item
                cv2.imwrite(str(path), image)
            finally:
                self.items.task_done()

    def save(self, path, image):
        self.items.put((path, image))

    def close(self):
        self.items.put(None)
        self.items.join()
        self.thread.join()


class NullVideoWriter:
    def write(self, frame):
        pass

    def release(self):
        pass


def parse_classes(classes_text):
    if classes_text is None or not classes_text.strip():
        return None
    return [int(item.strip()) for item in classes_text.split(",") if item.strip()]


def parse_class_weights(weights_text):
    if weights_text is None or not weights_text.strip():
        return {}
    weights = {}
    for item in weights_text.split(","):
        class_id, weight = item.split(":", 1)
        weights[int(class_id.strip())] = float(weight.strip())
    return weights


def get_count_region(size):
    base_w, base_h = 2304, 1296
    base_center = (1184, 724)
    base_radius = 150
    width, height = size
    scale_x = width / base_w
    scale_y = height / base_h
    center = (int(base_center[0] * scale_x), int(base_center[1] * scale_y))
    radius = int(base_radius * min(scale_x, scale_y))
    return center, radius


def scale_reference_distance(distance, size):
    base_w, base_h = 2304, 1296
    return float(distance) * min(size[0] / base_w, size[1] / base_h)


def draw_text(image, text, position, font_scale=0.8, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(round(font_scale * 2)))
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    cv2.rectangle(image, (x - 4, y - text_h - 8), (x + text_w + 4, y + baseline + 4), bg_color, -1)
    cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)
    return image


def get_class_names(model):
    names = getattr(model, "names", None) or DEFAULT_NAMES
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(name) for i, name in enumerate(names)}


def detections_from_results(results):
    mot_result = []
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy()
        for box, conf, cls_id in zip(xyxy, confs, clss):
            mot_result.append([*box.tolist(), float(conf), float(cls_id)])
    if not mot_result:
        return np.empty((0, 6), dtype=np.float32)
    return np.asarray(mot_result, dtype=np.float32)


def boxmot_tracks(tracker_output, scale_x=1.0, scale_y=1.0):
    tracks = []
    for row in tracker_output:
        x1, y1, x2, y2, track_id, conf, cls_id = row[:7]
        box = np.array([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y], dtype=np.float32)
        tracks.append((box, int(track_id), int(cls_id), float(conf)))
    return tracks


def draw_count_overlay(frame, class_names, center, radius, band_width, region_counts):
    inner_radius = max(1, int(round(radius - band_width)))
    outer_radius = max(inner_radius + 1, int(round(radius + band_width)))
    cv2.circle(frame, center, inner_radius, (0, 180, 255), 1, lineType=cv2.LINE_AA)
    cv2.circle(frame, center, radius, (0, 0, 255), 3, lineType=cv2.LINE_AA)
    cv2.circle(frame, center, outer_radius, (0, 180, 255), 1, lineType=cv2.LINE_AA)

    y = 34
    draw_text(frame, "Trap region counting by ByteTrack", (10, y), 0.8, (255, 255, 255), (0, 0, 0))
    y += 34
    for cls_id in sorted(class_names):
        name = class_names[cls_id]
        text = f"{name}: count={region_counts[cls_id]}"
        draw_text(frame, text, (10, y), 0.68, (255, 255, 255), (0, 0, 0))
        y += 30
    return frame


def save_image(image_saver, path, image):
    if image_saver is None:
        cv2.imwrite(str(path), image)
    else:
        image_saver.save(path, image.copy())


def save_event_frames(event_dir, event, previous_annotated, current_annotated, previous_raw, current_raw, save_raw, image_saver):
    video_stem = Path(event["video"]).stem
    event_name = (
        f"global{event['global_frame']:08d}_{video_stem}_frame{event['video_frame']:06d}_"
        f"id{event['track_id']}_logical{event['logical_track_id']}_cls{event['cls_id']}_{event['direction']}"
    )
    prev_annotated = previous_annotated if previous_annotated is not None else current_annotated
    save_image(image_saver, event_dir / f"{event_name}_01_prev_annotated.jpg", prev_annotated)
    save_image(image_saver, event_dir / f"{event_name}_02_curr_annotated.jpg", current_annotated)

    if save_raw:
        prev_raw = previous_raw if previous_raw is not None else current_raw
        save_image(image_saver, event_dir / f"{event_name}_01_prev_raw.jpg", prev_raw)
        save_image(image_saver, event_dir / f"{event_name}_02_curr_raw.jpg", current_raw)


def process_video(
    model,
    video_path,
    args,
    class_names,
    combined_writer,
    image_saver,
    output_size,
    region_counts,
    event_rows,
    segment_rows,
    global_state,
    track_classes,
    occupancy_counter,
):
    event_dir = args.output / "event_frames"
    if not args.no_event_frames:
        event_dir.mkdir(parents=True, exist_ok=True)

    input_size, fps, _ = get_video_info(video_path)
    center, radius = get_count_region(output_size)
    tracker = ByteTrack(frame_rate=max(1, int(round(fps))))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    scale_x = output_size[0] / input_size[0]
    scale_y = output_size[1] / input_size[1]

    previous_annotated = None
    previous_raw = None
    video_frame_index = 0
    segment_start_counts = {cls_id: region_counts[cls_id] for cls_id in sorted(class_names)}
    segment_start_global_frame = global_state["frame"] + 1

    while True:
        if global_state["max_frames"] and global_state["frame"] >= global_state["max_frames"]:
            break
        if args.max_frames and video_frame_index >= args.max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        video_frame_index += 1
        global_state["frame"] += 1

        results = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            half=args.half,
            classes=track_classes,
            verbose=False,
        )
        mot_input = detections_from_results(results)
        tracker_output = tracker.update(mot_input, frame)

        need_annotated_frame = not args.no_video or not args.no_event_frames
        raw_frame = frame
        annotated = tracker.plot_results(frame.copy(), show_trajectories=True) if need_annotated_frame else None
        if raw_frame.shape[1] != output_size[0] or raw_frame.shape[0] != output_size[1]:
            raw_frame = cv2.resize(raw_frame, output_size)
            if annotated is not None:
                annotated = cv2.resize(annotated, output_size)
        tracks = boxmot_tracks(tracker_output, scale_x, scale_y)
        _, crossing_events = occupancy_counter.update(
            tracks,
            raw_frame,
            global_state["frame"],
            namespace=video_path.name,
        )
        frame_events = []
        for crossing in crossing_events:
            count_cls_id = crossing["count_cls_id"]
            region_counts[count_cls_id] = max(0, region_counts[count_cls_id] + crossing["delta"])
            center_point = crossing["center"]
            event = {
                "video": video_path.name,
                "video_frame": video_frame_index,
                "global_frame": global_state["frame"],
                "track_id": crossing["raw_track_id"],
                "logical_track_id": crossing["logical_track_id"],
                "cls_id": count_cls_id,
                "class_name": class_names.get(count_cls_id, str(count_cls_id)),
                "detected_cls_id": crossing["cls_id"],
                "voted_cls_id": crossing["voted_class_id"],
                "direction": crossing["direction"],
                "delta": crossing["delta"],
                "count_after": region_counts[count_cls_id],
                "center_x": round(float(center_point[0]), 2),
                "center_y": round(float(center_point[1]), 2),
                "conf": round(float(crossing["confidence"]), 4),
                "stitched": int(crossing["stitched"]),
                "stitch_gap_frames": crossing["stitch_gap_frames"],
                "event_reason": crossing["event_reason"],
                "recovery_mode": crossing.get("recovery_mode", ""),
                "occupancy_token_id": crossing["occupancy_token_id"],
                "recovered_from_logical_id": crossing["recovered_from_logical_id"],
                "recovery_score": crossing.get("recovery_score", ""),
                "recovery_track_class_id": crossing.get("recovery_track_class_id", ""),
                "recovery_class_match": crossing.get("recovery_class_match", ""),
            }
            frame_events.append(event)
            event_rows.append(event)

        if annotated is not None:
            annotated = draw_count_overlay(
                annotated,
                class_names,
                center,
                radius,
                occupancy_counter.band_width,
                region_counts,
            )
            if not args.no_event_frames:
                for event in frame_events:
                    save_event_frames(
                        event_dir,
                        event,
                        previous_annotated,
                        annotated,
                        previous_raw,
                        raw_frame,
                        args.save_raw_event,
                        image_saver,
                    )

            if not args.no_video:
                combined_writer.write(annotated)
            previous_annotated = annotated.copy()
            previous_raw = raw_frame.copy()

    cap.release()

    segment_end_global_frame = global_state["frame"] if video_frame_index else segment_start_global_frame - 1
    for cls_id in sorted(class_names):
        segment_rows.append(
            {
                "video": video_path.name,
                "class_id": cls_id,
                "class_name": class_names[cls_id],
                "start_region_count": segment_start_counts[cls_id],
                "end_region_count": region_counts[cls_id],
                "frames": video_frame_index,
                "fps": round(float(fps), 3),
                "global_frame_start": segment_start_global_frame,
                "global_frame_end": segment_end_global_frame,
                "combined_video": "" if args.no_video else str(args.output / "combined_track_count.mp4"),
                "event_frames_dir": str(event_dir),
            }
        )

    print(f"完成: {video_path.name}，帧数: {video_frame_index}，连续区域计数: {dict(region_counts)}")


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    weights = resolve_weights(args.weights)
    videos = iter_videos(args.source)

    first_size, first_fps, _ = get_video_info(videos[0])
    combined_video = args.output / "combined_track_count.mp4"
    if args.no_video:
        combined_writer = NullVideoWriter()
    else:
        video_writer = create_video_writer(combined_video, first_size, first_fps)
        combined_writer = video_writer if args.sync_write else AsyncVideoWriter(video_writer)
    image_saver = None if args.sync_write or args.no_event_frames else AsyncImageSaver()

    print(f"模型权重: {weights}")
    print(f"视频目录: {args.source}")
    print(f"输出目录: {args.output}")
    print(f"合并视频: {'disabled' if args.no_video else combined_video}")
    print(f"视频数量: {len(videos)}")
    max_global_frames = int(round(first_fps * args.max_seconds)) if args.max_seconds else 0
    if max_global_frames:
        print(f"测试截断: {args.max_seconds:g} 秒，约 {max_global_frames} 帧")

    model = YOLO(str(weights))
    try:
        model.fuse()
    except Exception:
        pass
    class_names = get_class_names(model)
    track_classes = parse_classes(args.classes)
    center, radius = get_count_region(first_size)
    occupancy_counter = RegionOccupancyCounter(
        center=center,
        radius=radius,
        band_width=scale_reference_distance(args.count_band, first_size),
        stitch_max_gap_frames=max(1, int(round(first_fps * args.stitch_max_gap_seconds))),
        stitch_max_distance=scale_reference_distance(args.stitch_max_distance, first_size),
        stitch_max_size_ratio=args.stitch_max_size_ratio,
        stitch_min_score=args.stitch_min_score,
        enable_stitching=not args.disable_track_stitching,
        recovery_min_gap_frames=max(1, int(round(first_fps * args.recovery_min_gap_seconds))),
        recovery_max_size_ratio=args.recovery_max_size_ratio,
        recovery_min_score=args.recovery_min_score,
        recovery_class_match_required=parse_classes(args.recovery_class_match_required),
        recovery_class_min_scores=parse_class_weights(args.recovery_class_min_scores),
        recovery_require_class_match=args.recovery_require_class_match,
        enable_band_start_recovery=args.enable_band_start_recovery,
        band_start_max_offset=scale_reference_distance(args.band_start_max_offset, first_size),
        enable_outer_start_recovery=args.enable_outer_start_recovery,
        outer_start_max_offset=scale_reference_distance(args.outer_start_max_offset, first_size),
        outer_start_min_outward=scale_reference_distance(args.outer_start_min_outward, first_size),
        outer_start_min_frames=args.outer_start_min_frames,
        outer_start_max_frames=args.outer_start_max_frames,
        reentry_cooldown_frames=int(round(first_fps * args.reentry_cooldown_seconds)),
        enable_exit_recovery=not args.disable_exit_recovery,
        class_vote_weights=parse_class_weights(args.class_vote_weights),
        entry_detected_class_override=parse_classes(args.entry_detected_class_override),
    )
    print(
        "计数优化: "
        f"band={occupancy_counter.band_width:.1f}px, "
        f"stitch_gap={occupancy_counter.stitch_max_gap_frames}帧, "
        f"stitch_distance={occupancy_counter.stitch_max_distance:.1f}px, "
        f"stitch_score={occupancy_counter.stitch_min_score:.2f}, "
        f"recovery_gap={occupancy_counter.recovery_min_gap_frames}帧, "
        f"recovery_score={occupancy_counter.recovery_min_score:.2f}, "
        f"recovery_class_scores={occupancy_counter.recovery_class_min_scores or 'default'}, "
        f"recovery_class_match={occupancy_counter.recovery_require_class_match}, "
        f"recovery_class_match_required={occupancy_counter.recovery_class_match_required or 'default'}, "
        f"band_start_recovery={occupancy_counter.enable_band_start_recovery}, "
        f"outer_start_recovery={occupancy_counter.enable_outer_start_recovery}, "
        f"reentry_cooldown={occupancy_counter.reentry_cooldown_frames}帧, "
        f"class_weights={occupancy_counter.class_vote_weights or 'default'}, "
        f"entry_detected_override={occupancy_counter.entry_detected_class_override or 'default'}"
    )
    region_counts = defaultdict(int)
    event_rows = []
    segment_rows = []
    global_state = {"frame": 0, "max_frames": max_global_frames}

    try:
        for video_path in videos:
            if global_state["max_frames"] and global_state["frame"] >= global_state["max_frames"]:
                break
            process_video(
                model,
                video_path,
                args,
                class_names,
                combined_writer,
                image_saver,
                first_size,
                region_counts,
                event_rows,
                segment_rows,
                global_state,
                track_classes,
                occupancy_counter,
            )
    finally:
        combined_writer.release()
        if image_saver is not None:
            image_saver.close()

    summary_csv = args.output / "summary_counts.csv"
    events_csv = args.output / "count_events.csv"
    final_csv = args.output / "final_counts.csv"

    write_csv(
        summary_csv,
        [
            "video",
            "class_id",
            "class_name",
            "start_region_count",
            "end_region_count",
            "frames",
            "fps",
            "global_frame_start",
            "global_frame_end",
            "combined_video",
            "event_frames_dir",
        ],
        segment_rows,
    )
    write_csv(
        events_csv,
        [
            "video",
            "video_frame",
            "global_frame",
            "track_id",
            "logical_track_id",
            "cls_id",
            "class_name",
            "detected_cls_id",
            "voted_cls_id",
            "direction",
            "delta",
            "count_after",
            "center_x",
            "center_y",
            "conf",
            "stitched",
            "stitch_gap_frames",
            "event_reason",
            "recovery_mode",
            "occupancy_token_id",
            "recovered_from_logical_id",
            "recovery_score",
            "recovery_track_class_id",
            "recovery_class_match",
        ],
        event_rows,
    )
    write_csv(
        final_csv,
        ["class_id", "class_name", "final_region_count"],
        [{"class_id": cls_id, "class_name": class_names[cls_id], "final_region_count": region_counts[cls_id]} for cls_id in sorted(class_names)],
    )

    print(f"全部处理完成，合并视频: {'disabled' if args.no_video else combined_video}")
    print(f"分段汇总: {summary_csv}")
    print(f"计数事件: {events_csv}")
    print(f"最终计数: {final_csv}")


if __name__ == "__main__":
    main()

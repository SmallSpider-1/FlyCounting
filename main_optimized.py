import argparse
import csv
import hashlib
import importlib
import json
import queue
import sys
import threading
import time
import types
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from counting_experiments import (
    STRATEGY_BASELINE,
    STRATEGY_CHOICES,
    SlowFastCircleCounter,
    get_square_roi_bounds,
    get_strategy_spec,
    offset_detections,
)

warnings.filterwarnings("ignore")


ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO_DIR = ROOT / "two_class_61_118"
DEFAULT_OUTPUT_DIR = ROOT / "project_results" / "result_boxmot"


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
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_VIDEO_DIR,
        help="单个视频或视频目录；目录模式保留历史连续累计，逐视频实验请使用运行脚本。",
    )
    parser.add_argument("--weights", type=Path, default=None, help="模型权重路径，例如 runs/.../weights/best.pt。")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="结果保存目录。")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸。")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值。")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值。")
    parser.add_argument("--device", default="0", help="推理设备，例如 0、cpu。")
    parser.add_argument("--half", action="store_true", help="使用 FP16 半精度推理，CUDA 上通常更快。")
    parser.add_argument("--classes", default="", help="只跟踪指定类别，逗号分隔；默认不过滤。")
    parser.add_argument("--sync-write", action="store_true", help="同步写视频/图片；默认异步写出以提高速度。")
    parser.add_argument("--save-raw-event", action="store_true", help="同时保存未标注的事件帧。")
    parser.add_argument("--csv-only", action="store_true", help="只输出 CSV，不写合并视频和事件帧，速度更快。")
    parser.add_argument(
        "--strategy",
        choices=STRATEGY_CHOICES,
        default=STRATEGY_BASELINE,
        help="消融策略：B0 冻结基准；E1 仅加圆周 ROI；E2 再加时序/直接跨界确认；E3 再加类别投票与进入类别锁定。",
    )
    parser.add_argument("--roi-scale", type=float, default=2.0, help="ROI 正方形边长相对圆直径的倍数，默认 2.0。")
    parser.add_argument(
        "--boundary-margin",
        type=float,
        default=8.0,
        help="E2/E3 的边界容差（按 2304x1296 原图像素定义），不改变标称圆。",
    )
    parser.add_argument("--confirm-frames", type=int, default=3, help="E2/E3 慢速通道在目标一侧连续确认帧数。")
    parser.add_argument("--slow-max-gap", type=int, default=2, help="慢速通道连续观测允许的最大帧间隔。")
    parser.add_argument("--fast-gap-seconds", type=float, default=0.12, help="快速飞入/飞出允许跨越边界的最长时间。")
    parser.add_argument(
        "--fast-min-radial",
        type=float,
        default=0.0,
        help="快速通道最小径向位移；0 表示自动取 3 倍 boundary-margin。",
    )
    parser.add_argument("--class-vote-window", type=int, default=15, help="E3 每条原始 ID 最多保留的类别投票观测数。")
    parser.add_argument(
        "--class-vote-max-age-seconds",
        type=float,
        default=0.5,
        help="E3 类别投票观测的最长保留时间。",
    )
    parser.add_argument(
        "--tracks-dir",
        type=Path,
        default=None,
        help="轨迹缓存目录；配合 --dump-tracks 写出，或 --from-tracks 读入。默认 <output>/track_cache。",
    )
    parser.add_argument(
        "--dump-tracks",
        action="store_true",
        help="正常跑检测+跟踪，同时把每帧每条轨迹的最小观测写入缓存，供之后 --from-tracks 重放计数。",
    )
    parser.add_argument(
        "--from-tracks",
        action="store_true",
        help="跳过 YOLO 与 ByteTrack，直接读轨迹缓存重放计数；检测/跟踪配置改变时指纹校验会拒绝。",
    )
    return parser.parse_args()


def validate_args(args):
    if args.roi_scale <= 1.0:
        raise ValueError("--roi-scale 必须大于 1.0，确保圆外仍有跟踪空间。")
    if not 0 < args.boundary_margin < 150:
        raise ValueError("--boundary-margin 必须大于 0 且小于基准半径 150。")
    if args.confirm_frames < 1 or args.slow_max_gap < 1:
        raise ValueError("--confirm-frames 和 --slow-max-gap 必须至少为 1。")
    if args.fast_gap_seconds <= 0 or args.fast_min_radial < 0:
        raise ValueError("--fast-gap-seconds 必须大于 0，--fast-min-radial 不能小于 0。")
    if args.class_vote_window < 1 or args.class_vote_max_age_seconds <= 0:
        raise ValueError("类别投票窗口必须至少为 1，最长保留时间必须大于 0。")
    if args.dump_tracks and args.from_tracks:
        raise ValueError("--dump-tracks 与 --from-tracks 不能同时使用：一个写缓存，一个读缓存。")
    if args.from_tracks and not args.csv_only:
        # 重放没有图像帧，无法绘制合并视频或事件帧。
        raise ValueError("--from-tracks 只重放计数，请同时加 --csv-only（重放不产生视频/事件帧）。")


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


def iter_videos(source):
    if not source.exists():
        raise FileNotFoundError(f"视频或目录不存在: {source}")
    if source.is_file():
        if source.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f"不支持的视频格式: {source}")
        return [source]

    videos = [p for p in sorted(source.iterdir()) if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES]
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


def parse_classes(classes_text):
    if classes_text is None or not classes_text.strip():
        return None
    return [int(item.strip()) for item in classes_text.split(",") if item.strip()]


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


# 轨迹缓存版本；缓存文件结构变化时递增，读到旧版本会被指纹校验拒绝。
TRACK_CACHE_VERSION = 1
# 跟踪器签名：ByteTrack 的关键默认参数。改跟踪器就必须重跑，缓存随之失效。
TRACKER_SIG = (
    "boxmot-bytetrack;min_conf=0.1;track_thresh=0.45;match_thresh=0.8;track_buffer=25;"
    "det_thresh=0.3;max_age=30;max_obs=50;min_hits=3;iou_threshold=0.3;per_class=False;asso_func=iou"
)


def weights_signature(weights_path):
    """用路径+大小+修改时间标识权重；换权重或重训后缓存会失配。"""
    st = weights_path.stat()
    return {"path": str(weights_path.resolve()), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def build_track_fingerprint(args, strategy_spec, weights, roi_bounds, input_size, fps):
    """检测+跟踪输出唯一由这些量决定；计数参数不在其中，所以改计数不会失配缓存。"""
    return {
        "cache_version": TRACK_CACHE_VERSION,
        "weights": weights_signature(weights),
        "imgsz": int(args.imgsz),
        "conf": float(args.conf),
        "iou": float(args.iou),
        "half": bool(args.half),
        "classes": args.classes or "",
        "use_roi": bool(strategy_spec.use_roi),
        "roi_scale": float(args.roi_scale) if strategy_spec.use_roi else None,
        "roi_bounds": list(roi_bounds) if roi_bounds is not None else None,
        "input_size": list(input_size),
        "fps": round(float(fps), 6),
        "tracker_sig": TRACKER_SIG,
    }


def fingerprint_hash(fingerprint):
    payload = json.dumps(fingerprint, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def track_cache_path(tracks_dir, video_path, fingerprint):
    detect_tag = "roi" if fingerprint["use_roi"] else "full"
    return Path(tracks_dir) / f"{video_path.stem}__{detect_tag}__{fingerprint_hash(fingerprint)}.jsonl"


# 指纹中必须逐字段匹配才允许复用缓存的键；device 不在内（只有 half 会改变数值结果）。
FINGERPRINT_STRICT_KEYS = (
    "cache_version",
    "weights",
    "imgsz",
    "conf",
    "iou",
    "half",
    "classes",
    "use_roi",
    "roi_scale",
    "roi_bounds",
    "input_size",
    "fps",
    "tracker_sig",
)


def write_track_cache(path, fingerprint, video_name, class_names, total_frames, cache_frames):
    """一行 JSON header（含指纹）+ 每帧一行观测。观测仅存计数需要的最小列。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = dict(fingerprint)
    header["video"] = video_name
    header["class_names"] = {str(k): str(v) for k, v in class_names.items()}
    header["total_frames"] = int(total_frames)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"header": header}, ensure_ascii=False) + "\n")
        for frame_idx, observations in cache_frames:
            f.write(json.dumps({"f": int(frame_idx), "o": observations}, ensure_ascii=False) + "\n")


def read_track_cache(path):
    with open(path, encoding="utf-8") as f:
        header = json.loads(f.readline())["header"]
        frames = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            frames.append((int(record["f"]), record["o"]))
    return header, frames


def fingerprint_diffs(expected, header):
    """返回 (字段, 期望值, 缓存值) 列表；空列表表示缓存与当前检测/跟踪配置一致。"""
    diffs = []
    for key in FINGERPRINT_STRICT_KEYS:
        if header.get(key) != expected.get(key):
            diffs.append((key, expected.get(key), header.get(key)))
    return diffs


def class_names_from_header(header):
    return {int(k): str(v) for k, v in header.get("class_names", {}).items()}


def _class_names_for_replay(args, videos, strategy_spec):
    """重放不加载模型，从任一视频的缓存 header 取类别名；缺失则回退到 DEFAULT_NAMES。"""
    tracks_dir = args.tracks_dir if args.tracks_dir is not None else (args.output / "track_cache")
    detect_tag = "roi" if strategy_spec.use_roi else "full"
    for video_path in videos:
        for candidate in sorted(Path(tracks_dir).glob(f"{video_path.stem}__{detect_tag}__*.jsonl")):
            try:
                with open(candidate, encoding="utf-8") as f:
                    header = json.loads(f.readline())["header"]
                names = class_names_from_header(header)
                if names:
                    return names
            except (OSError, KeyError, json.JSONDecodeError):
                continue
    return dict(DEFAULT_NAMES)


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


def point_in_circle(point, center, radius):
    return np.hypot(point[0] - center[0], point[1] - center[1]) <= radius


def point_circle_state(point, center, radius):
    # 二元判断：只有圈内 / 圈外两种状态，边界（含）算圈内。
    return "inside" if point_in_circle(point, center, radius) else "outside"


def build_confirmed_counter(args, radius, fps, use_class_vote, lock_entry_class):
    reference_scale = float(radius) / 150.0
    margin = args.boundary_margin * reference_scale
    base_fast_min = args.fast_min_radial if args.fast_min_radial > 0 else 3.0 * args.boundary_margin
    fast_min_radial = base_fast_min * reference_scale
    fast_max_gap_frames = max(1, int(round(args.fast_gap_seconds * fps)))
    vote_max_age_frames = max(1, int(round(args.class_vote_max_age_seconds * fps)))
    counter = SlowFastCircleCounter(
        radius=radius,
        margin=margin,
        confirm_frames=args.confirm_frames,
        slow_max_observation_gap=args.slow_max_gap,
        fast_max_gap_frames=fast_max_gap_frames,
        fast_min_radial_displacement=fast_min_radial,
        use_class_vote=use_class_vote,
        lock_entry_class=lock_entry_class,
        vote_window=args.class_vote_window,
        vote_max_age_frames=vote_max_age_frames,
    )
    return counter, {
        "boundary_margin_px": round(margin, 3),
        "fast_max_gap_frames": fast_max_gap_frames,
        "fast_min_radial_px": round(fast_min_radial, 3),
        "vote_max_age_frames": vote_max_age_frames,
    }


def draw_count_overlay(frame, class_names, center, radius, region_counts):
    cv2.circle(frame, center, radius, (0, 0, 255), 3, lineType=cv2.LINE_AA)

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
        f"id{event['track_id']}_cls{event['cls_id']}_{event['direction']}"
    )
    prev_annotated = previous_annotated if previous_annotated is not None else current_annotated
    save_image(image_saver, event_dir / f"{event_name}_01_prev_annotated.jpg", prev_annotated)
    save_image(image_saver, event_dir / f"{event_name}_02_curr_annotated.jpg", current_annotated)

    if save_raw:
        prev_raw = previous_raw if previous_raw is not None else current_raw
        save_image(image_saver, event_dir / f"{event_name}_01_prev_raw.jpg", prev_raw)
        save_image(image_saver, event_dir / f"{event_name}_02_curr_raw.jpg", current_raw)


def count_frame_observations(
    observations,
    video_frame_index,
    global_frame,
    center,
    radius,
    confirmed_counter,
    strategy_spec,
    track_circle_state,
    region_counts,
    video_stats,
    class_names,
    video_name,
    strategy_name,
    event_rows,
):
    """单帧计数：把观测列表送进策略计数器，产出事件字典列表。

    观测项为 (track_id, cls_id, conf, center_x, center_y)。实时路径与缓存重放
    路径都调这里，计数逻辑只有一份，两条路径结果逐位一致。
    """
    frame_events = []
    for track_id, cls_id, conf, cx, cy in observations:
        center_point = (float(cx), float(cy))
        distance = float(np.hypot(center_point[0] - center[0], center_point[1] - center[1]))

        if confirmed_counter is None:
            current_state = point_circle_state(center_point, center, radius)
            previous_state = track_circle_state.get(track_id)
            # B0/E1 基准逻辑：只比较该原始 ID 上一观测和当前观测与红圈的位置关系。
            if previous_state == "outside" and current_state == "inside":
                delta = 1
                direction = "enter"
            elif previous_state == "inside" and current_state == "outside":
                delta = -1
                direction = "exit"
            else:
                delta = 0
            track_circle_state[track_id] = current_state
            count_cls_id = cls_id
            event_reason = "instant"
            entry_cls_id = ""
            vote_n = ""
            vote_score = ""
            vote_share = ""
            paired_exit = ""
            class_source = "trigger_frame"
            diagnostic_reason = ""
        else:
            decision = confirmed_counter.update(
                track_id=track_id,
                frame_index=video_frame_index,
                distance=distance,
                cls_id=cls_id,
                conf=conf,
            )
            if decision is None or not decision.accepted:
                continue
            delta = decision.delta
            direction = decision.direction
            count_cls_id = decision.cls_id
            event_reason = decision.trigger
            current_state = decision.zone
            entry_cls_id = decision.entry_cls_id if decision.entry_cls_id is not None else ""
            vote_n = decision.vote_n
            vote_score = decision.vote_score
            vote_share = decision.vote_share
            if not strategy_spec.use_class_vote:
                vote_n = ""
                vote_score = ""
                vote_share = ""
            paired_exit = decision.paired_exit if direction == "exit" else ""
            class_source = decision.class_source
            diagnostic_reason = decision.diagnostic_reason

        if delta:
            region_counts[count_cls_id] += delta
            video_stats[f"accepted_{direction}_{event_reason}"] += 1
            if direction == "exit" and confirmed_counter is not None:
                video_stats["paired_exit"] += int(bool(paired_exit))
                video_stats["unpaired_exit"] += int(not paired_exit)
            event = {
                "video": video_name,
                "strategy": strategy_name,
                "video_frame": video_frame_index,
                "global_frame": global_frame,
                "track_id": track_id,
                "cls_id": count_cls_id,
                "class_name": class_names.get(count_cls_id, str(count_cls_id)),
                "direction": direction,
                "delta": delta,
                "count_after": region_counts[count_cls_id],
                "center_x": round(center_point[0], 2),
                "center_y": round(center_point[1], 2),
                "radial_distance": round(distance, 3),
                "conf": round(float(conf), 4),
                "raw_cls_id": cls_id,
                "entry_cls_id": entry_cls_id,
                "event_reason": event_reason,
                "circle_state": current_state,
                "vote_n": vote_n,
                "vote_score": "" if vote_score == "" else round(float(vote_score), 4),
                "vote_share": "" if vote_share == "" else round(float(vote_share), 4),
                "paired_exit": paired_exit,
                "class_source": class_source,
                "diagnostic_reason": diagnostic_reason,
            }
            frame_events.append(event)
            event_rows.append(event)
    return frame_events


def append_segment_and_metric_rows(
    video_name,
    strategy_name,
    class_names,
    region_counts,
    segment_start_counts,
    segment_rows,
    video_metric_rows,
    video_frame_index,
    fps,
    input_size,
    center,
    radius,
    roi_text,
    strategy_spec,
    strategy_params,
    args,
    video_stats,
    counter_stats,
    segment_start_global_frame,
    segment_end_global_frame,
    processing_seconds,
    combined_video_field,
    event_frames_field,
):
    """写 summary/segment 行与 video_metrics 行；实时与重放路径共用，保证列一致。"""
    for cls_id in sorted(class_names):
        segment_rows.append(
            {
                "video": video_name,
                "strategy": strategy_name,
                "class_id": cls_id,
                "class_name": class_names[cls_id],
                "start_region_count": segment_start_counts[cls_id],
                "end_region_count": region_counts[cls_id],
                "frames": video_frame_index,
                "fps": round(float(fps), 3),
                "global_frame_start": segment_start_global_frame,
                "global_frame_end": segment_end_global_frame,
                "combined_video": combined_video_field,
                "event_frames_dir": event_frames_field,
            }
        )

    video_metric_rows.append(
        {
            "video": video_name,
            "strategy": strategy_name,
            "frames": video_frame_index,
            "fps": round(float(fps), 3),
            "input_width": input_size[0],
            "input_height": input_size[1],
            "circle_center_x": center[0],
            "circle_center_y": center[1],
            "circle_radius": radius,
            "roi_bounds": roi_text,
            "boundary_margin_px": strategy_params["boundary_margin_px"],
            "confirm_frames": args.confirm_frames if strategy_spec.use_confirmed_crossing else "",
            "slow_max_gap_frames": args.slow_max_gap if strategy_spec.use_confirmed_crossing else "",
            "fast_max_gap_frames": strategy_params["fast_max_gap_frames"],
            "fast_min_radial_px": strategy_params["fast_min_radial_px"],
            "vote_window": args.class_vote_window if strategy_spec.use_class_vote else "",
            "vote_max_age_frames": strategy_params["vote_max_age_frames"] if strategy_spec.use_class_vote else "",
            "accepted_enter_total": sum(
                video_stats[key]
                for key in ("accepted_enter_instant", "accepted_enter_slow", "accepted_enter_fast")
            ),
            "accepted_exit_total": sum(
                video_stats[key]
                for key in ("accepted_exit_instant", "accepted_exit_slow", "accepted_exit_fast")
            ),
            "accepted_enter_instant": video_stats["accepted_enter_instant"],
            "accepted_exit_instant": video_stats["accepted_exit_instant"],
            "accepted_enter_slow": video_stats["accepted_enter_slow"],
            "accepted_exit_slow": video_stats["accepted_exit_slow"],
            "accepted_enter_fast": video_stats["accepted_enter_fast"],
            "accepted_exit_fast": video_stats["accepted_exit_fast"],
            "paired_exit": video_stats["paired_exit"],
            "unpaired_exit": video_stats["unpaired_exit"],
            "overwritten_entry_credit": counter_stats["overwritten_entry_credit"],
            "processing_seconds": round(processing_seconds, 3),
            "processing_fps": round(video_frame_index / processing_seconds, 3) if processing_seconds > 0 else 0.0,
        }
    )


def replay_video_from_cache(
    video_path,
    args,
    class_names,
    region_counts,
    event_rows,
    segment_rows,
    video_metric_rows,
    global_state,
):
    """从轨迹缓存重放计数：不开视频、不跑 YOLO/ByteTrack，只重算计数策略。"""
    strategy_spec = get_strategy_spec(args.strategy)
    tracks_dir = args.tracks_dir if args.tracks_dir is not None else (args.output / "track_cache")

    input_size, fps, roi_bounds = _resolve_replay_geometry(tracks_dir, video_path, args, strategy_spec)
    input_center, input_radius = get_count_region(input_size)
    center, radius = input_center, input_radius

    fingerprint = build_track_fingerprint(args, strategy_spec, args._resolved_weights, roi_bounds, input_size, fps)
    cache_file = track_cache_path(tracks_dir, video_path, fingerprint)
    if not cache_file.exists():
        raise FileNotFoundError(
            f"找不到轨迹缓存: {cache_file}\n请先用相同检测/跟踪配置执行 --dump-tracks 生成缓存。"
        )
    header, cache_frames = read_track_cache(cache_file)
    diffs = fingerprint_diffs(fingerprint, header)
    if diffs:
        detail = "\n".join(f"  - {key}: 期望 {want!r}，缓存 {got!r}" for key, want, got in diffs)
        raise ValueError(
            f"轨迹缓存与当前检测/跟踪配置不一致，拒绝重放（防止用过期缓存）:\n{cache_file}\n{detail}\n"
            "检测或跟踪配置变了就必须重新 --dump-tracks。"
        )

    confirmed_counter = None
    strategy_params = {
        "boundary_margin_px": "",
        "fast_max_gap_frames": "",
        "fast_min_radial_px": "",
        "vote_max_age_frames": "",
    }
    if strategy_spec.use_confirmed_crossing:
        confirmed_counter, strategy_params = build_confirmed_counter(
            args, radius, fps, strategy_spec.use_class_vote, strategy_spec.lock_entry_class
        )

    track_circle_state = {}
    video_stats = Counter()
    video_frame_index = 0
    segment_start_counts = {cls_id: region_counts[cls_id] for cls_id in sorted(class_names)}
    segment_start_global_frame = global_state["frame"] + 1
    processing_started = time.perf_counter()

    roi_text = "full-frame" if roi_bounds is None else ",".join(str(value) for value in roi_bounds)
    print(f"重放: {video_path.name}，策略: {args.strategy}，缓存: {cache_file.name}")

    for frame_idx, observations in cache_frames:
        video_frame_index += 1
        global_state["frame"] += 1
        obs_tuples = [(int(o["id"]), int(o["cls"]), float(o["conf"]), float(o["cx"]), float(o["cy"])) for o in observations]
        count_frame_observations(
            obs_tuples,
            video_frame_index,
            global_state["frame"],
            center,
            radius,
            confirmed_counter,
            strategy_spec,
            track_circle_state,
            region_counts,
            video_stats,
            class_names,
            video_path.name,
            args.strategy,
            event_rows,
        )

    processing_seconds = time.perf_counter() - processing_started
    segment_end_global_frame = global_state["frame"] if video_frame_index else segment_start_global_frame - 1
    counter_stats = confirmed_counter.stats if confirmed_counter is not None else Counter()
    append_segment_and_metric_rows(
        video_path.name,
        args.strategy,
        class_names,
        region_counts,
        segment_start_counts,
        segment_rows,
        video_metric_rows,
        video_frame_index,
        fps,
        input_size,
        center,
        radius,
        roi_text,
        strategy_spec,
        strategy_params,
        args,
        video_stats,
        counter_stats,
        segment_start_global_frame,
        segment_end_global_frame,
        processing_seconds,
        "",
        "",
    )
    print(f"重放完成: {video_path.name}，帧数: {video_frame_index}，连续区域计数: {dict(region_counts)}")


def _resolve_replay_geometry(tracks_dir, video_path, args, strategy_spec):
    """重放优先从缓存 header 拿几何信息，避免打开视频；缓存缺失时回退到视频。"""
    # 先按当前配置尝试匹配缓存文件名前缀，读取 header 里的 input_size/fps。
    detect_tag = "roi" if strategy_spec.use_roi else "full"
    candidates = sorted(Path(tracks_dir).glob(f"{video_path.stem}__{detect_tag}__*.jsonl"))
    for candidate in candidates:
        try:
            with open(candidate, encoding="utf-8") as f:
                header = json.loads(f.readline())["header"]
            input_size = tuple(header["input_size"])
            fps = float(header["fps"])
            roi_bounds = tuple(header["roi_bounds"]) if header.get("roi_bounds") is not None else None
            return input_size, fps, roi_bounds
        except (OSError, KeyError, json.JSONDecodeError):
            continue
    # 回退：缓存不可用时读视频元信息（仍可给出清晰的指纹失配报错）。
    input_size, fps, _ = get_video_info(video_path)
    input_center, input_radius = get_count_region(input_size)
    roi_bounds = (
        get_square_roi_bounds(input_size, input_center, input_radius, args.roi_scale)
        if strategy_spec.use_roi
        else None
    )
    return input_size, fps, roi_bounds


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
    video_metric_rows,
    global_state,
    track_classes,
):
    event_dir = args.output / "event_frames"
    if not args.csv_only:
        event_dir.mkdir(parents=True, exist_ok=True)

    input_size, fps, _ = get_video_info(video_path)
    strategy_spec = get_strategy_spec(args.strategy)
    input_center, input_radius = get_count_region(input_size)
    center, radius = input_center, input_radius
    roi_bounds = None
    if strategy_spec.use_roi:
        roi_bounds = get_square_roi_bounds(input_size, input_center, input_radius, args.roi_scale)

    # --dump-tracks：在实时跑的同时把每帧最小观测缓存下来，供之后 --from-tracks 重放。
    dump_tracks = getattr(args, "dump_tracks", False)
    cache_frames = [] if dump_tracks else None
    track_fingerprint = None
    cache_file = None
    if dump_tracks:
        tracks_dir = args.tracks_dir if args.tracks_dir is not None else (args.output / "track_cache")
        track_fingerprint = build_track_fingerprint(
            args, strategy_spec, args._resolved_weights, roi_bounds, input_size, fps
        )
        cache_file = track_cache_path(tracks_dir, video_path, track_fingerprint)

    confirmed_counter = None
    strategy_params = {
        "boundary_margin_px": "",
        "fast_max_gap_frames": "",
        "fast_min_radial_px": "",
        "vote_max_age_frames": "",
    }
    if strategy_spec.use_confirmed_crossing:
        confirmed_counter, strategy_params = build_confirmed_counter(
            args,
            radius,
            fps,
            strategy_spec.use_class_vote,
            strategy_spec.lock_entry_class,
        )

    tracker = ByteTrack(frame_rate=max(1, int(round(fps))))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    # 每个分段重置跟踪相关状态；region_counts 不重置，用来延续诱集器内计数。
    track_circle_state = {}
    video_stats = Counter()
    previous_annotated = None
    previous_raw = None
    video_frame_index = 0
    segment_start_counts = {cls_id: region_counts[cls_id] for cls_id in sorted(class_names)}
    segment_start_global_frame = global_state["frame"] + 1
    processing_started = time.perf_counter()

    roi_text = "full-frame" if roi_bounds is None else ",".join(str(value) for value in roi_bounds)
    print(f"开始: {video_path.name}，策略: {args.strategy}，推理区域: {roi_text}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        video_frame_index += 1
        global_state["frame"] += 1

        if roi_bounds is None:
            inference_frame = frame
        else:
            roi_x0, roi_y0, roi_x1, roi_y1 = roi_bounds
            inference_frame = frame[roi_y0:roi_y1, roi_x0:roi_x1]

        results = model.predict(
            inference_frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            half=args.half,
            classes=track_classes,
            verbose=False,
        )
        mot_input = detections_from_results(results)
        if roi_bounds is not None:
            mot_input = offset_detections(mot_input, (roi_bounds[0], roi_bounds[1]), input_size)
        tracker_output = tracker.update(mot_input, frame)

        if args.csv_only:
            raw_frame = None
            annotated = None
        else:
            raw_frame = frame.copy()
            annotated = tracker.plot_results(frame.copy(), show_trajectories=True)
        tracks = boxmot_tracks(tracker_output)

        observations = []
        for xyxy, track_id, cls_id, conf in tracks:
            x1, y1, x2, y2 = xyxy
            cx = float((x1 + x2) / 2)
            cy = float((y1 + y2) / 2)
            observations.append((int(track_id), int(cls_id), float(conf), cx, cy))

        if cache_frames is not None:
            cache_frames.append(
                (
                    video_frame_index,
                    [
                        {"id": int(tid), "cls": int(cid), "conf": float(cf), "cx": float(cx), "cy": float(cy)}
                        for tid, cid, cf, cx, cy in observations
                    ],
                )
            )

        frame_events = count_frame_observations(
            observations,
            video_frame_index,
            global_state["frame"],
            center,
            radius,
            confirmed_counter,
            strategy_spec,
            track_circle_state,
            region_counts,
            video_stats,
            class_names,
            video_path.name,
            args.strategy,
            event_rows,
        )

        if not args.csv_only:
            annotated = draw_count_overlay(annotated, class_names, center, radius, region_counts)
            if raw_frame.shape[1] != output_size[0] or raw_frame.shape[0] != output_size[1]:
                raw_frame = cv2.resize(raw_frame, output_size)
                annotated = cv2.resize(annotated, output_size)
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

            combined_writer.write(annotated)
            previous_annotated = annotated.copy()
            previous_raw = raw_frame.copy()

    cap.release()
    processing_seconds = time.perf_counter() - processing_started

    if cache_frames is not None:
        write_track_cache(
            cache_file,
            track_fingerprint,
            video_path.name,
            class_names,
            video_frame_index,
            cache_frames,
        )
        print(f"轨迹缓存已写出: {cache_file}（{video_frame_index} 帧，可用 --from-tracks 重放）")

    segment_end_global_frame = global_state["frame"] if video_frame_index else segment_start_global_frame - 1
    counter_stats = confirmed_counter.stats if confirmed_counter is not None else Counter()
    append_segment_and_metric_rows(
        video_path.name,
        args.strategy,
        class_names,
        region_counts,
        segment_start_counts,
        segment_rows,
        video_metric_rows,
        video_frame_index,
        fps,
        input_size,
        center,
        radius,
        roi_text,
        strategy_spec,
        strategy_params,
        args,
        video_stats,
        counter_stats,
        segment_start_global_frame,
        segment_end_global_frame,
        processing_seconds,
        "" if args.csv_only else str(args.output / "combined_track_count.mp4"),
        "" if args.csv_only else str(event_dir),
    )

    print(f"完成: {video_path.name}，帧数: {video_frame_index}，连续区域计数: {dict(region_counts)}")


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    validate_args(args)
    args.output.mkdir(parents=True, exist_ok=True)
    weights = resolve_weights(args.weights)
    # 指纹只用权重的路径/大小/时间戳，重放模式也要它，但不会把模型载入显存。
    args._resolved_weights = weights
    videos = iter_videos(args.source)
    strategy_spec = get_strategy_spec(args.strategy)

    first_size, first_fps, _ = get_video_info(videos[0])
    combined_video = args.output / "combined_track_count.mp4"
    if args.csv_only:
        combined_writer = None
        image_saver = None
    else:
        video_writer = create_video_writer(combined_video, first_size, first_fps)
        combined_writer = video_writer if args.sync_write else AsyncVideoWriter(video_writer)
        image_saver = None if args.sync_write else AsyncImageSaver()

    print(f"模型权重: {weights}")
    print(f"视频来源: {args.source}")
    print(f"输出目录: {args.output}")
    print(f"合并视频: {'不生成（--csv-only）' if args.csv_only else combined_video}")
    print(f"视频数量: {len(videos)}")
    if len(videos) > 1:
        print(
            "警告: main.py 的目录模式保留历史连续累计行为。"
            "论文逐视频独立计数请使用 run_counting_ablation.sh，或逐个视频传给 --source。"
        )
    print(
        f"实验策略: {args.strategy} "
        f"(ROI={strategy_spec.use_roi}, 慢/快确认={strategy_spec.use_confirmed_crossing}, "
        f"类别投票={strategy_spec.use_class_vote}, 进入类别锁定={strategy_spec.lock_entry_class})"
    )
    if args.from_tracks:
        # 重放模式：不加载 YOLO，不碰 GPU。类别名从缓存 header 取，回退到权重名。
        print("重放模式（--from-tracks）：跳过 YOLO 与 ByteTrack，直接读轨迹缓存重算计数。")
        model = None
        class_names = _class_names_for_replay(args, videos, strategy_spec)
    else:
        model = YOLO(str(weights))
        try:
            model.fuse()
        except Exception:
            pass
        class_names = get_class_names(model)
        if args.dump_tracks:
            print("缓存模式（--dump-tracks）：本次实时跑，同时写出轨迹缓存供之后重放。")
    track_classes = parse_classes(args.classes)
    region_counts = defaultdict(int)
    event_rows = []
    segment_rows = []
    video_metric_rows = []
    global_state = {"frame": 0}

    try:
        for video_path in videos:
            if args.from_tracks:
                replay_video_from_cache(
                    video_path,
                    args,
                    class_names,
                    region_counts,
                    event_rows,
                    segment_rows,
                    video_metric_rows,
                    global_state,
                )
                continue
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
                video_metric_rows,
                global_state,
                track_classes,
            )
    finally:
        if combined_writer is not None:
            combined_writer.release()
        if image_saver is not None:
            image_saver.close()

    summary_csv = args.output / "summary_counts.csv"
    events_csv = args.output / "count_events.csv"
    final_csv = args.output / "final_counts.csv"
    config_csv = args.output / "experiment_config.csv"
    video_metrics_csv = args.output / "video_metrics.csv"

    write_csv(
        summary_csv,
        [
            "video",
            "strategy",
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
        video_metrics_csv,
        [
            "video",
            "strategy",
            "frames",
            "fps",
            "input_width",
            "input_height",
            "circle_center_x",
            "circle_center_y",
            "circle_radius",
            "roi_bounds",
            "boundary_margin_px",
            "confirm_frames",
            "slow_max_gap_frames",
            "fast_max_gap_frames",
            "fast_min_radial_px",
            "vote_window",
            "vote_max_age_frames",
            "accepted_enter_total",
            "accepted_exit_total",
            "accepted_enter_instant",
            "accepted_exit_instant",
            "accepted_enter_slow",
            "accepted_exit_slow",
            "accepted_enter_fast",
            "accepted_exit_fast",
            "paired_exit",
            "unpaired_exit",
            "overwritten_entry_credit",
            "processing_seconds",
            "processing_fps",
        ],
        video_metric_rows,
    )
    write_csv(
        events_csv,
        [
            "video",
            "strategy",
            "video_frame",
            "global_frame",
            "track_id",
            "cls_id",
            "class_name",
            "direction",
            "delta",
            "count_after",
            "center_x",
            "center_y",
            "radial_distance",
            "conf",
            "raw_cls_id",
            "entry_cls_id",
            "event_reason",
            "circle_state",
            "vote_n",
            "vote_score",
            "vote_share",
            "paired_exit",
            "class_source",
            "diagnostic_reason",
        ],
        event_rows,
    )
    write_csv(
        final_csv,
        ["video", "strategy", "class_id", "class_name", "final_region_count"],
        [
            {
                "video": videos[0].name if len(videos) == 1 else "__directory_total__",
                "strategy": args.strategy,
                "class_id": cls_id,
                "class_name": class_names[cls_id],
                "final_region_count": region_counts[cls_id],
            }
            for cls_id in sorted(class_names)
        ],
    )
    write_csv(
        config_csv,
        [
            "strategy",
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
            "nominal_circle_reference",
            "roi_enabled",
            "roi_scale",
            "confirmed_crossing_enabled",
            "boundary_margin_reference_px",
            "confirm_frames",
            "slow_max_gap_frames",
            "fast_gap_seconds",
            "fast_min_radial_reference_px",
            "class_vote_enabled",
            "entry_class_lock_enabled",
            "class_vote_window",
            "class_vote_max_age_seconds",
            "tracker_id_policy",
            "tracker_defaults",
            "id_stitching",
            "reid",
            "unpaired_exit_policy",
            "exit_class_policy",
        ],
        [
            {
                "strategy": args.strategy,
                "source": str(args.source.resolve()),
                "source_videos": "|".join(str(path.resolve()) for path in videos),
                "weights": str(weights.resolve()),
                "output": str(args.output.resolve()),
                "imgsz": args.imgsz,
                "conf": args.conf,
                "iou": args.iou,
                "device": args.device,
                "half": args.half,
                "classes": args.classes,
                "nominal_circle_reference": "center=(1184,724);radius=150;frame=2304x1296",
                "roi_enabled": strategy_spec.use_roi,
                "roi_scale": args.roi_scale if strategy_spec.use_roi else "",
                "confirmed_crossing_enabled": strategy_spec.use_confirmed_crossing,
                "boundary_margin_reference_px": (
                    args.boundary_margin if strategy_spec.use_confirmed_crossing else ""
                ),
                "confirm_frames": args.confirm_frames if strategy_spec.use_confirmed_crossing else "",
                "slow_max_gap_frames": args.slow_max_gap if strategy_spec.use_confirmed_crossing else "",
                "fast_gap_seconds": args.fast_gap_seconds if strategy_spec.use_confirmed_crossing else "",
                "fast_min_radial_reference_px": (
                    (args.fast_min_radial if args.fast_min_radial > 0 else 3.0 * args.boundary_margin)
                    if strategy_spec.use_confirmed_crossing
                    else ""
                ),
                "class_vote_enabled": strategy_spec.use_class_vote,
                "entry_class_lock_enabled": strategy_spec.lock_entry_class,
                "class_vote_window": args.class_vote_window if strategy_spec.use_class_vote else "",
                "class_vote_max_age_seconds": (
                    args.class_vote_max_age_seconds if strategy_spec.use_class_vote else ""
                ),
                "tracker_id_policy": "raw ByteTrack ID only",
                "tracker_defaults": (
                    "min_conf=0.1;track_thresh=0.45;match_thresh=0.8;track_buffer=25;"
                    "det_thresh=0.3;max_age=30;max_obs=50;min_hits=3;iou_threshold=0.3;"
                    "per_class=False;asso_func=iou;frame_rate=per-video"
                ),
                "id_stitching": False,
                "reid": False,
                "unpaired_exit_policy": (
                    "count exit with current raw-ID vote fallback"
                    if strategy_spec.use_class_vote
                    else "count exit with trigger-frame class"
                ),
                "exit_class_policy": (
                    "paired exit uses locked entry vote; unpaired exit uses current raw-ID vote"
                    if strategy_spec.lock_entry_class
                    else "trigger-frame class"
                ),
            }
        ],
    )

    if args.csv_only:
        print("全部处理完成，本次为 --csv-only，不生成合并视频。")
    else:
        print(f"全部处理完成，合并视频: {combined_video}")
    print(f"分段汇总: {summary_csv}")
    print(f"视频级指标: {video_metrics_csv}")
    print(f"计数事件: {events_csv}")
    print(f"最终计数: {final_csv}")
    print(f"实验配置: {config_csv}")


if __name__ == "__main__":
    main()

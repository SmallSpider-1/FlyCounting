from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np

from tracking_model_benchmark._common.tracker_interface import FrameGeometry, UnifiedTrackerAdapter


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = ROOT / "tracking_model_benchmark"

TRACKER_PROJECTS = {
    "sort": "01_sort",
    "bytetrack": "02_bytetrack",
    "ocsort": "03_ocsort",
    "sfsort": "04_sfsort",
    "fasttracker": "05_fasttracker",
    "boosttrack": "06_boosttrack",
    "hybridsort": "07_hybridsort",
    "botsort": "08_botsort",
}

TRACKER_COMMITS = {
    "sort": "2236dff5019565958b84df7d871d41cc1db58ac7",
    "bytetrack": "d1bf0191adff59bc8fcfeaa0b33d3d1642552a99",
    "ocsort": "8462e7e729a93ccd3bd995c0a79a890336cb3a0b",
    "sfsort": "b1abdeca5ee361c59a20a7fa3937e830846eea82",
    "fasttracker": "a7e7e65a3e4086c5d2104dca06f6593564eda1ae",
    "boosttrack": "fb5bfc3a8f067476565e753b3a73df4d757c9d03",
    "hybridsort": "396f8d30db13304c0cbaf1dcf2e16ded93ce1701",
    "botsort": "251985436d6712aaf682aaaf5f71edb4987224bd",
}

DEFAULT_CONFIGS = {
    "sort": {
        "max_age": 1,
        "min_hits": 3,
        "iou_threshold": 0.3,
        "metadata_match_iou": 0.1,
    },
    "bytetrack": {
        "track_thresh": 0.6,
        "track_buffer": 30,
        "match_thresh": 0.9,
        "mot20": False,
        "metadata_match_iou": 0.1,
    },
    "ocsort": {
        "det_thresh": 0.6,
        "max_age": 30,
        "min_hits": 3,
        "iou_threshold": 0.3,
        "delta_t": 3,
        "asso_func": "iou",
        "inertia": 0.2,
        "use_byte": False,
        "metadata_match_iou": 0.1,
    },
    "sfsort": {
        "high_th": 0.6,
        "match_th_first": 0.67,
        "new_track_th": 0.7,
        "low_th": 0.1,
        "match_th_second": 0.3,
        "dynamic_tuning": False,
        "marginal_timeout": 0,
        "central_timeout": 0,
        "horizontal_margin_ratio": 0.0,
        "vertical_margin_ratio": 0.0,
        "metadata_match_iou": 0.1,
    },
    "fasttracker": {
        "track_high_thresh": 0.25,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.25,
        "track_buffer": 30,
        "match_thresh": 0.8,
        "fuse_score": True,
        "reset_velocity_offset_occ": 5,
        "reset_pos_offset_occ": 3,
        "enlarge_bbox_occ": 1.1,
        "dampen_motion_occ": 0.5,
        "active_occ_to_lost_thresh": 10,
        "occ_cover_thresh": 0.7,
        "occ_reappear_window": 40,
        "init_iou_suppress": 0.7,
        "metadata_match_iou": 0.1,
    },
    "boosttrack": {
        "max_age": 30,
        "min_hits": 3,
        "det_thresh": 0.6,
        "iou_threshold": 0.3,
        "lambda_iou": 0.5,
        "lambda_mhd": 0.25,
        "lambda_shape": 0.25,
        "use_dlo_boost": True,
        "use_duo_boost": True,
        "dlo_boost_coef": 0.65,
        "use_rich_s": True,
        "use_sb": True,
        "use_vt": True,
        "metadata_match_iou": 0.1,
    },
    "hybridsort": {
        "track_thresh": 0.6,
        "TCM_first_step": True,
        "TCM_byte_step": True,
        "TCM_first_step_weight": 1.0,
        "TCM_byte_step_weight": 1.0,
        "det_thresh": 0.6,
        "max_age": 30,
        "min_hits": 3,
        "iou_threshold": 0.25,
        "delta_t": 3,
        "asso_func": "Height_Modulated_IoU",
        "inertia": 0.05,
        "use_byte": True,
        "metadata_match_iou": 0.1,
    },
    "botsort": {
        "track_high_thresh": 0.6,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.7,
        "track_buffer": 30,
        "proximity_thresh": 0.5,
        "appearance_thresh": 0.25,
        "match_thresh": 0.8,
        "mot20": False,
        "metadata_match_iou": 0.1,
    },
}


def resolved_config(tracker_name: str, overrides: dict | None = None) -> dict:
    if tracker_name not in TRACKER_PROJECTS:
        raise ValueError(f"未知跟踪器: {tracker_name}")
    overrides = overrides or {}
    unknown = set(overrides) - set(DEFAULT_CONFIGS[tracker_name])
    if unknown:
        raise ValueError(f"{tracker_name} 配置包含未知字段: {sorted(unknown)}")
    return {**DEFAULT_CONFIGS[tracker_name], **overrides}


def enter_source(project: str, *extra_paths: Path) -> Path:
    source = BENCHMARK_ROOT / project / "src"
    os.chdir(source)
    for path in reversed((source, *extra_paths)):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
    return source


def native_rows_from_objects(objects) -> np.ndarray:
    rows = []
    for track in objects:
        box = np.asarray(track.tlbr, dtype=np.float32).reshape(4)
        rows.append([*box.tolist(), int(track.track_id)])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 5)


class ResultsLikeDetections:
    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xywh = np.asarray(xywh, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, index):
        return ResultsLikeDetections(self.xywh[index], self.conf[index], self.cls[index])


def fasttracker_detections(detections: np.ndarray) -> ResultsLikeDetections:
    xyxy = detections[:, :4]
    xywh = np.empty_like(xyxy)
    xywh[:, :2] = (xyxy[:, :2] + xyxy[:, 2:]) / 2.0
    xywh[:, 2:] = xyxy[:, 2:] - xyxy[:, :2]
    return ResultsLikeDetections(xywh, detections[:, 4], detections[:, 5])


class OfficialTrackerAdapter(UnifiedTrackerAdapter):
    def __init__(self, tracker_name: str, geometry: FrameGeometry, config: dict):
        self.tracker_name = tracker_name
        self.project = TRACKER_PROJECTS[tracker_name]
        self.implementation = str(BENCHMARK_ROOT / self.project / "src")
        self.implementation_commit = TRACKER_COMMITS[tracker_name]
        self.placeholder_image = None
        self.placeholder_tensor = None
        super().__init__(geometry, config)

    def build_native_tracker(self):
        config = self.config
        width, height = self.geometry.frame_size
        if self.tracker_name == "sort":
            enter_source(self.project)
            from sort import KalmanBoxTracker, Sort

            KalmanBoxTracker.count = 0
            return Sort(
                max_age=int(config["max_age"]),
                min_hits=int(config["min_hits"]),
                iou_threshold=float(config["iou_threshold"]),
            )
        if self.tracker_name == "bytetrack":
            enter_source(self.project)
            from yolox.tracker.basetrack import BaseTrack
            from yolox.tracker.byte_tracker import BYTETracker

            BaseTrack._count = 0
            return BYTETracker(types.SimpleNamespace(**config), frame_rate=int(round(self.geometry.fps)))
        if self.tracker_name == "ocsort":
            enter_source(self.project)
            from trackers.ocsort_tracker.ocsort import KalmanBoxTracker, OCSort

            KalmanBoxTracker.count = 0
            return OCSort(**config)
        if self.tracker_name == "sfsort":
            enter_source(self.project)
            from SFSORT import SFSORT

            sfsort_config = dict(config)
            horizontal_ratio = float(sfsort_config.pop("horizontal_margin_ratio"))
            vertical_ratio = float(sfsort_config.pop("vertical_margin_ratio"))
            sfsort_config.update(
                frame_width=width,
                frame_height=height,
                horizontal_margin=int(round(width * horizontal_ratio)),
                vertical_margin=int(round(height * vertical_ratio)),
            )
            return SFSORT(sfsort_config)
        if self.tracker_name == "fasttracker":
            os.chdir(ROOT)
            root_text = str(ROOT)
            if root_text not in sys.path:
                sys.path.insert(0, root_text)
            from ultralytics.trackers import FASTTracker

            return FASTTracker(args=types.SimpleNamespace(**config))
        if self.tracker_name == "boosttrack":
            enter_source(self.project, BENCHMARK_ROOT / self.project / "src" / "external")
            import torch
            from default_settings import BoostTrackPlusPlusSettings, BoostTrackSettings, GeneralSettings
            from tracker.boost_track import BoostTrack, KalmanBoxTracker

            GeneralSettings.values.update(
                max_age=int(config["max_age"]),
                min_hits=int(config["min_hits"]),
                det_thresh=float(config["det_thresh"]),
                iou_threshold=float(config["iou_threshold"]),
                use_embedding=False,
                use_ecc=False,
                dataset="custom",
                test_dataset=False,
            )
            BoostTrackSettings.values.update(
                lambda_iou=float(config["lambda_iou"]),
                lambda_mhd=float(config["lambda_mhd"]),
                lambda_shape=float(config["lambda_shape"]),
                use_dlo_boost=bool(config["use_dlo_boost"]),
                use_duo_boost=bool(config["use_duo_boost"]),
                dlo_boost_coef=float(config["dlo_boost_coef"]),
            )
            BoostTrackPlusPlusSettings.values.update(
                use_rich_s=bool(config["use_rich_s"]),
                use_sb=bool(config["use_sb"]),
                use_vt=bool(config["use_vt"]),
            )
            KalmanBoxTracker.count = 0
            self.placeholder_image = np.zeros((height, width, 3), dtype=np.uint8)
            self.placeholder_tensor = torch.zeros((1, 3, height, width), dtype=torch.float32)
            return BoostTrack(None)
        if self.tracker_name == "hybridsort":
            enter_source(self.project)
            from trackers.hybrid_sort_tracker.hybrid_sort import Hybrid_Sort, KalmanBoxTracker

            KalmanBoxTracker.count = 0
            argument_names = {
                "track_thresh",
                "TCM_first_step",
                "TCM_byte_step",
                "TCM_first_step_weight",
                "TCM_byte_step_weight",
            }
            tracker_args = {key: config[key] for key in argument_names}
            tracker_args.update(ECC=False, hybrid_sort_with_reid=False, with_fastreid=False)
            constructor = {key: value for key, value in config.items() if key not in argument_names}
            return Hybrid_Sort(types.SimpleNamespace(**tracker_args), **constructor)
        if self.tracker_name == "botsort":
            enter_source(self.project)
            torch_six = types.ModuleType("torch._six")
            torch_six.string_classes = (str,)
            sys.modules.setdefault("torch._six", torch_six)
            from tracker.bot_sort import BoTSORT

            args = dict(config)
            args.update(
                with_reid=False,
                cmc_method="none",
                name="fruitfly_counting",
                ablation=False,
                device="cpu",
                fast_reid_config="",
                fast_reid_weights="",
            )
            self.placeholder_image = np.zeros((height, width, 3), dtype=np.uint8)
            return BoTSORT(types.SimpleNamespace(**args), frame_rate=int(round(self.geometry.fps)))
        raise AssertionError(self.tracker_name)

    def update_native(self, detections: np.ndarray, frame_index: int) -> np.ndarray:
        input_nx5 = detections[:, :5].copy()
        width, height = self.geometry.frame_size
        if self.tracker_name == "sort":
            return self.native_tracker.update(input_nx5)
        if self.tracker_name == "bytetrack":
            return native_rows_from_objects(self.native_tracker.update(input_nx5, (height, width), (height, width)))
        if self.tracker_name == "ocsort":
            return self.native_tracker.update(input_nx5, (height, width), (height, width))
        if self.tracker_name == "sfsort":
            native = self.native_tracker.update(input_nx5[:, :4], input_nx5[:, 4])
            rows = []
            for item in native:
                rows.append([*np.asarray(item[0], dtype=np.float32).reshape(4).tolist(), int(item[1])])
            return np.asarray(rows, dtype=np.float32).reshape(-1, 5)
        if self.tracker_name == "fasttracker":
            output = self.native_tracker.update(fasttracker_detections(detections))
            return np.asarray(output, dtype=np.float32).reshape(-1, 8)[:, :5] if len(output) else np.empty((0, 5))
        if self.tracker_name == "boosttrack":
            return self.native_tracker.update(
                input_nx5,
                self.placeholder_tensor,
                self.placeholder_image,
                f"frame:{frame_index}",
            )[:, :5]
        if self.tracker_name == "hybridsort":
            return self.native_tracker.update(input_nx5, (height, width), (height, width))
        if self.tracker_name == "botsort":
            return native_rows_from_objects(self.native_tracker.update(input_nx5, self.placeholder_image))
        raise AssertionError(self.tracker_name)


def create_tracker_adapter(
    tracker_name: str,
    frame_size: tuple[int, int],
    fps: float,
    config_overrides: dict | None = None,
) -> OfficialTrackerAdapter:
    width, height = map(int, frame_size)
    if width <= 0 or height <= 0:
        raise ValueError(f"无效 frame_size: {frame_size}")
    geometry = FrameGeometry((width, height), float(fps))
    return OfficialTrackerAdapter(tracker_name, geometry, resolved_config(tracker_name, config_overrides))

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


DETECTION_FIELDS = ("x1", "y1", "x2", "y2", "confidence", "class_id")
TRACK_FIELDS = (
    "x1",
    "y1",
    "x2",
    "y2",
    "track_id",
    "confidence",
    "class_id",
    "detection_index",
)


@dataclass(frozen=True)
class FrameGeometry:
    frame_size: tuple[int, int]
    fps: float


def validate_detections(detections) -> np.ndarray:
    array = np.asarray(detections, dtype=np.float32)
    if array.size == 0:
        return np.empty((0, len(DETECTION_FIELDS)), dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != len(DETECTION_FIELDS):
        raise ValueError(f"统一检测输入必须是 Nx6，实际形状为 {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("统一检测输入包含 NaN 或 Inf。")
    if np.any(array[:, 2] < array[:, 0]) or np.any(array[:, 3] < array[:, 1]):
        raise ValueError("统一检测输入包含无效 xyxy 框。")
    return array


def validate_native_tracks(native_tracks) -> np.ndarray:
    array = np.asarray(native_tracks, dtype=np.float32)
    if array.size == 0:
        return np.empty((0, 5), dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 5:
        raise ValueError(f"适配器原生输出至少应为 Nx5 [xyxy,id]，实际形状为 {array.shape}")
    array = array[:, :5]
    if not np.isfinite(array).all():
        raise ValueError("跟踪器原生输出包含 NaN 或 Inf。")
    return array


def box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.empty((len(boxes_a), len(boxes_b)), dtype=np.float32)
    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:4], boxes_b[None, :, 2:4])
    intersection_wh = np.clip(bottom_right - top_left, 0.0, None)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    area_a = np.clip(boxes_a[:, 2] - boxes_a[:, 0], 0.0, None) * np.clip(
        boxes_a[:, 3] - boxes_a[:, 1], 0.0, None
    )
    area_b = np.clip(boxes_b[:, 2] - boxes_b[:, 0], 0.0, None) * np.clip(
        boxes_b[:, 3] - boxes_b[:, 1], 0.0, None
    )
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


class UnifiedTrackerAdapter(ABC):
    """Tracker-independent Nx6 input to Nx8 output contract."""

    tracker_name = "unknown"
    implementation = "unknown"
    implementation_commit = "unknown"
    image_dependency = "none"

    def __init__(self, geometry: FrameGeometry, config: dict):
        self.geometry = geometry
        self.config = dict(config)
        self.metadata_match_iou = float(self.config.pop("metadata_match_iou", 0.1))
        self.track_metadata: dict[int, tuple[float, int]] = {}
        self.last_frame_index = 0
        self.native_tracker = self.build_native_tracker()

    @abstractmethod
    def build_native_tracker(self):
        raise NotImplementedError

    @abstractmethod
    def update_native(self, detections: np.ndarray, frame_index: int) -> np.ndarray:
        """Return Nx5 [x1,y1,x2,y2,track_id] in full-frame coordinates."""
        raise NotImplementedError

    def update(self, detections, frame_index: int) -> np.ndarray:
        detections = validate_detections(detections)
        expected_frame = self.last_frame_index + 1
        if int(frame_index) != expected_frame:
            raise ValueError(f"跟踪器帧号必须连续: 期望 {expected_frame}，实际 {frame_index}")
        self.last_frame_index = int(frame_index)

        native_tracks = validate_native_tracks(self.update_native(detections, int(frame_index)))
        if len(native_tracks) == 0:
            return np.empty((0, len(TRACK_FIELDS)), dtype=np.float32)

        unified = np.empty((len(native_tracks), len(TRACK_FIELDS)), dtype=np.float32)
        unified[:, :5] = native_tracks
        matches = self._match_metadata(native_tracks[:, :4], detections[:, :4])
        for track_row, track_id_value in enumerate(native_tracks[:, 4]):
            track_id = int(round(float(track_id_value)))
            detection_index = matches.get(track_row, -1)
            if detection_index >= 0:
                confidence = float(detections[detection_index, 4])
                class_id = int(round(float(detections[detection_index, 5])))
                self.track_metadata[track_id] = (confidence, class_id)
            elif track_id in self.track_metadata:
                confidence, class_id = self.track_metadata[track_id]
            else:
                raise ValueError(
                    f"{self.tracker_name} 输出了无法关联当前检测、且无历史类别的 track_id={track_id}"
                )
            unified[track_row, 4] = track_id
            unified[track_row, 5] = confidence
            unified[track_row, 6] = class_id
            unified[track_row, 7] = detection_index

        return unified

    def _match_metadata(self, track_boxes: np.ndarray, detection_boxes: np.ndarray) -> dict[int, int]:
        if len(track_boxes) == 0 or len(detection_boxes) == 0:
            return {}
        ious = box_iou_matrix(track_boxes, detection_boxes)
        track_indices, detection_indices = linear_sum_assignment(1.0 - ious)
        return {
            int(track_index): int(detection_index)
            for track_index, detection_index in zip(track_indices, detection_indices)
            if ious[track_index, detection_index] >= self.metadata_match_iou
        }

    def cache_metadata(self) -> dict:
        return {
            "tracker_name": self.tracker_name,
            "implementation": self.implementation,
            "implementation_commit": self.implementation_commit,
            "adapter_contract": "Nx6_full_frame_detections_to_Nx8_full_frame_tracks_v1",
            "tracking_space": "full_frame",
            "per_class": False,
            "class_policy": "current_matched_detection_else_last_observation",
            "metadata_match_iou": self.metadata_match_iou,
            "image_dependency": self.image_dependency,
            "config": self.config,
        }

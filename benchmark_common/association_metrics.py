"""Shared spatial association metrics for the eight-tracker benchmark."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


IOU = "iou"
NORMALIZED_EUCLIDEAN = "normalized_euclidean"
SUPPORTED_ASSOCIATION_METRICS = (IOU, NORMALIZED_EUCLIDEAN)


def _box_array(boxes: Sequence | np.ndarray) -> np.ndarray:
    """Return an ``Nx4`` float array while accepting native tracker box rows."""
    array = np.asarray(boxes, dtype=np.float32)
    if array.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] < 4:
        raise ValueError(f"Bounding boxes must have shape Nx4+, got {array.shape}")
    return array[:, :4]


def normalized_center_distance(boxes_a: Sequence | np.ndarray, boxes_b: Sequence | np.ndarray) -> np.ndarray:
    """Return pairwise center Euclidean distance normalized by each pair's enclosing-box diagonal.

    This follows the spatial term in the project's AEM definition. The result is a bounded
    ``[0, 1]`` cost: zero means coincident centers and smaller values are better matches.
    """
    boxes_a = _box_array(boxes_a)
    boxes_b = _box_array(boxes_b)
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.empty((len(boxes_a), len(boxes_b)), dtype=np.float32)

    center_a = (boxes_a[:, :2] + boxes_a[:, 2:4]) / 2.0
    center_b = (boxes_b[:, :2] + boxes_b[:, 2:4]) / 2.0
    center_distance = np.linalg.norm(center_a[:, None, :] - center_b[None, :, :], axis=2)

    enclosing_min = np.minimum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    enclosing_max = np.maximum(boxes_a[:, None, 2:4], boxes_b[None, :, 2:4])
    enclosing_diagonal = np.linalg.norm(enclosing_max - enclosing_min, axis=2)
    distance = np.divide(
        center_distance,
        enclosing_diagonal,
        out=np.zeros_like(center_distance),
        where=enclosing_diagonal > np.finfo(np.float32).eps,
    )
    return np.clip(distance, 0.0, 1.0).astype(np.float32, copy=False)


def normalized_center_similarity(boxes_a: Sequence | np.ndarray, boxes_b: Sequence | np.ndarray) -> np.ndarray:
    """Return ``1 - normalized_center_distance`` as an IoU-range similarity."""
    return 1.0 - normalized_center_distance(boxes_a, boxes_b)


def _native_track_boxes(tracks: Sequence | np.ndarray) -> np.ndarray:
    """Extract ``tlbr``/``xyxy`` boxes from ByteTrack-family objects or native arrays."""
    if isinstance(tracks, np.ndarray):
        return _box_array(tracks)
    rows = []
    for track in tracks:
        if isinstance(track, np.ndarray):
            rows.append(track[:4])
        elif hasattr(track, "tlbr"):
            rows.append(track.tlbr)
        elif hasattr(track, "xyxy"):
            rows.append(track.xyxy)
        else:
            raise TypeError(f"Cannot extract a box from {type(track).__name__}")
    return _box_array(rows)


def normalized_center_distance_from_tracks(tracks_a, tracks_b) -> np.ndarray:
    """ByteTrack-family compatible cost function accepting boxes or track objects."""
    return normalized_center_distance(_native_track_boxes(tracks_a), _native_track_boxes(tracks_b))


def normalized_center_distance_from_sfsort(tracks, boxes, iou_only: bool = False) -> np.ndarray:
    """SF-SORT compatible cost function; ``iou_only`` is retained for API parity."""
    del iou_only
    track_boxes = [track.bbox for track in tracks]
    return normalized_center_distance(track_boxes, boxes)

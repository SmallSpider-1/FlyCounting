import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


INNER = "inner"
BAND = "band"
OUTER = "outer"


def circle_zone(point, center, radius, band_width):
    """Classify a point using stable inner/outer zones separated by a hysteresis band."""
    distance = math.hypot(point[0] - center[0], point[1] - center[1])
    if distance < radius - band_width:
        return INNER
    if distance > radius + band_width:
        return OUTER
    return BAND


def extract_appearance(frame, xyxy):
    """Build a compact color/brightness histogram from a tracked object crop."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    padding_x = 0.08 * box_width
    padding_y = 0.08 * box_height
    left = max(0, int(math.floor(x1 - padding_x)))
    top = max(0, int(math.floor(y1 - padding_y)))
    right = min(width, int(math.ceil(x2 + padding_x)))
    bottom = min(height, int(math.ceil(y2 + padding_y)))
    if right <= left or bottom <= top:
        return None

    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Prefer insect-like pixels so the white trap background does not dominate.
    foreground = ((hsv[:, :, 1] > 30) | (hsv[:, :, 2] < 210)).astype(np.uint8) * 255
    mask = foreground if cv2.countNonZero(foreground) >= 32 else None
    hs_hist = cv2.calcHist([hsv], [0, 1], mask, [12, 8], [0, 180, 0, 256]).reshape(-1)
    gray_hist = cv2.calcHist([gray], [0], mask, [16], [0, 256]).reshape(-1)
    descriptor = np.concatenate([hs_hist, gray_hist]).astype(np.float32)
    total = float(descriptor.sum())
    return descriptor / total if total > 0 else None


def appearance_similarity(first, second):
    if first is None or second is None:
        return 0.5
    return float(np.clip(np.sqrt(first * second).sum(), 0.0, 1.0))


@dataclass
class LogicalTrack:
    logical_id: int
    last_frame: int
    last_center: np.ndarray
    last_box: np.ndarray
    last_raw_key: tuple
    stable_zone: Optional[str] = None
    counted_inside: bool = False
    counted_class_id: Optional[int] = None
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    appearance: Optional[np.ndarray] = None
    class_scores: dict = field(default_factory=lambda: defaultdict(float))
    raw_track_ids: set = field(default_factory=set)
    observations: int = 0
    last_stitch_gap_frames: int = 0
    recovery_eligible: bool = False
    band_start_recovery_eligible: bool = False
    outer_start_recovery_eligible: bool = False
    outer_start_frame: int = 0
    outer_start_offset: float = 0.0
    last_exit_frame: int = -1

    @property
    def voted_class_id(self):
        if not self.class_scores:
            return 0
        return max(self.class_scores, key=self.class_scores.get)


@dataclass
class OccupancyToken:
    token_id: int
    owner_logical_id: int
    class_id: int
    entry_frame: int
    appearance: Optional[np.ndarray]
    size: np.ndarray


class RegionOccupancyCounter:
    """Count stable circle crossings while joining short fragments from different tracker IDs."""

    def __init__(
        self,
        center,
        radius,
        band_width=15.0,
        stitch_max_gap_frames=28,
        stitch_max_distance=80.0,
        stitch_max_size_ratio=2.5,
        stitch_min_score=0.6,
        enable_stitching=True,
        recovery_min_gap_frames=6,
        recovery_max_size_ratio=2.5,
        recovery_min_score=0.4,
        enable_exit_recovery=True,
        recovery_require_class_match=False,
        recovery_class_match_required=None,
        recovery_class_min_scores=None,
        enable_band_start_recovery=False,
        band_start_max_offset=0.0,
        enable_outer_start_recovery=False,
        outer_start_max_offset=35.0,
        outer_start_min_outward=10.0,
        outer_start_min_frames=2,
        outer_start_max_frames=30,
        reentry_cooldown_frames=0,
        class_vote_weights=None,
        entry_detected_class_override=None,
    ):
        if band_width < 0 or band_width >= radius:
            raise ValueError("band_width must be non-negative and smaller than radius")
        self.center = np.asarray(center, dtype=np.float32)
        self.radius = float(radius)
        self.band_width = float(band_width)
        self.stitch_max_gap_frames = max(1, int(stitch_max_gap_frames))
        self.stitch_max_distance = max(1.0, float(stitch_max_distance))
        self.stitch_max_size_ratio = max(1.0, float(stitch_max_size_ratio))
        self.stitch_min_score = float(stitch_min_score)
        self.enable_stitching = bool(enable_stitching)
        self.recovery_min_gap_frames = max(1, int(recovery_min_gap_frames))
        self.recovery_max_size_ratio = max(1.0, float(recovery_max_size_ratio))
        self.recovery_min_score = float(recovery_min_score)
        self.enable_exit_recovery = bool(enable_exit_recovery)
        self.recovery_require_class_match = bool(recovery_require_class_match)
        self.recovery_class_match_required = (
            None
            if recovery_class_match_required is None
            else {int(class_id) for class_id in recovery_class_match_required}
        )
        self.recovery_class_min_scores = {
            int(key): float(value) for key, value in (recovery_class_min_scores or {}).items()
        }
        self.enable_band_start_recovery = bool(enable_band_start_recovery)
        self.band_start_max_offset = float(band_start_max_offset)
        self.enable_outer_start_recovery = bool(enable_outer_start_recovery)
        self.outer_start_max_offset = max(0.0, float(outer_start_max_offset))
        self.outer_start_min_outward = max(0.0, float(outer_start_min_outward))
        self.outer_start_min_frames = max(1, int(outer_start_min_frames))
        self.outer_start_max_frames = max(self.outer_start_min_frames, int(outer_start_max_frames))
        self.reentry_cooldown_frames = max(0, int(reentry_cooldown_frames))
        self.class_vote_weights = {int(key): float(value) for key, value in (class_vote_weights or {}).items()}
        self.entry_detected_class_override = (
            None
            if entry_detected_class_override is None
            else {int(class_id) for class_id in entry_detected_class_override}
        )
        self.logical_tracks = {}
        self.raw_to_logical = {}
        self.recent_updates = deque()
        self.next_logical_id = 1
        self.occupancy_tokens = {}
        self.owner_to_token = {}
        self.next_token_id = 1

    def update(self, tracks, frame, frame_index, namespace=""):
        observations = [self._make_observation(track, frame, namespace) for track in tracks]
        assignments = self._assign_observations(observations, frame_index)
        results = []
        events = []

        for observation, logical_id, stitched, stitch_gap in assignments:
            logical_track = self.logical_tracks.get(logical_id)
            if logical_track is None:
                logical_track = self._create_logical_track(logical_id, observation, frame_index)
                event = None
            else:
                event = self._update_logical_track(logical_track, observation, frame_index)

            result = {
                **observation,
                "logical_track_id": logical_id,
                "stitched": len(logical_track.raw_track_ids) > 1,
                "stitched_now": stitched,
                "stitch_gap_frames": logical_track.last_stitch_gap_frames,
                "voted_class_id": logical_track.voted_class_id,
            }
            results.append(result)
            if event is not None:
                events.append({**result, **event})

        return results, events

    def _make_observation(self, track, frame, namespace):
        xyxy, raw_track_id, cls_id, confidence = track
        xyxy = np.asarray(xyxy, dtype=np.float32)
        center = np.asarray(((xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2), dtype=np.float32)
        size = np.asarray((max(1.0, xyxy[2] - xyxy[0]), max(1.0, xyxy[3] - xyxy[1])), dtype=np.float32)
        return {
            "xyxy": xyxy,
            "raw_track_id": int(raw_track_id),
            "raw_key": (str(namespace), int(raw_track_id)),
            "cls_id": int(cls_id),
            "confidence": float(confidence),
            "center": center,
            "size": size,
            "zone": circle_zone(center, self.center, self.radius, self.band_width),
            "radial_offset": math.hypot(center[0] - self.center[0], center[1] - self.center[1]) - self.radius,
            "appearance": extract_appearance(frame, xyxy),
        }

    def _assign_observations(self, observations, frame_index):
        assignments = []
        assigned_observations = set()
        assigned_logical_ids = set()

        for observation_index, observation in enumerate(observations):
            logical_id = self.raw_to_logical.get(observation["raw_key"])
            if logical_id is None or logical_id in assigned_logical_ids:
                continue
            assignments.append((observation_index, logical_id, False, 0))
            assigned_observations.add(observation_index)
            assigned_logical_ids.add(logical_id)

        if self.enable_stitching:
            candidates = []
            recent_logical_ids = self._recent_logical_ids(frame_index)
            for observation_index, observation in enumerate(observations):
                if observation_index in assigned_observations:
                    continue
                for logical_id in recent_logical_ids:
                    if logical_id in assigned_logical_ids:
                        continue
                    logical_track = self.logical_tracks[logical_id]
                    score = self._match_score(logical_track, observation, frame_index)
                    if score is not None and score >= self.stitch_min_score:
                        candidates.append((score, observation_index, logical_id))

            for score, observation_index, logical_id in sorted(candidates, reverse=True):
                if observation_index in assigned_observations or logical_id in assigned_logical_ids:
                    continue
                gap = frame_index - self.logical_tracks[logical_id].last_frame
                assignments.append((observation_index, logical_id, True, gap))
                assigned_observations.add(observation_index)
                assigned_logical_ids.add(logical_id)

        for observation_index in range(len(observations)):
            if observation_index in assigned_observations:
                continue
            logical_id = self.next_logical_id
            self.next_logical_id += 1
            assignments.append((observation_index, logical_id, False, 0))

        assignments.sort(key=lambda item: item[0])
        return [
            (observations[observation_index], logical_id, stitched, stitch_gap)
            for observation_index, logical_id, stitched, stitch_gap in assignments
        ]

    def _recent_logical_ids(self, frame_index):
        while self.recent_updates and frame_index - self.recent_updates[0][0] > self.stitch_max_gap_frames:
            self.recent_updates.popleft()
        return {logical_id for _, logical_id in self.recent_updates}

    def _match_score(self, logical_track, observation, frame_index):
        gap = frame_index - logical_track.last_frame
        if gap <= 0 or gap > self.stitch_max_gap_frames:
            return None

        predicted_center = logical_track.last_center + logical_track.velocity * gap
        predicted_distance = float(np.linalg.norm(observation["center"] - predicted_center))
        direct_distance = float(np.linalg.norm(observation["center"] - logical_track.last_center))
        speed = float(np.linalg.norm(logical_track.velocity))
        allowed_predicted_distance = self.stitch_max_distance
        allowed_direct_distance = self.stitch_max_distance + min(speed * gap, self.stitch_max_distance)
        if predicted_distance > allowed_predicted_distance and direct_distance > allowed_direct_distance:
            return None

        previous_size = np.maximum(logical_track.last_box[2:4] - logical_track.last_box[0:2], 1.0)
        size_ratio = np.maximum(observation["size"] / previous_size, previous_size / observation["size"])
        if float(size_ratio.max()) > self.stitch_max_size_ratio:
            return None

        distance_ratio = min(
            predicted_distance / allowed_predicted_distance,
            direct_distance / allowed_direct_distance,
        )
        distance_score = math.exp(-(distance_ratio**2))
        size_score = math.exp(-float(np.abs(np.log(observation["size"] / previous_size)).mean()))
        appearance_score = appearance_similarity(logical_track.appearance, observation["appearance"])

        displacement = observation["center"] - logical_track.last_center
        displacement_norm = float(np.linalg.norm(displacement))
        if speed > 0.5 and displacement_norm > 1.0:
            cosine = float(np.dot(logical_track.velocity, displacement) / (speed * displacement_norm))
            motion_score = (np.clip(cosine, -1.0, 1.0) + 1.0) / 2.0
        else:
            motion_score = 0.5

        class_total = sum(logical_track.class_scores.values())
        class_score = logical_track.class_scores.get(observation["cls_id"], 0.0) / class_total if class_total else 0.5
        time_score = 1.0 - (gap - 1) / self.stitch_max_gap_frames
        return float(
            0.40 * distance_score
            + 0.15 * size_score
            + 0.20 * appearance_score
            + 0.10 * motion_score
            + 0.10 * class_score
            + 0.05 * time_score
        )

    def _create_logical_track(self, logical_id, observation, frame_index):
        logical_track = LogicalTrack(
            logical_id=logical_id,
            last_frame=frame_index,
            last_center=observation["center"].copy(),
            last_box=observation["xyxy"].copy(),
            last_raw_key=observation["raw_key"],
            stable_zone=observation["zone"] if observation["zone"] != BAND else None,
            appearance=observation["appearance"],
            recovery_eligible=observation["zone"] == INNER,
            band_start_recovery_eligible=self._is_recoverable_band_start(observation),
            outer_start_recovery_eligible=self._is_recoverable_outer_start(observation),
            outer_start_frame=frame_index,
            outer_start_offset=float(observation["radial_offset"]),
        )
        self.logical_tracks[logical_id] = logical_track
        self._record_observation(logical_track, observation, frame_index, update_motion=False)
        return logical_track

    def _update_logical_track(self, logical_track, observation, frame_index):
        previous_stable_zone = logical_track.stable_zone
        current_zone = observation["zone"]
        raw_id_changed = observation["raw_key"] != logical_track.last_raw_key
        stitch_gap = frame_index - logical_track.last_frame if raw_id_changed else 0
        if raw_id_changed:
            logical_track.last_stitch_gap_frames = stitch_gap
        self._record_observation(logical_track, observation, frame_index, update_motion=True)

        if current_zone == BAND:
            return None
        if previous_stable_zone is None:
            if (
                current_zone == OUTER
                and logical_track.band_start_recovery_eligible
                and self.enable_exit_recovery
            ):
                recovery = self._recover_occupancy_token(logical_track, observation, frame_index)
                logical_track.recovery_eligible = False
                logical_track.band_start_recovery_eligible = False
                logical_track.stable_zone = current_zone
                if recovery is not None:
                    token, recovery_score, recovery_class_match = recovery
                    return {
                        "direction": "exit",
                        "delta": -1,
                        "count_cls_id": token.class_id,
                        "raw_id_changed": raw_id_changed,
                        "transition_gap_frames": stitch_gap,
                        "event_reason": "recovered_exit",
                        "recovery_mode": "band_start",
                        "occupancy_token_id": token.token_id,
                        "recovered_from_logical_id": token.owner_logical_id,
                        "recovery_score": recovery_score,
                        "recovery_track_class_id": logical_track.voted_class_id,
                        "recovery_class_match": int(recovery_class_match),
                    }
            logical_track.stable_zone = current_zone
            logical_track.recovery_eligible = current_zone == INNER
            logical_track.band_start_recovery_eligible = False
            return None
        if current_zone == previous_stable_zone:
            if previous_stable_zone == OUTER:
                event = self._maybe_recover_outer_start_exit(logical_track, observation, frame_index, raw_id_changed, stitch_gap)
                if event is not None:
                    return event
            return None

        # Stable endpoints imply a full crossing even when detections inside the band were missed.
        logical_track.stable_zone = current_zone
        logical_track.band_start_recovery_eligible = False
        logical_track.outer_start_recovery_eligible = False
        if previous_stable_zone == OUTER and current_zone == INNER:
            if logical_track.counted_inside:
                return None
            if self._is_reentry_cooldown_active(logical_track, frame_index):
                logical_track.counted_class_id = None
                logical_track.recovery_eligible = False
                return None
            class_id = self._entry_class_id(logical_track, observation)
            logical_track.counted_inside = True
            logical_track.counted_class_id = class_id
            logical_track.recovery_eligible = False
            token = self._create_occupancy_token(logical_track, class_id, observation, frame_index)
            return {
                "direction": "enter",
                "delta": 1,
                "count_cls_id": class_id,
                "raw_id_changed": raw_id_changed,
                "transition_gap_frames": stitch_gap,
                "event_reason": "tracked_enter",
                "recovery_mode": "",
                "occupancy_token_id": token.token_id,
                "recovered_from_logical_id": "",
                "recovery_score": "",
                "recovery_track_class_id": "",
                "recovery_class_match": "",
            }

        if previous_stable_zone == INNER and current_zone == OUTER:
            if logical_track.counted_inside:
                token = self._consume_owned_token(logical_track.logical_id)
                logical_track.counted_inside = False
                logical_track.counted_class_id = None
                logical_track.recovery_eligible = False
                logical_track.last_exit_frame = frame_index
                if token is None:
                    return None
                return {
                    "direction": "exit",
                    "delta": -1,
                    "count_cls_id": token.class_id,
                    "raw_id_changed": raw_id_changed,
                    "transition_gap_frames": stitch_gap,
                    "event_reason": "tracked_exit",
                    "recovery_mode": "",
                    "occupancy_token_id": token.token_id,
                    "recovered_from_logical_id": logical_track.logical_id,
                    "recovery_score": "",
                    "recovery_track_class_id": "",
                    "recovery_class_match": "",
                }

            recovery = None
            if logical_track.recovery_eligible and self.enable_exit_recovery:
                recovery = self._recover_occupancy_token(logical_track, observation, frame_index)
            logical_track.recovery_eligible = False
            logical_track.last_exit_frame = frame_index
            if recovery is None:
                return None
            token, recovery_score, recovery_class_match = recovery
            return {
                "direction": "exit",
                "delta": -1,
                "count_cls_id": token.class_id,
                "raw_id_changed": raw_id_changed,
                "transition_gap_frames": stitch_gap,
                "event_reason": "recovered_exit",
                "recovery_mode": "inner_start",
                "occupancy_token_id": token.token_id,
                "recovered_from_logical_id": token.owner_logical_id,
                "recovery_score": recovery_score,
                "recovery_track_class_id": logical_track.voted_class_id,
                "recovery_class_match": int(recovery_class_match),
            }
        return None

    def _is_reentry_cooldown_active(self, logical_track, frame_index):
        return (
            self.reentry_cooldown_frames > 0
            and logical_track.last_exit_frame >= 0
            and frame_index - logical_track.last_exit_frame <= self.reentry_cooldown_frames
        )

    def _entry_class_id(self, logical_track, observation):
        detected_class_id = observation["cls_id"]
        if self.entry_detected_class_override is not None and detected_class_id in self.entry_detected_class_override:
            return detected_class_id
        return logical_track.voted_class_id

    def _is_recoverable_band_start(self, observation):
        return (
            self.enable_band_start_recovery
            and observation["zone"] == BAND
            and observation["radial_offset"] <= self.band_start_max_offset
        )

    def _is_recoverable_outer_start(self, observation):
        return (
            self.enable_outer_start_recovery
            and observation["zone"] == OUTER
            and observation["radial_offset"] <= self.outer_start_max_offset
        )

    def _maybe_recover_outer_start_exit(self, logical_track, observation, frame_index, raw_id_changed, stitch_gap):
        if not (self.enable_exit_recovery and logical_track.outer_start_recovery_eligible):
            return None
        age = frame_index - logical_track.outer_start_frame
        if age > self.outer_start_max_frames:
            logical_track.outer_start_recovery_eligible = False
            return None
        if age < self.outer_start_min_frames:
            return None
        outward = float(observation["radial_offset"]) - logical_track.outer_start_offset
        if outward < self.outer_start_min_outward:
            return None

        logical_track.outer_start_recovery_eligible = False
        recovery = self._recover_occupancy_token(logical_track, observation, frame_index)
        if recovery is None:
            return None
        token, recovery_score, recovery_class_match = recovery
        return {
            "direction": "exit",
            "delta": -1,
            "count_cls_id": token.class_id,
            "raw_id_changed": raw_id_changed,
            "transition_gap_frames": stitch_gap,
            "event_reason": "recovered_exit",
            "recovery_mode": "outer_start",
            "occupancy_token_id": token.token_id,
            "recovered_from_logical_id": token.owner_logical_id,
            "recovery_score": recovery_score,
            "recovery_track_class_id": logical_track.voted_class_id,
            "recovery_class_match": int(recovery_class_match),
        }

    def _create_occupancy_token(self, logical_track, class_id, observation, frame_index):
        token = OccupancyToken(
            token_id=self.next_token_id,
            owner_logical_id=logical_track.logical_id,
            class_id=class_id,
            entry_frame=frame_index,
            appearance=None if observation["appearance"] is None else observation["appearance"].copy(),
            size=observation["size"].copy(),
        )
        self.next_token_id += 1
        self.occupancy_tokens[token.token_id] = token
        self.owner_to_token[logical_track.logical_id] = token.token_id
        return token

    def _consume_owned_token(self, owner_logical_id):
        token_id = self.owner_to_token.pop(owner_logical_id, None)
        if token_id is None:
            return None
        return self.occupancy_tokens.pop(token_id, None)

    def _recover_occupancy_token(self, logical_track, observation, frame_index):
        candidates = []
        voted_class_id = logical_track.voted_class_id
        for token in self.occupancy_tokens.values():
            if token.owner_logical_id == logical_track.logical_id:
                continue
            class_match = voted_class_id == token.class_id
            require_class_match = self.recovery_require_class_match
            if self.recovery_class_match_required is not None:
                require_class_match = token.class_id in self.recovery_class_match_required
            if require_class_match and not class_match:
                continue
            owner = self.logical_tracks.get(token.owner_logical_id)
            if owner is None or frame_index - owner.last_frame < self.recovery_min_gap_frames:
                continue

            size_ratio = np.maximum(observation["size"] / token.size, token.size / observation["size"])
            if float(size_ratio.max()) > self.recovery_max_size_ratio:
                continue
            size_score = math.exp(-float(np.abs(np.log(observation["size"] / token.size)).mean()))
            appearance_score = appearance_similarity(owner.appearance, observation["appearance"])
            class_score = 1.0 if class_match else 0.0
            score = 0.55 * appearance_score + 0.25 * size_score + 0.20 * class_score
            min_score = self.recovery_class_min_scores.get(token.class_id, self.recovery_min_score)
            if score >= min_score:
                candidates.append((score, token.entry_frame, int(class_match), token.token_id))

        if not candidates:
            return None
        score, _, class_match, token_id = max(candidates)
        token = self.occupancy_tokens.pop(token_id)
        self.owner_to_token.pop(token.owner_logical_id, None)
        owner = self.logical_tracks.get(token.owner_logical_id)
        if owner is not None:
            owner.counted_inside = False
            owner.counted_class_id = None
            owner.recovery_eligible = False
        return token, float(score), bool(class_match)

    def _record_observation(self, logical_track, observation, frame_index, update_motion):
        gap = frame_index - logical_track.last_frame
        if update_motion and gap > 0:
            measured_velocity = (observation["center"] - logical_track.last_center) / gap
            if logical_track.observations <= 1:
                logical_track.velocity = measured_velocity.astype(np.float32)
            else:
                logical_track.velocity = (0.7 * logical_track.velocity + 0.3 * measured_velocity).astype(np.float32)

        if logical_track.appearance is None:
            logical_track.appearance = observation["appearance"]
        elif observation["appearance"] is not None:
            logical_track.appearance = 0.8 * logical_track.appearance + 0.2 * observation["appearance"]
            logical_track.appearance /= max(float(logical_track.appearance.sum()), 1e-12)

        class_weight = self.class_vote_weights.get(observation["cls_id"], 1.0)
        logical_track.class_scores[observation["cls_id"]] += max(observation["confidence"], 0.01) * class_weight
        logical_track.raw_track_ids.add(observation["raw_key"])
        logical_track.last_frame = frame_index
        logical_track.last_center = observation["center"].copy()
        logical_track.last_box = observation["xyxy"].copy()
        logical_track.last_raw_key = observation["raw_key"]
        logical_track.observations += 1
        self.raw_to_logical[observation["raw_key"]] = logical_track.logical_id
        self.recent_updates.append((frame_index, logical_track.logical_id))

        token_id = self.owner_to_token.get(logical_track.logical_id)
        token = self.occupancy_tokens.get(token_id)
        if token is not None:
            token.size = observation["size"].copy()
            if logical_track.appearance is not None:
                token.appearance = logical_track.appearance.copy()

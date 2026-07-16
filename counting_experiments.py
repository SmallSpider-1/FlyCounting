"""Independent building blocks for reproducible circular-counting ablations."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

import numpy as np


STRATEGY_BASELINE = "b0_baseline"
STRATEGY_ROI = "e1_roi"
STRATEGY_SLOW_FAST = "e2_slow_fast"
STRATEGY_VOTE = "e3_vote"
STRATEGY_CHOICES = (STRATEGY_BASELINE, STRATEGY_ROI, STRATEGY_SLOW_FAST, STRATEGY_VOTE)


@dataclass(frozen=True)
class StrategySpec:
    name: str
    use_roi: bool
    use_confirmed_crossing: bool
    use_class_vote: bool
    lock_entry_class: bool


STRATEGY_SPECS = {
    STRATEGY_BASELINE: StrategySpec(STRATEGY_BASELINE, False, False, False, False),
    STRATEGY_ROI: StrategySpec(STRATEGY_ROI, True, False, False, False),
    STRATEGY_SLOW_FAST: StrategySpec(STRATEGY_SLOW_FAST, True, True, False, False),
    STRATEGY_VOTE: StrategySpec(STRATEGY_VOTE, True, True, True, True),
}


def get_strategy_spec(name: str) -> StrategySpec:
    try:
        return STRATEGY_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"未知实验策略: {name}") from exc


def get_square_roi_bounds(
    frame_size: Tuple[int, int],
    center: Tuple[int, int],
    radius: float,
    roi_scale: float = 2.0,
) -> Tuple[int, int, int, int]:
    """Return a centered square ROI, shifting it at image edges instead of clipping one side."""
    if roi_scale <= 1.0:
        raise ValueError("roi_scale 必须大于 1.0，确保计数圆外仍有跟踪空间。")
    width, height = frame_size
    if width <= 0 or height <= 0:
        raise ValueError(f"无效帧尺寸: {frame_size}")

    side = max(2, int(round(2.0 * float(radius) * roi_scale)))
    crop_w = min(side, width)
    crop_h = min(side, height)
    x0 = int(round(center[0] - crop_w / 2.0))
    y0 = int(round(center[1] - crop_h / 2.0))
    x0 = min(max(x0, 0), width - crop_w)
    y0 = min(max(y0, 0), height - crop_h)
    return x0, y0, x0 + crop_w, y0 + crop_h


def offset_detections(
    detections: np.ndarray,
    offset: Tuple[int, int],
    frame_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Map crop-local xyxy detections back to the source-frame coordinate system."""
    if detections.size == 0:
        return detections.copy()
    mapped = detections.copy()
    offset_x, offset_y = offset
    mapped[:, [0, 2]] += float(offset_x)
    mapped[:, [1, 3]] += float(offset_y)
    if frame_size is not None:
        width, height = frame_size
        mapped[:, [0, 2]] = np.clip(mapped[:, [0, 2]], 0, width)
        mapped[:, [1, 3]] = np.clip(mapped[:, [1, 3]], 0, height)
    return mapped


@dataclass(frozen=True)
class ClassDecision:
    cls_id: int
    vote_n: int
    score: float
    share: float


@dataclass(frozen=True)
class CrossingDecision:
    accepted: bool
    direction: str
    delta: int
    cls_id: int
    trigger: str
    distance: float
    zone: str
    stable_before: str
    stable_after: str
    raw_cls_id: int
    entry_cls_id: Optional[int]
    vote_n: int
    vote_score: float
    vote_share: float
    paired_exit: bool
    class_source: str
    diagnostic_reason: str = ""


@dataclass(frozen=True)
class EntryCredit:
    cls_id: int
    vote_n: int
    vote_score: float
    vote_share: float


@dataclass
class _TrackState:
    stable_side: Optional[str] = None
    candidate_side: Optional[str] = None
    candidate_streak: int = 0
    candidate_last_frame: Optional[int] = None
    last_observation_frame: Optional[int] = None
    last_strong_frame: Optional[int] = None
    last_strong_zone: Optional[str] = None
    last_strong_distance: Optional[float] = None
    active_credit: Optional[EntryCredit] = None
    votes: Deque[Tuple[int, int, float]] = field(default_factory=deque)


class SlowFastCircleCounter:
    """Per-raw-ID crossing confirmation without ID stitching or ReID.

    Every confirmed enter/exit contributes +1/-1. Entry credit is only used to
    diagnose pairing and, in E3, lock the exit class; it never gates an exit.
    """

    INSIDE = "inside"
    BAND = "band"
    OUTSIDE = "outside"

    def __init__(
        self,
        radius: float,
        margin: float = 8.0,
        confirm_frames: int = 3,
        slow_max_observation_gap: int = 2,
        fast_max_gap_frames: int = 7,
        fast_min_radial_displacement: Optional[float] = None,
        use_class_vote: bool = False,
        lock_entry_class: bool = False,
        vote_window: int = 15,
        vote_max_age_frames: int = 28,
    ) -> None:
        if radius <= 0:
            raise ValueError("radius 必须为正数。")
        if not 0 < margin < radius:
            raise ValueError("margin 必须大于 0 且小于 radius。")
        if confirm_frames < 1:
            raise ValueError("confirm_frames 必须至少为 1。")
        if slow_max_observation_gap < 1 or fast_max_gap_frames < 1:
            raise ValueError("观测间隔参数必须至少为 1 帧。")
        if vote_window < 1 or vote_max_age_frames < 1:
            raise ValueError("类别投票窗口必须至少为 1。")

        self.radius = float(radius)
        self.margin = float(margin)
        self.confirm_frames = int(confirm_frames)
        self.slow_max_observation_gap = int(slow_max_observation_gap)
        self.fast_max_gap_frames = int(fast_max_gap_frames)
        self.fast_min_radial_displacement = float(
            3.0 * margin if fast_min_radial_displacement is None else fast_min_radial_displacement
        )
        self.use_class_vote = bool(use_class_vote)
        self.lock_entry_class = bool(lock_entry_class)
        self.vote_window = int(vote_window)
        self.vote_max_age_frames = int(vote_max_age_frames)
        self.states: Dict[int, _TrackState] = {}
        self.stats: Counter = Counter()

    def zone_for_distance(self, distance: float) -> str:
        if distance <= self.radius - self.margin:
            return self.INSIDE
        if distance >= self.radius + self.margin:
            return self.OUTSIDE
        return self.BAND

    def update(
        self,
        track_id: int,
        frame_index: int,
        distance: float,
        cls_id: int,
        conf: float,
    ) -> Optional[CrossingDecision]:
        state = self.states.setdefault(int(track_id), _TrackState())
        self._append_vote(state, int(frame_index), int(cls_id), float(conf))
        zone = self.zone_for_distance(float(distance))
        observation_gap = (
            None if state.last_observation_frame is None else int(frame_index) - state.last_observation_frame
        )

        decision = self._try_fast_transition(
            state,
            frame_index=int(frame_index),
            distance=float(distance),
            zone=zone,
            raw_cls_id=int(cls_id),
            conf=float(conf),
        )
        if decision is None:
            decision = self._update_slow_candidate(
                state,
                frame_index=int(frame_index),
                distance=float(distance),
                zone=zone,
                raw_cls_id=int(cls_id),
                conf=float(conf),
                observation_gap=observation_gap,
            )

        state.last_observation_frame = int(frame_index)
        if zone != self.BAND:
            state.last_strong_frame = int(frame_index)
            state.last_strong_zone = zone
            state.last_strong_distance = float(distance)
        return decision

    def _append_vote(self, state: _TrackState, frame_index: int, cls_id: int, conf: float) -> None:
        state.votes.append((frame_index, cls_id, conf))
        while len(state.votes) > self.vote_window:
            state.votes.popleft()
        oldest_allowed = frame_index - self.vote_max_age_frames
        while state.votes and state.votes[0][0] < oldest_allowed:
            state.votes.popleft()

    def _class_decision(self, state: _TrackState, raw_cls_id: int, conf: float) -> ClassDecision:
        if not self.use_class_vote or not state.votes:
            return ClassDecision(raw_cls_id, 1, float(conf), 1.0)

        scores: Dict[int, float] = defaultdict(float)
        for _, vote_cls, vote_conf in state.votes:
            scores[vote_cls] += vote_conf
        best_score = max(scores.values())
        tied = {vote_cls for vote_cls, score in scores.items() if abs(score - best_score) <= 1e-12}
        winner = raw_cls_id
        for _, vote_cls, _ in reversed(state.votes):
            if vote_cls in tied:
                winner = vote_cls
                break
        total_score = sum(scores.values())
        share = best_score / total_score if total_score > 0 else 0.0
        return ClassDecision(winner, len(state.votes), best_score, share)

    def _try_fast_transition(
        self,
        state: _TrackState,
        frame_index: int,
        distance: float,
        zone: str,
        raw_cls_id: int,
        conf: float,
    ) -> Optional[CrossingDecision]:
        if zone == self.BAND or state.last_strong_zone not in {self.INSIDE, self.OUTSIDE}:
            return None
        # 快速通道只允许“上一有效观测就在相反稳定区”。如果中间明确看见过
        # BAND，说明这是可观测的渐进跨界，应交给慢速连续观测通道确认。
        if state.last_observation_frame != state.last_strong_frame:
            return None
        if state.last_strong_zone == zone or state.last_strong_frame is None:
            return None
        gap = frame_index - state.last_strong_frame
        radial_displacement = abs(distance - float(state.last_strong_distance))
        if gap < 1 or gap > self.fast_max_gap_frames:
            return None
        if radial_displacement < self.fast_min_radial_displacement:
            return None

        source_side = state.stable_side or state.last_strong_zone
        if source_side != state.last_strong_zone:
            return None
        return self._confirm_transition(
            state,
            source_side=source_side,
            target_side=zone,
            trigger="fast",
            distance=distance,
            raw_cls_id=raw_cls_id,
            conf=conf,
        )

    def _update_slow_candidate(
        self,
        state: _TrackState,
        frame_index: int,
        distance: float,
        zone: str,
        raw_cls_id: int,
        conf: float,
        observation_gap: Optional[int],
    ) -> Optional[CrossingDecision]:
        if zone == self.BAND:
            self._reset_candidate(state)
            return None

        if state.stable_side == zone:
            self._reset_candidate(state)
            return None

        candidate_continues = (
            state.candidate_side == zone
            and observation_gap is not None
            and 1 <= observation_gap <= self.slow_max_observation_gap
        )
        if candidate_continues:
            state.candidate_streak += 1
        else:
            state.candidate_side = zone
            state.candidate_streak = 1
        state.candidate_last_frame = frame_index

        if state.candidate_streak < self.confirm_frames:
            return None
        source_side = state.stable_side
        if source_side is None:
            state.stable_side = zone
            self._reset_candidate(state)
            self.stats[f"initialized_{zone}"] += 1
            return None
        return self._confirm_transition(
            state,
            source_side=source_side,
            target_side=zone,
            trigger="slow",
            distance=distance,
            raw_cls_id=raw_cls_id,
            conf=conf,
        )

    def _confirm_transition(
        self,
        state: _TrackState,
        source_side: str,
        target_side: str,
        trigger: str,
        distance: float,
        raw_cls_id: int,
        conf: float,
    ) -> CrossingDecision:
        stable_before = source_side
        state.stable_side = target_side
        self._reset_candidate(state)

        if source_side == self.OUTSIDE and target_side == self.INSIDE:
            if state.active_credit is not None:
                # 正常状态迁移下不会发生；保留统计便于暴露异常状态，而不是
                # 静默改变“每次确认进入都 +1”的净流量语义。
                self.stats["overwritten_entry_credit"] += 1
            class_decision = self._class_decision(state, raw_cls_id, conf)
            state.active_credit = EntryCredit(
                class_decision.cls_id,
                class_decision.vote_n,
                class_decision.score,
                class_decision.share,
            )
            self.stats[f"accepted_enter_{trigger}"] += 1
            return CrossingDecision(
                accepted=True,
                direction="enter",
                delta=1,
                cls_id=class_decision.cls_id,
                trigger=trigger,
                distance=distance,
                zone=target_side,
                stable_before=stable_before,
                stable_after=target_side,
                raw_cls_id=raw_cls_id,
                entry_cls_id=class_decision.cls_id,
                vote_n=class_decision.vote_n,
                vote_score=class_decision.score,
                vote_share=class_decision.share,
                paired_exit=False,
                class_source="entry_vote" if self.use_class_vote else "entry_trigger",
            )

        if source_side == self.INSIDE and target_side == self.OUTSIDE:
            credit = state.active_credit
            state.active_credit = None
            paired_exit = credit is not None
            if paired_exit and self.lock_entry_class:
                cls_id = credit.cls_id
                vote_n = credit.vote_n
                vote_score = credit.vote_score
                vote_share = credit.vote_share
                class_source = "entry_locked"
            else:
                class_decision = self._class_decision(state, raw_cls_id, conf)
                cls_id = class_decision.cls_id
                vote_n = class_decision.vote_n
                vote_score = class_decision.score
                vote_share = class_decision.share
                class_source = "exit_vote" if self.use_class_vote else "exit_trigger"
            if not paired_exit:
                self.stats["accepted_unpaired_exit"] += 1
            self.stats[f"accepted_exit_{trigger}"] += 1
            return CrossingDecision(
                accepted=True,
                direction="exit",
                delta=-1,
                cls_id=cls_id,
                trigger=trigger,
                distance=distance,
                zone=target_side,
                stable_before=stable_before,
                stable_after=target_side,
                raw_cls_id=raw_cls_id,
                entry_cls_id=credit.cls_id if credit is not None else None,
                vote_n=vote_n,
                vote_score=vote_score,
                vote_share=vote_share,
                paired_exit=paired_exit,
                class_source=class_source,
                diagnostic_reason="" if paired_exit else "unpaired_exit_fallback",
            )

        raise RuntimeError(f"非法稳定区转换: {source_side} -> {target_side}")

    @staticmethod
    def _reset_candidate(state: _TrackState) -> None:
        state.candidate_side = None
        state.candidate_streak = 0
        state.candidate_last_frame = None

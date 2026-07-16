import unittest

import numpy as np

from counting_experiments import (
    STRATEGY_BASELINE,
    STRATEGY_ROI,
    STRATEGY_SLOW_FAST,
    STRATEGY_VOTE,
    SlowFastCircleCounter,
    get_square_roi_bounds,
    get_strategy_spec,
    offset_detections,
)


class TestStrategyMatrix(unittest.TestCase):
    def test_ablation_layers_are_cumulative_and_isolated(self):
        baseline = get_strategy_spec(STRATEGY_BASELINE)
        roi = get_strategy_spec(STRATEGY_ROI)
        slow_fast = get_strategy_spec(STRATEGY_SLOW_FAST)
        vote = get_strategy_spec(STRATEGY_VOTE)

        self.assertEqual(
            (baseline.use_roi, baseline.use_confirmed_crossing, baseline.use_class_vote, baseline.lock_entry_class),
            (False, False, False, False),
        )
        self.assertEqual(
            (roi.use_roi, roi.use_confirmed_crossing, roi.use_class_vote, roi.lock_entry_class),
            (True, False, False, False),
        )
        self.assertEqual(
            (slow_fast.use_roi, slow_fast.use_confirmed_crossing, slow_fast.use_class_vote, slow_fast.lock_entry_class),
            (True, True, False, False),
        )
        self.assertEqual(
            (vote.use_roi, vote.use_confirmed_crossing, vote.use_class_vote, vote.lock_entry_class),
            (True, True, True, True),
        )


class TestRoiHelpers(unittest.TestCase):
    def test_reference_roi(self):
        self.assertEqual(get_square_roi_bounds((2304, 1296), (1184, 724), 150, 2.0), (884, 424, 1484, 1024))

    def test_roi_shifts_at_edges_without_shrinking(self):
        self.assertEqual(get_square_roi_bounds((1000, 800), (80, 70), 100, 2.0), (0, 0, 400, 400))
        self.assertEqual(get_square_roi_bounds((1000, 800), (950, 760), 100, 2.0), (600, 400, 1000, 800))

    def test_roi_scale_must_leave_outer_context(self):
        with self.assertRaises(ValueError):
            get_square_roi_bounds((2304, 1296), (1184, 724), 150, 1.0)

    def test_detection_offset_and_clipping(self):
        dets = np.array([[10, 20, 30, 40, 0.8, 1]], dtype=np.float32)
        mapped = offset_detections(dets, (884, 424), (900, 450))
        np.testing.assert_allclose(mapped[0], [894, 444, 900, 450, 0.8, 1])
        np.testing.assert_allclose(dets[0], [10, 20, 30, 40, 0.8, 1])

    def test_empty_detection_offset(self):
        dets = np.empty((0, 6), dtype=np.float32)
        mapped = offset_detections(dets, (10, 20), (100, 100))
        self.assertEqual(mapped.shape, (0, 6))
        self.assertEqual(mapped.dtype, np.float32)

    def test_reference_roi_maps_local_center_and_radius_to_full_frame(self):
        dets = np.array(
            [
                [290, 290, 310, 310, 0.9, 0],
                [440, 290, 460, 310, 0.9, 0],
            ],
            dtype=np.float32,
        )
        mapped = offset_detections(dets, (884, 424), (2304, 1296))
        centers = np.column_stack(
            (
                (mapped[:, 0] + mapped[:, 2]) / 2.0,
                (mapped[:, 1] + mapped[:, 3]) / 2.0,
            )
        )
        np.testing.assert_allclose(centers[0], [1184, 724])
        np.testing.assert_allclose(centers[1], [1334, 724])


class TestSlowFastCircleCounter(unittest.TestCase):
    def make_slow_counter(self, vote=False):
        return SlowFastCircleCounter(
            radius=150,
            margin=8,
            confirm_frames=3,
            slow_max_observation_gap=2,
            fast_max_gap_frames=1,
            fast_min_radial_displacement=100,
            use_class_vote=vote,
            vote_window=15,
            vote_max_age_frames=30,
        )

    def test_zone_boundaries_are_inclusive(self):
        counter = self.make_slow_counter()
        self.assertEqual(counter.zone_for_distance(142), counter.INSIDE)
        self.assertEqual(counter.zone_for_distance(158), counter.OUTSIDE)
        self.assertEqual(counter.zone_for_distance(150), counter.BAND)

    def test_band_jitter_has_no_event(self):
        counter = self.make_slow_counter()
        decisions = [counter.update(1, i, distance, 0, 0.8) for i, distance in enumerate([149, 151, 148, 152, 150], 1)]
        self.assertTrue(all(decision is None for decision in decisions))

    def test_slow_enter_and_paired_exit(self):
        counter = self.make_slow_counter()
        decisions = []
        frame = 0
        for distance in [170, 170, 170, 150, 140, 140, 140, 150, 170, 170, 170]:
            frame += 1
            decision = counter.update(1, frame, distance, 0, 0.8)
            if decision is not None:
                decisions.append(decision)
        self.assertEqual([(d.accepted, d.direction, d.delta, d.trigger) for d in decisions], [(True, "enter", 1, "slow"), (True, "exit", -1, "slow")])
        self.assertEqual(decisions[1].cls_id, decisions[0].cls_id)

    def test_slow_candidate_resets_in_band(self):
        counter = self.make_slow_counter()
        for frame, distance in enumerate([170, 170, 170, 140, 140, 150, 140, 140], 1):
            self.assertIsNone(counter.update(1, frame, distance, 0, 0.8))
        decision = counter.update(1, 9, 140, 0, 0.8)
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.direction, "enter")

    def test_fast_enter_and_exit_without_class_lock(self):
        counter = SlowFastCircleCounter(radius=150, margin=8, confirm_frames=3, fast_max_gap_frames=3)
        self.assertIsNone(counter.update(1, 1, 170, 0, 0.8))
        enter = counter.update(1, 2, 130, 0, 0.8)
        exit_event = counter.update(1, 3, 170, 1, 0.9)
        self.assertEqual((enter.accepted, enter.direction, enter.trigger), (True, "enter", "fast"))
        self.assertEqual((exit_event.accepted, exit_event.direction, exit_event.trigger), (True, "exit", "fast"))
        self.assertTrue(exit_event.paired_exit)
        self.assertEqual(exit_event.cls_id, 1, "E2 只改变跨界确认，仍使用退出触发帧类别")

    def test_band_observation_blocks_fast_and_requires_slow_confirmation(self):
        counter = SlowFastCircleCounter(radius=150, margin=8, confirm_frames=3, fast_max_gap_frames=7)
        decisions = []
        for frame, distance in enumerate([170, 170, 170, 150, 140, 140, 140, 150, 170, 170, 170], 1):
            decision = counter.update(1, frame, distance, 0, 0.8)
            if decision is not None:
                decisions.append(decision)
        self.assertEqual(
            [(decision.direction, decision.trigger, decision.paired_exit) for decision in decisions],
            [("enter", "slow", False), ("exit", "slow", True)],
        )

    def test_fast_transition_rejects_long_gap(self):
        counter = SlowFastCircleCounter(radius=150, margin=8, confirm_frames=3, fast_max_gap_frames=3)
        self.assertIsNone(counter.update(1, 1, 170, 0, 0.8))
        self.assertIsNone(counter.update(1, 6, 130, 0, 0.8))

    def test_born_inside_exit_uses_unpaired_fallback(self):
        counter = SlowFastCircleCounter(radius=150, margin=8, confirm_frames=3, fast_max_gap_frames=3)
        for frame in range(1, 4):
            self.assertIsNone(counter.update(1, frame, 130, 1, 0.8))
        decision = counter.update(1, 4, 170, 1, 0.8)
        self.assertTrue(decision.accepted)
        self.assertFalse(decision.paired_exit)
        self.assertEqual(
            (decision.direction, decision.delta, decision.cls_id, decision.diagnostic_reason),
            ("exit", -1, 1, "unpaired_exit_fallback"),
        )

    def test_real_reentry_is_allowed(self):
        counter = SlowFastCircleCounter(radius=150, margin=8, confirm_frames=3, fast_max_gap_frames=3)
        sequence = [(1, 170), (2, 130), (3, 170), (4, 130)]
        events = [counter.update(1, frame, distance, 0, 0.8) for frame, distance in sequence]
        events = [event for event in events if event and event.accepted]
        self.assertEqual([(event.direction, event.delta) for event in events], [("enter", 1), ("exit", -1), ("enter", 1)])

    def test_different_ids_are_never_stitched(self):
        counter = SlowFastCircleCounter(radius=150, margin=8, confirm_frames=3, fast_max_gap_frames=3)
        self.assertIsNone(counter.update(1, 1, 170, 0, 0.8))
        self.assertIsNone(counter.update(2, 2, 130, 0, 0.8))

    def test_vote_overrides_trigger_frame_and_exit_uses_locked_class(self):
        counter = SlowFastCircleCounter(
            radius=150,
            margin=8,
            confirm_frames=3,
            fast_max_gap_frames=3,
            use_class_vote=True,
            lock_entry_class=True,
            vote_window=15,
            vote_max_age_frames=30,
        )
        self.assertIsNone(counter.update(1, 1, 170, 1, 0.8))
        self.assertIsNone(counter.update(1, 2, 170, 1, 0.8))
        enter = counter.update(1, 3, 130, 0, 0.9)
        exit_event = counter.update(1, 4, 170, 0, 0.9)
        self.assertEqual(enter.cls_id, 1)
        self.assertEqual(enter.vote_n, 3)
        self.assertEqual(exit_event.cls_id, 1)
        self.assertTrue(exit_event.paired_exit)
        self.assertEqual(exit_event.class_source, "entry_locked")

    def test_unpaired_exit_in_vote_strategy_uses_current_track_vote(self):
        counter = SlowFastCircleCounter(
            radius=150,
            margin=8,
            confirm_frames=3,
            fast_max_gap_frames=3,
            use_class_vote=True,
            lock_entry_class=True,
            vote_window=15,
            vote_max_age_frames=30,
        )
        for frame, cls_id in [(1, 0), (2, 1), (3, 1)]:
            self.assertIsNone(counter.update(1, frame, 130, cls_id, 0.8))
        exit_event = counter.update(1, 4, 170, 1, 0.8)
        self.assertFalse(exit_event.paired_exit)
        self.assertEqual(exit_event.cls_id, 1)
        self.assertEqual(exit_event.class_source, "exit_vote")

    def test_vote_expires_old_observations(self):
        counter = SlowFastCircleCounter(
            radius=150,
            margin=8,
            confirm_frames=3,
            fast_max_gap_frames=3,
            use_class_vote=True,
            vote_window=15,
            vote_max_age_frames=2,
        )
        self.assertIsNone(counter.update(1, 1, 170, 1, 0.99))
        self.assertIsNone(counter.update(1, 10, 170, 0, 0.6))
        enter = counter.update(1, 11, 130, 0, 0.6)
        self.assertEqual(enter.cls_id, 0)
        self.assertEqual(enter.vote_n, 2)


if __name__ == "__main__":
    unittest.main()

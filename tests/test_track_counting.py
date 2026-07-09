import numpy as np

from track_counting import RegionOccupancyCounter


def make_track(center, track_id=1, cls_id=0, confidence=0.9, size=16):
    x, y = center
    half = size / 2
    return (np.array([x - half, y - half, x + half, y + half]), track_id, cls_id, confidence)


def update(counter, frame, frame_index, *tracks):
    _, events = counter.update(list(tracks), frame, frame_index, namespace="clip")
    return events


def test_hysteresis_band_ignores_boundary_jitter():
    frame = np.full((240, 240, 3), 180, dtype=np.uint8)
    counter = RegionOccupancyCounter(center=(120, 120), radius=50, band_width=10, enable_stitching=False)

    assert not update(counter, frame, 1, make_track((181, 120)))
    for frame_index, x in enumerate((178, 171, 169, 175, 172, 179), start=2):
        assert not update(counter, frame, frame_index, make_track((x, 120)))

    events = update(counter, frame, 8, make_track((159, 120)))
    assert [(event["direction"], event["delta"]) for event in events] == [("enter", 1)]

    for frame_index, x in enumerate((162, 169, 173, 168), start=9):
        assert not update(counter, frame, frame_index, make_track((x, 120)))

    events = update(counter, frame, 13, make_track((181, 120)))
    assert [(event["direction"], event["delta"]) for event in events] == [("exit", -1)]


def test_id_fragment_is_stitched_across_a_crossing():
    frame = np.full((240, 240, 3), 200, dtype=np.uint8)
    counter = RegionOccupancyCounter(
        center=(120, 120),
        radius=50,
        band_width=10,
        stitch_max_gap_frames=10,
        stitch_max_distance=40,
        stitch_min_score=0.45,
    )

    assert not update(counter, frame, 1, make_track((181, 120), track_id=10))
    assert not update(counter, frame, 2, make_track((174, 120), track_id=10))
    events = update(counter, frame, 5, make_track((158, 120), track_id=77))

    assert len(events) == 1
    assert events[0]["direction"] == "enter"
    assert events[0]["logical_track_id"] == 1
    assert events[0]["raw_track_id"] == 77
    assert events[0]["stitched"] is True
    assert events[0]["stitched_now"] is True
    assert events[0]["stitch_gap_frames"] == 3
    assert counter.logical_tracks[1].raw_track_ids == {("clip", 10), ("clip", 77)}


def test_distant_new_id_is_not_stitched_or_counted():
    frame = np.full((240, 240, 3), 200, dtype=np.uint8)
    counter = RegionOccupancyCounter(
        center=(120, 120),
        radius=50,
        band_width=10,
        stitch_max_gap_frames=10,
        stitch_max_distance=30,
        stitch_min_score=0.45,
    )

    assert not update(counter, frame, 1, make_track((181, 120), track_id=10))
    events = update(counter, frame, 3, make_track((82, 120), track_id=77))

    assert not events
    assert len(counter.logical_tracks) == 2


def test_tracker_id_reset_between_video_segments_can_be_stitched():
    frame = np.full((240, 240, 3), 200, dtype=np.uint8)
    counter = RegionOccupancyCounter(
        center=(120, 120),
        radius=50,
        band_width=10,
        stitch_max_gap_frames=5,
        stitch_max_distance=40,
        stitch_min_score=0.45,
    )

    _, first_events = counter.update([make_track((181, 120), track_id=1)], frame, 1, namespace="part-a")
    _, second_events = counter.update([make_track((159, 120), track_id=1)], frame, 2, namespace="part-b")

    assert not first_events
    assert second_events[0]["direction"] == "enter"
    assert second_events[0]["logical_track_id"] == 1
    assert second_events[0]["stitched"] is True


def test_exit_uses_class_frozen_at_entry():
    frame = np.full((240, 240, 3), 200, dtype=np.uint8)
    counter = RegionOccupancyCounter(center=(120, 120), radius=50, band_width=10, enable_stitching=False)

    assert not update(counter, frame, 1, make_track((181, 120), cls_id=1))
    assert not update(counter, frame, 2, make_track((170, 120), cls_id=1))
    enter = update(counter, frame, 3, make_track((159, 120), cls_id=0, confidence=0.6))
    assert enter[0]["count_cls_id"] == 1

    assert not update(counter, frame, 4, make_track((170, 120), cls_id=0))
    exit_events = update(counter, frame, 5, make_track((181, 120), cls_id=0))
    assert exit_events[0]["count_cls_id"] == 1


def test_lost_inside_track_exit_is_recovered_by_inventory_token():
    frame = np.full((240, 240, 3), 200, dtype=np.uint8)
    counter = RegionOccupancyCounter(
        center=(120, 120),
        radius=50,
        band_width=10,
        enable_stitching=False,
        recovery_min_gap_frames=3,
    )

    assert not update(counter, frame, 1, make_track((181, 120), track_id=10))
    enter = update(counter, frame, 2, make_track((159, 120), track_id=10))
    assert enter[0]["event_reason"] == "tracked_enter"

    assert not update(counter, frame, 6, make_track((159, 120), track_id=77))
    recovered_exit = update(counter, frame, 7, make_track((181, 120), track_id=77))

    assert recovered_exit[0]["event_reason"] == "recovered_exit"
    assert recovered_exit[0]["recovered_from_logical_id"] == 1
    assert recovered_exit[0]["count_cls_id"] == 0
    assert not counter.occupancy_tokens


def test_recent_visible_owner_token_cannot_be_stolen():
    frame = np.full((240, 240, 3), 200, dtype=np.uint8)
    counter = RegionOccupancyCounter(
        center=(120, 120),
        radius=50,
        band_width=10,
        enable_stitching=False,
        recovery_min_gap_frames=5,
    )

    assert not update(counter, frame, 1, make_track((181, 120), track_id=10))
    assert update(counter, frame, 2, make_track((159, 120), track_id=10))
    assert not update(counter, frame, 3, make_track((158, 120), track_id=10), make_track((159, 120), track_id=77))
    events = update(counter, frame, 4, make_track((157, 120), track_id=10), make_track((181, 120), track_id=77))

    assert not events
    assert len(counter.occupancy_tokens) == 1


def test_initial_inside_track_does_not_exit_without_inventory():
    frame = np.full((240, 240, 3), 200, dtype=np.uint8)
    counter = RegionOccupancyCounter(center=(120, 120), radius=50, band_width=10, enable_stitching=False)

    assert not update(counter, frame, 1, make_track((159, 120)))
    assert not update(counter, frame, 2, make_track((181, 120)))
    assert not counter.occupancy_tokens


def test_class_vote_weights_can_correct_detector_prior():
    frame = np.full((240, 240, 3), 200, dtype=np.uint8)
    counter = RegionOccupancyCounter(
        center=(120, 120),
        radius=50,
        band_width=10,
        enable_stitching=False,
        class_vote_weights={0: 0.35, 1: 1.0},
    )

    assert not update(counter, frame, 1, make_track((181, 120), cls_id=0, confidence=0.9))
    assert not update(counter, frame, 2, make_track((170, 120), cls_id=1, confidence=0.7))
    enter = update(counter, frame, 3, make_track((159, 120), cls_id=0, confidence=0.9))

    assert enter[0]["count_cls_id"] == 1


def test_entry_detected_class_can_override_historical_vote_for_selected_classes():
    frame = np.full((240, 240, 3), 200, dtype=np.uint8)
    counter = RegionOccupancyCounter(
        center=(120, 120),
        radius=50,
        band_width=10,
        enable_stitching=False,
        entry_detected_class_override={1},
    )

    assert not update(counter, frame, 1, make_track((181, 120), cls_id=0, confidence=0.9))
    assert not update(counter, frame, 2, make_track((170, 120), cls_id=0, confidence=0.9))
    enter = update(counter, frame, 3, make_track((159, 120), cls_id=1, confidence=0.6))

    assert enter[0]["voted_class_id"] == 0
    assert enter[0]["count_cls_id"] == 1
    assert next(iter(counter.occupancy_tokens.values())).class_id == 1


def test_recovery_can_require_class_match():
    frame = np.full((240, 240, 3), 200, dtype=np.uint8)
    counter = RegionOccupancyCounter(
        center=(120, 120),
        radius=50,
        band_width=10,
        enable_stitching=False,
        recovery_min_gap_frames=3,
        recovery_require_class_match=True,
    )

    assert not update(counter, frame, 1, make_track((181, 120), track_id=10, cls_id=1))
    assert update(counter, frame, 2, make_track((159, 120), track_id=10, cls_id=1))

    assert not update(counter, frame, 6, make_track((159, 120), track_id=77, cls_id=0))
    assert not update(counter, frame, 7, make_track((181, 120), track_id=77, cls_id=0))
    assert len(counter.occupancy_tokens) == 1


def test_recovery_class_match_can_be_required_for_selected_classes_only():
    frame = np.full((240, 240, 3), 200, dtype=np.uint8)
    counter = RegionOccupancyCounter(
        center=(120, 120),
        radius=50,
        band_width=10,
        enable_stitching=False,
        recovery_min_gap_frames=3,
        recovery_class_match_required={1},
    )

    assert not update(counter, frame, 1, make_track((181, 120), track_id=10, cls_id=0))
    assert update(counter, frame, 2, make_track((159, 120), track_id=10, cls_id=0))

    assert not update(counter, frame, 6, make_track((159, 120), track_id=77, cls_id=1))
    recovered_exit = update(counter, frame, 7, make_track((181, 120), track_id=77, cls_id=1))

    assert recovered_exit[0]["event_reason"] == "recovered_exit"
    assert recovered_exit[0]["count_cls_id"] == 0
    assert recovered_exit[0]["recovery_class_match"] == 0
    assert not counter.occupancy_tokens


def test_recovery_class_min_score_can_block_one_class():
    frame = np.full((240, 240, 3), 200, dtype=np.uint8)
    counter = RegionOccupancyCounter(
        center=(120, 120),
        radius=50,
        band_width=10,
        enable_stitching=False,
        recovery_min_gap_frames=3,
        recovery_class_min_scores={1: 1.1},
    )

    assert not update(counter, frame, 1, make_track((181, 120), track_id=10, cls_id=1))
    assert update(counter, frame, 2, make_track((159, 120), track_id=10, cls_id=1))

    assert not update(counter, frame, 6, make_track((159, 120), track_id=77, cls_id=1))
    assert not update(counter, frame, 7, make_track((181, 120), track_id=77, cls_id=1))
    assert len(counter.occupancy_tokens) == 1


def test_band_inner_half_start_can_recover_exit_when_enabled():
    frame = np.full((240, 240, 3), 200, dtype=np.uint8)
    counter = RegionOccupancyCounter(
        center=(120, 120),
        radius=50,
        band_width=10,
        enable_stitching=False,
        recovery_min_gap_frames=3,
        enable_band_start_recovery=True,
    )

    assert not update(counter, frame, 1, make_track((181, 120), track_id=10))
    assert update(counter, frame, 2, make_track((159, 120), track_id=10))

    assert not update(counter, frame, 6, make_track((165, 120), track_id=77))
    recovered_exit = update(counter, frame, 7, make_track((181, 120), track_id=77))

    assert recovered_exit[0]["event_reason"] == "recovered_exit"
    assert not counter.occupancy_tokens


def test_band_outer_half_start_does_not_recover_exit():
    frame = np.full((240, 240, 3), 200, dtype=np.uint8)
    counter = RegionOccupancyCounter(
        center=(120, 120),
        radius=50,
        band_width=10,
        enable_stitching=False,
        recovery_min_gap_frames=3,
        enable_band_start_recovery=True,
    )

    assert not update(counter, frame, 1, make_track((181, 120), track_id=10))
    assert update(counter, frame, 2, make_track((159, 120), track_id=10))

    assert not update(counter, frame, 6, make_track((175, 120), track_id=77))
    assert not update(counter, frame, 7, make_track((181, 120), track_id=77))
    assert len(counter.occupancy_tokens) == 1


def test_outer_near_boundary_start_can_recover_after_moving_outward():
    frame = np.full((240, 240, 3), 200, dtype=np.uint8)
    counter = RegionOccupancyCounter(
        center=(120, 120),
        radius=50,
        band_width=10,
        enable_stitching=False,
        recovery_min_gap_frames=3,
        enable_outer_start_recovery=True,
        outer_start_max_offset=20,
        outer_start_min_outward=8,
        outer_start_min_frames=2,
        outer_start_max_frames=10,
    )

    assert not update(counter, frame, 1, make_track((181, 120), track_id=10))
    assert update(counter, frame, 2, make_track((159, 120), track_id=10))

    assert not update(counter, frame, 6, make_track((181, 120), track_id=77))
    assert not update(counter, frame, 7, make_track((185, 120), track_id=77))
    recovered_exit = update(counter, frame, 8, make_track((193, 120), track_id=77))

    assert recovered_exit[0]["event_reason"] == "recovered_exit"
    assert recovered_exit[0]["recovery_mode"] == "outer_start"
    assert not counter.occupancy_tokens


def test_outer_start_recovery_requires_outward_motion():
    frame = np.full((240, 240, 3), 200, dtype=np.uint8)
    counter = RegionOccupancyCounter(
        center=(120, 120),
        radius=50,
        band_width=10,
        enable_stitching=False,
        recovery_min_gap_frames=3,
        enable_outer_start_recovery=True,
        outer_start_max_offset=20,
        outer_start_min_outward=8,
        outer_start_min_frames=2,
        outer_start_max_frames=10,
    )

    assert not update(counter, frame, 1, make_track((181, 120), track_id=10))
    assert update(counter, frame, 2, make_track((159, 120), track_id=10))

    assert not update(counter, frame, 6, make_track((181, 120), track_id=77))
    assert not update(counter, frame, 7, make_track((182, 120), track_id=77))
    assert not update(counter, frame, 8, make_track((183, 120), track_id=77))
    assert len(counter.occupancy_tokens) == 1


def test_same_logical_track_reentry_can_be_suppressed_by_cooldown():
    frame = np.full((240, 240, 3), 200, dtype=np.uint8)
    counter = RegionOccupancyCounter(
        center=(120, 120),
        radius=50,
        band_width=10,
        enable_stitching=False,
        reentry_cooldown_frames=5,
    )

    assert not update(counter, frame, 1, make_track((181, 120)))
    assert update(counter, frame, 2, make_track((159, 120)))
    assert update(counter, frame, 3, make_track((181, 120)))
    assert not update(counter, frame, 4, make_track((159, 120)))

    assert not counter.occupancy_tokens


def test_same_logical_track_reentry_after_cooldown_is_counted():
    frame = np.full((240, 240, 3), 200, dtype=np.uint8)
    counter = RegionOccupancyCounter(
        center=(120, 120),
        radius=50,
        band_width=10,
        enable_stitching=False,
        reentry_cooldown_frames=2,
    )

    assert not update(counter, frame, 1, make_track((181, 120)))
    assert update(counter, frame, 2, make_track((159, 120)))
    assert update(counter, frame, 3, make_track((181, 120)))
    assert not update(counter, frame, 6, make_track((170, 120)))
    reenter = update(counter, frame, 7, make_track((159, 120)))

    assert reenter[0]["direction"] == "enter"
    assert len(counter.occupancy_tokens) == 1

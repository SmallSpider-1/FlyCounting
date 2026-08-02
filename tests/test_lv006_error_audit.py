from __future__ import annotations

import csv
import inspect
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_lv006_error_audit import (
    HUMAN_COLUMNS,
    REVIEW_COLUMNS,
    Case,
    Obs,
    Seed,
    Segment,
    TrackState,
    assign_formal_events,
    build_residual_rows,
    compute_formal_delta,
    deduplicate_seeds,
    derive_expected_delta,
    invalidate_existing_reviews,
    parse_packet_pts_lines,
    read_csv,
    scheduled_source_frames,
    write_contact_sheet,
    write_review_sheet,
)


def reviewed(outcome: str, **updates: str) -> dict[str, str]:
    row = {column: "" for column in HUMAN_COLUMNS}
    row.update(
        {
            "review_status": "reviewed",
            "review_outcome": outcome,
            "reviewer": "tester",
            "confidence": "high",
        }
    )
    row.update(updates)
    return row


def test_formal_delta_summary() -> None:
    assert compute_formal_delta(
        [
            {"class_id": 0, "delta": 1},
            {"class_id": 0, "delta": -1},
            {"class_id": 1, "delta": 1},
        ]
    ) == (0, 1)


def test_formal_correct_maps_expected_to_formal() -> None:
    result = derive_expected_delta(reviewed("formal_correct"), -2, 3)
    assert (result["expected_delta_D"], result["expected_delta_C"]) == (-2, 3)
    assert (result["bias_D"], result["bias_C"]) == (0, 0)


def test_true_crossing_missed_requires_and_maps_species_direction() -> None:
    result = derive_expected_delta(
        reviewed("true_crossing_missed", expected_species="dorsalis", expected_direction="enter"), 0, 0
    )
    assert (result["expected_delta_D"], result["expected_delta_C"]) == (1, 0)
    assert (result["bias_D"], result["correction_D"]) == (-1, 1)


def test_wrong_species_creates_cross_species_bias() -> None:
    result = derive_expected_delta(
        reviewed("wrong_species", expected_species="dorsalis", expected_direction="enter"), 0, 1
    )
    assert (result["expected_delta_D"], result["expected_delta_C"]) == (1, 0)
    assert (result["bias_D"], result["bias_C"]) == (-1, 1)


def test_wrong_direction_uses_confirmed_direction() -> None:
    result = derive_expected_delta(
        reviewed("wrong_direction", expected_species="dorsalis", expected_direction="exit"), 1, 0
    )
    assert (result["expected_delta_D"], result["expected_delta_C"]) == (-1, 0)
    assert result["bias_D"] == 2


def test_false_formal_event_expected_is_zero() -> None:
    result = derive_expected_delta(reviewed("formal_event_not_real"), -1, 1)
    assert (result["expected_delta_D"], result["expected_delta_C"]) == (0, 0)
    assert (result["bias_D"], result["bias_C"]) == (-1, 1)


def test_uncertain_leaves_numeric_fields_empty() -> None:
    result = derive_expected_delta(reviewed("uncertain"), 1, -1)
    assert result["confirmation_state"] == "uncertain"
    for key in ("expected_delta_D", "expected_delta_C", "bias_D", "bias_C", "correction_D", "correction_C"):
        assert result[key] is None


def test_override_takes_precedence() -> None:
    result = derive_expected_delta(
        reviewed(
            "wrong_species",
            expected_species="dorsalis",
            expected_direction="enter",
            expected_delta_D_override="2",
            expected_delta_C_override="-1",
        ),
        1,
        0,
    )
    assert (result["expected_delta_D"], result["expected_delta_C"]) == (2, -1)
    assert result["delta_source"] == "manual_override"


def test_bias_and_correction_have_opposite_signs() -> None:
    result = derive_expected_delta(
        reviewed("true_crossing_missed", expected_species="correcta", expected_direction="enter"), 0, 0
    )
    assert result["bias_D"] == -result["correction_D"]
    assert result["bias_C"] == -result["correction_C"]


def test_residual_summary_excludes_uncertain_and_pending() -> None:
    ledger = [
        {"segment_group": "S007", "confirmation_state": "confirmed", "bias_D": -2, "bias_C": 1},
        {"segment_group": "S007", "confirmation_state": "uncertain", "bias_D": "", "bias_C": ""},
        {"segment_group": "S007", "confirmation_state": "pending_second_review", "bias_D": -5, "bias_C": -5},
    ]
    summary = {row["segment_group"]: row for row in build_residual_rows(ledger)}
    assert summary["S007"]["confirmed_bias_D"] == -2
    assert summary["S007"]["residual_D"] == -21
    assert summary["S007"]["uncertain_cases"] == 1
    assert summary["S007"]["needs_second_review_cases"] == 1


def fake_segment(index: int = 7) -> Segment:
    return Segment(index, Path("video.mp4"), Path("tracks.jsonl"), Path("detections.jsonl"), 1000, 2304, 1296, 50.0)


def test_case_dedup_merges_same_id_within_two_seconds() -> None:
    segment = fake_segment()
    obs = Obs(80, 1160, 700, 1200, 740, 9, 0.9, 0, 0)
    states = {9: TrackState(9, obs, obs)}
    seeds = [
        Seed(7, 100, {9}, {"repeated_crossing"}, 30.0, ["a"], 1180, 720, [100]),
        Seed(7, 130, {9}, {"box_center_jitter"}, 40.0, ["b"], 1182, 722, [130]),
    ]
    cases = deduplicate_seeds(segment, seeds, states, [])
    assert len(cases) == 1
    assert set(cases[0].candidate_flags) == {"repeated_crossing", "box_center_jitter"}


def test_formal_event_is_assigned_only_once() -> None:
    event = {
        "event_key": "E000001",
        "formal_row_index": 1,
        "segment_index": 7,
        "frame": 110,
        "track_id": 9,
        "class_id": 0,
        "delta": 1,
    }
    cases = [
        Case("A", 7, "S007", 90, 100, 140, [9], ["a"], 50.0, "a"),
        Case("B", 7, "S007", 90, 115, 140, [9], ["b"], 40.0, "b"),
    ]
    assign_formal_events(cases, [event])
    assert sum(len(case.formal_events) for case in cases) == 1
    assert cases[1].formal_events[0]["event_key"] == "E000001"


def test_regeneration_preserves_human_review_fields(tmp_path: Path) -> None:
    review_path = tmp_path / "review_sheet.csv"
    existing = {column: "" for column in REVIEW_COLUMNS}
    existing.update(
        {
            "case_id": "LV006_S007_C0001",
            "review_status": "reviewed",
            "review_outcome": "formal_correct",
            "reviewer": "alice",
            "confidence": "high",
            "notes": "keep me",
        }
    )
    with review_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerow(existing)
    manifest = {
        "case_id": "LV006_S007_C0001",
        "segment_group": "S007",
        "segment_index": "7",
        "clip_path": "/tmp/a.mp4",
        "thumbnail_path": "/tmp/a.jpg",
        "anchor_timestamp": "00:00:01.000",
        "track_ids": "9",
        "candidate_flags": "repeated_crossing",
        "priority_score": "50",
        "priority_reason": "test",
        "formal_event_rows": "E000001",
        "formal_delta_D": "1",
        "formal_delta_C": "0",
        "render_selected": "yes",
    }
    write_review_sheet(tmp_path, [manifest])
    row = read_csv(review_path)[0]
    assert row["reviewer"] == "alice"
    assert row["notes"] == "keep me"


def test_contact_sheet_uses_http_safe_relative_paths(tmp_path: Path) -> None:
    clip = tmp_path / "cases" / "segment_007" / "LV006_S007_C0001.mp4"
    thumbnail = tmp_path / "thumbnails" / "segment_007" / "LV006_S007_C0001.jpg"
    row = {
        "case_id": "LV006_S007_C0001",
        "segment_index": "7",
        "anchor_timestamp": "00:00:01.000",
        "candidate_flags": "repeated_crossing",
        "priority_score": "50",
        "priority_reason": "test",
        "priority_rank_global": "1",
        "render_selected": "yes",
        "clip_path": str(clip),
        "thumbnail_path": str(thumbnail),
    }
    write_contact_sheet(tmp_path, [row])
    content = (tmp_path / "contact_sheet.html").read_text(encoding="utf-8")
    assert "file://" not in content
    assert "href='cases/segment_007/LV006_S007_C0001.mp4'" in content
    assert "src='thumbnails/segment_007/LV006_S007_C0001.jpg'" in content


def test_invalidated_reviews_are_archived_then_cleared(tmp_path: Path) -> None:
    review_path = tmp_path / "review_sheet.csv"
    archive_path = tmp_path / "invalidated.csv"
    row = {column: "" for column in REVIEW_COLUMNS}
    row.update(
        {
            "case_id": "LV006_S007_C0001",
            "review_status": "reviewed",
            "review_outcome": "formal_event_not_real",
            "notes": "made from a bad clip",
        }
    )
    with review_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    assert invalidate_existing_reviews(review_path, archive_path) == 1
    cleared = read_csv(review_path)[0]
    assert all(cleared[column] == "" for column in HUMAN_COLUMNS)
    archived = read_csv(archive_path)[0]
    assert archived["case_id"] == "LV006_S007_C0001"
    assert archived["notes"] == "made from a bad clip"


def test_packet_pts_are_sorted_into_display_order() -> None:
    assert parse_packet_pts_lines(["0.040000", "0.000000", "0.020000"], 3) == [0.0, 0.02, 0.04]


def test_source_reader_seeks_by_pts_and_preserves_ordinal_frames() -> None:
    frame_pts = [1.00, 1.01, 1.02, 1.03]

    class FakeCapture:
        def __init__(self, _: str):
            self.next_index = 0
            self.last_index = -1
            self.seek_values: list[float] = []

        def isOpened(self) -> bool:
            return True

        def set(self, prop: int, value: float) -> bool:
            assert prop == cv2.CAP_PROP_POS_MSEC
            self.seek_values.append(value)
            self.next_index = min(range(len(frame_pts)), key=lambda index: abs(frame_pts[index] * 1000 - value))
            self.last_index = self.next_index - 1
            return True

        def read(self) -> tuple[bool, np.ndarray]:
            if self.next_index >= len(frame_pts):
                return False, np.empty((0, 0, 3), dtype=np.uint8)
            self.last_index = self.next_index
            self.next_index += 1
            return True, np.full((1, 1, 3), self.last_index + 1, dtype=np.uint8)

        def get(self, prop: int) -> float:
            assert prop == cv2.CAP_PROP_POS_MSEC
            return frame_pts[self.last_index] * 1000

        def release(self) -> None:
            pass

    capture = FakeCapture("unused")
    with patch("scripts.build_lv006_error_audit.cv2.VideoCapture", return_value=capture):
        rows = list(scheduled_source_frames(Path("video.mp4"), [2, 4, 4, 3], frame_pts))
    assert [frame for frame, _ in rows] == [2, 4, 4, 3]
    assert [int(image[0, 0, 0]) for _, image in rows] == [2, 4, 4, 3]
    assert capture.seek_values == [1000.0, 1000.0]


def test_segments_one_and_two_share_only_combined_target_group() -> None:
    assert fake_segment(1).group == "S001-002"
    assert fake_segment(2).group == "S001-002"
    assert "S001" not in {row["segment_group"] for row in build_residual_rows([])}
    assert "S002" not in {row["segment_group"] for row in build_residual_rows([])}


def test_illegal_review_combinations_are_reported() -> None:
    missing_direction = derive_expected_delta(
        reviewed("true_crossing_missed", expected_species="dorsalis"), 0, 0
    )
    one_override = derive_expected_delta(
        reviewed("wrong_species", expected_delta_D_override="1"), 0, 1
    )
    assert missing_direction["confirmation_state"] == "invalid"
    assert one_override["confirmation_state"] == "invalid"


def main() -> int:
    """Run the same tests without requiring pytest in the benchmark environment."""
    passed = []
    for name, function in sorted(globals().items()):
        if not name.startswith("test_") or not callable(function):
            continue
        parameters = list(inspect.signature(function).parameters)
        if not parameters:
            function()
        elif parameters == ["tmp_path"]:
            with tempfile.TemporaryDirectory(prefix="lv006_error_audit_test_") as directory:
                function(Path(directory))
        else:
            raise RuntimeError(f"Unsupported test signature: {name}{inspect.signature(function)}")
        passed.append(name)
    print(f"PASS {len(passed)} tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

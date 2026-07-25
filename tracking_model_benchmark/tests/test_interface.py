import json
import unittest
from pathlib import Path

import numpy as np

from benchmark_common.numeric_cache import CACHE_FIELDS
from tracking_model_benchmark._common.tracker_adapters import (
    DEFAULT_CONFIGS,
    TRACKER_PROJECTS,
    resolved_config,
)
from tracking_model_benchmark._common.tracker_interface import (
    DETECTION_FIELDS,
    TRACK_FIELDS,
    FrameGeometry,
    UnifiedTrackerAdapter,
)


ROOT = Path(__file__).resolve().parents[2]


class FakeAdapter(UnifiedTrackerAdapter):
    tracker_name = "fake"

    def build_native_tracker(self):
        return object()

    def update_native(self, local_detections, frame_index):
        if len(local_detections):
            return np.array([[*local_detections[0, :4], 9]], dtype=np.float32)
        return np.array([[5, 5, 25, 25, 9]], dtype=np.float32)


class TestUnifiedTrackerContract(unittest.TestCase):
    def test_fields_match_numeric_cache_schema(self):
        self.assertEqual(list(DETECTION_FIELDS), CACHE_FIELDS["detections"])
        self.assertEqual(list(TRACK_FIELDS), CACHE_FIELDS["tracks"])

    def test_full_frame_coordinates_pass_through(self):
        adapter = FakeAdapter(FrameGeometry((100, 80), 30.0), {"metadata_match_iou": 0.1})
        detection = np.array([[25, 15, 45, 35, 0.8, 1]], dtype=np.float32)
        tracks = adapter.update(detection, 1)
        np.testing.assert_allclose(tracks[0], [25, 15, 45, 35, 9, 0.8, 1, 0])

    def test_unmatched_track_carries_last_class_without_voting(self):
        adapter = FakeAdapter(FrameGeometry((100, 80), 30.0), {"metadata_match_iou": 0.1})
        adapter.update(np.array([[25, 15, 45, 35, 0.8, 1]], dtype=np.float32), 1)
        tracks = adapter.update(np.empty((0, 6), dtype=np.float32), 2)
        self.assertEqual(int(tracks[0, 6]), 1)
        self.assertEqual(int(tracks[0, 7]), -1)

    def test_frame_numbers_must_be_continuous(self):
        adapter = FakeAdapter(FrameGeometry((100, 80), 30.0), {})
        with self.assertRaisesRegex(ValueError, "帧号必须连续"):
            adapter.update(np.empty((0, 6), dtype=np.float32), 2)


class TestCandidateConfigs(unittest.TestCase):
    def test_all_eight_candidate_files_match_adapter_defaults(self):
        for tracker, project in TRACKER_PROJECTS.items():
            path = ROOT / "tracking_model_benchmark" / project / "configs" / "counting_candidate_v1.json"
            with self.subTest(tracker=tracker):
                with open(path, encoding="utf-8") as handle:
                    self.assertEqual(json.load(handle), DEFAULT_CONFIGS[tracker])

    def test_unknown_tracker_option_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "未知字段"):
            resolved_config("sort", {"reid": True})


if __name__ == "__main__":
    unittest.main()

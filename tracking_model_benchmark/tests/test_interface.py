import json
import unittest
from pathlib import Path

import numpy as np

from benchmark_common.numeric_cache import CACHE_FIELDS
from tracking_model_benchmark._common.tracker_adapters import (
    DEFAULT_CONFIGS,
    TRACKER_PROJECTS,
    create_tracker_adapter,
    resolved_config,
)
from tracking_model_benchmark._common.tracker_interface import (
    DETECTION_FIELDS,
    TRACK_FIELDS,
    FrameGeometry,
    UnifiedTrackerAdapter,
)


ROOT = Path(__file__).resolve().parents[2]

EXPECTED_PROTOCOL_CONFIGS = {
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
        "dynamic_tuning": True,
        "cth": 0.5,
        "high_th": 0.6,
        "high_th_m": 0.1,
        "match_th_first": 0.67,
        "match_th_first_m": 0.05,
        "match_th_second": 0.2,
        "low_th": 0.1,
        "new_track_th": 0.7,
        "new_track_th_m": 0.08,
        "marginal_timeout": {"fps_ratio": 0.7},
        "central_timeout": {"fps_ratio": 1.0},
        "horizontal_margin_ratio": 0.1,
        "vertical_margin_ratio": 0.1,
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
    def test_all_eight_defaults_and_candidate_files_match_protocol(self):
        for tracker, project in TRACKER_PROJECTS.items():
            path = ROOT / "tracking_model_benchmark" / project / "configs" / "counting_candidate_v1.json"
            with self.subTest(tracker=tracker):
                self.assertEqual(DEFAULT_CONFIGS[tracker], EXPECTED_PROTOCOL_CONFIGS[tracker])
                with open(path, encoding="utf-8") as handle:
                    self.assertEqual(json.load(handle), EXPECTED_PROTOCOL_CONFIGS[tracker])

    def test_sfsort_resolves_protocol_geometry_and_fps_parameters(self):
        adapter = create_tracker_adapter("sfsort", (2304, 1296), 55.9)
        native = adapter.native_tracker
        self.assertTrue(native.dynamic_tuning)
        self.assertEqual(native.cth, 0.5)
        self.assertEqual(native.high_th_m, 0.1)
        self.assertEqual(native.match_th_first_m, 0.05)
        self.assertEqual(native.new_track_th_m, 0.08)
        self.assertEqual(native.match_th_second, 0.2)
        self.assertEqual(native.marginal_timeout, 39)
        self.assertEqual(native.central_timeout, 55)
        self.assertEqual((native.l_margin, native.r_margin), (230, 2074))
        self.assertEqual((native.t_margin, native.b_margin), (129, 1167))

    def test_sfsort_rejects_fixed_timeout_frame_overrides(self):
        with self.assertRaisesRegex(ValueError, "不接受固定帧数"):
            create_tracker_adapter(
                "sfsort",
                (2304, 1296),
                55.9,
                {"marginal_timeout": 38, "central_timeout": 55},
            )

    def test_resolved_sfsort_formula_cannot_mutate_defaults(self):
        config = resolved_config("sfsort")
        config["marginal_timeout"]["fps_ratio"] = 9.0
        self.assertEqual(DEFAULT_CONFIGS["sfsort"]["marginal_timeout"], {"fps_ratio": 0.7})

    def test_unknown_tracker_option_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "未知字段"):
            resolved_config("sort", {"reid": True})


if __name__ == "__main__":
    unittest.main()

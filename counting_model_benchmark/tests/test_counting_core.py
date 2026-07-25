import json
import unittest
from collections import defaultdict
from pathlib import Path

from counting_model_benchmark.counting_core import (
    REFERENCE_CIRCLE_CENTER,
    REFERENCE_CIRCLE_RADIUS,
    REFERENCE_FRAME_SIZE,
    apply_signed_count_delta,
    circle_transition,
    get_count_region,
    point_circle_state,
)


class TestUnifiedCircle(unittest.TestCase):
    def test_reference_circle_is_fixed(self):
        self.assertEqual(REFERENCE_FRAME_SIZE, (2304, 1296))
        self.assertEqual(REFERENCE_CIRCLE_CENTER, (1185, 732))
        self.assertEqual(REFERENCE_CIRCLE_RADIUS, 215)
        self.assertEqual(get_count_region(REFERENCE_FRAME_SIZE), ((1185, 732), 215))

    def test_circle_scales_with_frame_size(self):
        self.assertEqual(get_count_region((1920, 1080)), ((988, 610), 179))

    def test_boundary_is_inside(self):
        center, radius = get_count_region(REFERENCE_FRAME_SIZE)
        self.assertEqual(point_circle_state((center[0] + radius, center[1]), center, radius), "inside")
        self.assertEqual(point_circle_state((center[0] + radius + 0.01, center[1]), center, radius), "outside")

    def test_only_binary_crossings_create_events(self):
        self.assertEqual(circle_transition(None, "inside"), (None, 0))
        self.assertEqual(circle_transition(None, "outside"), (None, 0))
        self.assertEqual(circle_transition("outside", "inside"), ("enter", 1))
        self.assertEqual(circle_transition("inside", "outside"), ("exit", -1))
        self.assertEqual(circle_transition("inside", "inside"), (None, 0))

    def test_signed_count_is_not_clamped_at_zero(self):
        counts = defaultdict(int)
        self.assertEqual(apply_signed_count_delta(counts, 0, -1), -1)
        self.assertEqual(apply_signed_count_delta(counts, 0, 1), 0)

    def test_baseline_config_matches_core_constants(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "baseline_circle_v1.json"
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)
        self.assertEqual(config["reference_frame_size"], list(REFERENCE_FRAME_SIZE))
        self.assertEqual(config["reference_center"], list(REFERENCE_CIRCLE_CENTER))
        self.assertEqual(config["reference_radius"], REFERENCE_CIRCLE_RADIUS)

if __name__ == "__main__":
    unittest.main()

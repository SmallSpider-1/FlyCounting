import unittest

import numpy as np

from benchmark_common.association_metrics import (
    normalized_center_distance,
    normalized_center_distance_from_sfsort,
    normalized_center_distance_from_tracks,
    normalized_center_similarity,
)


class TrackLike:
    def __init__(self, box):
        self.tlbr = np.asarray(box, dtype=np.float32)


class SFTrackLike:
    def __init__(self, box):
        self.bbox = np.asarray(box, dtype=np.float32)


class TestNormalizedCenterDistance(unittest.TestCase):
    def test_identical_centers_have_zero_cost_and_unit_similarity(self):
        first = np.array([[0, 0, 10, 10]], dtype=np.float32)
        second = np.array([[2, 2, 8, 8]], dtype=np.float32)
        np.testing.assert_allclose(normalized_center_distance(first, second), [[0.0]])
        np.testing.assert_allclose(normalized_center_similarity(first, second), [[1.0]])

    def test_nonoverlapping_boxes_keep_finite_matchable_distance(self):
        first = np.array([[0, 0, 10, 10]], dtype=np.float32)
        second = np.array([[20, 0, 30, 10]], dtype=np.float32)
        expected = 20.0 / np.sqrt(30.0**2 + 10.0**2)
        np.testing.assert_allclose(normalized_center_distance(first, second), [[expected]], rtol=1e-6)
        self.assertGreater(float(normalized_center_similarity(first, second)[0, 0]), 0.3)

    def test_empty_inputs_preserve_pairwise_shape(self):
        result = normalized_center_distance(np.empty((0, 4)), np.ones((3, 4)))
        self.assertEqual(result.shape, (0, 3))

    def test_byte_family_track_objects_are_supported(self):
        result = normalized_center_distance_from_tracks(
            [TrackLike([0, 0, 10, 10])],
            [TrackLike([20, 0, 30, 10])],
        )
        self.assertEqual(result.shape, (1, 1))
        self.assertTrue(0.0 < result[0, 0] < 1.0)

    def test_sfsort_signature_is_supported(self):
        result = normalized_center_distance_from_sfsort(
            [SFTrackLike([0, 0, 10, 10])],
            np.array([[20, 0, 30, 10]], dtype=np.float32),
            iou_only=True,
        )
        self.assertEqual(result.shape, (1, 1))

if __name__ == "__main__":
    unittest.main()

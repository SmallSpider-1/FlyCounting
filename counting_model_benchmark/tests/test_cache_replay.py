import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from benchmark_common.numeric_cache import (
    CACHE_FIELDS,
    CACHE_VERSION,
    FIELD_DECIMALS,
    NumericCacheWriter,
    iter_cache_frames,
    read_cache_header,
)
from counting_model_benchmark.counting_core import (
    REFERENCE_CIRCLE_CENTER,
    REFERENCE_CIRCLE_RADIUS,
    REFERENCE_FRAME_SIZE,
)
from counting_model_benchmark.replay_counting import replay_cache


def cache_metadata():
    return {
        "video": {"path": "/data/example.mp4", "size": 123, "mtime_ns": 456},
        "frame_size": list(REFERENCE_FRAME_SIZE),
        "fps": 30.0,
        "reported_frame_count": 3,
        "coordinate_space": "full_frame_xyxy",
        "class_names": {"0": "correcta", "1": "dorsalis"},
        "producer": {"stage": "test"},
    }


class TestNumericCache(unittest.TestCase):
    def test_complete_cache_round_trip_keeps_empty_frames(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "example.detections.jsonl"
            writer = NumericCacheWriter(path, "detections", cache_metadata())
            writer.write_frame(1, [])
            writer.write_frame(2, [[1, 2, 3, 4, 0.9, 1]])
            writer.write_frame(3, [])
            writer.close()

            header = read_cache_header(path, "detections")
            self.assertEqual(header["cache_id"], writer.cache_id)
            self.assertEqual(list(iter_cache_frames(path, "detections")), [(1, []), (2, [[1, 2, 3, 4, 0.9, 1]]), (3, [])])

    def test_incomplete_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "incomplete.tracks.jsonl"
            metadata = cache_metadata()
            header_without_id = {
                "cache_version": CACHE_VERSION,
                "kind": "tracks",
                "fields": ["x1", "y1", "x2", "y2", "track_id", "confidence", "class_id", "detection_index"],
                "field_decimals": {field: FIELD_DECIMALS[field] for field in CACHE_FIELDS["tracks"]},
                **metadata,
            }
            from benchmark_common.numeric_cache import stable_hash

            header = {**header_without_id, "cache_id": stable_hash(header_without_id)}
            path.write_text(
                json.dumps({"header": header}) + "\n" + json.dumps({"f": 1, "t": []}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "未完整写出"):
                list(iter_cache_frames(path, "tracks"))

    def test_float_noise_below_precision_keeps_identical_payload(self):
        """同一四舍五入区间内的浮点尾数差异不改变缓存字节。

        注意：截断只吸收区间内的噪声。若两次推理的数值正好落在 .xx5 分界线两侧，
        仍会写出不同结果，因此不能据此声称跨环境哈希必然一致。协议要求的
        “同一环境重跑哈希一致”依赖的是推理本身逐位确定，而不是这里的截断。
        """
        rows_a = [[1184.521999, 731.090001, 1205.331999, 752.870001, 0.9311999678611755, 1]]
        rows_b = [[1184.522001, 731.089999, 1205.332001, 752.869999, 0.9312000021338463, 1]]
        payloads = []
        with tempfile.TemporaryDirectory() as temp_dir:
            for name, rows in (("a", rows_a), ("b", rows_b)):
                path = Path(temp_dir) / f"{name}.detections.jsonl"
                writer = NumericCacheWriter(path, "detections", cache_metadata())
                writer.write_frame(1, rows)
                writer.close()
                payloads.append(path.read_bytes())
            self.assertEqual(payloads[0], payloads[1])
            self.assertEqual(
                list(iter_cache_frames(path, "detections")),
                [(1, [[1184.52, 731.09, 1205.33, 752.87, 0.9312, 1]])],
            )

    def test_non_integer_id_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "badclass.detections.jsonl"
            writer = NumericCacheWriter(path, "detections", cache_metadata())
            with self.assertRaisesRegex(ValueError, "整数字段收到非整数值"):
                writer.write_frame(1, [[1, 2, 3, 4, 0.9, 0.5]])
            writer.abort()

    def test_missing_frame_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "missing.detections.jsonl"
            writer = NumericCacheWriter(path, "detections", cache_metadata())
            writer.write_frame(1, [])
            with self.assertRaisesRegex(ValueError, "帧号必须连续"):
                writer.write_frame(3, [])
            writer.abort()


class TestCountingReplay(unittest.TestCase):
    def test_track_cache_replays_signed_enter_and_exit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "example.tracks.jsonl"
            metadata = {**cache_metadata(), "upstream_detection_cache_id": "det-cache"}
            writer = NumericCacheWriter(path, "tracks", metadata)
            cx, cy = REFERENCE_CIRCLE_CENTER
            writer.write_frame(1, [[cx + 300 - 5, cy - 5, cx + 300 + 5, cy + 5, 7, 0.9, 1, 0]])
            writer.write_frame(2, [[cx - 5, cy - 5, cx + 5, cy + 5, 7, 0.9, 1, 0]])
            writer.write_frame(3, [[cx + 300 - 5, cy - 5, cx + 300 + 5, cy + 5, 7, 0.9, 1, 0]])
            writer.close()

            counts = defaultdict(int)
            events = []
            segments = []
            replay_cache(path, REFERENCE_CIRCLE_CENTER, REFERENCE_CIRCLE_RADIUS, counts, events, segments)

            self.assertEqual(counts[1], 0)
            self.assertEqual([event["direction"] for event in events], ["enter", "exit"])
            self.assertEqual([event["count_after"] for event in events], [1, 0])
            self.assertEqual(segments[1]["track_cache_id"], writer.cache_id)
            self.assertEqual(segments[1]["upstream_detection_cache_id"], "det-cache")
if __name__ == "__main__":
    unittest.main()

import csv
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from benchmark_common.numeric_cache import NumericCacheWriter
from counting_model_benchmark.evaluate_counting import (
    EXPECTED_TRACKERS,
    collect_result_files,
    load_manifest_tracker_roots,
    main as evaluate_main,
    validate_manifest_tracker_set,
)
from counting_model_benchmark.replay_counting import (
    main as replay_main,
    prepare_output_directory,
)
from counting_model_benchmark.run_counting_benchmark import (
    MANIFEST_FIELDS,
    append_manifest_row,
    clear_count_outputs,
    inspect_manifest,
    plan_tracker_actions,
    selected_trackers,
)


CLASS_NAMES = {
    "0": "Bactrocera dorsalis",
    "1": "Bactrocera correcta",
}


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_empty_track_cache(path: Path) -> None:
    writer = NumericCacheWriter(
        path,
        "tracks",
        {
            "video": {"path": "/data/segment_001.mp4", "size": 123, "mtime_ns": 456},
            "frame_size": [2304, 1296],
            "fps": 55.0,
            "reported_frame_count": 1,
            "coordinate_space": "full_frame_xyxy",
            "class_names": CLASS_NAMES,
            "upstream_detection_cache_id": "det-cache",
            "producer": {"stage": "test"},
        },
    )
    writer.write_frame(1, [])
    writer.close()


def manifest_context(root: Path, detection_name: str = "detections") -> dict:
    return {
        "detection_source": str((root / detection_name).resolve()),
        "counting_config": str((root / "counting.json").resolve()),
        "counting_config_id": "count-config-id",
        "reference_center_x": 1185.0,
        "reference_center_y": 732.0,
        "reference_radius": 215.0,
        "initial_count_policy": "fixed_zero_per_class",
        "initial_counts": "all_classes=0",
    }


def manifest_row(root: Path, tracker: str, run_id: str, context: dict) -> dict:
    row = {field: "" for field in MANIFEST_FIELDS}
    row.update(
        {
            **context,
            "run_id": run_id,
            "run_started_at": "2026-07-28T12:00:00+08:00",
            "tracker": tracker,
            "config": str((root / f"{tracker}.json").resolve()),
            "track_root": str((root / tracker / "tracks").resolve()),
            "count_root": str((root / tracker / "counts").resolve()),
            "logical_groups": 1,
            "counting_enabled": True,
            "tracks_reused": False,
            "track_run_id": run_id,
        }
    )
    return row


class TestResultSafety(unittest.TestCase):
    def test_duplicate_tracker_selection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "同一次运行内覆盖"):
            selected_trackers("sort,sfsort,sort")

    def test_replay_output_conflict_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "counts"
            output.mkdir()
            conflict = output / "final_counts.csv"
            conflict.write_text("old\n", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "拒绝静默覆盖") as raised:
                prepare_output_directory(output, overwrite=False)
            self.assertIn(str(conflict.resolve()), str(raised.exception))
            self.assertEqual(prepare_output_directory(output, overwrite=True), output.resolve())

    def test_batch_action_requires_explicit_reuse_or_purge(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "run"
            (output / "sfsort").mkdir(parents=True)

            with self.assertRaisesRegex(FileExistsError, "拒绝混写或覆盖") as raised:
                plan_tracker_actions(
                    output,
                    ["sort", "sfsort"],
                    manifest_rows=[],
                    overwrite=False,
                    purge=False,
                )
            self.assertIn(str((output / "sfsort").resolve()), str(raised.exception))
            self.assertEqual(
                plan_tracker_actions(
                    output,
                    ["sort", "sfsort"],
                    manifest_rows=[],
                    overwrite=True,
                    purge=False,
                ),
                {"sort": "run", "sfsort": "reuse"},
            )
            self.assertEqual(
                plan_tracker_actions(
                    output,
                    ["sort", "sfsort"],
                    manifest_rows=[],
                    overwrite=False,
                    purge=True,
                ),
                {"sort": "run", "sfsort": "purge"},
            )
            self.assertTrue((output / "sfsort").is_dir())

    def test_overwrite_cleanup_removes_only_count_csv_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker_root = Path(temp_dir) / "sort"
            track_cache = tracker_root / "tracks" / "segment.tracks.jsonl"
            track_cache.parent.mkdir(parents=True)
            track_cache.write_bytes(b"track-cache-must-survive")
            unrelated = tracker_root / "counts" / "logical_video" / "notes.txt"
            unrelated.parent.mkdir(parents=True)
            unrelated.write_text("keep", encoding="utf-8")
            expected_outputs = []
            for name in ("final_counts.csv", "count_events.csv", "segment_summary.csv", "replay_config.csv"):
                path = unrelated.parent / name
                path.write_text("old", encoding="utf-8")
                expected_outputs.append(path)

            removed = clear_count_outputs(tracker_root / "counts")

            self.assertEqual(set(removed), set(expected_outputs))
            self.assertEqual(track_cache.read_bytes(), b"track-cache-must-survive")
            self.assertEqual(unrelated.read_text(encoding="utf-8"), "keep")

    def test_manifest_appends_batches_and_rejects_incompatible_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "benchmark_run_manifest.csv"
            context = manifest_context(root)
            append_manifest_row(manifest, manifest_row(root, "sort", "batch-1", context), context)
            append_manifest_row(manifest, manifest_row(root, "bytetrack", "batch-2", context), context)

            rows = inspect_manifest(manifest, context, include_counting_context=True)
            self.assertEqual([row["tracker"] for row in rows], ["sort", "bytetrack"])
            self.assertEqual([row["run_id"] for row in rows], ["batch-1", "batch-2"])

            with self.assertRaisesRegex(ValueError, "已存在 tracker"):
                append_manifest_row(manifest, manifest_row(root, "sort", "batch-3", context), context)

            changed_detection = {**context, "detection_source": str((root / "other").resolve())}
            with self.assertRaisesRegex(ValueError, "detection_source"):
                inspect_manifest(manifest, changed_detection, include_counting_context=True)

            changed_counting = {**context, "counting_config_id": "different"}
            with self.assertRaisesRegex(ValueError, "counting_config_id"):
                inspect_manifest(manifest, changed_counting, include_counting_context=True)

    def test_manifest_is_required(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(FileNotFoundError, "不再扫描结果目录"):
                load_manifest_tracker_roots(Path(temp_dir))

    def test_evaluation_requires_all_protocol_trackers_by_default(self):
        roots = [("sort", Path("/tmp/sort-counts"))]
        with self.assertRaisesRegex(ValueError, "协议规定的 8 个算法") as raised:
            validate_manifest_tracker_set(roots, allow_partial=False)
        self.assertIn("bytetrack", str(raised.exception))
        self.assertIn("--allow-partial-trackers", str(raised.exception))

        complete = [(tracker, Path(f"/tmp/{tracker}-counts")) for tracker in EXPECTED_TRACKERS]
        self.assertEqual(validate_manifest_tracker_set(complete, allow_partial=False), [])

    def test_partial_tracker_diagnostics_are_explicit_and_reject_unknown(self):
        roots = [("sort", Path("/tmp/sort-counts"))]
        missing = validate_manifest_tracker_set(roots, allow_partial=True)
        self.assertEqual(missing, list(EXPECTED_TRACKERS[1:]))

        with self.assertRaisesRegex(ValueError, "unexpected=.*unknown"):
            validate_manifest_tracker_set(
                [*roots, ("unknown", Path("/tmp/unknown-counts"))],
                allow_partial=True,
            )

    def test_duplicate_logical_video_results_are_rejected_with_all_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            count_root = Path(temp_dir) / "counts"
            first = count_root / "logical_video" / "final_counts.csv"
            second = count_root / "archive" / "logical_video" / "final_counts.csv"
            for path in (first, second):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("class_name,final_region_count\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "匹配到多个") as raised:
                collect_result_files("sort", count_root, {"logical_video"})
            self.assertIn(str(first.resolve()), str(raised.exception))
            self.assertIn(str(second.resolve()), str(raised.exception))

    def test_unmatched_result_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            count_root = Path(temp_dir) / "counts"
            invalid = count_root / "archive" / "final_counts.csv"
            invalid.parent.mkdir(parents=True)
            invalid.write_text("class_name,final_region_count\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "无法唯一归属") as raised:
                collect_result_files("sort", count_root, {"logical_video"})
            self.assertIn(str(invalid.resolve()), str(raised.exception))

    def test_fixed_zero_replay_is_explicit_and_evaluation_uses_only_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            results = root / "results"
            directory_name = "dorsalis_0-correcta_0"
            track_cache = root / "segment.tracks.jsonl"
            count_root = results / "sort" / "counts"
            count_output = count_root / directory_name
            write_empty_track_cache(track_cache)

            with patch.object(
                sys,
                "argv",
                [
                    "replay_counting.py",
                    "--tracks",
                    str(track_cache),
                    "--output",
                    str(count_output),
                ],
            ), redirect_stdout(io.StringIO()):
                replay_main()

            with open(count_output / "replay_config.csv", encoding="utf-8") as handle:
                replay_rows = list(csv.DictReader(handle))
            self.assertEqual(replay_rows[0]["initial_count_policy"], "fixed_zero_per_class")
            self.assertEqual(replay_rows[0]["initial_counts"], "0=0|1=0")
            with open(count_output / "final_counts.csv", encoding="utf-8") as handle:
                final_rows = list(csv.DictReader(handle))
            self.assertEqual([row["final_region_count"] for row in final_rows], ["0", "0"])

            write_csv(
                results / "benchmark_run_manifest.csv",
                ["tracker", "count_root", "counting_enabled"],
                [{"tracker": "sort", "count_root": str(count_root), "counting_enabled": True}],
            )
            stale = results / "old_tracker" / "counts" / directory_name / "final_counts.csv"
            write_csv(
                stale,
                ["class_id", "class_name", "final_region_count"],
                [{"class_id": 0, "class_name": CLASS_NAMES["0"], "final_region_count": 999}],
            )
            long_videos = root / "long_videos.csv"
            ground_truth = root / "ground_truth.csv"
            write_csv(
                long_videos,
                ["video_id", "directory_name"],
                [{"video_id": "LV001", "directory_name": directory_name}],
            )
            write_csv(
                ground_truth,
                ["video_id", "species", "count"],
                [
                    {"video_id": "LV001", "species": CLASS_NAMES["0"], "count": 0},
                    {"video_id": "LV001", "species": CLASS_NAMES["1"], "count": 0},
                ],
            )

            with patch.object(
                sys,
                "argv",
                [
                    "evaluate_counting.py",
                    "--results",
                    str(results),
                    "--long-videos",
                    str(long_videos),
                    "--ground-truth",
                    str(ground_truth),
                    "--allow-partial-trackers",
                ],
            ), redirect_stdout(io.StringIO()):
                evaluate_main()

            with open(results / "evaluation" / "counting_metrics.csv", encoding="utf-8") as handle:
                metrics = list(csv.DictReader(handle))
            self.assertEqual(len(metrics), 1)
            self.assertEqual(metrics[0]["tracker"], "sort")
            self.assertEqual(metrics[0]["class_mae"], "0.0")


if __name__ == "__main__":
    unittest.main()

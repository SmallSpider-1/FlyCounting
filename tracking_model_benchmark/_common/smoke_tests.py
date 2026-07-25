#!/usr/bin/env python3
"""Unified-interface smoke test for one archived tracker per isolated process."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracking_model_benchmark._common.tracker_adapters import TRACKER_PROJECTS, create_tracker_adapter


def cached_detection(cls_id: int, x_offset: float = 0.0) -> np.ndarray:
    return np.array([[10 + x_offset, 10, 30 + x_offset, 30, 0.90, cls_id]], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project", choices=TRACKER_PROJECTS)
    args = parser.parse_args()

    adapter = create_tracker_adapter(
        args.project,
        frame_size=(64, 64),
        fps=30,
    )
    first = adapter.update(cached_detection(0), 1)
    second = adapter.update(cached_detection(1, 1.0), 2)
    empty = adapter.update(np.empty((0, 6), dtype=np.float32), 3)

    assert first.shape == second.shape == (1, 8), (first, second)
    assert empty.shape == (0, 8), empty
    assert int(first[0, 4]) == int(second[0, 4]), (first, second)
    assert int(first[0, 6]) == 0 and int(second[0, 6]) == 1, (first, second)
    assert int(first[0, 7]) == int(second[0, 7]) == 0, (first, second)
    assert np.allclose(first[0, :4], [10, 10, 30, 30], atol=1.0), first

    metadata = adapter.cache_metadata()
    assert metadata["per_class"] is False
    assert metadata["adapter_contract"] == "Nx6_full_frame_detections_to_Nx8_full_frame_tracks_v1"

    print(f"project={args.project}")
    print(f"python={sys.version.split()[0]}")
    print(f"numpy={np.__version__}")
    print(f"stable_track_id={int(first[0, 4])}")
    print("input=Nx6 [xyxy,confidence,class_id]")
    print("output=Nx8 [xyxy,track_id,confidence,class_id,detection_index]")
    print("per_class=false")
    print("reid=false")
    print("gmc=false")
    print("validation_status=READY_UNIFIED_INTERFACE")


if __name__ == "__main__":
    main()

# Counting model benchmark

This directory owns tracker-independent counting logic, eight-tracker counting
orchestration, evaluation, configurations, future ablations, and counting tests.

```text
counting_model_benchmark/
  counting_core.py              pure circle and signed-transition functions
  replay_counting.py            replay one logical video from numeric tracks
  run_counting_benchmark.py     run all selected trackers and the same counter
  evaluate_counting.py          compare final counts with logical-video truth
  configs/                      runnable, versioned counting configurations
  ablations/                    future one-factor ablation definitions
  tests/                        automated regression checks, not runtime inputs
```

The first baseline uses full-frame detections and one reference circle at
`(1185, 732)` with radius `215` in a `2304 x 1296` frame. ROI selection is not
part of the first tracker baseline.

Generate the shared detector cache:

```bash
python detection_model_benchmark/cache_detections.py \
  --source video_data \
  --weights <frozen_detector.pt> \
  --output project_results/detection_cache/<detector_run_id> \
  --conf 0.10
```

The cache floor is `0.10` so trackers with a native low-score recovery stage can
use it. Normal/new-track thresholds remain tracker-specific; the cache floor is
not a universal acceptance threshold.

Run the eight tracker candidates and the same counter:

```bash
python counting_model_benchmark/run_counting_benchmark.py \
  --detections project_results/detection_cache/<detector_run_id>/numeric_cache \
  --output project_results/counting_model_benchmark/baseline_circle_v1 \
  --counting-config counting_model_benchmark/configs/baseline_circle_v1.json
```

Evaluate after every required logical video is present:

```bash
python counting_model_benchmark/evaluate_counting.py \
  --results project_results/counting_model_benchmark/baseline_circle_v1
```

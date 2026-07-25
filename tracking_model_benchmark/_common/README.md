# Unified no-ReID tracking and counting benchmark

This benchmark compares eight online trackers through one detection-cache input and one track-cache output contract. The obsolete root `main.py` and local BoxMOT ByteTrack test path have been removed.

## Eight candidates

| Key | Official workspace | Selected variant |
|---|---|---|
| `sort` | `01_sort/` | geometry-only SORT |
| `bytetrack` | `02_bytetrack/` | official ByteTrack |
| `ocsort` | `03_ocsort/` | official pure-motion OC-SORT |
| `sfsort` | `04_sfsort/` | online SF-SORT |
| `fasttracker` | `05_fasttracker/` | official Ultralytics no-appearance FastTracker |
| `boosttrack` | `06_boosttrack/` | BoostTrack with embedding/ECC disabled |
| `hybridsort` | `07_hybridsort/` | pure-motion HybridSORT |
| `botsort` | `08_botsort/` | BoT-SORT with ReID/GMC disabled |

The comparison conditions are online only, AABB only, `per_class=false`, no future-frame interpolation, no offline reconnection, no appearance embeddings, and no GMC/ECC/optical flow. All trackers consume exactly the same cached detections.

## Unified interface

`tracker_interface.py` defines the only public tracker boundary:

```text
input D : Nx6 [x1,y1,x2,y2,confidence,class_id]
output T: Nx8 [x1,y1,x2,y2,track_id,confidence,class_id,detection_index]
```

Detection and output track coordinates are always full-frame `xyxy`. The first baseline does not crop an ROI; all eight trackers receive the same full-frame detection arrays directly. ROI selection is deliberately deferred to a later counting-optimization experiment.

`tracker_adapters.py` owns all native API differences. The counter never imports a tracker implementation. Class association is uniformly `current matched detection, otherwise last observation`; this is metadata propagation, not class voting. The absolute ID number is native to each tracker and is not compared across algorithms.

## Candidate parameters

Each `configs/counting_candidate_v1.json` is the official-default starting point
for the selected detection-only variant. Defaults come from the official main
evaluation entry, constructor, registered YAML, or canonical no-ReID experiment
as appropriate:

| Tracker | Official-default baseline parameters |
|---|---|
| SORT | `max_age=1`, `min_hits=3`, `iou_threshold=0.3`; no score filter added |
| ByteTrack | `track_thresh=0.6`, `track_buffer=30`, `match_thresh=0.9` |
| OC-SORT | `det_thresh=0.6`, constructor defaults, `use_byte=false` |
| SF-SORT | constructor defaults: `high=0.6`, `low=0.1`, `new=0.7`, dynamic tuning/timeouts/margins off |
| FastTracker | registered `fasttrack.yaml` defaults: `high=0.25`, `low=0.1`, `new=0.25` and official occlusion settings |
| BoostTrack | default MOT17 score settings (`det=0.6`, `dlo_boost_coef=0.65`) and official detection-score boosts |
| HybridSORT | canonical MOT17 no-ReID experiment: `track/det=0.6`, HMIoU, IoU `0.25`, inertia `0.05`, BYTE and TCM on |
| BoT-SORT | main evaluation defaults: `high=0.6`, `low=0.1`, `new=0.7`, buffer `30`, match `0.8` |

ReID, embeddings and camera-motion features remain disabled because they cannot
be reconstructed from numeric detections. This is the only intentional deviation
from variants whose full official default expects images or external motion data.
These are baseline parameters, not validation-optimal parameters. Tune on the
future validation split, then create a separately named frozen config before test
evaluation.

## Commands

Generate the shared detection-cache tree once in the detector environment. When the
source is a directory, `cache_detections.py` scans it recursively and preserves each logical
video directory, so equal segment stems in different directories cannot overwrite
one another:

```bash
python detection_model_benchmark/cache_detections.py \
  --source video_data \
  --weights <frozen_detector.pt> \
  --output project_results/detection_cache/<detector_run_id> \
  --conf 0.10
```

`0.10` is the information-preserving cache floor for this detection-only tracker
benchmark. It does not make every tracker use the same confidence threshold:
each official tracker applies its own defaults, and SORT natively has no score
filter. All eight trackers must consume the same cache in any one comparison.

Then use the dedicated tracker environment:

```bash
conda activate bench_tracking
```

Run all isolated unified-interface smokes:

```bash
tracking_model_benchmark/_common/run_smoke_tests.sh
```

Run one tracker over a detection-cache tree:

```bash
python tracking_model_benchmark/_common/run_tracker.py \
  --tracker ocsort \
  --detections project_results/detection_cache/<detector_run_id>/numeric_cache \
  --output project_results/tracking_model_benchmark/ocsort/tracks \
  --config tracking_model_benchmark/03_ocsort/configs/counting_candidate_v1.json
```

Run all eight trackers and the same counter:

```bash
python counting_model_benchmark/run_counting_benchmark.py \
  --detections project_results/detection_cache/<detector_run_id>/numeric_cache \
  --output project_results/counting_model_benchmark/baseline_circle_v1 \
  --counting-config counting_model_benchmark/configs/baseline_circle_v1.json
```

After all six logical videos are present, calculate diagnostic counting metrics:

```bash
python counting_model_benchmark/evaluate_counting.py \
  --results project_results/counting_model_benchmark/baseline_circle_v1
```

The evaluator reports per-species MAE, Class-MAE, Total-MAE, exact-video rate, and negative-video rate. These are not formal test conclusions until the detector, `N_start`, data split, counting rule, and tracker parameters are frozen.

## Environment and provenance

The independent `bench_tracking` environment was cloned from `bench_detection_torch201` without modifying existing environments:

- Python 3.9.23
- NumPy 1.23.5
- PyTorch 2.0.1+cu118
- torchvision 0.15.2+cu118
- CUDA runtime 11.8

Tracker-only additions are recorded in `bench_tracking.requirements.txt`; core constraints are in `bench_tracking.constraints.txt`. Each track cache records the official source path, pinned commit, resolved parameters, tracking bounds, upstream detection cache ID, and unified adapter contract.

BoostTrack retains an official unconditional ReID-package import, but `use_embedding=false` prevents model construction or weight use. BoT-SORT keeps its official source clean and uses the documented `torch._six.string_classes` import shim; the runtime path remains `with_reid=false`, `cmc_method=none`.

All eight adapters have passed import, construction, two associated frames with a class change, an empty frame, full-frame coordinate preservation, and unified Nx8 output checks. This proves interface execution only, not tracking or counting accuracy.

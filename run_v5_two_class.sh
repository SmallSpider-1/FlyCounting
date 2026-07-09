#!/usr/bin/env bash
set -u

cd /home/admin1/Projects/ultralytics-main || exit 1
mkdir -p result_two_class_v5

LOG=result_two_class_v5/run.log
EXIT_FILE=result_two_class_v5/run.exit
RUNNER_PID=result_two_class_v5/run.runner.pid
PYTHON_PID=result_two_class_v5/run.pid

: > "$LOG"
rm -f "$EXIT_FILE"
echo "$$" > "$RUNNER_PID"

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

/home/admin1/.conda/envs/yolov26/bin/python -u main.py \
  --source video_data/two_class_61_118 \
  --weights yolo8n_two-class.pt \
  --output result_two_class_v5 \
  --device 0 \
  --half \
  --class-vote-weights 0:0.18,1:1.0 \
  --entry-detected-class-override 1 \
  --recovery-class-match-required 1 \
  --recovery-class-min-scores 0:0.60,1:0.97 \
  --enable-outer-start-recovery \
  --outer-start-max-offset 35 \
  --outer-start-min-outward 10 \
  --outer-start-min-frames 2 \
  --outer-start-max-frames 30 \
  --no-video \
  --no-event-frames \
  >> "$LOG" 2>&1 &

child=$!
echo "$child" > "$PYTHON_PID"
wait "$child"
code=$?
echo "EXIT_CODE=$code" > "$EXIT_FILE"
exit "$code"

#!/usr/bin/env bash
set -euo pipefail

UNIT_NAME=fruitfly-benchmark-e100-s42.service
PIPELINE_ROOT=/home/admin1/Projects/ultralytics-main/detection_model_benchmark/e100_pipeline
UNIT_SOURCE="$PIPELINE_ROOT/systemd/$UNIT_NAME"
UNIT_TARGET=/home/admin1/.config/systemd/user/$UNIT_NAME

systemctl --user show-environment >/dev/null
install -D -m 0644 "$UNIT_SOURCE" "$UNIT_TARGET"
systemctl --user daemon-reload
systemctl --user enable "$UNIT_NAME"
systemctl --user start "$UNIT_NAME"
systemctl --user show "$UNIT_NAME" --no-pager \
  --property=Id,ActiveState,SubState,MainPID,ExecMainStartTimestamp,FragmentPath

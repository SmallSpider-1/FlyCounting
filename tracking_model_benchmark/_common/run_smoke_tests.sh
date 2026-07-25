#!/usr/bin/env bash
set -uo pipefail

root="/home/admin1/Projects/ultralytics-main/tracking_model_benchmark"
runner="${root}/_common/smoke_tests.py"
env_name="bench_tracking"
projects=(
  "sort:01_sort"
  "bytetrack:02_bytetrack"
  "ocsort:03_ocsort"
  "sfsort:04_sfsort"
  "fasttracker:05_fasttracker"
  "boosttrack:06_boosttrack"
  "hybridsort:07_hybridsort"
  "botsort:08_botsort"
)

passed=0
failed=0
for item in "${projects[@]}"; do
  name="${item%%:*}"
  project_dir="${item#*:}"
  log="${root}/${project_dir}/logs/smoke_test.log"
  {
    echo "[$(date --iso-8601=seconds)] ${name} minimal no-ReID/no-GMC smoke test"
    echo "command=conda run -n ${env_name} python ${runner} ${name}"
    PYTHONDONTWRITEBYTECODE=1 YOLO_AUTOINSTALL=false \
      conda run -n "${env_name}" --no-capture-output python "${runner}" "${name}"
  } >"${log}" 2>&1
  status=$?
  if (( status == 0 )); then
    echo "PASS ${name}: ${log}"
    ((passed+=1))
  else
    echo "FAIL ${name}: ${log}"
    ((failed+=1))
  fi
done

echo "summary passed=${passed} failed=${failed} total=${#projects[@]}"
(( failed == 0 ))

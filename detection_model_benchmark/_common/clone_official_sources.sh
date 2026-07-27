#!/usr/bin/env bash
set -uo pipefail

root="/home/admin1/Projects/ultralytics-main/detection_model_benchmark"

projects=(
  "01_yolov5n|https://github.com/ultralytics/yolov5.git"
  "02_yolov7_tiny|https://github.com/WongKinYiu/yolov7.git"
  "03_yolov8n|https://github.com/ultralytics/ultralytics.git"
  "04_yolov9_t|https://github.com/WongKinYiu/yolov9.git"
  "05_yolov10n|https://github.com/THU-MIG/yolov10.git"
  "06_yolo11n|https://github.com/ultralytics/ultralytics.git"
  "07_yolo12n|https://github.com/sunsmarterjie/yolov12.git"
  "08_yolo26n|https://github.com/ultralytics/ultralytics.git"
  "09_deim_dfine_n|https://github.com/Intellindust-AI-Lab/DEIM.git"
  "10_rt_detr_r18|https://github.com/lyuwenyu/RT-DETR.git"
  "11_faster_rcnn_r50_fpn|https://github.com/pytorch/vision.git"
)

clone_one() {
  local project_dir="$1"
  local repo_url="$2"
  local src_dir="${root}/${project_dir}/src"
  local log_file="${root}/${project_dir}/logs/clone.log"
  {
    echo "[$(date --iso-8601=seconds)] git clone ${repo_url} ${src_dir}"
    if [ -d "${src_dir}/.git" ]; then
      echo "Source already cloned; leaving it unchanged."
    elif [ -n "$(find "${src_dir}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
      echo "BLOCKED: ${src_dir} is non-empty but is not a Git working tree."
      return 2
    else
      git clone "${repo_url}" "${src_dir}"
    fi
    git -C "${src_dir}" remote -v
    git -C "${src_dir}" branch --show-current
    git -C "${src_dir}" rev-parse HEAD
    git -C "${src_dir}" status --short
  } >"${log_file}" 2>&1
}

pids=()
labels=()
for entry in "${projects[@]}"; do
  IFS='|' read -r project_dir repo_url <<<"${entry}"
  clone_one "${project_dir}" "${repo_url}" &
  pids+=("$!")
  labels+=("${project_dir}")
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "OK ${labels[$index]}"
  else
    echo "FAILED ${labels[$index]} (see project logs/clone.log)"
    failed=$((failed + 1))
  fi
done

echo "Clone failures: ${failed}"
exit 0

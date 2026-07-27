#!/usr/bin/env bash
set -uo pipefail

root="/home/admin1/Projects/ultralytics-main/detection_model_benchmark"
summary="${root}/_logs/weight_download_summary.csv"

downloads=(
  "01_yolov5n|yolov5n.pt|https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt"
  "02_yolov7_tiny|yolov7-tiny.pt|https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt"
  "03_yolov8n|yolov8n.pt|https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt"
  "04_yolov9_t|yolov9-t-converted.pt|https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-t-converted.pt"
  "05_yolov10n|yolov10n.pt|https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt"
  "06_yolo11n|yolo11n.pt|https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt"
  "07_yolo12n|yolov12n.pt|https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12n.pt"
  "08_yolo26n|yolo26n.pt|https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"
  "09_deim_dfine_n|deim_dfine_hgnetv2_n_coco_160e.pth|https://drive.usercontent.google.com/download?id=1ZPEhiU9nhW4M5jLnYOFwTSLQC1Ugf62e&export=download&confirm=t"
  "10_rt_detr_r18|rtdetr_r18vd_dec3_6x_coco_from_paddle.pth|https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth"
  "11_faster_rcnn_r50_fpn|fasterrcnn_resnet50_fpn_coco-258fb6c6.pth|https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
)

download_one() {
  local project="$1"
  local filename="$2"
  local url="$3"
  local output="${root}/${project}/weights/${filename}"
  local partial="${output}.part"
  local log="${root}/${project}/logs/weight_download.log"
  {
    echo "[$(date --iso-8601=seconds)] URL=${url}"
    echo "TARGET=${output}"
    if [ -s "${output}" ]; then
      echo "Existing non-empty target retained."
    else
      curl --http1.1 --fail --location --retry 8 --retry-all-errors --retry-delay 3 --connect-timeout 30 --output "${partial}" "${url}" || return 1
      mv "${partial}" "${output}" || return 1
    fi
    stat --printf='size_bytes=%s\n' "${output}"
    sha256sum "${output}"
  } >>"${log}" 2>&1
}

labels=()
statuses=()
for item in "${downloads[@]}"; do
  IFS='|' read -r project filename url <<<"${item}"
  labels+=("${project}|${filename}|${url}")
  if download_one "${project}" "${filename}" "${url}"; then
    statuses+=("OK")
  else
    statuses+=("FAILED")
  fi
done

printf '%s\n' 'project,filename,url,status,size_bytes,sha256' >"${summary}"
for index in "${!labels[@]}"; do
  IFS='|' read -r project filename url <<<"${labels[$index]}"
  output="${root}/${project}/weights/${filename}"
  size="0"
  digest="N/A"
  if [ -f "${output}" ]; then
    size="$(stat --printf='%s' "${output}")"
    digest="$(sha256sum "${output}" | awk '{print $1}')"
  fi
  printf '"%s","%s","%s",%s,%s,%s\n' "${project}" "${filename}" "${url}" "${statuses[$index]}" "${size}" "${digest}" >>"${summary}"
  echo "${statuses[$index]} ${project} ${filename} ${size} bytes"
done

exit 0

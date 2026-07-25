#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRATEGY="${1:-b0_baseline}"
SOURCE_DIR="${2:-${ROOT_DIR}/video_data/dorsalis_79-correcta_27}"
OUTPUT_ROOT="${3:-${ROOT_DIR}/project_results/result_counting_ablation}"
DEVICE="${4:-0}"
WEIGHTS="${5:-${ROOT_DIR}/yolo8n_two-class.pt}"
CONDA_ENV="${CONDA_ENV:-yolov26}"

case "${STRATEGY}" in
    b0_baseline|e1_roi|e2_slow_fast|e3_vote) ;;
    *)
        echo "未知策略: ${STRATEGY}" >&2
        echo "可选: b0_baseline, e1_roi, e2_slow_fast, e3_vote" >&2
        exit 2
        ;;
esac

if [[ ! -d "${SOURCE_DIR}" ]]; then
    echo "视频目录不存在: ${SOURCE_DIR}" >&2
    exit 2
fi
if [[ ! -f "${WEIGHTS}" ]]; then
    echo "权重不存在: ${WEIGHTS}" >&2
    exit 2
fi

output_args=(--csv-only)
if [[ "${SAVE_VIDEO:-0}" == "1" ]]; then
    output_args=()
fi

precision_args=()
if [[ "${USE_HALF:-0}" == "1" ]]; then
    precision_args=(--half)
fi

video_count=0
while IFS= read -r -d '' video_path; do
    video_count=$((video_count + 1))
    video_name="$(basename "${video_path}")"
    video_stem="${video_name%.*}"
    output_dir="${OUTPUT_ROOT}/${STRATEGY}/${video_stem}"

    if [[ -e "${output_dir}/experiment_config.csv" ]]; then
        echo "输出已存在，拒绝覆盖: ${output_dir}" >&2
        echo "请更换 OUTPUT_ROOT，或先人工确认并移走旧结果。" >&2
        exit 2
    fi

    echo "[${STRATEGY}] ${video_name} -> ${output_dir}"
    conda run -n "${CONDA_ENV}" --no-capture-output python "${ROOT_DIR}/main.py" \
        --source "${video_path}" \
        --weights "${WEIGHTS}" \
        --output "${output_dir}" \
        --device "${DEVICE}" \
        --strategy "${STRATEGY}" \
        --imgsz 640 \
        --conf 0.25 \
        --iou 0.7 \
        "${output_args[@]}" \
        "${precision_args[@]}"
done < <(
    find "${SOURCE_DIR}" -maxdepth 1 -type f \
        \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mov' -o -iname '*.mkv' -o -iname '*.wmv' -o -iname '*.flv' -o -iname '*.m4v' \) \
        -print0 | sort -z
)

if [[ "${video_count}" -eq 0 ]]; then
    echo "目录中没有支持的视频: ${SOURCE_DIR}" >&2
    exit 2
fi

echo "策略 ${STRATEGY} 完成，共 ${video_count} 个视频；每个视频均为独立进程、从 0 计数。"

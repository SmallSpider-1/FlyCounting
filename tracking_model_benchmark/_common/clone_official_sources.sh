#!/usr/bin/env bash
set -uo pipefail

root="/home/admin1/Projects/ultralytics-main/tracking_model_benchmark"
max_attempts=3

projects=(
  "01_sort|https://github.com/abewley/sort.git"
  "02_bytetrack|https://github.com/ifzhang/ByteTrack.git"
  "03_ocsort|https://github.com/noahcao/OC_SORT.git"
  "04_sfsort|https://github.com/gitmehrdad/SFSORT.git"
  "06_boosttrack|https://github.com/vukasin-stanojevic/BoostTrack.git"
  "07_hybridsort|https://github.com/ymzis69/HybridSORT.git"
  "08_botsort|https://github.com/NirAharon/BoT-SORT.git"
)

clone_one() {
  local project_dir="$1"
  local repo_url="$2"
  local src_dir="${root}/${project_dir}/src"
  local log_file="${root}/${project_dir}/logs/clone.log"
  local attempt
  local tmp_parent
  {
    echo "[$(date --iso-8601=seconds)] git clone ${repo_url} ${src_dir}"
    if [ -d "${src_dir}/.git" ]; then
      echo "Source already cloned; leaving it unchanged."
    elif [ -n "$(find "${src_dir}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
      echo "BLOCKED: ${src_dir} is non-empty but is not a Git working tree."
      return 2
    else
      for attempt in $(seq 1 "${max_attempts}"); do
        echo "Clone attempt ${attempt}/${max_attempts}"
        tmp_parent="$(mktemp -d "${root}/${project_dir}/.clone_tmp.XXXXXX")" || return 3
        if git clone "${repo_url}" "${tmp_parent}/src"; then
          if [ -e "${src_dir}" ]; then
            rmdir "${src_dir}" 2>/dev/null || {
              echo "BLOCKED: ${src_dir} appeared while cloning and is not empty."
              rm -rf -- "${tmp_parent}"
              return 4
            }
          fi
          mv "${tmp_parent}/src" "${src_dir}"
          rmdir "${tmp_parent}"
          break
        fi
        echo "Clone attempt ${attempt} failed."
        rm -rf -- "${tmp_parent}"
        if [ "${attempt}" -lt "${max_attempts}" ]; then
          sleep 3
        fi
      done
      if [ ! -d "${src_dir}/.git" ]; then
        echo "FAILED: exhausted ${max_attempts} clone attempts."
        return 1
      fi
    fi
    git -C "${src_dir}" remote -v
    git -C "${src_dir}" branch --show-current
    git -C "${src_dir}" rev-parse HEAD
    git -C "${src_dir}" status --short
  } >"${log_file}" 2>&1
}

{
  echo "[$(date --iso-8601=seconds)] SKIPPED: local Ultralytics source pointer; no repository cloned here."
  echo "official_repo=https://github.com/ultralytics/ultralytics.git"
  echo "upstream_tag=v8.4.105"
  echo "upstream_commit=a7e7e65a3e4086c5d2104dca06f6593564eda1ae"
  echo "source_path=/home/admin1/Projects/ultralytics-main/ultralytics/trackers"
  echo "local_base_commit=b06fe71473c5541c2bfa06222e30ce4db379b4f9"
} >"${root}/05_fasttracker/logs/clone.log"

pids=()
labels=()
for entry in "${projects[@]}"; do
  IFS='|' read -r project_dir repo_url <<<"${entry}"
  clone_one "${project_dir}" "${repo_url}" &
  pids+=("$!")
  labels+=("${project_dir}")
done

succeeded=0
failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "OK ${labels[$index]}"
    succeeded=$((succeeded + 1))
  else
    echo "FAILED ${labels[$index]} (see project logs/clone.log)"
    failed=$((failed + 1))
  fi
done

echo "SKIPPED 05_fasttracker (local Ultralytics source pointer)"
echo "Clone summary: succeeded=${succeeded} failed=${failed} skipped=1"
exit 0

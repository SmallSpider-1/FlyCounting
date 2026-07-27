#!/usr/bin/env bash
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
map_file="${root}/_common/environment_map.yaml"
verify_log="${root}/_logs/verify_all.log"
failures=0
checked=0

record() {
  printf '%s\n' "$*" | tee -a "${verify_log}"
}

check_ok() {
  record "PASS $1"
}

check_fail() {
  record "FAIL $1"
  failures=$((failures + 1))
}

: >"${verify_log}"
record "verify_time=$(date --iso-8601=seconds)"

if [ ! -f "${map_file}" ]; then
  check_fail "environment map missing: ${map_file}"
  exit 1
fi

map_records="$({
  awk '
    /^  [a-z0-9_]+:$/ && in_projects { project=$1; sub(/:$/, "", project) }
    /^projects:$/ { in_projects=1; next }
    /^shared_environment_groups:$/ { in_projects=0 }
    in_projects && /^    project_dir:/ { dir=$2 }
    in_projects && /^    official_repo:/ { repo=$2 }
    in_projects && /^    conda_env:/ { env=$2 }
    in_projects && /^    weight_path:/ { weight=$2 }
    in_projects && /^    weight_sha256:/ { sha=$2 }
    in_projects && /^    smoke_log:/ { smoke=$2 }
    in_projects && /^    validation_status:/ { status=$2; print project "|" dir "|" repo "|" env "|" weight "|" sha "|" smoke "|" status }
  ' "${map_file}"
})"

project_count="$(printf '%s\n' "${map_records}" | sed '/^$/d' | wc -l)"
if [ "${project_count}" -eq 11 ]; then
  check_ok "environment map contains 11 projects"
else
  check_fail "environment map contains ${project_count}, expected 11"
fi

declare -A env_seen
declare -A env_project_count

while IFS='|' read -r project project_dir official_repo conda_env weight_path weight_sha smoke_log mapped_status; do
  [ -n "${project}" ] || continue
  checked=$((checked + 1))
  project_root="${root}/${project_dir}"
  src_dir="${project_root}/src"

  if [ -d "${project_root}" ]; then check_ok "${project}: project directory"; else check_fail "${project}: project directory missing"; fi
  if [ -d "${src_dir}/.git" ] && git -C "${src_dir}" cat-file -e 'HEAD^{commit}' 2>/dev/null; then
    actual_remote="$(git -C "${src_dir}" remote get-url origin 2>/dev/null || true)"
    if [ "${actual_remote}" = "${official_repo}" ]; then check_ok "${project}: official source and readable commit"; else check_fail "${project}: origin mismatch (${actual_remote})"; fi
  else
    check_fail "${project}: official Git source incomplete"
  fi

  if [ -f "${root}/${weight_path}" ]; then
    actual_sha="$(sha256sum "${root}/${weight_path}" | awk '{print $1}')"
    if [ "${actual_sha}" = "${weight_sha}" ]; then check_ok "${project}: weight exists and SHA256 matches"; else check_fail "${project}: weight SHA256 mismatch"; fi
  else
    check_fail "${project}: weight missing (${weight_path})"
  fi

  if conda env list --json | python -c 'import json,sys; env=sys.argv[1]; paths=json.load(sys.stdin)["envs"]; raise SystemExit(0 if any(p == env or p.endswith("/" + env) for p in paths) else 1)' "${conda_env}"; then
    check_ok "${project}: Conda environment ${conda_env} exists"
  else
    check_fail "${project}: Conda environment ${conda_env} missing"
  fi

  if conda run -n "${conda_env}" --no-capture-output python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda)' >>"${verify_log}" 2>&1; then
    check_ok "${project}: PyTorch import and CUDA"
  else
    check_fail "${project}: PyTorch import or CUDA"
  fi

  if [ -f "${root}/${smoke_log}" ] && tail -60 "${root}/${smoke_log}" | rg -q 'SMOKE_TEST_PASS'; then
    check_ok "${project}: recorded core import, weight load, and single-image inference"
  else
    check_fail "${project}: successful smoke marker missing"
  fi

  if [ "${mapped_status}" != "READY" ]; then check_fail "${project}: map status is ${mapped_status}"; fi
  env_seen["${conda_env}"]=1
  env_project_count["${conda_env}"]=$(( ${env_project_count["${conda_env}"]:-0} + 1 ))
done <<<"${map_records}"

for env_name in "${!env_seen[@]}"; do
  check_ok "shared mapping: ${env_name} serves ${env_project_count[${env_name}]} project(s)"
done

record "checked_projects=${checked}"
record "failures=${failures}"
if [ "${checked}" -ne 11 ] || [ "${failures}" -ne 0 ]; then
  exit 1
fi
record "VERIFY_ALL_PASS"

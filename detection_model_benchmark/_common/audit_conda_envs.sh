#!/usr/bin/env bash
set -uo pipefail

benchmark_root="/home/admin1/Projects/ultralytics-main/detection_model_benchmark"
log_dir="${benchmark_root}/_logs"
audit_csv="${log_dir}/conda_environment_audit.csv"
system_log="${log_dir}/system_info.txt"

mkdir -p "${log_dir}"

{
  echo "audit_time=$(date --iso-8601=seconds)"
  echo
  echo "[os-release]"
  cat /etc/os-release
  echo
  echo "[uname]"
  uname -a
  echo
  echo "[gpu-and-driver]"
  nvidia-smi || true
  echo
  echo "[nvcc]"
  nvcc --version || true
  echo
  echo "[conda]"
  conda --version || true
  conda env list || true
  echo
  echo "[disk]"
  df -h "${benchmark_root}"
  df -B1 "${benchmark_root}"
} >"${system_log}" 2>&1

printf '%s\n' 'environment,environment_path,is_base,python_version,torch_version,torchvision_version,torch_cuda_runtime,cuda_available,cuda_device_count,numpy_version,opencv_version,pillow_version,scipy_version,pandas_version,pyyaml_version,timm_version,transformers_version,onnxruntime_version,ultralytics_version,pycocotools_version,core_probe_status,probe_error' >"${audit_csv}"

conda env list --json | python -c 'import json,sys; [print(p) for p in json.load(sys.stdin)["envs"]]' | while IFS= read -r env_path; do
  [ -n "${env_path}" ] || continue
  env_name="$(basename "${env_path}")"
  is_base="false"
  if [ "${env_path}" = "$(conda info --base)" ]; then
    env_name="base"
    is_base="true"
  fi

  probe="$({ conda run -p "${env_path}" --no-capture-output python - <<'PY'
import csv
import importlib
import io
import json
import platform

def version(module_name, attr="__version__"):
    try:
        module = importlib.import_module(module_name)
        return str(getattr(module, attr, "installed"))
    except Exception:
        return "NOT_INSTALLED"

record = {
    "python_version": platform.python_version(),
    "torch_version": "NOT_INSTALLED",
    "torchvision_version": version("torchvision"),
    "torch_cuda_runtime": "N/A",
    "cuda_available": "false",
    "cuda_device_count": "0",
    "numpy_version": version("numpy"),
    "opencv_version": version("cv2"),
    "pillow_version": version("PIL"),
    "scipy_version": version("scipy"),
    "pandas_version": version("pandas"),
    "pyyaml_version": version("yaml"),
    "timm_version": version("timm"),
    "transformers_version": version("transformers"),
    "onnxruntime_version": version("onnxruntime"),
    "ultralytics_version": version("ultralytics"),
    "pycocotools_version": version("pycocotools"),
    "core_probe_status": "OK",
    "probe_error": "",
}
try:
    import torch
    record["torch_version"] = str(torch.__version__)
    record["torch_cuda_runtime"] = str(torch.version.cuda)
    record["cuda_available"] = str(torch.cuda.is_available()).lower()
    record["cuda_device_count"] = str(torch.cuda.device_count())
except Exception as exc:
    record["core_probe_status"] = "ERROR"
    record["probe_error"] = f"torch probe: {type(exc).__name__}: {exc}"

buffer = io.StringIO()
writer = csv.writer(buffer, lineterminator="")
writer.writerow(record.values())
print("__AUDIT__" + buffer.getvalue())
PY
  } 2>&1)"
  probe_status=$?
  audit_line="$(printf '%s\n' "${probe}" | sed -n 's/^__AUDIT__//p' | tail -n 1)"
  if [ -z "${audit_line}" ]; then
    escaped_error="$(printf '%s' "${probe}" | tr '\n\r' '  ' | sed 's/"/""/g')"
    audit_line="UNKNOWN,UNKNOWN,N/A,false,0,UNKNOWN,UNKNOWN,UNKNOWN,UNKNOWN,UNKNOWN,UNKNOWN,UNKNOWN,UNKNOWN,UNKNOWN,UNKNOWN,UNKNOWN,UNKNOWN,ERROR,\"probe exit ${probe_status}: ${escaped_error}\""
  fi
  escaped_name="$(printf '%s' "${env_name}" | sed 's/"/""/g')"
  escaped_path="$(printf '%s' "${env_path}" | sed 's/"/""/g')"
  printf '"%s","%s",%s,%s\n' "${escaped_name}" "${escaped_path}" "${is_base}" "${audit_line}" >>"${audit_csv}"
done

echo "Wrote ${system_log}"
echo "Wrote ${audit_csv}"

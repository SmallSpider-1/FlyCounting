#!/usr/bin/env bash
set -euo pipefail

root="/home/admin1/Projects/ultralytics-main/tracking_model_benchmark"
env_name="bench_tracking"
source_env="/home/admin1/.conda/envs/bench_detection_torch201"
requirements="${root}/_common/bench_tracking.requirements.txt"
constraints="${root}/_common/bench_tracking.constraints.txt"
log="${root}/_logs/create_${env_name}.log"

mkdir -p "${root}/_logs"

{
  echo "[$(date --iso-8601=seconds)] Creating ${env_name}"
  if conda env list --json | python -c 'import json,sys; print(any(p.endswith("/bench_tracking") for p in json.load(sys.stdin)["envs"]))' | grep -qx True; then
    echo "Environment already exists; reconciling only this task-created environment."
  else
    echo "Cloning the validated baseline without modifying its source environment: ${source_env}"
    conda create -y -n "${env_name}" --clone "${source_env}"
  fi

  echo "Verifying the immutable baseline versions before tracker dependency installation."
  conda run -n "${env_name}" --no-capture-output python - <<'PY'
import numpy
import torch
import torchvision

assert numpy.__version__ == "1.23.5", numpy.__version__
assert torch.__version__ == "2.0.1+cu118", torch.__version__
assert torchvision.__version__ == "0.15.2+cu118", torchvision.__version__
print(f"numpy={numpy.__version__}")
print(f"torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")
PY

  echo "Installing Cython before building cython-bbox."
  conda run -n "${env_name}" --no-capture-output python -m pip install \
    --constraint "${constraints}" "Cython==0.29.37"
  conda run -n "${env_name}" --no-capture-output python -m pip install \
    --constraint "${constraints}" --no-build-isolation "cython-bbox==0.1.5"

  echo "Installing the remaining tracker-only dependencies."
  conda run -n "${env_name}" --no-capture-output python -m pip install \
    --constraint "${constraints}" --requirement "${requirements}"

  echo "Validating the completed independent environment."
  conda run -n "${env_name}" --no-capture-output python - <<'PY'
import sys
import numpy
import torch
import torchvision

assert numpy.__version__ == "1.23.5", numpy.__version__
assert torch.__version__ == "2.0.1+cu118", torch.__version__
assert torchvision.__version__ == "0.15.2+cu118", torchvision.__version__
print(f"python={sys.version.split()[0]}")
print(f"numpy={numpy.__version__}")
print(f"torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
PY
  conda run -n "${env_name}" --no-capture-output python -m pip check
} >>"${log}" 2>&1

echo "Created and validated ${env_name}; log=${log}"

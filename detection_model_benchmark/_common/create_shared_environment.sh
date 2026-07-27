#!/usr/bin/env bash
set -euo pipefail

root="/home/admin1/Projects/ultralytics-main/detection_model_benchmark"
env_name="bench_detection_torch201"
log="${root}/_logs/create_${env_name}.log"

{
  echo "[$(date --iso-8601=seconds)] Creating ${env_name}"
  if conda env list --json | python -c 'import json,sys; print(any(p.endswith("/bench_detection_torch201") for p in json.load(sys.stdin)["envs"]))' | grep -qx True; then
    echo "Environment already exists; leaving it in place and reconciling only this task-created environment."
  else
    echo "Cloning the compatible torch 2.0.1+cu118 baseline without modifying its source environment."
    conda create -y -n "${env_name}" --clone /home/admin1/.openclaw/workspace/micromamba/envs/sugar
  fi

  conda run -n "${env_name}" --no-capture-output python -m pip install \
    -r "${root}/_common/bench_detection_torch201.requirements.txt"

  conda run -n "${env_name}" --no-capture-output python - <<'PY'
import sys
import torch
import torchvision
print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
assert torch.cuda.is_available()
PY
  conda run -n "${env_name}" --no-capture-output python -m pip check
} >>"${log}" 2>&1

echo "Created and validated ${env_name}; log=${log}"

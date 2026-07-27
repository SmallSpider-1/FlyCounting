#!/usr/bin/env python3
"""Print a compact benchmark scheduler snapshot."""

from __future__ import annotations

import json

from common import MODEL_SPECS, RESULT_ROOT, read_json, run_dir


def main() -> None:
    print(json.dumps(read_json(RESULT_ROOT / "scheduler/pipeline_status.json", {}), ensure_ascii=False, indent=2))
    for model, spec in MODEL_SPECS.items():
        status = read_json(run_dir(model) / "status.json", {})
        print(f"{spec['run_name']}\t{status.get('status', 'missing')}\tgpu={status.get('physical_gpu_id', '-')}")


if __name__ == "__main__":
    main()

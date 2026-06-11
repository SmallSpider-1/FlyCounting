import json
import sys
from datetime import datetime
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKSPACE.parent
RESULTS_DIR = WORKSPACE / "results"
EXPORTS_DIR = WORKSPACE / "exports"
LOGS_DIR = WORKSPACE / "logs"
CONFIG_PATH = WORKSPACE / "configs" / "experiment_matrix.json"

for path in (RESULTS_DIR, EXPORTS_DIR, LOGS_DIR):
    path.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_config(path=CONFIG_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def metric_value(obj, name, default=0.0):
    return float(getattr(obj, name, default) or 0.0)


def collect_val_metrics(metrics):
    box = getattr(metrics, "box", None)
    speed = getattr(metrics, "speed", {}) or {}
    out = {
        "precision": metric_value(box, "mp"),
        "recall": metric_value(box, "mr"),
        "map50": metric_value(box, "map50"),
        "map50_95": metric_value(box, "map"),
        "fitness": metric_value(metrics, "fitness"),
        "speed_ms": {k: float(v) for k, v in speed.items()},
        "save_dir": str(getattr(metrics, "save_dir", "")),
    }
    if box is not None:
        out["class_precision"] = [float(x) for x in getattr(box, "p", [])]
        out["class_recall"] = [float(x) for x in getattr(box, "r", [])]
        out["class_map50"] = [float(x) for x in getattr(box, "ap50", [])]
        out["class_map50_95"] = [float(x) for x in getattr(box, "ap", [])]
    return out

import json
import time
from pathlib import Path


def save_result(task, payload, out_dir="benchmarks/real/results"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def get_run_dir(task, base_dir="benchmarks/real/artifacts"):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base_dir) / task / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_results(out_dir="benchmarks/real/results"):
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return []
    results = []
    for path in sorted(out_dir.glob("*.json")):
        try:
            results.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return results

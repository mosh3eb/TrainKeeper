import json
import os
from pathlib import Path


def run_checks():
    results = {
        "write_access": _check_write_access(),
        "torch_available": _check_torch(),
        "cuda_available": _check_cuda(),
        "data_dir": _check_data_dir(),
    }
    return results


def _check_write_access():
    try:
        path = Path(".trainkeeper_write_test")
        path.write_text("ok", encoding="utf-8")
        path.unlink()
        return True
    except Exception:
        return False


def _check_torch():
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _check_cuda():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _check_data_dir():
    data_dir = os.environ.get("TRAINKEEPER_DATA_DIR", "data")
    try:
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        return str(Path(data_dir).resolve())
    except Exception:
        return None


def to_json():
    return json.dumps(run_checks(), indent=2)

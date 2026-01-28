import json
import time
from collections import Counter
from pathlib import Path

import numpy as np


def _js_divergence(p, q, eps=1e-8):
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))


class MetricsMonitor:
    def __init__(self):
        self.latencies = []
        self.predictions = []
        self.timestamps = []

    def record_latency(self, seconds):
        self.latencies.append(float(seconds))
        self.timestamps.append(time.time())

    def record_predictions(self, preds):
        if isinstance(preds, (list, tuple, np.ndarray)):
            self.predictions.extend(list(preds))
        else:
            self.predictions.append(preds)
        self.timestamps.append(time.time())

    def snapshot(self, bins=20):
        preds = np.asarray(self.predictions)
        if preds.size == 0:
            pred_stats = {"type": "empty"}
        elif np.issubdtype(preds.dtype, np.number):
            hist, edges = np.histogram(preds, bins=bins)
            pred_stats = {"type": "numeric", "hist": hist.tolist(), "edges": edges.tolist()}
        else:
            counts = Counter([str(x) for x in preds])
            pred_stats = {"type": "categorical", "counts": dict(counts)}
        return {
            "latency_ms_p50": float(np.percentile(self.latencies, 50) * 1000.0)
            if self.latencies
            else None,
            "latency_ms_p95": float(np.percentile(self.latencies, 95) * 1000.0)
            if self.latencies
            else None,
            "predictions": pred_stats,
        }

    def detect_drift(self, baseline_snapshot, threshold=0.2):
        current = self.snapshot()
        base_pred = baseline_snapshot.get("predictions", {})
        cur_pred = current.get("predictions", {})
        if base_pred.get("type") in (None, "empty") or cur_pred.get("type") in (None, "empty"):
            return None
        if base_pred.get("type") != cur_pred.get("type"):
            return {"issue": "type_mismatch"}
        if base_pred.get("type") == "numeric":
            preds = np.asarray(self.predictions)
            hist, _ = np.histogram(preds, bins=np.asarray(base_pred["edges"]))
            js = _js_divergence(base_pred["hist"], hist)
            if js > threshold:
                return {"issue": "drift_js", "js_divergence": js}
        if base_pred.get("type") == "categorical":
            keys = set(base_pred["counts"].keys()) | set(cur_pred["counts"].keys())
            p = np.array([base_pred["counts"].get(k, 0) for k in keys])
            q = np.array([cur_pred["counts"].get(k, 0) for k in keys])
            js = _js_divergence(p, q)
            if js > threshold:
                return {"issue": "drift_js", "js_divergence": js}
        return None

    def export_prometheus(self):
        snap = self.snapshot()
        lines = []
        if snap["latency_ms_p50"] is not None:
            lines.append(f"trainkeeper_latency_ms_p50 {snap['latency_ms_p50']}")
        if snap["latency_ms_p95"] is not None:
            lines.append(f"trainkeeper_latency_ms_p95 {snap['latency_ms_p95']}")
        return "\n".join(lines)

    def to_json(self):
        return json.dumps(self.snapshot(), indent=2)

    def save_snapshot(self, path):
        Path(path).write_text(self.to_json(), encoding="utf-8")
        return path

    @staticmethod
    def load_snapshot(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

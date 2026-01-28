import json
import re
from dataclasses import dataclass
from pathlib import Path


class HookManager:
    def __init__(self, log_layers=None, artifacts_dir="artifacts", compute_entropy=False):
        self.log_layers = log_layers or []
        self.artifacts_dir = Path(artifacts_dir)
        self._hooks = []
        self.compute_entropy = compute_entropy
        self._stats = {"activations": {}, "gradients": {}}

    def _matches(self, name):
        if not self.log_layers:
            return True
        return any(re.match(pat, name) for pat in self.log_layers)

    def _activation_hook(self, name):
        def hook(_, __, output):
            try:
                val = output.detach()
                self._stats["activations"][name] = {
                    "mean": float(val.mean().cpu().item()),
                    "std": float(val.std().cpu().item()),
                    "abs_max": float(val.abs().max().cpu().item()),
                }
                if self.compute_entropy:
                    self._stats["activations"][name]["entropy"] = _activation_entropy(val)
            except Exception:
                pass

        return hook

    def _gradient_hook(self, name):
        def hook(grad):
            try:
                val = grad.detach()
                self._stats["gradients"][name] = {
                    "mean": float(val.mean().cpu().item()),
                    "std": float(val.std().cpu().item()),
                    "norm": float(val.norm().cpu().item()),
                }
            except Exception:
                pass

        return hook

    def attach(self, model):
        for name, module in model.named_modules():
            if self._matches(name):
                self._hooks.append(module.register_forward_hook(self._activation_hook(name)))

                for param_name, param in module.named_parameters(recurse=False):
                    if param.requires_grad:
                        hook_name = f"{name}.{param_name}"
                        self._hooks.append(param.register_hook(self._gradient_hook(hook_name)))
        return self

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def analyze_gradients(self, explode_threshold=1000.0, vanish_threshold=1e-8):
        issues = []
        for name, stats in self._stats.get("gradients", {}).items():
            norm = stats.get("norm")
            if norm is None:
                continue
            if norm > explode_threshold:
                issues.append({"param": name, "issue": "exploding", "norm": norm})
            if norm < vanish_threshold:
                issues.append({"param": name, "issue": "vanishing", "norm": norm})
        return issues

    def detect_dead_neurons(self, zero_std_threshold=1e-12):
        issues = []
        for name, stats in self._stats.get("activations", {}).items():
            if stats.get("std", 1.0) <= zero_std_threshold:
                issues.append({"layer": name, "issue": "dead_neurons"})
        return issues

    def flush(self, filename="debug_stats.json"):
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.artifacts_dir / filename
        out_path.write_text(json.dumps(self._stats, indent=2))
        return out_path


class LossTracker:
    def __init__(self):
        self.records = []

    def update(self, loss, batch_id=None):
        try:
            value = float(loss)
        except Exception:
            value = float(loss.detach().cpu().item())
        self.records.append({"batch": batch_id, "loss": value})

    def to_json(self):
        return json.dumps(self.records, indent=2)


def detect_nan_inf(tensor, name="tensor"):
    try:
        import torch

        if torch.isnan(tensor).any():
            return {"issue": "nan", "name": name}
        if torch.isinf(tensor).any():
            return {"issue": "inf", "name": name}
    except Exception:
        return None
    return None


def capture_failure_batch(batch, reason, artifacts_dir="artifacts"):
    path = Path(artifacts_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = {"reason": reason, "batch": _safe_serialize(batch)}
    out_path = path / "failure_batch.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out_path


def capture_model_state(model, reason, artifacts_dir="artifacts"):
    try:
        import torch
    except Exception:
        return None
    path = Path(artifacts_dir)
    path.mkdir(parents=True, exist_ok=True)
    out_path = path / "failure_model.pt"
    torch.save({"reason": reason, "state_dict": model.state_dict()}, out_path)
    return out_path


def check_training_step(loss, outputs=None, batch=None, model=None, exception=None, artifacts_dir="artifacts"):
    issues = []
    issue = detect_nan_inf(loss, name="loss")
    if issue:
        issues.append(issue)
    if outputs is not None:
        issue = detect_nan_inf(outputs, name="outputs")
        if issue:
            issues.append(issue)
    if exception is not None:
        issues.append({"issue": "exception", "detail": str(exception)})
    if issues and batch is not None:
        capture_failure_batch(batch, issues, artifacts_dir=artifacts_dir)
    if issues and model is not None:
        capture_model_state(model, issues, artifacts_dir=artifacts_dir)
    return issues


def compare_debug_stats(path_a, path_b):
    data_a = json.loads(Path(path_a).read_text(encoding="utf-8"))
    data_b = json.loads(Path(path_b).read_text(encoding="utf-8"))
    diff = {"added": [], "removed": [], "changed": []}
    keys = set(data_a.keys()) | set(data_b.keys())
    for key in keys:
        if key not in data_a:
            diff["added"].append(key)
        elif key not in data_b:
            diff["removed"].append(key)
        elif data_a[key] != data_b[key]:
            diff["changed"].append(key)
    return diff


def _activation_entropy(tensor, bins=20):
    try:
        import torch

        data = tensor.detach().flatten().float()
        if data.numel() == 0:
            return 0.0
        hist = torch.histc(data, bins=bins)
        prob = hist / (hist.sum() + 1e-8)
        entropy = -(prob * (prob + 1e-8).log()).sum().item()
        return float(entropy)
    except Exception:
        return None


def update_to_weight_ratio(param, grad, eps=1e-8):
    try:
        import torch

        w_norm = torch.norm(param.detach()).item()
        g_norm = torch.norm(grad.detach()).item()
        return float(g_norm / (w_norm + eps))
    except Exception:
        return None


@dataclass
class TrainingHealthReport:
    issues: list
    summary: dict
    metrics: dict

    def to_json(self):
        return json.dumps(
            {"issues": self.issues, "summary": self.summary, "metrics": self.metrics},
            indent=2,
        )


def build_health_report(hook_manager, loss_tracker=None, explode_threshold=1000.0, vanish_threshold=1e-8):
    issues = []
    issues.extend(hook_manager.analyze_gradients(explode_threshold=explode_threshold, vanish_threshold=vanish_threshold))
    issues.extend(hook_manager.detect_dead_neurons())
    summary = {}
    for issue in issues:
        key = issue.get("issue", "unknown")
        summary[key] = summary.get(key, 0) + 1
    metrics = {"loss_records": len(loss_tracker.records) if loss_tracker else 0}
    return TrainingHealthReport(issues=issues, summary=summary, metrics=metrics)


def _safe_serialize(obj):
    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(x) for x in obj]
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    return str(obj)


def check_label_leakage(df, label_col, threshold=0.99):
    if label_col not in df.columns:
        raise ValueError("label_col not found in dataframe")
    label = df[label_col]
    issues = []
    for col in df.columns:
        if col == label_col:
            continue
        series = df[col]
        if series.equals(label):
            issues.append({"column": col, "issue": "perfect_match"})
        if series.dtype.kind in "iuf" and label.dtype.kind in "iuf":
            corr = abs(series.corr(label))
            if corr >= threshold:
                issues.append({"column": col, "issue": "high_correlation", "corr": corr})
    return issues


def diff_snapshots(path_a, path_b):
    data_a = json.loads(Path(path_a).read_text(encoding="utf-8"))
    data_b = json.loads(Path(path_b).read_text(encoding="utf-8"))
    changes = {"added": [], "removed": [], "changed": []}
    keys = set(data_a.keys()) | set(data_b.keys())
    for key in keys:
        if key not in data_a:
            changes["added"].append(key)
        elif key not in data_b:
            changes["removed"].append(key)
        elif data_a[key] != data_b[key]:
            changes["changed"].append(key)
    return changes

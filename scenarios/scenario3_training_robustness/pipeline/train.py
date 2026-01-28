import argparse
import json
import time
from pathlib import Path

import numpy as np

from trainkeeper.debugger import (
    HookManager,
    LossTracker,
    build_health_report,
    check_training_step,
)
from trainkeeper.experiment import run_reproducible


def _require_torch():
    try:
        import torch
    except Exception as exc:
        raise SystemExit("Scenario 3 requires torch. Install with: pip install -e .[torch]") from exc
    return torch


def _make_dataset(n=512, d=20, seed=123):
    torch = _require_torch()
    rng = np.random.RandomState(seed)
    x = rng.normal(0, 1, size=(n, d)).astype("float32")
    w = rng.normal(0, 1, size=(d,)).astype("float32")
    logits = x @ w + rng.normal(0, 0.5, size=(n,)).astype("float32")
    y = (logits > 0).astype("int64")
    return torch.utils.data.TensorDataset(torch.tensor(x), torch.tensor(y))


def _build_model(input_dim):
    torch = _require_torch()
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 2),
    )


def _write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _run_case(case, artifacts_dir):
    torch = _require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = _make_dataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model = _build_model(input_dim=20).to(device)
    if case == "representation_collapse":
        for param in model.parameters():
            param.data.zero_()

    lr = 0.05
    if case == "optimizer_divergence":
        lr = 10.0
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    hook = HookManager(artifacts_dir=artifacts_dir, compute_entropy=True)
    hook.attach(model)
    loss_tracker = LossTracker()

    issues = []
    blocked = False
    failure_report = None
    step = 0
    for batch in loader:
        step += 1
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        if case == "corrupted_batch" and step == 1:
            x[0, 0] = float("nan")

        logits = model(x)
        loss = loss_fn(logits, y)

        if case == "nan_loss" and step == 1:
            loss = loss * torch.tensor(float("nan"), device=loss.device)
        if case == "exploding_gradient" and step == 1:
            loss = loss * 1e6

        opt.zero_grad(set_to_none=True)
        loss.backward()

        if case == "optimizer_divergence" and step == 1:
            for param in model.parameters():
                param.data.mul_(1e3)

        grad_issues = hook.analyze_gradients(explode_threshold=1e3, vanish_threshold=1e-8)
        step_issues = check_training_step(
            loss=loss,
            outputs=logits,
            batch={"x": x[:4].detach().cpu(), "y": y[:4].detach().cpu()},
            model=model,
            artifacts_dir=artifacts_dir,
        )
        issues.extend(grad_issues)
        issues.extend(step_issues)

        if case == "optimizer_divergence":
            loss_value = float(loss.detach().cpu().item())
            if loss_value > 1e3:
                issues.append({"issue": "loss_spike", "loss": loss_value})
            total_norm = 0.0
            for param in model.parameters():
                total_norm += float(param.detach().norm().cpu().item())
            if total_norm > 1e3:
                issues.append({"issue": "param_divergence", "param_norm": total_norm})

        loss_tracker.update(loss.detach(), batch_id=step)
        if issues:
            blocked = True
            failure_report = {
                "case": case,
                "step": step,
                "issues": issues,
            }
            break

        opt.step()
        if step >= 5:
            break

    debug_path = hook.flush("debug_stats.json")
    health = build_health_report(hook, loss_tracker=loss_tracker)
    _write_json(Path(artifacts_dir) / "health_report.json", json.loads(health.to_json()))

    report_name = None
    if failure_report:
        if case == "exploding_gradient":
            report_name = "exploding_gradient_report.json"
        elif case == "nan_loss":
            report_name = "nan_loss_report.json"
        elif case == "corrupted_batch":
            report_name = "corrupted_batch_report.json"
        elif case == "optimizer_divergence":
            report_name = "optimizer_divergence_report.json"
        elif case == "representation_collapse":
            report_name = "representation_collapse_report.json"
        _write_json(Path(artifacts_dir) / report_name, failure_report)

    return {
        "case": case,
        "blocked": blocked,
        "detected": bool(issues),
        "issues": len(issues),
        "debug_stats": str(debug_path),
        "report": report_name,
    }


def run_case(case, artifacts_root):
    @run_reproducible(
        auto_capture_git=True,
        seed=123,
        config={"scenario": "training_instability_lab", "case": case},
        artifacts_dir=str(Path(artifacts_root) / case),
    )
    def _runner(run_ctx=None):
        start = time.time()
        metrics = _run_case(case, run_ctx.run_dir)
        metrics["detection_ms"] = int((time.time() - start) * 1000)
        return metrics

    return _runner()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        default="exploding_gradient",
        choices=[
            "exploding_gradient",
            "nan_loss",
            "corrupted_batch",
            "optimizer_divergence",
            "representation_collapse",
        ],
    )
    parser.add_argument("--artifacts-root", default="scenario3_results")
    args = parser.parse_args()
    return run_case(args.case, args.artifacts_root)


if __name__ == "__main__":
    main()

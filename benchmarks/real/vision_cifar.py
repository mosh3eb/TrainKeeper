import os
import time
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trainkeeper.experiment import run_reproducible
from trainkeeper.trainutils import efficient_train, log_data_fingerprint, ResourceTracker
from trainkeeper.debugger import HookManager, check_training_step

from benchmarks.real.utils import get_run_dir, save_result


def _require_torch():
    try:
        import torch
        import torchvision
        from torchvision import datasets, transforms
    except Exception as exc:
        raise SystemExit(
            "This benchmark requires torch and torchvision. "
            "Install with: pip install -e .[torch,vision]"
        ) from exc
    return torch, torchvision, datasets, transforms


def _make_loaders(batch_size=64, subset=2048):
    torch, _, datasets, transforms = _require_torch()
    transform = transforms.Compose([transforms.ToTensor()])
    root_dir = Path(os.environ.get("TRAINKEEPER_DATA_DIR", "data"))
    data_dir = root_dir / "cifar-10-batches-py"
    expected = [data_dir / f"data_batch_{i}" for i in range(1, 6)] + [data_dir / "test_batch"]
    download = not all(p.exists() for p in expected)
    try:
        train = datasets.CIFAR10(root=str(root_dir), train=True, download=download, transform=transform)
    except RuntimeError:
        # Handle corrupted download by removing and retrying once.
        tgz = root_dir / "cifar-10-python.tar.gz"
        if tgz.exists():
            tgz.unlink()
        train_dir = root_dir / "cifar-10-batches-py"
        if train_dir.exists():
            for path in train_dir.rglob("*"):
                if path.is_file():
                    path.unlink()
            for path in sorted(train_dir.rglob("*"), reverse=True):
                if path.is_dir():
                    path.rmdir()
            train_dir.rmdir()
        train = datasets.CIFAR10(root=str(root_dir), train=True, download=True, transform=transform)
    idx = torch.randperm(len(train))[:subset]
    subset_train = torch.utils.data.Subset(train, idx.tolist())
    loader = torch.utils.data.DataLoader(subset_train, batch_size=batch_size, shuffle=True)
    return loader


def _build_model():
    torch, _, _, _ = _require_torch()
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(16, 32, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Flatten(),
        torch.nn.Linear(32 * 8 * 8, 10),
    )


def _train_loop(epochs=1, mixed_precision=False, artifacts_dir=None):
    torch, _, _, _ = _require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = _make_loaders()
    model = _build_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    tracker = ResourceTracker()
    hm = HookManager(log_layers=[], artifacts_dir=artifacts_dir or "artifacts", compute_entropy=True).attach(model)
    if artifacts_dir:
        log_data_fingerprint(loader.dataset, artifacts_dir=artifacts_dir)

    total = 0
    correct = 0
    start = time.time()
    for _ in range(epochs):
        for batch in loader:
            t0 = time.time()
            x, y = batch
            x, y = x.to(device), y.to(device)
            with efficient_train(mixed_precision=mixed_precision, grad_accum_steps=1) as eff:
                logits = model(x)
                loss = loss_fn(logits, y)
                eff.backward(loss, opt)
            issues = check_training_step(loss, outputs=logits, batch=None, model=model, artifacts_dir=artifacts_dir or "artifacts")
            preds = logits.argmax(dim=1)
            total += y.numel()
            correct += (preds == y).sum().item()
            tracker.record(step_time=time.time() - t0)
    elapsed = time.time() - start
    if artifacts_dir:
        hm.flush(filename="debug_stats.json")
    acc = correct / max(1, total)
    return {"accuracy": acc, "elapsed_sec": elapsed, "resource": tracker.summary()}


def baseline_run():
    return _train_loop(epochs=1, mixed_precision=False)


def trainkeeper_run(run_dir):
    @run_reproducible(auto_capture_git=False, artifacts_dir=run_dir / "exp")
    def _run():
        return _train_loop(epochs=1, mixed_precision=True, artifacts_dir=run_dir / "exp")

    return _run()


def main():
    baseline = baseline_run()
    run_dir = get_run_dir("vision_cifar")
    tk = trainkeeper_run(run_dir)
    payload = {"task": "vision_cifar", "baseline": baseline, "trainkeeper": tk, "artifacts_dir": str(run_dir)}
    out_path = save_result("vision_cifar", payload)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

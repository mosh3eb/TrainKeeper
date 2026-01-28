import os
from pathlib import Path

import numpy as np

from trainkeeper.experiment import run_reproducible
from trainkeeper.trainutils import CheckpointManager, log_data_fingerprint, make_dataloader

_DEFAULT_DATASET = os.environ.get("TK_DATASET", "cifar100")
_DEFAULT_EPOCHS = int(os.environ.get("TK_EPOCHS", "2"))
_DEFAULT_BATCH_SIZE = int(os.environ.get("TK_BATCH_SIZE", "64"))
_DEFAULT_MAX_SAMPLES = int(os.environ.get("TK_MAX_SAMPLES", "512"))
_DEFAULT_AUGMENT = os.environ.get("TK_AUGMENT", "0").lower() in {"1", "true", "yes"}
_DEFAULT_ARTIFACTS_DIR = os.environ.get("TK_ARTIFACTS_DIR", "scenario1_runs")


def _require_torch():
    try:
        import torch
    except Exception as exc:
        raise SystemExit("This scenario requires torch. Install with: pip install -e .[torch]") from exc
    return torch


def _require_torchvision():
    try:
        import torchvision
    except Exception as exc:
        raise SystemExit(
            "This scenario requires torchvision. Install with: pip install -e .[torch]"
        ) from exc
    return torchvision


def _make_dataset(name="cifar100", root="data", max_samples=512, augment=False, seed=123):
    torch = _require_torch()
    torchvision = _require_torchvision()
    transforms = torchvision.transforms
    tfms = [transforms.ToTensor()]
    if augment:
        tfms.insert(0, transforms.RandomHorizontalFlip())
    tfms.append(transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)))
    transform = transforms.Compose(tfms)

    if name != "cifar100":
        raise SystemExit("Only cifar100 is supported in this scenario script.")

    dataset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
    if max_samples and max_samples > 0 and max_samples < len(dataset):
        indices = torch.arange(int(max_samples))
        dataset = torch.utils.data.Subset(dataset, indices.tolist())
    return dataset


def _model_hash(model):
    torch = _require_torch()
    total = torch.tensor(0.0)
    for p in model.parameters():
        total += p.detach().float().sum()
    return float(total.item())


@run_reproducible(
    auto_capture_git=True,
    seed=123,
    config={
        "scenario": "research_student",
        "dataset": _DEFAULT_DATASET,
        "epochs": _DEFAULT_EPOCHS,
        "batch_size": _DEFAULT_BATCH_SIZE,
        "max_samples": _DEFAULT_MAX_SAMPLES,
        "augment": _DEFAULT_AUGMENT,
    },
    artifacts_dir=_DEFAULT_ARTIFACTS_DIR,
)
def train(run_ctx=None):
    torch = _require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = run_ctx.config or {}
    dataset = _make_dataset(
        name=cfg.get("dataset", _DEFAULT_DATASET),
        root=os.environ.get("TRAINKEEPER_DATA_DIR", "data"),
        max_samples=int(cfg.get("max_samples", _DEFAULT_MAX_SAMPLES)),
        augment=bool(cfg.get("augment", _DEFAULT_AUGMENT)),
        seed=123,
    )
    log_data_fingerprint(dataset, artifacts_dir=run_ctx.run_dir, seed=123)
    loader = make_dataloader(
        dataset,
        batch_size=int(cfg.get("batch_size", _DEFAULT_BATCH_SIZE)),
        shuffle=True,
        seed=123,
        num_workers=0,
    )

    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Flatten(),
        torch.nn.Linear(32 * 8 * 8, 100),
    ).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    ckpt_dir = Path(run_ctx.run_dir) / "checkpoints"
    ckpt = CheckpointManager(directory=str(ckpt_dir), max_to_keep=2)
    resume_from = os.environ.get("RESUME_FROM")
    resumed = False
    if resume_from:
        ckpt_path = Path(resume_from) / "checkpoints" / "ckpt-1.pt"
        if ckpt_path.exists():
            ckpt.load(model, optimizer=opt, path=str(ckpt_path))
            resumed = True

    losses = []
    epochs = int(cfg.get("epochs", _DEFAULT_EPOCHS))
    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu().item()))
        if epoch == 0:
            ckpt.save(model, optimizer=opt, step=1)

    return {
        "final_loss": float(np.mean(losses[-10:])),
        "model_hash": _model_hash(model),
        "resumed": resumed,
        "dataset_size": int(len(dataset)),
    }


if __name__ == "__main__":
    train()

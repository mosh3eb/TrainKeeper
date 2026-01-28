import contextlib
import inspect
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path


class EfficientTrain:
    def __init__(self, mixed_precision=False, grad_accum_steps=1):
        self.mixed_precision = mixed_precision
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self._step = 0
        self.scaler = None
        self.autocast = contextlib.nullcontext

        if self.mixed_precision:
            try:
                import torch

                self.scaler = torch.cuda.amp.GradScaler()
                self.autocast = torch.cuda.amp.autocast
            except Exception:
                self.mixed_precision = False

    def backward(self, loss, optimizer):
        if self.mixed_precision and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        self._step += 1
        if self._step % self.grad_accum_steps == 0:
            if self.mixed_precision and self.scaler is not None:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    def __enter__(self):
        self._autocast_cm = self.autocast()
        self._autocast_cm.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._autocast_cm.__exit__(exc_type, exc, tb)

    def __call__(self, fn):
        def wrapper(*args, **kwargs):
            with self as ctx:
                if "eff" in inspect.signature(fn).parameters and "eff" not in kwargs:
                    kwargs["eff"] = ctx
                return fn(*args, **kwargs)

        return wrapper


def efficient_train(mixed_precision=False, grad_accum_steps=1):
    return EfficientTrain(mixed_precision=mixed_precision, grad_accum_steps=grad_accum_steps)


@dataclass
class CheckpointManager:
    directory: str = "checkpoints"
    max_to_keep: int = 3

    def _ensure_dir(self):
        Path(self.directory).mkdir(parents=True, exist_ok=True)

    def save(self, model, optimizer=None, scheduler=None, step=None, extra=None, name=None, metric=None, best_only=False):
        self._ensure_dir()
        step = int(step) if step is not None else 0
        filename = name or f"ckpt-{step}.pt"
        path = Path(self.directory) / filename

        try:
            import torch
        except Exception as exc:
            raise RuntimeError("torch is required for checkpointing") from exc

        payload = {"model": model.state_dict(), "step": step}
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler"] = scheduler.state_dict()
        if extra is not None:
            payload["extra"] = extra
        if metric is not None:
            payload["metric"] = metric

        if best_only:
            best_path = Path(self.directory) / "best.txt"
            prev = None
            if best_path.exists():
                prev = best_path.read_text(encoding="utf-8").strip()
            if prev and Path(prev).exists() and metric is not None:
                prev_payload = torch.load(prev, map_location="cpu")
                if prev_payload.get("metric") is not None and metric <= prev_payload["metric"]:
                    return Path(prev)
            torch.save(payload, path)
            best_path.write_text(str(path), encoding="utf-8")
        else:
            torch.save(payload, path)
            self._prune()
            (Path(self.directory) / "latest.txt").write_text(str(path), encoding="utf-8")
        return path

    def load(self, model, optimizer=None, scheduler=None, path=None, strict=True, map_location="cpu"):
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("torch is required for checkpointing") from exc

        if path is None:
            latest = Path(self.directory) / "latest.txt"
            if not latest.exists():
                raise FileNotFoundError("latest checkpoint not found")
            path = latest.read_text(encoding="utf-8").strip()

        payload = torch.load(path, map_location=map_location)
        model.load_state_dict(payload.get("model", {}), strict=strict)
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and "scheduler" in payload:
            scheduler.load_state_dict(payload["scheduler"])
        return payload

    def _prune(self):
        if self.max_to_keep <= 0:
            return
        ckpts = sorted(Path(self.directory).glob("ckpt-*.pt"), key=os.path.getmtime)
        for old in ckpts[:-self.max_to_keep]:
            old.unlink(missing_ok=True)


def warm_start(model, checkpoint_path, map_location="cpu", filter_fn=None):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch is required for warm start") from exc

    payload = torch.load(checkpoint_path, map_location=map_location)
    state = payload.get("model", payload)
    if filter_fn is not None:
        state = {k: v for k, v in state.items() if filter_fn(k, v)}
    missing, unexpected = model.load_state_dict(state, strict=False)
    return {"missing_keys": missing, "unexpected_keys": unexpected}


def seed_worker(seed):
    def _seed_worker(worker_id):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        try:
            import numpy as np

            np.random.seed(worker_seed)
        except Exception:
            pass

    return _seed_worker


def seeded_generator(seed):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch is required for generator seeding") from exc

    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return gen


def make_dataloader(dataset, batch_size=32, shuffle=True, seed=1337, num_workers=0, **kwargs):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch is required for dataloader") from exc

    generator = seeded_generator(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker(seed),
        generator=generator,
        **kwargs,
    )


def dataset_hash(dataset, max_items=1000):
    h = 0
    try:
        import pandas as pd

        if isinstance(dataset, pd.DataFrame):
            sample = dataset.head(max_items).to_numpy()
            for row in sample:
                h ^= hash(str(row.tolist()))
            return hex(h)
    except Exception:
        pass
    for idx in range(min(len(dataset), max_items)):
        item = dataset[idx]
        h ^= hash(str(item))
    return hex(h)


def shuffle_signature(length, seed=1337, take=50):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch is required for shuffle signature") from exc
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    perm = torch.randperm(length, generator=gen)
    sig = perm[:take].tolist()
    return sig


def verify_shuffle_determinism(dataset, seed=1337, take=50):
    sig_a = shuffle_signature(len(dataset), seed=seed, take=take)
    sig_b = shuffle_signature(len(dataset), seed=seed, take=take)
    return sig_a == sig_b


def data_fingerprint(dataset, seed=1337, take=50):
    fingerprint = {"dataset_hash": dataset_hash(dataset)}
    try:
        import pandas as pd

        if isinstance(dataset, pd.DataFrame):
            return fingerprint
    except Exception:
        pass
    fingerprint.update(
        {
            "shuffle_signature": shuffle_signature(len(dataset), seed=seed, take=take),
            "shuffle_deterministic": verify_shuffle_determinism(dataset, seed=seed, take=take),
        }
    )
    return fingerprint


def log_data_fingerprint(dataset, artifacts_dir="artifacts", seed=1337, take=50):
    fp = data_fingerprint(dataset, seed=seed, take=take)
    path = Path(artifacts_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / "data_fingerprint.json"
    out.write_text(json.dumps(fp, indent=2), encoding="utf-8")
    return out


class SmartBatchSampler:
    def __init__(self, lengths, max_tokens, shuffle=True, seed=1337):
        self.lengths = list(lengths)
        self.max_tokens = int(max_tokens)
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(indices)

        batch = []
        total = 0
        for idx in indices:
            length = int(self.lengths[idx])
            if total + length > self.max_tokens and batch:
                yield batch
                batch = []
                total = 0
            batch.append(idx)
            total += length
        if batch:
            yield batch

    def __len__(self):
        total_tokens = sum(self.lengths)
        return max(1, total_tokens // max(1, self.max_tokens))


@dataclass
class ResourceTracker:
    history: list = None

    def __post_init__(self):
        if self.history is None:
            self.history = []

    def record(self, step_time, data_time=None):
        stats = {"step_time": step_time, "data_time": data_time}
        try:
            import torch

            if torch.cuda.is_available():
                stats["gpu_mem_allocated"] = int(torch.cuda.memory_allocated())
                stats["gpu_mem_reserved"] = int(torch.cuda.memory_reserved())
        except Exception:
            pass
        self.history.append(stats)
        return stats

    def summary(self):
        if not self.history:
            return {}
        step_times = [x["step_time"] for x in self.history if x.get("step_time") is not None]
        return {
            "steps": len(self.history),
            "avg_step_time": sum(step_times) / max(1, len(step_times)),
            "max_step_time": max(step_times) if step_times else None,
        }


def build_optimizer(params, name="adamw", lr=1e-3, **kwargs):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch is required for optimizers") from exc

    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, **kwargs)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, **kwargs)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, **kwargs)
    if name == "adagrad":
        return torch.optim.Adagrad(params, lr=lr, **kwargs)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, name="cosine", **kwargs):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch is required for schedulers") from exc

    name = name.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    raise ValueError(f"Unknown scheduler: {name}")

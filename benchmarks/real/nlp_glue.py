import os
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trainkeeper.datacheck import DataCheck
from trainkeeper.experiment import run_reproducible
from trainkeeper.trainutils import efficient_train, log_data_fingerprint, ResourceTracker
from trainkeeper.debugger import HookManager, check_training_step

from benchmarks.real.utils import get_run_dir, save_result


def _require_nlp():
    try:
        import torch
        from datasets import load_dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:
        raise SystemExit(
            "This benchmark requires datasets + transformers. "
            "Install with: pip install -e .[nlp]"
        ) from exc
    return torch, load_dataset, AutoModelForSequenceClassification, AutoTokenizer


def _load_glue():
    _, load_dataset, _, _ = _require_nlp()
    cache_dir = os.environ.get("TRAINKEEPER_HF_CACHE")
    try:
        return load_dataset("glue", "sst2", cache_dir=cache_dir)
    except Exception:
        return load_dataset("glue", "sst2", revision="main", cache_dir=cache_dir)


def _prepare(batch_size=16, subset=256):
    torch, load_dataset, AutoModelForSequenceClassification, AutoTokenizer = _require_nlp()
    dataset = _load_glue()
    train = dataset["train"].shuffle(seed=0).select(range(subset))
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch["sentence"], padding="max_length", truncation=True, max_length=128)

    train = train.map(tokenize, batched=True)
    train = train.rename_column("label", "labels")
    train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    return loader, model


def _train_loop(epochs=1, mixed_precision=False, artifacts_dir=None):
    torch, _, _, _ = _require_nlp()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader, model = _prepare()
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    tracker = ResourceTracker()
    hm = HookManager(log_layers=[], artifacts_dir=artifacts_dir or "artifacts", compute_entropy=True).attach(model)
    if artifacts_dir:
        log_data_fingerprint(loader.dataset, artifacts_dir=artifacts_dir)

    start = time.time()
    losses = []
    for _ in range(epochs):
        for batch in loader:
            t0 = time.time()
            batch = {k: v.to(device) for k, v in batch.items()}
            with efficient_train(mixed_precision=mixed_precision, grad_accum_steps=1) as eff:
                out = model(**batch)
                loss = out.loss
                eff.backward(loss, opt)
            losses.append(float(loss.detach().cpu().item()))
            _ = check_training_step(loss, outputs=out.logits, batch=None, model=model, artifacts_dir=artifacts_dir or "artifacts")
            tracker.record(step_time=time.time() - t0)
    elapsed = time.time() - start
    if artifacts_dir:
        hm.flush(filename="debug_stats.json")
    return {"loss": float(np.mean(losses)), "elapsed_sec": elapsed, "resource": tracker.summary()}


def baseline_run():
    return _train_loop(epochs=1, mixed_precision=False)


def trainkeeper_run(run_dir):
    @run_reproducible(auto_capture_git=False, artifacts_dir=run_dir / "exp")
    def _run():
        return _train_loop(epochs=1, mixed_precision=True, artifacts_dir=run_dir / "exp")

    return _run()


def main():
    try:
        # Simple data sanity check on raw text lengths
        dataset = _load_glue()["train"].select(range(200))
        lengths = [len(x) for x in dataset["sentence"]]
        df = pd.DataFrame({"length": lengths, "label": dataset["label"]})
        dc = DataCheck.from_dataframe(df).infer_schema()
        _ = dc.validate(df)

        baseline = baseline_run()
        run_dir = get_run_dir("nlp_glue_sst2")
        tk = trainkeeper_run(run_dir)
        payload = {
            "task": "nlp_glue_sst2",
            "baseline": baseline,
            "trainkeeper": tk,
            "artifacts_dir": str(run_dir),
        }
    except Exception as exc:
        payload = {"task": "nlp_glue_sst2", "status": "skipped", "reason": str(exc)}
    out_path = save_result("nlp_glue_sst2", payload)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

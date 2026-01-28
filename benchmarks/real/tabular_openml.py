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
from trainkeeper.trainutils import ResourceTracker

from benchmarks.real.utils import get_run_dir, save_result


def _require_tabular():
    try:
        from sklearn.datasets import fetch_openml
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.linear_model import LogisticRegression
        import openml
    except Exception as exc:
        raise SystemExit(
            "This benchmark requires scikit-learn and openml. "
            "Install with: pip install -e .[tabular]"
        ) from exc
    return fetch_openml, accuracy_score, train_test_split, OneHotEncoder, LogisticRegression, openml


def _load_data(sample=5000):
    fetch_openml, _, _, _, _, openml = _require_tabular()
    cache_dir = os.environ.get("TRAINKEEPER_OPENML_CACHE")
    if cache_dir:
        try:
            openml.config.set_root_cache_directory(cache_dir)
        except Exception:
            pass
    data = fetch_openml("adult", version=2, as_frame=True)
    df = data.frame.sample(n=sample, random_state=0)
    target = df.pop("class")
    return df, target


def _train_loop():
    fetch_openml, accuracy_score, train_test_split, OneHotEncoder, LogisticRegression = _require_tabular()
    df, target = _load_data()
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=0)

    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns
    num_cols = X_train.select_dtypes(exclude=["object", "category"]).columns

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_cat = encoder.fit_transform(X_train[cat_cols])
    X_test_cat = encoder.transform(X_test[cat_cols])

    X_train_num = X_train[num_cols].to_numpy()
    X_test_num = X_test[num_cols].to_numpy()

    X_train_all = np.hstack([X_train_num, X_train_cat])
    X_test_all = np.hstack([X_test_num, X_test_cat])

    model = LogisticRegression(max_iter=200)
    tracker = ResourceTracker()
    start = time.time()
    model.fit(X_train_all, y_train)
    preds = model.predict(X_test_all)
    acc = accuracy_score(y_test, preds)
    elapsed = time.time() - start
    tracker.record(step_time=elapsed)
    return {"accuracy": float(acc), "elapsed_sec": elapsed, "resource": tracker.summary()}


def baseline_run():
    return _train_loop()


def trainkeeper_run(run_dir):
    df, _ = _load_data(sample=1000)
    dc = DataCheck.from_dataframe(df).infer_schema()
    _ = dc.validate(df)

    @run_reproducible(auto_capture_git=False, artifacts_dir=run_dir / "exp")
    def _run():
        return _train_loop()

    return _run()


def main():
    baseline = baseline_run()
    run_dir = get_run_dir("tabular_openml_adult")
    tk = trainkeeper_run(run_dir)
    payload = {"task": "tabular_openml_adult", "baseline": baseline, "trainkeeper": tk, "artifacts_dir": str(run_dir)}
    out_path = save_result("tabular_openml_adult", payload)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

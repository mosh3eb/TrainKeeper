import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trainkeeper.datacheck import DataCheck, inject_label_noise, inject_nans, inject_schema_shift
from trainkeeper.experiment import run_reproducible

from pipeline.dataloader import LABEL_COL, load_dataset, save_variant
from pipeline.features import detect_leakage, label_distribution, label_drift_report


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"


def _write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _train_model(df):
    X = df.drop(columns=[LABEL_COL])
    y = df[LABEL_COL].astype(int)
    try:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=200)
        model.fit(pd.get_dummies(X), y)
        acc = float(model.score(pd.get_dummies(X), y))
        return {"trained": True, "accuracy": acc}
    except Exception:
        majority = int(y.value_counts().idxmax())
        acc = float((y == majority).mean())
        return {"trained": False, "accuracy": acc}


def _baseline_paths():
    return REPORTS_DIR / "baseline_schema.json", REPORTS_DIR / "baseline_label.json"


def _save_baseline(dc, df, run_dir):
    profile = dc.profile(df)
    _write_json(Path(run_dir) / "data_profile.json", {k: vars(v) for k, v in profile.items()})
    schema_path = Path(run_dir) / "schema.json"
    dc.save(schema_path)
    _write_json(Path(run_dir) / "fingerprint.json", {"fingerprint": dc.dataset_fingerprint()})
    baseline_schema, baseline_label = _baseline_paths()
    baseline_schema.parent.mkdir(parents=True, exist_ok=True)
    baseline_schema.write_text(schema_path.read_text(encoding="utf-8"), encoding="utf-8")
    counts, dist = label_distribution(df, LABEL_COL)
    _write_json(baseline_label, {"counts": counts, "dist": dist})


def _load_baseline():
    baseline_schema, baseline_label = _baseline_paths()
    if not baseline_schema.exists():
        raise SystemExit("Baseline schema not found. Run clean case first.")
    dc = DataCheck.load(baseline_schema)
    baseline_label_data = json.loads(baseline_label.read_text(encoding="utf-8"))
    return dc, baseline_label_data


def _schema_diff(baseline_schema, df):
    expected = {k for k in baseline_schema.keys() if k != "_meta"}
    current = set(df.columns)
    return {
        "missing_columns": sorted(expected - current),
        "unknown_columns": sorted(current - expected),
    }


def _run_case(case, artifacts_root):
    data_root = DATA_DIR
    df = load_dataset(data_root, split="clean")
    label_policy = {"no_nulls": True, "min_classes": 2}
    reports = {}
    blocked = False
    detected = False

    if case == "clean":
        dc = DataCheck.from_dataframe(df).infer_schema().snapshot_with_label(df, LABEL_COL)
        return df, dc, reports, blocked, detected, label_policy

    dc, baseline_label = _load_baseline()
    if case == "nan_corruption":
        df = inject_nans(df, columns=["age", "bmi", "bp"], frac=0.03, seed=7)
        save_variant(df, data_root, "corrupted")
        policy = {col: {"nullable": False} for col in df.columns if col != LABEL_COL}
        dc.enforce_schema(policy)
        issues = dc.validate(df, label_col=LABEL_COL, label_policy=label_policy)
        nan_issues = [i for i in issues if i.get("issue") == "nulls_present"]
        if nan_issues:
            detected = True
            blocked = True
            reports["data_failure_report.json"] = {"issues": nan_issues}
    elif case == "schema_break":
        df = df.drop(columns=["bp"]).rename(columns={"glucose": "glucose_mg"})
        df = inject_schema_shift(df, column="age", to_type="string")
        save_variant(df, data_root, "corrupted")
        issues = dc.validate(df, label_col=LABEL_COL, label_policy=label_policy)
        if issues:
            detected = True
            blocked = True
            reports["schema_diff.json"] = {
                "issues": issues,
                "diff": _schema_diff(dc.schema, df),
            }
    elif case == "label_shift":
        df = df.copy()
        rng = np.random.RandomState(4)
        idx = rng.choice(df.index, size=max(1, int(len(df) * 0.3)), replace=False)
        df.loc[idx, LABEL_COL] = 1
        save_variant(df, data_root, "corrupted")
        counts, dist = label_distribution(df, LABEL_COL)
        drift = label_drift_report(baseline_label["dist"], dist, threshold=0.05, delta_threshold=0.1)
        if drift["drifted"]:
            detected = True
        reports["label_drift_report.json"] = {
            "baseline_counts": baseline_label["counts"],
            "current_counts": counts,
            "drift": drift,
        }
    elif case == "leakage_case":
        df = df.copy()
        df["leak_score"] = df[LABEL_COL] + np.random.RandomState(1).normal(0, 0.01, size=len(df))
        save_variant(df, data_root, "corrupted")
        leakage = detect_leakage(df, LABEL_COL, corr_threshold=0.95, match_threshold=0.98)
        if leakage["status"] == "flagged":
            detected = True
        reports["leakage_report.json"] = leakage
    elif case == "drift_case":
        df = df.copy()
        df["glucose"] = df["glucose"] + 35.0
        save_variant(df, data_root, "drifted")
        drift = dc.drift_report(df, threshold={"js_divergence": 0.15, "ks_statistic": 0.15, "psi": 0.15})
        if drift.issues:
            detected = True
        reports["drift_report.json"] = {"issues": drift.issues, "summary": drift.summary}
    else:
        raise SystemExit(f"Unknown case: {case}")

    return df, dc, reports, blocked, detected, label_policy


def run_case(case, artifacts_root):
    @run_reproducible(
        auto_capture_git=True,
        seed=123,
        config={"scenario": "data_lab", "case": case},
        artifacts_dir=str(Path(artifacts_root) / case),
    )
    def _runner(run_ctx=None):
        start = time.time()
        df, dc, reports, blocked, detected, label_policy = _run_case(case, artifacts_root)
        detect_ms = int((time.time() - start) * 1000)

        if case == "clean":
            _save_baseline(dc, df, run_ctx.run_dir)
        else:
            for name, payload in reports.items():
                _write_json(Path(run_ctx.run_dir) / name, payload)

        metrics = {
            "case": case,
            "rows": int(len(df)),
            "blocked": bool(blocked),
            "detected": bool(detected),
            "detection_ms": detect_ms,
        }
        if not blocked:
            metrics.update(_train_model(df))
        return metrics

    return _runner()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        default="clean",
        choices=["clean", "nan_corruption", "schema_break", "label_shift", "leakage_case", "drift_case"],
    )
    parser.add_argument("--artifacts-root", default="scenario2_results")
    args = parser.parse_args()
    return run_case(args.case, args.artifacts_root)


if __name__ == "__main__":
    main()

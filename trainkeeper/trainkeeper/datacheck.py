import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _js_divergence(p, q, eps=1e-8):
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))


def _ks_statistic(x, y):
    x = np.sort(np.asarray(x, dtype=np.float64))
    y = np.sort(np.asarray(y, dtype=np.float64))
    if x.size == 0 or y.size == 0:
        return 0.0
    data = np.concatenate([x, y])
    cdf_x = np.searchsorted(x, data, side="right") / x.size
    cdf_y = np.searchsorted(y, data, side="right") / y.size
    return float(np.max(np.abs(cdf_x - cdf_y)))


def _psi(expected, actual, eps=1e-8):
    expected = np.asarray(expected, dtype=np.float64) + eps
    actual = np.asarray(actual, dtype=np.float64) + eps
    expected = expected / expected.sum()
    actual = actual / actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _infer_column_schema(series):
    nullable = series.isna().any()
    null_fraction = float(series.isna().mean())
    if pd.api.types.is_numeric_dtype(series):
        data = series.dropna().to_numpy()
        if data.size == 0:
            col_min = None
            col_max = None
        else:
            col_min = float(np.nanmin(data))
            col_max = float(np.nanmax(data))
        return {
            "type": "numeric",
            "dtype_kind": series.dtype.kind,
            "nullable": nullable,
            "null_fraction": null_fraction,
            "min": col_min,
            "max": col_max,
        }
    return {
        "type": "categorical",
        "dtype_kind": str(series.dtype),
        "nullable": nullable,
        "null_fraction": null_fraction,
    }


def _snapshot_column(series, bins=20, top_k=50, sample_size=1000):
    if pd.api.types.is_numeric_dtype(series):
        data = series.dropna().to_numpy()
        if data.size == 0:
            return {"type": "numeric", "hist": [], "edges": [], "sample": []}
        hist, edges = np.histogram(data, bins=bins)
        sample = data
        if data.size > sample_size:
            rng = np.random.RandomState(0)
            sample = rng.choice(data, size=sample_size, replace=False)
        return {
            "type": "numeric",
            "hist": hist.tolist(),
            "edges": edges.tolist(),
            "sample": sample.tolist(),
        }
    counts = series.astype("string").value_counts(dropna=False)
    if len(counts) > top_k:
        top = counts.iloc[:top_k]
        other = int(counts.iloc[top_k:].sum())
        values = top.to_dict()
        values["__other__"] = other
        return {"type": "categorical", "counts": values, "top_k": top_k}
    return {"type": "categorical", "counts": counts.to_dict(), "top_k": top_k}


def _fingerprint_payload(payload):
    raw = json.dumps(_to_jsonable(payload), sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _to_jsonable(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    return str(obj)


def inject_nans(df, columns, frac=0.05, seed=0):
    df = df.copy()
    rng = np.random.RandomState(seed)
    for col in columns:
        idx = rng.choice(df.index, size=max(1, int(len(df) * frac)), replace=False)
        df.loc[idx, col] = np.nan
    return df


def inject_label_noise(df, label_col, frac=0.1, seed=0):
    df = df.copy()
    rng = np.random.RandomState(seed)
    idx = rng.choice(df.index, size=max(1, int(len(df) * frac)), replace=False)
    labels = df[label_col].astype("string").unique().tolist()
    if len(labels) < 2:
        return df
    df.loc[idx, label_col] = rng.choice(labels, size=len(idx))
    return df


def inject_schema_shift(df, column, to_type="string"):
    df = df.copy()
    if to_type == "string":
        df[column] = df[column].astype("string")
    elif to_type == "int":
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
    return df


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    null_fraction: float
    mean: float | None
    std: float | None
    min_value: float | None
    max_value: float | None
    unique: int


@dataclass
class DriftReport:
    issues: list
    summary: dict

    def to_json(self):
        return json.dumps({"issues": self.issues, "summary": self.summary}, indent=2)


def _summarize_issues(issues):
    summary = {}
    for issue in issues:
        key = issue.get("issue", "unknown")
        summary[key] = summary.get(key, 0) + 1
    return summary


@dataclass
class DataCheck:
    schema: dict = field(default_factory=dict)
    snapshot: dict = field(default_factory=dict)
    _df: Optional[pd.DataFrame] = None

    @classmethod
    def from_dataframe(cls, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("DataCheck only supports pandas.DataFrame for MVP")
        return cls(schema={}, snapshot={}, _df=df)

    @classmethod
    def from_pyarrow_table(cls, table):
        if not hasattr(table, "to_pandas"):
            raise TypeError("table must support to_pandas()")
        return cls.from_dataframe(table.to_pandas())

    @classmethod
    def from_hf_dataset(cls, dataset):
        if hasattr(dataset, "to_pandas"):
            return cls.from_dataframe(dataset.to_pandas())
        raise TypeError("dataset must support to_pandas()")

    @classmethod
    def from_tfds(cls, dataset):
        try:
            import tensorflow_datasets as tfds

            df = tfds.as_dataframe(dataset)
            return cls.from_dataframe(df)
        except Exception as exc:
            raise TypeError("dataset must be compatible with tfds.as_dataframe") from exc

    def infer_schema(self, df=None):
        df = df if df is not None else self._df
        if df is None:
            raise ValueError("df is required for schema inference")
        self.schema = {col: _infer_column_schema(df[col]) for col in df.columns}
        self.snapshot = {col: _snapshot_column(df[col]) for col in df.columns}
        self.schema["_meta"] = {
            "schema_version": 1,
            "fingerprint": self.dataset_fingerprint(),
        }
        return self

    def enforce_schema(self, schema_policy):
        for col, policy in schema_policy.items():
            if col not in self.schema:
                self.schema[col] = {"type": policy.get("type", "numeric")}
            self.schema[col].update(policy)
        return self

    def dataset_fingerprint(self):
        payload = {"schema": self.schema, "snapshot": self.snapshot}
        return _fingerprint_payload(payload)

    def snapshot_with_label(self, df, label_col):
        if label_col not in df.columns:
            raise ValueError("label_col not found in dataframe")
        label_counts = df[label_col].astype("string").value_counts(dropna=False).to_dict()
        per_class_stats = {}
        for col in df.columns:
            if col == label_col:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                per_class_stats[col] = (
                    df.groupby(label_col)[col].mean().fillna(0).to_dict()
                )
        self.snapshot["_label"] = {
            "label_col": label_col,
            "counts": label_counts,
            "per_class_mean": per_class_stats,
        }
        return self

    def profile(self, df):
        profiles = {}
        for col in df.columns:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                profiles[col] = ColumnProfile(
                    name=col,
                    dtype="numeric",
                    null_fraction=float(series.isna().mean()),
                    mean=float(series.mean(skipna=True)),
                    std=float(series.std(skipna=True)),
                    min_value=float(series.min(skipna=True)),
                    max_value=float(series.max(skipna=True)),
                    unique=int(series.nunique(dropna=False)),
                )
            else:
                profiles[col] = ColumnProfile(
                    name=col,
                    dtype="categorical",
                    null_fraction=float(series.isna().mean()),
                    mean=None,
                    std=None,
                    min_value=None,
                    max_value=None,
                    unique=int(series.nunique(dropna=False)),
                )
        return profiles

    def drift_report(self, df, threshold=None, label_col=None, label_policy=None):
        issues = self.validate(df, threshold=threshold, label_col=label_col, label_policy=label_policy)
        return DriftReport(issues=issues, summary=_summarize_issues(issues))

    def save(self, path):
        payload = _to_jsonable({"schema": self.schema, "snapshot": self.snapshot})
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(schema=data.get("schema", {}), snapshot=data.get("snapshot", {}))

    def validate(self, df, threshold=None, alert_fn=None, label_col=None, label_policy=None):
        if threshold is None:
            threshold = {"js_divergence": 0.2, "ks_statistic": 0.2, "psi": 0.2}
        issues = []

        for col in self.schema.keys():
            if col == "_meta":
                continue
            if col not in df.columns:
                issues.append({"column": col, "issue": "missing_column"})

        for col in df.columns:
            if col not in self.schema:
                issues.append({"column": col, "issue": "unknown_column"})
                continue
            col_schema = self.schema[col]
            series = df[col]

            if col_schema["type"] == "numeric":
                if not pd.api.types.is_numeric_dtype(series):
                    issues.append({"column": col, "issue": "type_mismatch", "expected": "numeric"})
                    continue

                if not col_schema.get("nullable", True) and series.isna().any():
                    issues.append({"column": col, "issue": "nulls_present"})

                data = series.dropna().to_numpy()
                if data.size:
                    col_min = float(np.nanmin(data))
                    col_max = float(np.nanmax(data))
                    if col_schema.get("min") is not None and col_min < col_schema["min"]:
                        issues.append({"column": col, "issue": "range_violation", "min": col_min})
                    if col_schema.get("max") is not None and col_max > col_schema["max"]:
                        issues.append({"column": col, "issue": "range_violation", "max": col_max})

                snap = self.snapshot.get(col, {})
                if snap.get("type") == "numeric" and snap.get("hist"):
                    hist, _ = np.histogram(series.dropna().to_numpy(), bins=len(snap["hist"]))
                    if hist.size and len(snap["hist"]):
                        js = _js_divergence(hist, snap["hist"])
                    else:
                        js = 0.0
                    if js > threshold.get("js_divergence", 0.2):
                        issues.append({"column": col, "issue": "drift_js", "js_divergence": js})
                    if snap.get("sample"):
                        ks = _ks_statistic(series.dropna().to_numpy(), np.asarray(snap["sample"]))
                        if ks > threshold.get("ks_statistic", 0.2):
                            issues.append({"column": col, "issue": "drift_ks", "ks_statistic": ks})
                    psi = _psi(snap.get("hist", []), hist) if hist.size else 0.0
                    if psi > threshold.get("psi", 0.2):
                        issues.append({"column": col, "issue": "drift_psi", "psi": psi})
            else:
                if pd.api.types.is_numeric_dtype(series):
                    issues.append({"column": col, "issue": "type_mismatch", "expected": "categorical"})
                    continue

                if not col_schema.get("nullable", True) and series.isna().any():
                    issues.append({"column": col, "issue": "nulls_present"})

                snap = self.snapshot.get(col, {})
                if snap.get("type") == "categorical":
                    counts = series.astype("string").value_counts(dropna=False)
                    keys = set(counts.keys()) | set(snap["counts"].keys())
                    p = np.array([counts.get(k, 0) for k in keys])
                    q = np.array([snap["counts"].get(k, 0) for k in keys])
                    js = _js_divergence(p, q)
                    if js > threshold.get("js_divergence", 0.2):
                        issues.append({"column": col, "issue": "drift_js", "js_divergence": js})

        if label_col is not None and label_col in df.columns:
            label_policy = label_policy or {}
            label_series = df[label_col]
            if label_policy.get("no_nulls") and label_series.isna().any():
                issues.append({"column": label_col, "issue": "label_nulls"})
            if label_policy.get("min_classes"):
                if label_series.nunique() < int(label_policy["min_classes"]):
                    issues.append({"column": label_col, "issue": "label_low_cardinality"})

        if issues and alert_fn is not None:
            try:
                alert_fn(issues)
            except Exception:
                pass
        return issues

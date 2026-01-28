import numpy as np
import pandas as pd


def detect_leakage(df, label_col="label", corr_threshold=0.95, match_threshold=0.98):
    if label_col not in df.columns:
        return {"status": "error", "reason": "label column missing", "suspicious": []}

    label = df[label_col]
    suspicious = []
    for col in df.columns:
        if col == label_col:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            corr = float(np.corrcoef(series.fillna(0), label)[0, 1])
            if np.isnan(corr):
                corr = 0.0
            if abs(corr) >= corr_threshold:
                suspicious.append({"feature": col, "reason": "high_correlation", "corr": corr})
            if series.nunique(dropna=True) <= 2:
                match = float((series.fillna(0).astype(int) == label.astype(int)).mean())
                if match >= match_threshold:
                    suspicious.append({"feature": col, "reason": "near_perfect_match", "match_rate": match})
        else:
            try:
                match = float((series.astype("string") == label.astype("string")).mean())
                if match >= match_threshold:
                    suspicious.append({"feature": col, "reason": "near_perfect_match", "match_rate": match})
            except Exception:
                continue
    status = "flagged" if suspicious else "clean"
    return {"status": status, "suspicious": suspicious}


def label_distribution(df, label_col="label"):
    counts = df[label_col].astype("string").value_counts(dropna=False).to_dict()
    total = sum(counts.values()) or 1
    return {str(k): int(v) for k, v in counts.items()}, {str(k): v / total for k, v in counts.items()}


def js_divergence(p, q, eps=1e-8):
    keys = sorted(set(p) | set(q))
    pa = np.array([p.get(k, 0.0) for k in keys], dtype=np.float64) + eps
    qa = np.array([q.get(k, 0.0) for k in keys], dtype=np.float64) + eps
    pa = pa / pa.sum()
    qa = qa / qa.sum()
    m = 0.5 * (pa + qa)
    return float(0.5 * (np.sum(pa * np.log(pa / m)) + np.sum(qa * np.log(qa / m))))


def label_drift_report(baseline_dist, current_dist, threshold=0.2, delta_threshold=0.1):
    js = js_divergence(baseline_dist, current_dist)
    base_pos = float(baseline_dist.get("1", 0.0))
    curr_pos = float(current_dist.get("1", 0.0))
    delta = abs(curr_pos - base_pos)
    return {
        "js_divergence": js,
        "threshold": threshold,
        "pos_rate_delta": delta,
        "delta_threshold": delta_threshold,
        "drifted": (js > threshold) or (delta > delta_threshold),
    }

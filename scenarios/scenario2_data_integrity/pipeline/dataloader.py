import os
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_COL = "label"


def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def generate_clean_dataset(path, n=2000, seed=123):
    rng = np.random.RandomState(seed)
    age = rng.normal(45, 12, size=n).clip(18, 90)
    bmi = rng.normal(27, 5, size=n).clip(15, 55)
    bp = rng.normal(120, 15, size=n).clip(80, 200)
    glucose = rng.normal(95, 18, size=n).clip(60, 240)
    visits = rng.poisson(2.5, size=n)
    churn_score = rng.normal(0.0, 1.0, size=n)
    region = rng.choice(["north", "south", "east", "west"], size=n, p=[0.3, 0.25, 0.25, 0.2])
    channel = rng.choice(["web", "mobile", "branch"], size=n, p=[0.5, 0.35, 0.15])

    logit = (
        0.02 * (age - 40)
        + 0.03 * (bmi - 25)
        + 0.01 * (bp - 120)
        + 0.02 * (glucose - 90)
        + 0.4 * churn_score
        + rng.normal(0, 0.5, size=n)
    )
    prob = 1 / (1 + np.exp(-logit))
    label = (prob > 0.5).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "bmi": bmi,
            "bp": bp,
            "glucose": glucose,
            "visits": visits,
            "churn_score": churn_score,
            "region": region,
            "channel": channel,
            LABEL_COL: label,
        }
    )

    path = Path(path)
    _ensure_dir(path.parent)
    df.to_csv(path, index=False)
    return df


def load_dataset(root, split="clean", seed=123):
    root = Path(root)
    if split == "clean":
        path = root / "clean" / "clean.csv"
        if not path.exists():
            return generate_clean_dataset(path, n=2000, seed=seed)
        return pd.read_csv(path)
    path = root / split / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return pd.read_csv(path)


def save_variant(df, root, split):
    path = Path(root) / split / f"{split}.csv"
    _ensure_dir(path.parent)
    df.to_csv(path, index=False)
    return path

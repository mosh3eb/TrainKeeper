import json

import numpy as np
import pandas as pd

from trainkeeper.datacheck import DataCheck


def inject_schema_swap(df):
    df = df.copy()
    df["age"] = df["age"].astype(str)
    return df


def inject_null_spike(df):
    df = df.copy()
    df.loc[df.sample(frac=0.2, random_state=0).index, "income"] = np.nan
    return df


def inject_range_violation(df):
    df = df.copy()
    df.loc[df.sample(frac=0.1, random_state=1).index, "age"] = 999
    return df


def main():
    base = pd.DataFrame(
        {
            "age": np.random.randint(18, 65, size=1000),
            "income": np.random.normal(50000, 10000, size=1000),
            "segment": np.random.choice(["a", "b", "c"], size=1000),
        }
    )
    dc = DataCheck.from_dataframe(base).infer_schema()

    cases = {
        "schema_swap": inject_schema_swap(base),
        "null_spike": inject_null_spike(base),
        "range_violation": inject_range_violation(base),
    }

    results = {}
    for name, df in cases.items():
        issues = dc.validate(df, threshold={"js_divergence": 0.2, "ks_statistic": 0.2})
        results[name] = issues

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

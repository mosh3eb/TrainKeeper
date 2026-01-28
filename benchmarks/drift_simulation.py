import json

import numpy as np
import pandas as pd

from trainkeeper.datacheck import DataCheck


def main():
    base = pd.DataFrame(
        {
            "value": np.random.normal(0, 1, size=2000),
            "category": np.random.choice(["a", "b", "c"], p=[0.6, 0.3, 0.1], size=2000),
        }
    )
    drifted = pd.DataFrame(
        {
            "value": np.random.normal(2.5, 1.2, size=2000),
            "category": np.random.choice(["a", "b", "c"], p=[0.2, 0.4, 0.4], size=2000),
        }
    )

    dc = DataCheck.from_dataframe(base).infer_schema()
    issues = dc.validate(drifted, threshold={"js_divergence": 0.1, "ks_statistic": 0.1})
    print(json.dumps({"issues": issues}, indent=2))


if __name__ == "__main__":
    main()

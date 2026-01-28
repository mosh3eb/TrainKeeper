import numpy as np
import pandas as pd

from trainkeeper.datacheck import DataCheck


def test_schema_inference_and_validation():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    dc = DataCheck.from_dataframe(df).infer_schema()
    issues = dc.validate(df)
    assert issues == []


def test_drift_detection_js():
    base = pd.DataFrame({"value": np.random.normal(0, 1, size=500)})
    new = pd.DataFrame({"value": np.random.normal(4, 1, size=500)})
    dc = DataCheck.from_dataframe(base).infer_schema()
    issues = dc.validate(new, threshold={"js_divergence": 0.05, "ks_statistic": 0.05})
    assert any(i["issue"].startswith("drift") for i in issues)

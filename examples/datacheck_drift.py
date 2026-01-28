import numpy as np
import pandas as pd

from trainkeeper.datacheck import DataCheck

base = pd.DataFrame({"value": np.random.normal(0, 1, size=1000)})
new = pd.DataFrame({"value": np.random.normal(3, 1, size=1000)})

dc = DataCheck.from_dataframe(base).infer_schema()
issues = dc.validate(new, threshold={"js_divergence": 0.2, "ks_statistic": 0.2})
print("issues:", issues)

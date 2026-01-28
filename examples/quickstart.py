import time

import numpy as np
import pandas as pd

from trainkeeper.debugger import HookManager, LossTracker
from trainkeeper.datacheck import DataCheck
from trainkeeper.experiment import run_reproducible
from trainkeeper.monitor import MetricsMonitor
from trainkeeper.trainutils import efficient_train


@run_reproducible(auto_capture_git=True)
def train():
    train_df = pd.DataFrame(
        {
            "x": np.random.randn(1000),
            "y": np.random.randn(1000),
            "label": np.random.randint(0, 2, size=1000),
        }
    )

    dc = DataCheck.from_dataframe(train_df).infer_schema()
    issues = dc.validate(train_df)
    if issues:
        print("data issues:", issues)

    # Fake training loop (replace with real PyTorch code)
    loss_tracker = LossTracker()
    monitor = MetricsMonitor()
    for step in range(5):
        time.sleep(0.01)
        loss = 1.0 / (step + 1)
        loss_tracker.update(loss, batch_id=step)
        monitor.record_latency(0.01)
        monitor.record_predictions([0, 1, 1, 0])

    print("loss records:", loss_tracker.records[:2])
    print("monitor snapshot:", monitor.snapshot())


if __name__ == "__main__":
    train()

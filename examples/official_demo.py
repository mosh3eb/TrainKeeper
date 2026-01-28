import time

import numpy as np
import pandas as pd

from trainkeeper.datacheck import DataCheck, inject_nans, inject_label_noise
from trainkeeper.debugger import HookManager, LossTracker, build_health_report, check_training_step
from trainkeeper.experiment import run_reproducible
from trainkeeper.pkg import create_repro_seal, generate_model_card
from trainkeeper.trainutils import ResourceTracker, log_data_fingerprint


@run_reproducible(auto_capture_git=True, config={"demo": True, "epochs": 2})
def demo(run_ctx=None):
    # Synthetic tabular data to demonstrate checks and artifacts
    df = pd.DataFrame(
        {
            "age": np.random.randint(18, 65, size=200),
            "income": np.random.normal(50000, 10000, size=200),
            "segment": np.random.choice(["a", "b", "c"], size=200),
            "label": np.random.randint(0, 2, size=200),
        }
    )

    # Data integrity
    dc = DataCheck.from_dataframe(df).infer_schema().snapshot_with_label(df, "label")
    df_bad = inject_nans(df, ["income"], frac=0.1)
    df_bad = inject_label_noise(df_bad, "label", frac=0.1)
    issues = dc.validate(df_bad, label_col="label", label_policy={"min_classes": 2, "no_nulls": True})

    # Training-style monitoring (simulated)
    tracker = ResourceTracker()
    loss_tracker = LossTracker()
    hm = HookManager(log_layers=[], artifacts_dir=str(run_ctx.run_dir), compute_entropy=True)

    for step in range(5):
        time.sleep(0.01)
        loss = 1.0 / (step + 1)
        loss_tracker.update(loss, batch_id=step)
        tracker.record(step_time=0.01)
        _ = check_training_step(loss, outputs=None, batch=None, model=None, artifacts_dir=str(run_ctx.run_dir))

    hm.flush(filename="debug_stats.json")
    health = build_health_report(hm, loss_tracker=loss_tracker)

    # Data fingerprint + reproducibility seal
    log_data_fingerprint(df, artifacts_dir=str(run_ctx.run_dir))
    create_repro_seal(run_ctx.run_dir)
    generate_model_card(
        output=str(run_ctx.run_dir / "MODEL_CARD.md"),
        model_name="trainkeeper-demo",
        description="Official TrainKeeper demo experiment",
        metrics={"issues_found": len(issues)},
        datasets=["synthetic-tabular"],
        training_details={"steps": 5},
    )

    return {
        "issues_found": len(issues),
        "resource": tracker.summary(),
        "health_summary": health.summary,
    }


if __name__ == "__main__":
    demo()

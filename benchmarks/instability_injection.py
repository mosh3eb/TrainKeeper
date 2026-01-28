import json
import time

import numpy as np

from trainkeeper.debugger import HookManager, LossTracker


def simulate_training(instability_step=5):
    loss_tracker = LossTracker()
    hm = HookManager(log_layers=[])
    fake_grad_norms = []
    for step in range(10):
        time.sleep(0.01)
        if step >= instability_step:
            loss = 10 ** (step - instability_step + 1)
            grad_norm = 10 ** (step - instability_step + 2)
        else:
            loss = 1.0 / (step + 1)
            grad_norm = 0.1 / (step + 1)
        loss_tracker.update(loss, batch_id=step)
        fake_grad_norms.append({"param": f"layer.{step}", "norm": grad_norm})

    hm._stats["gradients"] = {g["param"]: {"norm": g["norm"]} for g in fake_grad_norms}
    issues = hm.analyze_gradients(explode_threshold=100.0, vanish_threshold=1e-6)
    return {"issues": issues, "loss_records": loss_tracker.records}


def main():
    report = simulate_training(instability_step=4)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

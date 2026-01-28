# ![TrainKeeper logo](https://raw.githubusercontent.com/mosh3eb/TrainKeeper/main/assets/branding/trainkeeper-logo.png)
# TrainKeeper

TrainKeeper is a minimal-decision, high-signal toolkit for making ML training experiments reproducible, debuggable, resource-efficient, and production-ready with a composable API that fits into existing codebases.

## Why TrainKeeper

Most failures happen **inside** training loops: non-determinism, silent data drift, unstable gradients, and inconsistent packaging. TrainKeeper adds lightweight guardrails and artifacts without replacing your existing stack.

## Install

```bash
pip install trainkeeper
```

Optional extras:
- `trainkeeper[torch]` PyTorch helpers
- `trainkeeper[vision]` vision benchmarks
- `trainkeeper[nlp]` NLP benchmarks
- `trainkeeper[tabular]` tabular benchmarks
- `trainkeeper[wandb]` W&B integration
- `trainkeeper[mlflow]` MLflow integration

## Quick start

```python
from trainkeeper.experiment import run_reproducible

@run_reproducible(auto_capture_git=True)
def train():
    # your training code
    print("TrainKeeper is running.")

if __name__ == "__main__":
    train()
```

Artifacts created per run:
- `experiment.yaml`, `run.json`
- `seeds.json`, `system.json`, `env.txt`, `run.sh`

## Modules

- `experiment` deterministic runs + environment capture
- `datacheck` schema inference + drift detection
- `trainutils` mixed precision, checkpoints, dataloader determinism
- `debugger` hooks, loss tracking, failure capture
- `monitor` runtime metrics + prediction drift
- `pkg` export helpers (ONNX/TorchScript)

## CLI

```bash
tk init
tk run -- python train.py
tk replay <exp-id> -- python train.py
tk compare <exp-a> <exp-b>
tk doctor
tk repro-summary <runs-dir>
```

## License

Apache-2.0

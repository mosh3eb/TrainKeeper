# ![TrainKeeper logo](assets/branding/trainkeeper-logo.png)
# TrainKeeper

[![PyPI](https://img.shields.io/pypi/v/trainkeeper)](https://pypi.org/project/trainkeeper/)
[![Python](https://img.shields.io/pypi/pyversions/trainkeeper)](https://pypi.org/project/trainkeeper/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![CI](https://github.com/mosh3eb/TrainKeeper/actions/workflows/ci.yml/badge.svg)](https://github.com/mosh3eb/TrainKeeper/actions/workflows/ci.yml)

TrainKeeper is a minimal-decision, high-signal toolkit for building reproducible, debuggable, and efficient ML training systems. It adds guardrails **inside** training loops without replacing your existing stack.

## Positioning
TrainKeeper augments MLflow/W&B/DVC/Hydra/Lightning rather than replacing them. It focuses on the training-time failure surface: determinism, data issues, and instability.

## Design principles
- Zero-surprise defaults (deterministic seeds + environment capture)
- Composable modules (opt-in, independent)
- Minimal API surface (wraps existing loops)
- Observability-first (artifacts over dashboards)

## Who it's for
- ML engineers iterating on model training
- Researchers validating reproducibility and failures
- Teams shipping production-grade training pipelines

## What TrainKeeper solves
- **Reproducibility**: deterministic seeds, environment capture, replay
- **Data integrity**: schema inference, drift detection, corruption alarms
- **Training stability**: hooks for gradients/activations, failure snapshots
- **Efficiency**: mixed precision, accumulation, checkpoint utilities
- **Production handoff**: export helpers and runtime sanity checks

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

## Outputs per run
- `experiment.yaml`, `run.json`
- `seeds.json`, `system.json`, `env.txt`, `run.sh`

## Typical workflow
```bash
tk init
tk run -- python train.py
tk compare exp-aaa exp-bbb
tk repro-summary scenario_runs/
```

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

## Core modules
- `experiment` reproducible runs + environment capture
- `datacheck` schema inference + drift detection
- `trainutils` efficient training primitives
- `debugger` hooks + failure snapshots
- `monitor` runtime metrics + prediction drift
- `pkg` export helpers

## Scenarios & system tests (repo-only)
These are **not** included in the PyPI package.
- `scenarios/scenario1_reproducibility/`
- `scenarios/scenario2_data_integrity/`
- `scenarios/scenario3_training_robustness/`
- `system_tests/runner.py`

## System hardening
Run the cross-scenario validation suite:
```bash
tk system-check
```
Outputs: `scenario_results/system_summary.md` and `scenario_results/unified_failure_matrix.json`.

## CLI
```bash
tk init
tk run -- python train.py
tk replay <exp-id> -- python train.py
tk compare <exp-a> <exp-b>
tk doctor
tk repro-summary <runs-dir>
tk system-check
```

## Examples
- `examples/quickstart.py`
- `examples/datacheck_drift.py`
- `examples/official_demo.py`
- `examples/demo.py`

## Benchmarks
See `benchmarks/` for the baseline suite and real pipelines.

## Documentation
- `docs/architecture.md`
- `docs/benchmark_plan.md`
- `docs/benchmarks.md`
- `docs/hypotheses.md`
- `docs/research_problem.md`
- `docs/packaging.md`

## Release checklist
- `python -m build`
- `twine check dist/*`
- `tk system-check`

## Architecture diagram
See `docs/architecture.md` for the system overview and component boundaries.

## Development
```bash
pip install -e .[dev,torch]
pytest
mkdocs serve
```

## Contributing
We welcome issues and PRs. Please:
- open an issue with the problem or proposal
- keep changes scoped and tested
- run `pytest` and `tk system-check` before submitting

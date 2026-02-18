<div align="center">
  <img src="assets/branding/trainkeeper-logo.png" alt="TrainKeeper Logo" width="100%">
  
  <br>

  [![PyPI Version](https://img.shields.io/pypi/v/trainkeeper?style=for-the-badge&color=blue)](https://pypi.org/project/trainkeeper/)
  [![Python Versions](https://img.shields.io/pypi/pyversions/trainkeeper?style=for-the-badge&color=green)](https://pypi.org/project/trainkeeper/)
  [![License](https://img.shields.io/badge/license-Apache--2.0-orange?style=for-the-badge)](LICENSE)
  [![CI](https://img.shields.io/github/actions/workflow/status/mosh3eb/TrainKeeper/ci.yml?branch=main&style=for-the-badge&label=CI)](https://github.com/mosh3eb/TrainKeeper/actions/workflows/ci.yml)

  <h3>Production-Grade Training Guardrails for PyTorch</h3>

  <p>Reproducible • Debuggable • Distributed • Efficient</p>
</div>

---

**TrainKeeper** is a minimal-decision, high-signal toolkit for building reproducible, debuggable, and efficient ML training systems. It adds guardrails **inside** training loops without replacing your existing stack.

## ⚡️ Why TrainKeeper?

Most failures happen **silently** inside execution loops: non-determinism, data drift, unstable gradients, and inconsistent environments. TrainKeeper solves this with zero-config composable modules.

- **🔒 Zero-Surprise Reproducibility**: Automatic seed setting, environment capture, and git state locking.
- **🛡️ Data Integrity**: Schema inference and drift detection caught *before* training wastes GPU hours.
- **🚅 Distributed Made Easy**: Auto-configured DDP and FSDP with a single line of code.
- **📉 Resource Efficiency**: GPU memory profiling and smart checkpointing that respects disk limits.

## 🚀 Quick start
```python
from trainkeeper.experiment import run_reproducible

@run_reproducible(auto_capture_git=True)
def train():
    # your training code
    print("TrainKeeper is running.")

if __name__ == "__main__":
    train()
```

Minimal check:
```bash
pip install trainkeeper
tk --help
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

## 📦 Install
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
- `trainkeeper[dashboard]` **NEW**: Interactive Streamlit dashboard
- `trainkeeper[all]` All features

## 🎨 Interactive Dashboard (NEW in v0.3.0)

Launch a beautiful, interactive dashboard to explore your experiments:

```bash
pip install trainkeeper[dashboard]
tk dashboard
```

**Features:**
- 🔍 **Experiment Explorer**: Browse and filter all experiments with metadata
- 📈 **Metric Comparison**: Interactive Plotly charts comparing metrics across runs
- 🌊 **Data Drift Analysis**: Visualize schema changes and data quality
- 💻 **System Monitor**: Track GPU usage, dependencies, and reproducibility score

The dashboard provides a modern, gradient-based UI with:
- Real-time filtering and search
- Interactive visualizations
- Reproducibility scoring
- Export capabilities

Open `http://localhost:8501` after running `tk dashboard` to access the interface.

## 🚀 Production-Grade Features (NEW)

### Distributed Training Made Easy

Stop fighting with `torch.distributed`. TrainKeeper handles everything:

```python
from trainkeeper.distributed import distributed_training, wrap_model_ddp, wrap_model_fsdp

with distributed_training() as dist_config:
    model = MyModel()
    model = wrap_model_ddp(model, dist_config)  # That's it!
    # Or for large models:
    # model = wrap_model_fsdp(model, dist_config)
    # Your training code works exactly the same
```

**Features:**
- 🔄 Auto-detects `torchrun`, SLURM, or manual setup
- 🎯 DDP support with one function call
- 🚀 **NEW**: FSDP (Fully Sharded Data Parallel) support for large models
    - Just replace `wrap_model_ddp` with `wrap_model_fsdp`!
- 💾 Smart distributed checkpointing
- 📊 Distributed sampler creation

```bash
# Single GPU → Multi-GPU with ZERO code changes
torchrun --nproc_per_node=4 train.py
```

---

### GPU Memory Profiler

**The #1 pain point in deep learning = solved.**

```python
from trainkeeper.gpu_profiler import GPUProfiler

profiler = GPUProfiler()
profiler.start()

for batch in dataloader:
    profiler.step("forward")
    loss = model(batch)
    profiler.step("backward")
    loss.backward()

report = profiler.stop()
print(report.summary())  # Get actionable recommendations!
```

**What you get:**
- 🔍 Memory leak detection
- 💡 Automatic optimization recommendations
- 📊 Peak/average/fragmentation analysis
- 🎯 Optimal batch size finder

Example output:
```
💡 Recommendations:
  1. Memory fragmentation detected (35%). Try:
     • torch.cuda.empty_cache() periodically
  2. Consider gradient checkpointing to trade compute for memory
```

---

### Smart Checkpoint Manager

Never run out of disk space again. **Automatic cleanup** based on your metrics:

```python
from trainkeeper.checkpoint_manager import CheckpointManager

manager = CheckpointManager(
    keep_best=3,       # Keep top 3 by metric
    keep_last=2,       # Keep 2 most recent
    metric="val_acc",
    mode="max",        # Higher is better
    compress=True      # Auto-compress old checkpoints
)

# During training
manager.save(
    model=model,
    optimizer=optimizer,
    epoch=epoch,
    metrics={"val_acc": 0.95, "loss": 0.05}
)
# Old checkpoints automatically cleaned up!
```

**Features:**
- 🧹 Automatic cleanup (keep best N + last N)
- 📦 Optional compression (gzip)
- 🔐 Checkpoint integrity hashing
- ☁️  Cloud sync ready (S3, GCS, Azure)

---

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

## Scenarios summary
| Scenario | Purpose | Key output | Outcome |
|---|---|---|---|
| reproducibility | deterministic runs | `run.json`, `metrics.json` | consistent hashing |
| data integrity | silent data bugs | schema + drift reports | detected corruptions |
| robustness | model instability | debug reports | captured failures |

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
tk dashboard  # NEW: Launch interactive dashboard
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

## How it works (diagram)
![TrainKeeper architecture for training-time guardrails](assets/branding/architecture-diagram.png)

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

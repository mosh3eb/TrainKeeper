<div align="center">
  <img src="https://raw.githubusercontent.com/mosh3eb/TrainKeeper/main/assets/branding/trainkeeper-logo.png" alt="TrainKeeper Logo" width="100%">
  
  <br>

  [![PyPI Version](https://img.shields.io/pypi/v/trainkeeper?style=for-the-badge&color=blue)](https://pypi.org/project/trainkeeper/)
  [![Python Versions](https://img.shields.io/pypi/pyversions/trainkeeper?style=for-the-badge&color=green)](https://pypi.org/project/trainkeeper/)
  [![License](https://img.shields.io/badge/license-Apache--2.0-orange?style=for-the-badge)](LICENSE)

  <h3>Production-Grade Training Guardrails for PyTorch</h3>

  <p>Reproducible • Debuggable • Distributed • Efficient</p>
</div>

---

**TrainKeeper** is a minimal-decision, high-signal toolkit for building robust ML training systems. It adds guardrails **inside** your training loops without replacing your existing stack (PyTorch, Lightning, Accelerate).

## ⚡️ Why TrainKeeper?

Most failures happen **silently** inside execution loops: non-determinism, data drift, unstable gradients, and inconsistent environments. TrainKeeper solves this with zero-config composable modules.

- **🔒 Zero-Surprise Reproducibility**: Automatic seed setting, environment capture, and git state locking.
- **🛡️ Data Integrity**: Schema inference and drift detection caught *before* training wastes GPU hours.
- **🚅 Distributed Made Easy**: Auto-configured DDP and FSDP with a single line of code.
- **📉 Resource Efficiency**: GPU memory profiling and smart checkpointing that respects disk limits.

## 📦 Installation

```bash
pip install trainkeeper
```

## 🚀 Quick Start

Wrap your entry point to effectively "freeze" the experimental conditions:

```python
from trainkeeper.experiment import run_reproducible

@run_reproducible(auto_capture_git=True)
def train():
    print("TrainKeeper is running: Experiment is now reproducible.")

if __name__ == "__main__":
    train()
```

## ✨ Features at a Glance

### 1. Distributed Training (DDP & FSDP)
Stop fighting with `torchrun`.

```python
from trainkeeper.distributed import distributed_training, wrap_model_fsdp

with distributed_training() as dist_config:
    model = MyModel()
    model = wrap_model_fsdp(model, dist_config)  # FSDP with auto-wrapping!
```

### 2. GPU Memory Profiler
Find leaks and optimize batch sizes automatically.

```python
from trainkeeper.gpu_profiler import GPUProfiler

profiler = GPUProfiler()
profiler.start()
# ... training loop ...
print(profiler.stop().summary())
# Output: "Fragmentation detected (35%). Suggestion: Empty cache at epoch end."
```

### 3. Interactive Dashboard
Explore experiments, compare metrics, and analyze drift.

```bash
pip install trainkeeper[dashboard]
tk dashboard
```

## 🔗 Links

- **GitHub Repository**: [mosh3eb/TrainKeeper](https://github.com/mosh3eb/TrainKeeper)
- **Full Documentation**: [Read the Docs](https://github.com/mosh3eb/TrainKeeper/tree/main/docs)


# TrainKeeper Benchmarks

These scripts provide a lightweight, reproducible harness for evaluating TrainKeeper's
core claims: reproducibility, data failure detection, drift detection, and debugging
latency for instabilities.

## Setup
```bash
pip install -e .[dev,torch]
```

## Run
```bash
python benchmarks/reproducibility.py
python benchmarks/data_corruption.py
python benchmarks/drift_simulation.py
python benchmarks/instability_injection.py
```

## Real pipelines (PyTorch / GLUE / OpenML)
```bash
python benchmarks/real/vision_cifar.py
python benchmarks/real/nlp_glue.py
python benchmarks/real/tabular_openml.py
python benchmarks/real/aggregate_results.py
python benchmarks/real/report.py
python benchmarks/real/run_all.py
```

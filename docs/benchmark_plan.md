 # TrainKeeper Benchmark Plan
 
 ## Metrics
 - Reproducibility score: variance of metrics across repeated seeded runs.
 - Drift detection accuracy: true/false positive rates on synthetic shifts.
 - Debug latency: time to identify and localize instability events.
 - Training efficiency: throughput (samples/sec), memory use, time-to-checkpoint.
 - Failure coverage: number of injected failures detected.
 
 ## Datasets and Tasks
 - Vision: CIFAR-10/100 (subset), ImageNet subset.
 - NLP: GLUE (subset), SQuAD (subset).
 - Tabular: OpenML/UCI datasets.
 - Synthetic drift: controlled distribution shifts.
 
 ## Baselines
 - Raw PyTorch training loop.
 - PyTorch + W&B.
 - PyTorch + TrainKeeper.
 
 ## Experimental Design
 - 3 model families: CNN (vision), Transformer (NLP), MLP/GBDT (tabular).
 - 3–5 datasets, 3 seeds each.
 - Injected failures: data schema swap, label leakage, gradient explosion.
 - Report: mean/variance metrics, drift detection rates, latency breakdowns.
 
 ## Outputs
 - Tables: reproducibility variance, detection accuracy, overhead.
 - Plots: drift detection curves, stability diagnostics.
 - Artifacts: experiment manifests, run scripts, telemetry logs.

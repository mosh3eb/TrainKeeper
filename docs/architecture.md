 # TrainKeeper Architecture (Draft)
 
 This file provides the system architecture content for conversion to `docs/architecture.pdf`.
 
 ## Overview
 TrainKeeper is a modular training-time system with five core pillars:
 - Reproducibility and environment capture
 - Data validation and drift detection
 - Training efficiency primitives
 - Debugging and interpretability hooks
 - Packaging and monitoring for deployment

![TrainKeeper architecture diagram](assets/architecture-diagram.png)
 
 ## Components
 
 ### experiment
 Captures environment metadata, locks seeds, and writes a reproducibility manifest and replay script.
 
 ### datacheck
 Infers data schemas, validates types/ranges/nulls, and detects distribution drift with JS/KS statistics.
 
 ### trainutils
 Provides mixed precision, gradient accumulation, checkpointing, warm-start, and deterministic data loading.
 
 ### debugger
 Hooks into activations/gradients, detects instability, and surfaces failure signatures.
 
 ### pkg
 Exports to ONNX/TorchScript, provides quantization helpers, and emits minimal Dockerfiles.
 
 ### monitor
 Collects runtime metrics and drift signals; exports Prometheus-compatible metrics.
 
 ## Data/Control Flow
 1. `experiment` wraps training entrypoint and captures environment.
 2. `datacheck` validates dataset batches pre-training.
 3. `trainutils` manages training loop efficiency and checkpointing.
 4. `debugger` collects telemetry and instability indicators.
 5. `pkg` exports model artifacts for deployment targets.
 6. `monitor` tracks runtime predictions and latency.
 
 ## Integration Strategy
 TrainKeeper augments MLflow/W&B/DVC rather than replacing them, using light integrations and artifact logging.

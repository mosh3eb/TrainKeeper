# Scenario 1 — Reproducibility Lab

This scenario demonstrates deterministic training and reproducible artifacts.

## Goal
Prove that identical runs produce identical metrics and hashes.

## What is tested
- deterministic seeding and replay
- artifact capture and comparison

## Run

```bash
python3 scenarios/scenario1_reproducibility/train.py
```

## Expected outputs
- `scenario1_runs/exp-*/experiment.yaml`
- `scenario1_runs/exp-*/metrics.json`

Artifacts are written to `scenario1_runs/` (ignored by git).

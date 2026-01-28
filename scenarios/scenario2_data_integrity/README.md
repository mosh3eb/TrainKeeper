# Scenario 2 — Data Corruption & Silent Failure Lab

This scenario simulates real-world tabular data failures and verifies that
TrainKeeper detects and reports them before training.

## Goal
Catch silent data corruption before training.

## What is tested
- NaNs in features
- schema mismatches
- label shift
- leakage detection
- distribution drift

## Structure

```
scenarios/scenario2_data_integrity/
  data/
    clean/
    corrupted/
    drifted/
  pipeline/
    dataloader.py
    features.py
    train.py
  reports/
```

## Run all cases

```bash
python3 scenarios/scenario2_data_integrity/run_all.py
```

Outputs are written to `scenario2_results/` with per-case runs plus:
- `global_data_report.json`
- `failure_matrix.json`
- `scenario2_summary.json`
- `scenario2_summary.md`

## Run a single case

```bash
python3 scenarios/scenario2_data_integrity/pipeline/train.py --case nan_corruption
```

## Expected outputs
- `scenario2_results/*/exp-*/metrics.json`
- `scenario2_results/*/exp-*/schema_diff.json` (schema break)
- `scenario2_results/*/exp-*/drift_report.json` (drift)

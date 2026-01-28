# Scenario 2 — Data Corruption & Silent Failure Lab

This scenario simulates real-world tabular data failures and verifies that
TrainKeeper detects and reports them before training.

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

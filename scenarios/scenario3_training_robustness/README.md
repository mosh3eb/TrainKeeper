# Scenario 3 — Training Instability & Robustness Lab

This scenario simulates training-time failures and verifies that TrainKeeper
detects, reports, and snapshots training instability.

## Run all cases

```bash
python3 scenarios/scenario3_training_robustness/run_all.py
```

Outputs are written to `scenario3_results/` with:
- `failure_matrix.json`
- `scenario3_summary.json`
- `scenario3_summary.md`

## Run a single case

```bash
python3 scenarios/scenario3_training_robustness/pipeline/train.py --case nan_loss
```

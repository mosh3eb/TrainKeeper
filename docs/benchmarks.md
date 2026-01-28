# Benchmark Results

After running `python benchmarks/real/run_all.py`, results are stored under:
- `benchmarks/real/results/`
- `benchmarks/real/report.md`
- `benchmarks/real/plots/metrics.png`
- `benchmarks/real/plots/runtime.png`

If the GLUE dataset is unavailable, the NLP benchmark will be marked as **skipped** and still produce a report.

## Optional HF cache
Set `TRAINKEEPER_HF_CACHE` to reuse cached datasets:
```bash
export TRAINKEEPER_HF_CACHE="$HOME/.cache/huggingface/datasets"
```

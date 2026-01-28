import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.real.utils import load_results


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit("This report requires matplotlib. Install with: pip install -e .[bench]") from exc
    return plt


def _extract_metric(item):
    b = item.get("baseline", {})
    t = item.get("trainkeeper", {})
    b_metric = b.get("accuracy", b.get("loss"))
    t_metric = t.get("accuracy", t.get("loss"))
    return b_metric, t_metric


def make_table(results, out_path="benchmarks/real/report.md"):
    lines = [
        "| task | baseline_metric | trainkeeper_metric | baseline_time | trainkeeper_time |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in results:
        task = item.get("task", "unknown")
        if item.get("status") == "skipped":
            lines.append(f"| {task} | skipped | skipped | - | - |")
            continue
        b = item.get("baseline", {})
        t = item.get("trainkeeper", {})
        b_metric, t_metric = _extract_metric(item)
        lines.append(
            f"| {task} | {b_metric:.4f} | {t_metric:.4f} | {b.get('elapsed_sec',0):.4f} | {t.get('elapsed_sec',0):.4f} |"
        )
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    return out_path


def make_plots(results, out_dir="benchmarks/real/plots"):
    plt = _require_matplotlib()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = [r for r in results if r.get("status") != "skipped"]
    if not results:
        return {}
    tasks = [r.get("task") for r in results]
    b_metrics = []
    t_metrics = []
    for r in results:
        b, t = _extract_metric(r)
        b_metrics.append(b)
        t_metrics.append(t)

    x = range(len(tasks))
    plt.figure(figsize=(8, 4))
    plt.bar(x, b_metrics, width=0.4, label="baseline")
    plt.bar([i + 0.4 for i in x], t_metrics, width=0.4, label="trainkeeper")
    plt.xticks([i + 0.2 for i in x], tasks, rotation=20, ha="right")
    plt.title("Baseline vs TrainKeeper metric")
    plt.tight_layout()
    metric_path = out_dir / "metrics.png"
    plt.savefig(metric_path)
    plt.close()

    b_time = [r.get("baseline", {}).get("elapsed_sec", 0) for r in results]
    t_time = [r.get("trainkeeper", {}).get("elapsed_sec", 0) for r in results]
    plt.figure(figsize=(8, 4))
    plt.bar(x, b_time, width=0.4, label="baseline")
    plt.bar([i + 0.4 for i in x], t_time, width=0.4, label="trainkeeper")
    plt.xticks([i + 0.2 for i in x], tasks, rotation=20, ha="right")
    plt.title("Baseline vs TrainKeeper runtime (sec)")
    plt.tight_layout()
    time_path = out_dir / "runtime.png"
    plt.savefig(time_path)
    plt.close()

    return {"metrics_plot": str(metric_path), "runtime_plot": str(time_path)}


def main():
    results = load_results()
    if not results:
        print("No results found. Run real benchmarks first.")
        return
    table_path = make_table(results)
    plots = make_plots(results)
    summary = {"table": str(table_path), "plots": plots}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

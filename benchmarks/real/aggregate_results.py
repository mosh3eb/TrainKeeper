from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.real.utils import load_results


def _fmt(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def to_markdown(results):
    lines = ["| task | baseline_metric | trainkeeper_metric | baseline_time | trainkeeper_time |", "|---|---:|---:|---:|---:|"]
    for item in results:
        task = item.get("task", "unknown")
        if item.get("status") == "skipped":
            lines.append(f"| {task} | skipped | skipped | - | - |")
            continue
        b = item.get("baseline", {})
        t = item.get("trainkeeper", {})
        b_metric = b.get("accuracy", b.get("loss", ""))
        t_metric = t.get("accuracy", t.get("loss", ""))
        lines.append(
            f"| {task} | {_fmt(b_metric)} | {_fmt(t_metric)} | {_fmt(b.get('elapsed_sec',''))} | {_fmt(t.get('elapsed_sec',''))} |"
        )
    return "\n".join(lines)


def main():
    results = load_results()
    if not results:
        print("No results found. Run benchmarks first.")
        return
    print(to_markdown(results))


if __name__ == "__main__":
    main()

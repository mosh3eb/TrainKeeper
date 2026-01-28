import json
from pathlib import Path

from pipeline.train import run_case


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
ARTIFACTS_ROOT = REPO_ROOT / "scenario3_results"


def _latest_run_dir(case_dir):
    runs = sorted(case_dir.glob("exp-*"), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def _collect_runs(case_dir):
    return sorted(case_dir.glob("exp-*"), key=lambda p: p.stat().st_mtime)


def main():
    cases = [
        "exploding_gradient",
        "nan_loss",
        "corrupted_batch",
        "optimizer_divergence",
        "representation_collapse",
    ]
    for case in cases:
        run_case(case, str(ARTIFACTS_ROOT))

    matrix = []
    summary = {"cases": []}
    verdict = {"status": "PASS", "reasons": []}

    for case in cases:
        case_dir = ARTIFACTS_ROOT / case
        run_dir = _latest_run_dir(case_dir)
        metrics_path = run_dir / "metrics.json" if run_dir else None
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path and metrics_path.exists() else {}
        report_name = metrics.get("report") or "missing"
        detected = bool(metrics.get("detected"))
        blocked = bool(metrics.get("blocked"))

        matrix.append(
            {
                "failure": case,
                "detected": "✅" if detected else "❌",
                "blocked": "✅" if blocked else "⚠️",
                "report_generated": report_name,
            }
        )
        runs = _collect_runs(case_dir)
        summary["cases"].append(
            {
                "failure": case,
                "runs": [r.name for r in runs],
                "latest_run": run_dir.name if run_dir else None,
                "detected": detected,
                "blocked": blocked,
                "report_generated": report_name,
                "metrics": metrics,
            }
        )
        if not detected:
            verdict["status"] = "FAIL"
            verdict["reasons"].append(f"{case}: not detected")
        if report_name == "missing":
            verdict["status"] = "FAIL"
            verdict["reasons"].append(f"{case}: report missing")

    summary["verdict"] = verdict
    summary_path = ARTIFACTS_ROOT / "scenario3_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    matrix_path = ARTIFACTS_ROOT / "failure_matrix.json"
    matrix_path.write_text(json.dumps(matrix, indent=2), encoding="utf-8")

    md_path = ARTIFACTS_ROOT / "scenario3_summary.md"
    lines = [
        "# Scenario 3 Summary",
        "",
        f"- Verdict: {verdict['status']}",
    ]
    if verdict["reasons"]:
        lines.append(f"- Reasons: {', '.join(verdict['reasons'])}")
    lines.extend(
        [
            "",
            "## Failure Matrix",
            "| failure | detected | blocked | report | latest_run |",
            "|---|---|---|---|---|",
        ]
    )
    for entry in summary["cases"]:
        lines.append(
            f"| {entry['failure']} | "
            f"{'✅' if entry['detected'] else '❌'} | "
            f"{'✅' if entry['blocked'] else '⚠️'} | "
            f"{entry['report_generated']} | "
            f"{entry['latest_run'] or 'n/a'} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(str(matrix_path))
    print(str(summary_path))
    print(str(md_path))


if __name__ == "__main__":
    main()

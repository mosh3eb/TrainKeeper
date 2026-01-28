import json
from pathlib import Path

from pipeline.train import run_case


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
ARTIFACTS_ROOT = REPO_ROOT / "scenario2_results"


def _latest_run_dir(case_dir):
    runs = sorted(case_dir.glob("exp-*"), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def _collect_runs(case_dir):
    return sorted(case_dir.glob("exp-*"), key=lambda p: p.stat().st_mtime)


def _report_exists(run_dir, name):
    if not run_dir:
        return False
    return (run_dir / name).exists()


def main():
    cases = [
        "clean",
        "nan_corruption",
        "schema_break",
        "label_shift",
        "leakage_case",
        "drift_case",
    ]
    for case in cases:
        run_case(case, str(ARTIFACTS_ROOT))

    matrix = []
    results = {}
    summary = {"cases": []}
    verdict = {"status": "PASS", "reasons": []}
    for case in cases:
        case_dir = ARTIFACTS_ROOT / case
        run_dir = _latest_run_dir(case_dir)
        metrics_path = run_dir / "metrics.json" if run_dir else None
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path and metrics_path.exists() else {}
        results[case] = metrics

        if case == "nan_corruption":
            report = "data_failure_report.json"
        elif case == "schema_break":
            report = "schema_diff.json"
        elif case == "label_shift":
            report = "label_drift_report.json"
        elif case == "leakage_case":
            report = "leakage_report.json"
        elif case == "drift_case":
            report = "drift_report.json"
        else:
            report = None

        report_generated = report if report and _report_exists(run_dir, report) else "missing"
        matrix.append(
            {
                "failure": case,
                "detected": "✅" if metrics.get("detected") else "❌",
                "blocked": "✅" if metrics.get("blocked") else "⚠️",
                "report_generated": report_generated,
            }
        )
        runs = _collect_runs(case_dir)
        summary["cases"].append(
            {
                "failure": case,
                "runs": [r.name for r in runs],
                "latest_run": run_dir.name if run_dir else None,
                "detected": bool(metrics.get("detected")),
                "blocked": bool(metrics.get("blocked")),
                "report_generated": report_generated,
                "metrics": metrics,
            }
        )
        if case != "clean":
            if not metrics.get("detected"):
                verdict["status"] = "FAIL"
                verdict["reasons"].append(f"{case}: not detected")
            if report_generated == "missing":
                verdict["status"] = "FAIL"
                verdict["reasons"].append(f"{case}: report missing")
        if case in {"nan_corruption", "schema_break"} and not metrics.get("blocked"):
            verdict["status"] = "FAIL"
            verdict["reasons"].append(f"{case}: not blocked")

    report_path = ARTIFACTS_ROOT / "global_data_report.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    matrix_path = ARTIFACTS_ROOT / "failure_matrix.json"
    matrix_path.write_text(json.dumps(matrix, indent=2), encoding="utf-8")

    summary["verdict"] = verdict
    summary_path = ARTIFACTS_ROOT / "scenario2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_md = ARTIFACTS_ROOT / "scenario2_summary.md"
    lines = [
        "# Scenario 2 Summary",
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
    summary_md.write_text("\n".join(lines), encoding="utf-8")
    print(str(report_path))
    print(str(matrix_path))
    print(str(summary_path))
    print(str(summary_md))


if __name__ == "__main__":
    main()

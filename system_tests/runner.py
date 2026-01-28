import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCENARIO2 = ROOT / "scenarios" / "scenario2_data_integrity" / "run_all.py"
SCENARIO3 = ROOT / "scenarios" / "scenario3_training_robustness" / "run_all.py"
RESULTS_DIR = ROOT / "scenario_results"


def _run(cmd):
    return subprocess.call(cmd)


def _load(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_matrix(path):
    data = _load(path) or []
    return data if isinstance(data, list) else []


def _health_metrics(summary):
    cases = summary.get("cases", []) if isinstance(summary, dict) else []
    if not cases:
        return {"detected_rate": 0.0, "blocked_rate": 0.0, "avg_detection_ms": 0.0}
    detected = [c for c in cases if c.get("detected")]
    blocked = [c for c in cases if c.get("blocked")]
    times = [c.get("metrics", {}).get("detection_ms") for c in cases if isinstance(c.get("metrics"), dict)]
    times = [t for t in times if isinstance(t, (int, float))]
    return {
        "detected_rate": len(detected) / len(cases),
        "blocked_rate": len(blocked) / len(cases),
        "avg_detection_ms": sum(times) / len(times) if times else 0.0,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    exit_codes = []

    exit_codes.append(_run(["python3", str(SCENARIO2)]))
    exit_codes.append(_run(["python3", str(SCENARIO3)]))

    s2 = _load(ROOT / "scenario2_results" / "scenario2_summary.json") or {}
    s3 = _load(ROOT / "scenario3_results" / "scenario3_summary.json") or {}
    m2 = _load_matrix(ROOT / "scenario2_results" / "failure_matrix.json")
    m3 = _load_matrix(ROOT / "scenario3_results" / "failure_matrix.json")

    verdicts = {
        "scenario2": s2.get("verdict", {}).get("status", "UNKNOWN"),
        "scenario3": s3.get("verdict", {}).get("status", "UNKNOWN"),
    }
    overall = "PASS" if all(v == "PASS" for v in verdicts.values()) else "FAIL"

    unified_matrix = [
        {"scenario": "scenario2", **entry} for entry in m2
    ] + [
        {"scenario": "scenario3", **entry} for entry in m3
    ]
    health = {
        "scenario2": _health_metrics(s2),
        "scenario3": _health_metrics(s3),
    }
    summary = {
        "overall": overall,
        "verdicts": verdicts,
        "exit_codes": exit_codes,
        "health": health,
    }
    summary_path = RESULTS_DIR / "system_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    unified_path = RESULTS_DIR / "unified_failure_matrix.json"
    unified_path.write_text(json.dumps(unified_matrix, indent=2), encoding="utf-8")

    health_path = RESULTS_DIR / "health_metrics.json"
    health_path.write_text(json.dumps(health, indent=2), encoding="utf-8")

    md_path = RESULTS_DIR / "system_summary.md"
    lines = [
        "# System Summary",
        "",
        f"- Overall: {overall}",
        "",
        "## Scenarios",
        "| scenario | verdict |",
        "|---|---|",
        f"| scenario2 | {verdicts['scenario2']} |",
        f"| scenario3 | {verdicts['scenario3']} |",
        "",
        "## Health Metrics",
        "| scenario | detected_rate | blocked_rate | avg_detection_ms |",
        "|---|---:|---:|---:|",
        f"| scenario2 | {health['scenario2']['detected_rate']:.2f} | {health['scenario2']['blocked_rate']:.2f} | {health['scenario2']['avg_detection_ms']:.1f} |",
        f"| scenario3 | {health['scenario3']['detected_rate']:.2f} | {health['scenario3']['blocked_rate']:.2f} | {health['scenario3']['avg_detection_ms']:.1f} |",
    ]
    lines.extend(
        [
            "",
            "## Unified Failure Matrix",
            "| scenario | failure | detected | blocked | report_generated |",
            "|---|---|---|---|---|",
        ]
    )
    for entry in unified_matrix:
        lines.append(
            f"| {entry.get('scenario')} | {entry.get('failure')} | "
            f"{entry.get('detected')} | {entry.get('blocked')} | {entry.get('report_generated')} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(str(summary_path))
    print(str(unified_path))
    print(str(health_path))
    print(str(md_path))

    if overall != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

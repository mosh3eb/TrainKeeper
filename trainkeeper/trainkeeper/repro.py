import json
import statistics
from collections import defaultdict
from pathlib import Path

from trainkeeper.experiment import compare_experiments


def build_repro_report(runs_dir, output="repro_report.json"):
    runs_dir = Path(runs_dir)
    runs = sorted(runs_dir.glob("exp-*"))
    metrics = []
    for r in runs:
        path = r / "metrics.json"
        if path.exists():
            metrics.append((r.name, json.loads(path.read_text(encoding="utf-8"))))

    final_losses = [m[1].get("final_loss") for m in metrics if isinstance(m[1], dict)]
    model_hashes = [m[1].get("model_hash") for m in metrics if isinstance(m[1], dict)]
    total = len(metrics)
    groups = defaultdict(list)
    for run_id, m in metrics:
        key = (m.get("model_hash"), m.get("final_loss"))
        groups[key].append(run_id)
    largest_group = max((len(ids) for ids in groups.values()), default=0)
    determinism_score = (largest_group / total) if total else 0.0
    outliers = []
    if groups:
        for key, ids in groups.items():
            if len(ids) == 1:
                outliers.extend(ids)

    report = {
        "runs": total,
        "identical_runs": largest_group,
        "determinism_score": determinism_score,
        "loss_variance": statistics.pvariance(final_losses) if len(final_losses) > 1 else 0.0,
        "hash_variance": statistics.pvariance(model_hashes) if len(model_hashes) > 1 else 0.0,
        "outliers": outliers,
    }
    out_path = runs_dir / output
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out_path


def _diff_reason(diff):
    reasons = []
    config = diff.get("config") or {}
    if isinstance(config, dict) and config.get("a") is not None and config.get("b") is not None:
        a_cfg = config.get("a") or {}
        b_cfg = config.get("b") or {}
        if isinstance(a_cfg, dict) and isinstance(b_cfg, dict):
            changed = sorted({*a_cfg.keys(), *b_cfg.keys()} - {k for k in a_cfg if a_cfg.get(k) == b_cfg.get(k)})
            if changed:
                reasons.append(f"config differs: {', '.join(changed)}")
        else:
            reasons.append("config differs")

    metrics = diff.get("metrics") or {}
    if isinstance(metrics, dict) and metrics.get("a") is not None and metrics.get("b") is not None:
        a_m = metrics.get("a") or {}
        b_m = metrics.get("b") or {}
        if isinstance(a_m, dict) and isinstance(b_m, dict):
            changed = sorted({*a_m.keys(), *b_m.keys()} - {k for k in a_m if a_m.get(k) == b_m.get(k)})
            if changed:
                reasons.append(f"metrics differ: {', '.join(changed)}")
        else:
            reasons.append("metrics differ")

    env = diff.get("env") or {}
    if env:
        reasons.append(f"environment differs: {', '.join(sorted(env.keys()))}")

    seeds = diff.get("seeds") or {}
    if seeds:
        reasons.append("seed differs")

    resumed = diff.get("resumed") or {}
    if resumed:
        reasons.append("resume status differs")

    return reasons or ["no differences detected"]


def build_repro_summary(runs_dir, output_json="repro_summary.json", output_md="repro_summary.md"):
    runs_dir = Path(runs_dir)
    runs = sorted(runs_dir.glob("exp-*"))
    entries = []
    for r in runs:
        metrics_path = r / "metrics.json"
        run_path = r / "run.json"
        if not metrics_path.exists() or not run_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        run_meta = json.loads(run_path.read_text(encoding="utf-8"))
        entries.append(
            {
                "id": r.name,
                "model_hash": metrics.get("model_hash"),
                "final_loss": metrics.get("final_loss"),
                "resumed": metrics.get("resumed"),
                "seed": run_meta.get("seed"),
            }
        )

    groups = defaultdict(list)
    for e in entries:
        key = (e.get("model_hash"), e.get("final_loss"))
        groups[key].append(e["id"])

    group_list = []
    for idx, (key, ids) in enumerate(sorted(groups.items(), key=lambda x: len(x[1]), reverse=True), start=1):
        group_list.append(
            {
                "group": idx,
                "count": len(ids),
                "model_hash": key[0],
                "final_loss": key[1],
                "runs": ids,
            }
        )

    total_runs = len(entries)
    largest_group = group_list[0]["count"] if group_list else 0
    determinism_score = (largest_group / total_runs) if total_runs else 0.0

    diffs = []
    group_diffs = []
    if len(entries) >= 2 and group_list:
        base = group_list[0]["runs"][0]
        for other in [e["id"] for e in entries[1:]]:
            diff = compare_experiments(runs_dir / base, runs_dir / other)
            diffs.append({"base": base, "other": other, "diff": diff})
        for group in group_list[1:]:
            rep = group["runs"][0]
            diff = compare_experiments(runs_dir / base, runs_dir / rep)
            group_diffs.append(
                {
                    "base": base,
                    "group": group["group"],
                    "representative": rep,
                    "diff": diff,
                    "reasons": _diff_reason(diff),
                }
            )

    summary = {
        "total_runs": total_runs,
        "identical_groups": len(group_list),
        "largest_group": largest_group,
        "determinism_score": determinism_score,
        "groups": group_list,
        "group_diffs": group_diffs,
        "diffs": diffs,
    }

    runs_dir.mkdir(parents=True, exist_ok=True)
    json_path = runs_dir / output_json
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Reproducibility Summary",
        "",
        f"- Total runs: {summary['total_runs']}",
        f"- Identical groups: {summary['identical_groups']}",
        f"- Largest identical group: {summary['largest_group']}",
        f"- Determinism score: {summary['determinism_score']:.3f}",
        "",
        "## Groups",
        "| group | count | model_hash | final_loss | runs |",
        "|---|---:|---|---:|---|",
    ]
    for g in group_list:
        runs_str = ", ".join(g["runs"])
        lines.append(f"| {g['group']} | {g['count']} | {g['model_hash']} | {g['final_loss']} | {runs_str} |")

    if group_diffs:
        lines.extend(["", "## Why groups differ"])
        for diff in group_diffs:
            reasons = "; ".join(diff["reasons"])
            lines.append(f"- Group {diff['group']} ({diff['representative']}): {reasons}")

    md_path = runs_dir / output_md
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": json_path, "md": md_path}

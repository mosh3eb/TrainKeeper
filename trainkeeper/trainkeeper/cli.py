import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from trainkeeper.diagnostics import to_json as diagnostics_json
from trainkeeper.repro import build_repro_report, build_repro_summary
from trainkeeper.experiment import (
    capture_environment,
    compare_experiments,
    load_experiment,
    lock_seeds,
    replay_from_id,
    _to_serializable,
)


def _init_project(path="."):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    template_dir = Path(__file__).resolve().parent / "templates" / "basic"
    if template_dir.exists():
        shutil.copytree(template_dir, path, dirs_exist_ok=True)
    else:
        (path / "artifacts").mkdir(parents=True, exist_ok=True)
        (path / "train.py").write_text(
            "from trainkeeper.experiment import run_reproducible\n\n"
            "@run_reproducible(auto_capture_git=True)\n"
            "def train():\n"
            "    print('hello, trainkeeper')\n\n"
            "if __name__ == '__main__':\n"
            "    train()\n",
            encoding="utf-8",
        )


def _run_with_capture(cmd):
    lock_seeds()
    env = capture_environment(auto_capture_git=True)
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    (Path("artifacts") / "cli_environment.json").write_text(
        json.dumps(_to_serializable(env), indent=2),
        encoding="utf-8",
    )
    return subprocess.call(cmd)


def _replay(exp_id, cmd, artifacts_dir="artifacts", verify=False):
    with replay_from_id(exp_id, artifacts_dir=artifacts_dir, apply_env=True) as data:
        if cmd:
            exit_code = subprocess.call(cmd)
            if verify:
                _verify_replay(exp_id, artifacts_dir=artifacts_dir)
            return exit_code
        entry = data.get("entrypoint", {})
        module_name = entry.get("module")
        func_name = entry.get("name")
        if not module_name or not func_name:
            raise SystemExit("Entry point not found in experiment metadata")
        if module_name == "__main__":
            run_sh = Path(artifacts_dir) / f"exp-{exp_id}" / "run.sh"
            if run_sh.exists():
                exit_code = subprocess.call(["bash", str(run_sh)])
                if verify:
                    _verify_replay(exp_id, artifacts_dir=artifacts_dir)
                return exit_code
            raise SystemExit("run.sh not found for replay")
        try:
            module = __import__(module_name, fromlist=[func_name])
            fn = getattr(module, func_name)
            result = fn()
            if verify:
                _verify_replay(exp_id, artifacts_dir=artifacts_dir)
            return result
        except Exception:
            run_sh = Path(artifacts_dir) / f"exp-{exp_id}" / "run.sh"
            if run_sh.exists():
                exit_code = subprocess.call(["bash", str(run_sh)])
                if verify:
                    _verify_replay(exp_id, artifacts_dir=artifacts_dir)
                return exit_code
            raise


def _verify_replay(exp_id, artifacts_dir="artifacts"):
    exp_dir = Path(artifacts_dir) / f"exp-{exp_id}"
    original = exp_dir / "metrics.json"
    if not original.exists():
        print("verify: metrics.json not found in original run")
        return
    baseline = json.loads(original.read_text(encoding="utf-8"))
    runs = sorted(Path(artifacts_dir).glob("exp-*"), key=lambda p: p.stat().st_mtime)
    latest = runs[-1] if runs else None
    if not latest or latest == exp_dir:
        print("verify: no new run found")
        return
    latest_metrics = latest / "metrics.json"
    if not latest_metrics.exists():
        print("verify: metrics.json not found in replay run")
        return
    replayed = json.loads(latest_metrics.read_text(encoding="utf-8"))
    if baseline == replayed:
        print("verify: PASS (metrics match)")
    else:
        print("verify: FAIL (metrics differ)")


def _system_check():
    root = Path(__file__).resolve().parents[2]
    runner = root / "system_tests" / "runner.py"
    if not runner.exists():
        raise SystemExit("system-check runner not found")
    return subprocess.call([sys.executable, str(runner)])


def _dashboard(port=8501, artifacts_dir="artifacts"):
    """Launch the TrainKeeper interactive dashboard"""
    try:
        import streamlit.web.cli as stcli
    except ImportError:
        raise SystemExit(
            "Dashboard requires streamlit. Install with: pip install trainkeeper[dashboard]"
        )
    
    dashboard_app = Path(__file__).resolve().parent / "dashboard" / "app.py"
    if not dashboard_app.exists():
        raise SystemExit(f"Dashboard app not found at {dashboard_app}")
    
    # Set environment variable for artifacts directory
    import os
    os.environ["TRAINKEEPER_ARTIFACTS_DIR"] = artifacts_dir
    
    sys.argv = ["streamlit", "run", str(dashboard_app), "--server.port", str(port)]
    sys.exit(stcli.main())


def main():
    parser = argparse.ArgumentParser(prog="tk")
    sub = parser.add_subparsers(dest="subcommand")

    p_init = sub.add_parser("init", help="initialize a minimal project")
    p_init.add_argument("--path", default=".")

    p_run = sub.add_parser("run", help="run command with environment capture")
    p_run.add_argument("cmd", nargs=argparse.REMAINDER)

    p_replay = sub.add_parser("replay", help="replay a captured experiment")
    p_replay.add_argument("exp_id")
    p_replay.add_argument("--artifacts-dir", default="artifacts")
    p_replay.add_argument("--verify", action="store_true")
    p_replay.add_argument("cmd", nargs=argparse.REMAINDER)

    p_compare = sub.add_parser("compare", help="compare two experiment runs")
    p_compare.add_argument("exp_a")
    p_compare.add_argument("exp_b")

    p_doctor = sub.add_parser("doctor", help="run environment diagnostics")

    p_repro = sub.add_parser("repro-report", help="generate reproducibility report")
    p_repro.add_argument("runs_dir")

    p_summary = sub.add_parser("repro-summary", help="generate reproducibility summary")
    p_summary.add_argument("runs_dir")

    p_system = sub.add_parser("system-check", help="run scenario system checks")

    p_dashboard = sub.add_parser("dashboard", help="launch interactive dashboard")
    p_dashboard.add_argument("--port", type=int, default=8501, help="port for dashboard (default: 8501)")
    p_dashboard.add_argument("--artifacts-dir", default="artifacts", help="artifacts directory path")

    args = parser.parse_args()

    if args.subcommand == "init":
        _init_project(args.path)
        return 0

    if args.subcommand == "run":
        if not args.cmd:
            raise SystemExit("tk run -- <command>")
        cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
        return _run_with_capture(cmd)

    if args.subcommand == "replay":
        cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
        return _replay(args.exp_id, cmd, artifacts_dir=args.artifacts_dir, verify=args.verify)

    if args.subcommand == "compare":
        diff = compare_experiments(args.exp_a, args.exp_b)
        print(json.dumps(diff, indent=2))
        return 0

    if args.subcommand == "doctor":
        print(diagnostics_json())
        return 0

    if args.subcommand == "repro-report":
        path = build_repro_report(args.runs_dir)
        print(str(path))
        return 0

    if args.subcommand == "repro-summary":
        result = build_repro_summary(args.runs_dir)
        print(str(result["json"]))
        print(str(result["md"]))
        return 0

    if args.subcommand == "system-check":
        return _system_check()

    if args.subcommand == "dashboard":
        return _dashboard(port=args.port, artifacts_dir=args.artifacts_dir)

    parser.print_help()
    return 0

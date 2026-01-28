import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    commands = [
        [sys.executable, "benchmarks/real/vision_cifar.py"],
        [sys.executable, "benchmarks/real/nlp_glue.py"],
        [sys.executable, "benchmarks/real/tabular_openml.py"],
        [sys.executable, "benchmarks/real/aggregate_results.py"],
        [sys.executable, "benchmarks/real/report.py"],
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env.get('PYTHONPATH','')}"
    for cmd in commands:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.call(cmd, env=env)
        if result != 0:
            raise SystemExit(result)


if __name__ == "__main__":
    main()

import json
import math
from statistics import mean, pstdev

import numpy as np

from trainkeeper.experiment import run_reproducible


def _run_once(seed):
    @run_reproducible(seed=seed, auto_capture_git=False, artifacts_dir="artifacts")
    def train():
        rng = np.random.RandomState(seed)
        metrics = rng.normal(loc=0.8, scale=0.02, size=50)
        return float(metrics.mean())

    return train()


def reproducibility_score(runs):
    mu = mean(runs)
    sigma = pstdev(runs) if len(runs) > 1 else 0.0
    score = 1.0 - min(1.0, sigma / max(1e-6, abs(mu)))
    return {"mean": mu, "std": sigma, "score": score}


def main():
    seeds = [7, 7, 7, 7, 7]
    results = [_run_once(seed) for seed in seeds]
    report = reproducibility_score(results)
    print(json.dumps({"results": results, "report": report}, indent=2))


if __name__ == "__main__":
    main()

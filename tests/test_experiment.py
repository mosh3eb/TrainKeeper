import os
from pathlib import Path

from trainkeeper.experiment import capture_environment, run_reproducible


def test_capture_environment_basic():
    env = capture_environment(auto_capture_git=False)
    assert "python_version" in env
    assert "platform" in env
    assert "pip_freeze" in env


def test_run_reproducible_writes_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    @run_reproducible(auto_capture_git=False, artifacts_dir="artifacts")
    def train():
        return 123

    assert train() == 123
    exp_dirs = list((tmp_path / "artifacts").glob("exp-*"))
    assert exp_dirs, "expected experiment directory"
    exp_dir = exp_dirs[0]
    assert (exp_dir / "experiment.yaml").exists()
    run_sh = exp_dir / "run.sh"
    assert run_sh.exists()
    assert os.access(run_sh, os.X_OK)

import contextlib
import datetime as _dt
import functools
import json
import os
import platform
import random
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
import hashlib
import tempfile
from typing import Any, Dict, Optional


def _safe_run(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception:
        return None


def _git_info():
    return {
        "commit": _safe_run(["git", "rev-parse", "HEAD"]),
        "branch": _safe_run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "status": _safe_run(["git", "status", "--porcelain"]),
        "diff": _safe_run(["git", "diff"]),
    }


def _pip_freeze():
    return _safe_run([sys.executable, "-m", "pip", "freeze"])


def _conda_info():
    info = _safe_run(["conda", "info", "--json"])
    pkgs = _safe_run(["conda", "list", "--json"])
    try:
        info = json.loads(info) if info else None
    except Exception:
        info = info
    try:
        pkgs = json.loads(pkgs) if pkgs else None
    except Exception:
        pkgs = pkgs
    return {"info": info, "list": pkgs}


def _cuda_info():
    return {
        "nvidia_smi": _safe_run(["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"]),
        "nvcc": _safe_run(["nvcc", "--version"]),
    }


def _torch_state():
    try:
        import torch

        cuda_states = None
        if torch.cuda.is_available():
            cuda_states = [s.tolist() for s in torch.cuda.get_rng_state_all()]
        return {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "random_state": torch.get_rng_state().tolist(),
            "cuda_random_state": cuda_states,
        }
    except Exception:
        return None


def _numpy_state():
    try:
        import numpy as np

        return np.random.get_state()
    except Exception:
        return None


def _seed_state():
    state = {
        "python_random_state": random.getstate(),
        "numpy_state": _numpy_state(),
        "torch_state": _torch_state(),
    }
    return _to_serializable(state)


def _write_yaml(path, payload):
    try:
        import yaml

        _atomic_write_text(path, yaml.safe_dump(payload, sort_keys=False))
    except Exception:
        _atomic_write_text(path, json.dumps(payload, indent=2))


def _to_serializable(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8", errors="ignore")
        except Exception:
            return str(obj)
    return str(obj)


def _hash_bytes(data):
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def hash_file(path):
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_directory(path, ignore=None):
    path = Path(path)
    if not path.exists():
        return None
    ignore = set(ignore or [])
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_dir():
            continue
        if any(part in ignore for part in p.parts):
            continue
        h.update(str(p.relative_to(path)).encode("utf-8"))
        h.update(hash_file(p).encode("utf-8"))
    return h.hexdigest()


def _atomic_write_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tmp:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
    Path(tmp.name).replace(path)


@dataclass
class RunContext:
    run_dir: Path
    exp_id: str
    config: Optional[Dict[str, Any]] = None
    _drift_detector: Optional[Any] = None

    @property
    def run_json(self):
        return self.run_dir / "run.json"

    @property
    def experiment_yaml(self):
        return self.run_dir / "experiment.yaml"

    def write(self, payload):
        payload = _to_serializable(payload)
        _write_yaml(self.experiment_yaml, payload)
        _atomic_write_text(self.run_json, json.dumps(payload, indent=2))

    def check_drift(self, current_data: Any, feature_name: str, reference_data: Any = None, threshold: float = 0.1):
        """
        Check for data drift against a reference dataset.
        
        Args:
            current_data: The new data to check (numpy array or list)
            feature_name: Name of the feature/column
            reference_data: Reference/baseline data. If None, must have been set previously.
            threshold: Drift score threshold (default 0.1)
        """
        if self._drift_detector is None:
            # Lazy init default detector if none configured
            try:
                from trainkeeper.drift import DriftDetector
                self._drift_detector = DriftDetector()
            except ImportError:
                print("⚠️  Drift detection module not found.")
                return None

        if reference_data is not None:
             self._drift_detector.set_reference(reference_data)

        try:
            report = self._drift_detector.check_drift(
                current_data, 
                feature_name=feature_name, 
                threshold=threshold
            )
            
            # Log drift report
            report_file = self.run_dir / f"drift_report_{feature_name}.json"
            _atomic_write_text(
                report_file, 
                json.dumps({
                    "feature": feature_name,
                    "drifted": report.is_drifted,
                    "score": report.drift_score,
                    "metric": report.metric,
                    "timestamp": _dt.datetime.utcnow().isoformat()
                }, indent=2)
            )
            
            if report.is_drifted:
                print(f"🚨 Drift detected in {feature_name}! Score: {report.drift_score:.4f} > {threshold}")
            
            return report
        except Exception as e:
            print(f"⚠️  Drift check failed for {feature_name}: {e}")
            return None


def lock_seeds(seed=1337):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def capture_environment(auto_capture_git=True):
    env_vars = {
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUDNN_DETERMINISTIC": os.environ.get("CUDNN_DETERMINISTIC"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
    }
    env = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
        "pip_freeze": _pip_freeze(),
        "conda": _conda_info(),
        "cuda": _cuda_info(),
        "random_state": random.getstate(),
        "numpy_state": _numpy_state(),
        "torch_state": _torch_state(),
        "env_vars": env_vars,
    }
    if auto_capture_git:
        env["git"] = _git_info()
    return env


def _write_run_script(run_path, cmd):
    run_path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + " ".join(cmd) + "\n",
        encoding="utf-8",
    )
    run_path.chmod(0o755)


def _write_env_script(run_path, env_vars):
    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    for key, value in env_vars.items():
        if value is None:
            continue
        lines.append(f'export {key}="{value}"')
    run_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    run_path.chmod(0o755)


def _write_text(path, content):
    _atomic_write_text(path, content)


def run_reproducible(
    auto_capture_git=True,
    capture_env=True,
    seed=1337,
    artifacts_dir="artifacts",
    wandb_project=None,
    mlflow_experiment=None,
    write_env_script=True,
    config=None,
    storage_uri=None,
    drift_config=None,
):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            exp_id = uuid.uuid4().hex[:8]
            base = Path(artifacts_dir) / f"exp-{exp_id}"
            base.mkdir(parents=True, exist_ok=True)
            
            # Initialize drift detector if config provided
            drift_detector = None
            if drift_config:
                try:
                    from trainkeeper.drift import DriftDetector, EmailAlert, WebhookAlert
                    alerts = []
                    for alert_cfg in drift_config.get("alerts", []):
                        if alert_cfg["type"] == "email":
                            alerts.append(EmailAlert(
                                smtp_server=alert_cfg["smtp_server"],
                                smtp_port=alert_cfg.get("smtp_port", 587),
                                from_addr=alert_cfg["from_addr"],
                                to_addrs=alert_cfg["to_addrs"],
                                username=alert_cfg.get("username"),
                                password=alert_cfg.get("password")
                            ))
                        elif alert_cfg["type"] == "webhook":
                            alerts.append(WebhookAlert(url=alert_cfg["url"]))
                    
                    drift_detector = DriftDetector(alerts=alerts)
                except ImportError:
                    print("⚠️  Drift module not found. Drift detection disabled.")
                except Exception as e:
                    print(f"⚠️  Failed to init drift detector: {e}")

            ctx = RunContext(
                run_dir=base, 
                exp_id=exp_id, 
                config=config or {},
                _drift_detector=drift_detector
            )

            # Initialize storage backend if URI provided
            storage_backend = None
            if storage_uri:
                try:
                    from trainkeeper.storage import get_storage_backend
                    storage_backend = get_storage_backend(storage_uri)
                except ImportError:
                    print("⚠️  Storage module not found. Cloud sync disabled.")
                except Exception as e:
                    print(f"⚠️  Failed to initialize storage backend: {e}")

            lock_seeds(seed=seed)

            payload = {
                "schema_version": 1,
                "exp_id": exp_id,
                "seed": seed,
                "entrypoint": {"module": fn.__module__, "name": fn.__name__},
                "argv": sys.argv,
                "entrypoint_file": sys.argv[0] if sys.argv else None,
            }
            if config is not None:
                payload["config"] = config
            if capture_env:
                payload["environment"] = capture_environment(auto_capture_git=auto_capture_git)

            payload_serializable = _to_serializable(payload)
            ctx.write(payload_serializable)
            _atomic_write_text(
                base / "environment.json",
                json.dumps(payload_serializable.get("environment", {}), indent=2),
            )
            if capture_env:
                _write_text(
                    base / "env.txt",
                    "\n".join(
                        f"{k}={v}"
                        for k, v in payload.get("environment", {}).get("env_vars", {}).items()
                    ),
                )
                _write_text(
                    base / "system.json",
                    json.dumps(_to_serializable(payload.get("environment", {})), indent=2),
                )
                _write_text(base / "seeds.json", json.dumps(_seed_state(), indent=2))
            if auto_capture_git:
                git_diff = payload.get("environment", {}).get("git", {}).get("diff")
                if git_diff:
                    _write_text(base / "git.diff", git_diff)
            _write_run_script(base / "run.sh", [sys.executable] + sys.argv)
            if capture_env and write_env_script:
                env_vars = payload.get("environment", {}).get("env_vars", {})
                _write_env_script(base / "env.sh", env_vars)

            if wandb_project:
                _maybe_log_wandb(wandb_project, payload)
            if mlflow_experiment:
                _maybe_log_mlflow(mlflow_experiment, payload, base / "experiment.yaml")

            if "run_ctx" in fn.__code__.co_varnames and "run_ctx" not in kwargs:
                kwargs["run_ctx"] = ctx
            result = fn(*args, **kwargs)
            if isinstance(result, dict):
                _atomic_write_text(
                    base / "metrics.json",
                    json.dumps(_to_serializable(result), indent=2),
                )

            # Sync artifacts to cloud storage if configured
            if storage_backend:
                try:
                    print(f"☁️  Syncing artifacts to {storage_uri}...")
                    # Upload the entire experiment directory
                    # Structure: {storage_uri}/experiments/{exp_id}/
                    remote_path = f"experiments/{ctx.exp_id}"
                    storage_backend.upload(ctx.run_dir, remote_path)
                    print(f"✅  Synced to: {storage_uri}/{remote_path}")
                except Exception as e:
                    print(f"❌  Cloud sync failed: {e}")

            return result

        return wrapper

    return decorator


def _maybe_log_wandb(project, payload):
    try:
        import wandb

        wandb.init(project=project, config=payload, reinit=True)
    except Exception:
        return None


def _maybe_log_mlflow(experiment, payload, artifact_path):
    try:
        import mlflow

        mlflow.set_experiment(experiment)
        with mlflow.start_run():
            mlflow.log_params(
                {
                    "exp_id": payload.get("exp_id"),
                    "seed": payload.get("seed"),
                    "entrypoint": payload.get("entrypoint", {}).get("name"),
                }
            )
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))
    except Exception:
        return None


def load_experiment(exp_dir):
    exp_dir = Path(exp_dir)
    exp_file = exp_dir / "experiment.yaml"
    if not exp_file.exists():
        raise FileNotFoundError(f"experiment.yaml not found in {exp_dir}")
    try:
        import yaml

        return yaml.safe_load(exp_file.read_text(encoding="utf-8"))
    except Exception:
        return json.loads(exp_file.read_text(encoding="utf-8"))


def compare_experiments(exp_a, exp_b):
    a = load_experiment(exp_a)
    b = load_experiment(exp_b)
    diff = {"env": {}, "config": {}, "metrics": {}, "seeds": {}, "resumed": {}}
    for key in ["python_version", "platform", "hostname", "executable"]:
        av = a.get("environment", {}).get(key)
        bv = b.get("environment", {}).get(key)
        if av != bv:
            diff["env"][key] = {"a": av, "b": bv}
    if a.get("config") != b.get("config"):
        diff["config"] = {"a": a.get("config"), "b": b.get("config")}

    def _load_metrics(exp_dir):
        path = Path(exp_dir) / "metrics.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    ma = _load_metrics(exp_a)
    mb = _load_metrics(exp_b)
    if ma != mb:
        diff["metrics"] = {"a": ma, "b": mb}

    sa = a.get("seed")
    sb = b.get("seed")
    if sa != sb:
        diff["seeds"] = {"a": sa, "b": sb}
    ra = ma.get("resumed") if isinstance(ma, dict) else None
    rb = mb.get("resumed") if isinstance(mb, dict) else None
    if ra != rb:
        diff["resumed"] = {"a": ra, "b": rb}
    return diff


@contextlib.contextmanager
def replay_experiment(exp_dir, apply_env=True):
    data = load_experiment(exp_dir)
    if apply_env:
        env_vars = data.get("environment", {}).get("env_vars", {})
        for key, value in env_vars.items():
            if value is not None:
                os.environ[key] = str(value)
    seed = data.get("seed", 1337)
    lock_seeds(seed=seed)
    yield data


def replay_from_id(exp_id, artifacts_dir="artifacts", apply_env=True):
    exp_dir = Path(artifacts_dir) / f"exp-{exp_id}"
    return replay_experiment(exp_dir, apply_env=apply_env)

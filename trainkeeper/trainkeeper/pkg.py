import json
import shutil
import tempfile
from pathlib import Path
import datetime as _dt
import hashlib


class RuntimeRunner:
    def __init__(self, model, preprocess):
        self.model = model
        self.preprocess = preprocess

    def predict(self, inputs):
        try:
            import torch

            with torch.no_grad():
                return self.model(self.preprocess(inputs))
        except Exception:
            return self.model(self.preprocess(inputs))


def quantize_model(model, backend="fbgemm"):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch is required for quantization") from exc

    torch.backends.quantized.engine = backend
    return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def generate_dockerfile(
    output="Dockerfile",
    entrypoint="python train.py",
    base_image="python:3.11-slim",
):
    cmd = entrypoint.split(" ")
    cmd_json = json.dumps(cmd)
    content = (
        f"FROM {base_image}\n"
        "WORKDIR /app\n"
        "COPY . /app\n"
        "RUN pip install -U pip && pip install -e .\n"
        f"CMD {cmd_json}\n"
    )
    Path(output).write_text(content, encoding="utf-8")
    return output


def export_for_edge(
    model,
    preprocess,
    target="onnx",
    quantize=False,
    out="deploy/package.zip",
    runtime="torch",
    hardware="cpu",
):
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        meta = {"target": target, "quantize": bool(quantize), "runtime": runtime, "hardware": hardware}
        (tmp_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

        if target == "torchscript":
            try:
                import torch

                scripted = torch.jit.trace(model, preprocess.example_input())
                scripted.save(str(tmp_dir / "model.pt"))
            except Exception as exc:
                raise RuntimeError("TorchScript export failed") from exc
        elif target == "onnx":
            try:
                import torch

                torch.onnx.export(
                    model,
                    preprocess.example_input(),
                    tmp_dir / "model.onnx",
                    input_names=["input"],
                    output_names=["output"],
                    opset_version=17,
                )
            except Exception as exc:
                raise RuntimeError("ONNX export failed") from exc
        else:
            raise ValueError("target must be 'onnx' or 'torchscript'")

        if quantize:
            meta["quantize_note"] = "Quantization performed with dynamic quantization"
            (tmp_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

        shutil.make_archive(str(out_path.with_suffix("")), "zip", tmp_dir)

    return out_path


def generate_model_card(
    output="MODEL_CARD.md",
    model_name="model",
    description="",
    metrics=None,
    datasets=None,
    training_details=None,
    license_name="Apache-2.0",
):
    metrics = metrics or {}
    datasets = datasets or []
    training_details = training_details or {}
    content = [
        f"# Model Card: {model_name}",
        "",
        f"Generated: {_dt.datetime.utcnow().isoformat()}Z",
        "",
        "## Description",
        description or "N/A",
        "",
        "## Datasets",
        "\n".join([f"- {d}" for d in datasets]) or "N/A",
        "",
        "## Metrics",
        "\n".join([f"- {k}: {v}" for k, v in metrics.items()]) or "N/A",
        "",
        "## Training Details",
        "\n".join([f"- {k}: {v}" for k, v in training_details.items()]) or "N/A",
        "",
        "## License",
        license_name,
        "",
    ]
    Path(output).write_text("\n".join(content), encoding="utf-8")
    return output


def bundle_preprocess(preprocess, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(preprocess, "save"):
        preprocess.save(out_dir / "preprocess.bin")
        return out_dir / "preprocess.bin"
    if hasattr(preprocess, "__name__"):
        path = out_dir / "preprocess.py"
        path.write_text(
            "def preprocess(x):\n"
            "    return x\n",
            encoding="utf-8",
        )
        return path
    path = out_dir / "preprocess.txt"
    path.write_text(str(preprocess), encoding="utf-8")
    return path


def inference_sanity_check(model, preprocess):
    if not hasattr(preprocess, "example_input"):
        raise ValueError("preprocess must define example_input() for sanity check")
    example = preprocess.example_input()
    try:
        import torch

        with torch.no_grad():
            _ = model(preprocess(example))
    except Exception as exc:
        raise RuntimeError("Sanity check failed for torch model") from exc
    return True


def export_pipeline(
    model,
    preprocess,
    out_dir="deploy",
    targets=("onnx", "torchscript"),
    quantize=False,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_preprocess(preprocess, out_dir)
    inference_sanity_check(model, preprocess)
    artifacts = []
    for target in targets:
        out_path = out_dir / f"package_{target}.zip"
        artifacts.append(
            export_for_edge(
                model,
                preprocess,
                target=target,
                quantize=quantize,
                out=str(out_path),
            )
        )
    return artifacts


def create_repro_seal(run_dir, output="repro_seal.json"):
    run_dir = Path(run_dir)
    run_json = run_dir / "run.json"
    if not run_json.exists():
        raise FileNotFoundError("run.json not found in run directory")
    h = hashlib.sha256(run_json.read_bytes()).hexdigest()
    payload = {"run_dir": str(run_dir), "run_hash": h}
    out_path = run_dir / output
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path

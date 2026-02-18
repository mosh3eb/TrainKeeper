import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

logger = logging.getLogger(__name__)

def export_to_onnx(
    model: Any, 
    input_sample: Any, 
    path: Union[str, Path], 
    opset_version: int = 14,
    input_names: List[str] = ["input"],
    output_names: List[str] = ["output"],
    dynamic_axes: Optional[Dict[str, Any]] = None,
    verify: bool = True
) -> str:
    """
    Export a PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model (nn.Module)
        input_sample: Sample input tensor(s) for the model
        path: Output path for the .onnx file
        opset_version: ONNX opset version
        input_names: List of input names
        output_names: List of output names
        dynamic_axes: Dictionary defining dynamic axes
        verify: Whether to verify the exported model with ONNX Runtime
        
    Returns:
        Path to the exported model
    """
    try:
        import torch
    except ImportError:
        logger.error("PyTorch is required for ONNX export.")
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set model to eval mode
    training = model.training
    model.eval()
    
    try:
        logger.info(f"Exporting model to {path}...")
        
        # Workaround for broken transformers/huggingface_hub dependency in torch.onnx
        import sys
        _old_transformers = sys.modules.get("transformers")
        sys.modules["transformers"] = None # Force ModuleNotFoundError for transformers
        
        try:
            torch.onnx.export(
                model,
                input_sample,
                str(path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes
            )
        finally:
            # Restore transformers if it was present
            if _old_transformers is not None:
                sys.modules["transformers"] = _old_transformers
            else:
                sys.modules.pop("transformers", None)
                
        logger.info("ONNX export successful.")
        
        if verify:
            _verify_onnx(str(path), model, input_sample)
            
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        raise e
    finally:
        # Restore training state
        model.train(training)
        
    return str(path)

def _verify_onnx(onnx_path, model, input_sample):
    """Verify ONNX model against PyTorch model"""
    try:
        import onnx
        import onnxruntime as ort
        import numpy as np
        import torch
    except ImportError:
        logger.warning("onnx or onnxruntime not found. Skipping verification.")
        return

    # Check model structure
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    # Check outputs
    ort_session = ort.InferenceSession(onnx_path)
    
    # Prepare inputs
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
    if isinstance(input_sample, torch.Tensor):
        inputs = {ort_session.get_inputs()[0].name: to_numpy(input_sample)}
        input_tensors = (input_sample,)
    elif isinstance(input_sample, tuple):
        inputs = {ort_session.get_inputs()[i].name: to_numpy(inp) for i, inp in enumerate(input_sample)}
        input_tensors = input_sample
    else:
        logger.warning("Complex input types not fully supported for auto verification.")
        return

    # Run ONNX Runtime
    ort_outs = ort_session.run(None, inputs)
    
    # Run PyTorch
    with torch.no_grad():
        torch_out = model(*input_tensors)
    
    if isinstance(torch_out, torch.Tensor):
        torch_outs = [to_numpy(torch_out)]
    else:
        torch_outs = [to_numpy(x) for x in torch_out]
        
    # Compare
    for i, (t_out, o_out) in enumerate(zip(torch_outs, ort_outs)):
        np.testing.assert_allclose(t_out, o_out, rtol=1e-03, atol=1e-05)
        logger.info(f"Output {i} matched within tolerance.")
        
    logger.info("✅ ONNX verification passed.")

def create_model_archive(
    model_name: str,
    version: str,
    serialized_file: str,
    handler: str,
    extra_files: Optional[str] = None,
    export_path: str = "model_store",
    requirements_file: Optional[str] = None
):
    """
    Wrap torch-model-archiver to create a .mar file for TorchServe.
    """
    import subprocess
    import shutil
    
    if not shutil.which("torch-model-archiver"):
         logger.error("torch-model-archiver not found. Install it lightly: pip install torch-model-archiver")
         return None

    cmd = [
        "torch-model-archiver",
        "--model-name", model_name,
        "--version", version,
        "--serialized-file", serialized_file,
        "--handler", handler,
        "--export-path", export_path
    ]
    
    if extra_files:
        cmd.extend(["--extra-files", extra_files])
    
    if requirements_file:
         cmd.extend(["--requirements-file", requirements_file])
         
    path = Path(export_path)
    path.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"Creating .mar archive for {model_name}...")
        subprocess.check_call(cmd)
        logger.info(f"Successfully created {model_name}.mar in {export_path}")
        return str(path / f"{model_name}.mar")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create model archive: {e}")
        return None

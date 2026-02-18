import pytest
import torch
import torch.nn as nn
from pathlib import Path
from trainkeeper.serving import export_to_onnx

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

def test_export_to_onnx(tmp_path):
    model = SimpleModel()
    input_sample = torch.randn(1, 10)
    output_path = tmp_path / "model.onnx"
    
    # Export without verification (simpler test env)
    exported_path = export_to_onnx(
        model, 
        input_sample, 
        output_path, 
        verify=False
    )
    
    assert Path(exported_path).exists()
    assert str(exported_path) == str(output_path)

def test_export_to_onnx_with_dynamic_axes(tmp_path):
    model = SimpleModel()
    input_sample = torch.randn(1, 10)
    output_path = tmp_path / "dynamic_model.onnx"
    
    exported_path = export_to_onnx(
        model,
        input_sample,
        output_path,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        verify=False
    )
    
    assert Path(exported_path).exists()

import pytest
import json
from trainkeeper.reports import ReportGenerator
from trainkeeper.validation import ValidationReport
from trainkeeper.drift import DriftReport

def test_report_generation(tmp_path):
    generator = ReportGenerator(run_dir=tmp_path)
    
    config = {"batch_size": 32, "lr": 0.001}
    metrics = {"accuracy": 0.95, "loss": 0.1}
    
    val_report = ValidationReport(
        passed=True,
        checks={"smoke": True},
        metrics={"latency": 10.0}
    )
    
    drift_report = DriftReport(
        is_drifted=False,
        drift_score=0.01,
        metric="PSI",
        feature_name="feature_x",
        threshold=0.1,
        details={}
    )
    
    report_path = generator.generate(
        experiment_id="exp_001",
        config=config,
        metrics=metrics,
        validation_report=val_report,
        drift_report=drift_report,
        artifacts=["model.pt"]
    )
    
    assert report_path.exists()
    assert (tmp_path / "report.json").exists()
    
    # Check content
    with open(report_path, "r") as f:
        content = f.read()
        assert "# Experiment Report: exp_001" in content
        assert "accuracy" in content
        assert "0.95" in content
        assert "Passed" in content
        assert "No Drift" in content
        assert "model.pt" in content

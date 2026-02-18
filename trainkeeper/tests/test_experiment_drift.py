import json
import pytest
import numpy as np
from pathlib import Path
from trainkeeper.experiment import run_reproducible

def test_experiment_drift_detection(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    
    # Drift configuration
    drift_config = {
        "alerts": [
            {
                "type": "email",
                "smtp_server": "localhost",
                "smtp_port": 25,
                "from_addr": "test@example.com",
                "to_addrs": ["admin@example.com"]
            }
        ]
    }

    @run_reproducible(artifacts_dir=str(artifacts_dir), drift_config=drift_config)
    def train_with_drift(run_ctx=None):
        # Simulate data
        ref_data = np.random.normal(0, 1, 1000)
        current_data = np.random.normal(2, 1, 1000) # Drifted data
        
        # Check drift
        report = run_ctx.check_drift(current_data, "feature_1", reference_data=ref_data)
        return {"drift_detected": report.is_drifted}

    result = train_with_drift()
    
    # Assert return value
    assert result["drift_detected"] is True
    
    # Assert report file created
    exp_dir = list(artifacts_dir.glob("exp-*"))[0]
    report_file = exp_dir / "drift_report_feature_1.json"
    assert report_file.exists()
    
    report_data = json.loads(report_file.read_text())
    assert report_data["drifted"] is True
    assert report_data["feature"] == "feature_1"
    assert report_data["score"] > 0.1

def test_experiment_no_drift(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    
    @run_reproducible(artifacts_dir=str(artifacts_dir))
    def train_no_drift(run_ctx=None):
        # Simulate data
        ref_data = np.random.normal(0, 1, 1000)
        current_data = np.random.normal(0, 1, 1000) # Similar data
        
        # Check drift
        report = run_ctx.check_drift(current_data, "feature_2", reference_data=ref_data)
        return {"drift_detected": report.is_drifted}

    result = train_no_drift()
    
    assert result["drift_detected"] is False
    
    exp_dir = list(artifacts_dir.glob("exp-*"))[0]
    report_file = exp_dir / "drift_report_feature_2.json"
    assert report_file.exists()
    
    report_data = json.loads(report_file.read_text())
    assert report_data["drifted"] is False

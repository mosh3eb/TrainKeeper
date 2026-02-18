import pytest
import numpy as np
from trainkeeper.drift import DriftDetector, DriftReport

def test_psi_calculation():
    detector = DriftDetector()
    
    # Identical distributions should have 0 PSI
    data = np.random.normal(0, 1, 1000)
    psi = detector.calculate_psi(data, data)
    assert psi < 0.01

    # Different distributions
    data1 = np.random.normal(0, 1, 1000)
    data2 = np.random.normal(2, 1, 1000)
    psi = detector.calculate_psi(data1, data2)
    assert psi > 0.1

def test_ks_calculation():
    detector = DriftDetector()
    
    # Identical distributions
    data = np.random.normal(0, 1, 1000)
    ks = detector.calculate_ks(data, data)
    assert ks < 0.05

    # Different distributions
    data1 = np.random.normal(0, 1, 1000)
    data2 = np.random.normal(0.5, 1, 1000)
    ks = detector.calculate_ks(data1, data2)
    assert ks > 0.1

def test_check_drift():
    ref_data = np.random.normal(0, 1, 1000)
    drift_data = np.random.normal(2, 1, 1000)
    
    detector = DriftDetector(reference_data=ref_data)
    
    # Check no drift with similar data
    similar_data = np.random.normal(0, 1, 1000)
    report = detector.check_drift(similar_data, "feature_x", method="psi", threshold=0.1)
    assert not report.is_drifted
    
    # Check drift with different data
    report = detector.check_drift(drift_data, "feature_x", method="psi", threshold=0.1)
    assert report.is_drifted
    assert report.drift_score > 0.1

class MockAlert:
    def __init__(self):
        self.sent_reports = []
    
    def send(self, report):
        self.sent_reports.append(report)

def test_alerting():
    ref_data = np.random.normal(0, 1, 1000)
    drift_data = np.random.normal(2, 1, 1000)
    
    alert = MockAlert()
    detector = DriftDetector(reference_data=ref_data, alerts=[alert])
    
    # Should trigger alert
    detector.check_drift(drift_data, "feature_x", threshold=0.1)
    assert len(alert.sent_reports) == 1
    assert alert.sent_reports[0].is_drifted

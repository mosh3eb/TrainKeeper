import pytest
import time
from trainkeeper.validation import ModelValidator

class SimpleModel:
    def __call__(self, x):
        return x

class SlowModel:
    def __call__(self, x):
        time.sleep(0.01) # 10ms delay
        return x

def test_validation_smoke_test():
    model = SimpleModel()
    validator = ModelValidator(model)
    
    # Pass input sample
    report = validator.validate(smoke_test=True, input_sample=1.0)
    assert report.passed
    assert report.checks["smoke_test"]
    assert not report.failures

def test_validation_latency_check(tmp_path):
    model = SlowModel()
    validator = ModelValidator(model)
    
    # 10ms model, check limit 1ms (should fail)
    report = validator.validate(
        max_latency_ms=1.0, 
        smoke_test=False, 
        input_sample=1.0
    )
    assert not report.passed
    assert not report.checks["latency"]
    assert "Latency" in report.failures[0]
    
    # 10ms model, check limit 100ms (should pass)
    report_pass = validator.validate(
        max_latency_ms=100.0, 
        smoke_test=False, 
        input_sample=1.0
    )
    assert report_pass.passed
    assert report_pass.checks["latency"]

def test_validation_data_loader():
    model = SimpleModel()
    data = [1.0, 2.0, 3.0]
    validator = ModelValidator(model, data_loader=iter([data])) # Batch is list
    
    report = validator.validate(smoke_test=True)
    assert report.passed
    assert report.checks["smoke_test"]

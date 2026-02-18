import pytest
from trainkeeper.ab_testing import ExperimentManager, ExperimentVariant
from trainkeeper.monitoring import ModelMonitor

class SimpleModel:
    def __init__(self, val):
        self.val = val
        
    def __call__(self, x):
        return x + self.val

def test_ab_routing():
    model_a = SimpleModel(1)
    model_b = SimpleModel(2)
    
    variant_a = ExperimentVariant(name="A", model=model_a, traffic_percentage=0.5)
    variant_b = ExperimentVariant(name="B", model=model_b, traffic_percentage=0.5)
    
    manager = ExperimentManager(variants=[variant_a, variant_b])
    
    # Test deterministic routing
    user1 = "user123"
    variant1 = manager.route(user1)
    
    variant1_again = manager.route(user1)
    assert variant1.name == variant1_again.name
    
    user2 = "user456"
    variant2 = manager.route(user2)
    
    # Not guaranteed to be different, but statistically likely to have some distribution
    # Let's check distribution over many users
    counts = {"A": 0, "B": 0}
    for i in range(1000):
        v = manager.route(f"user_{i}")
        counts[v.name] += 1
        
    assert 450 < counts["A"] < 550
    assert 450 < counts["B"] < 550

def test_ab_prediction_logging():
    model_a = SimpleModel(1)
    variant_a = ExperimentVariant(name="A", model=model_a, traffic_percentage=1.0)
    
    manager = ExperimentManager(variants=[variant_a])
    
    # Predict
    result = manager.predict("user1", 10)
    assert result == 11
    
    # Check monitor stats
    health = variant_a.monitor.get_health()
    assert health.request_count == 1  # Actually ModelMonitor logs errors/latency via log_request separately from preiction
    # My simple predict wrapper logged both?
    # In predict implementation:
    # variant.monitor.log_prediction(prediction)
    # return prediction
    # Ah, I didn't verify log_request was called in predict implementation yet.
    # Let's check implementation of predict in ab_testing.py

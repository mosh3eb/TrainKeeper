import pytest
import numpy as np
from trainkeeper.preprocessing import PreprocessingPipeline

class MockTransformer:
    def __init__(self, add_val=1):
        self.add_val = add_val
        self.fitted = False

    def fit(self, X, y=None):
        self.fitted = True
        return self

    def transform(self, X):
        return X + self.add_val
        
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def test_pipeline_basic():
    pipeline = PreprocessingPipeline()
    t1 = MockTransformer(add_val=2)
    t2 = MockTransformer(add_val=3)
    
    pipeline.add_step(t1)
    pipeline.add_step(t2)
    
    data = np.array([1, 2, 3])
    
    # Test fit_transform
    transformed = pipeline.fit_transform(data)
    # 1+2+3 = 6
    assert np.all(transformed == data + 5)
    assert t1.fitted
    assert t2.fitted
    
    # Test transform
    data2 = np.array([10, 20])
    transformed2 = pipeline.transform(data2)
    assert np.all(transformed2 == data2 + 5)

def test_pipeline_function_step():
    pipeline = PreprocessingPipeline()
    pipeline.add_step(lambda x: x * 2)
    
    data = np.array([1, 2, 3])
    transformed = pipeline.fit_transform(data)
    assert np.all(transformed == data * 2)

def test_save_load(tmp_path):
    pipeline = PreprocessingPipeline()
    pipeline.add_step(MockTransformer(add_val=10))
    
    path = tmp_path / "pipeline.pkl"
    pipeline.save(path)
    
    loaded_pipeline = PreprocessingPipeline.load(path)
    
    data = np.array([1, 2, 3])
    transformed = loaded_pipeline.transform(data)
    assert np.all(transformed == data + 10)

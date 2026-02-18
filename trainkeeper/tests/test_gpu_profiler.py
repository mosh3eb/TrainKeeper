"""
Tests for GPU profiler functionality
"""

import pytest


def test_gpu_profiler_disabled_without_cuda():
    """Test that profiler gracefully handles missing CUDA"""
    from trainkeeper.gpu_profiler import GPUProfiler
    
    profiler = GPUProfiler(device=0)
    # Should not crash even if CUDA is unavailable
    profiler.start()
    profiler.step("test")
    report = profiler.stop()
    
    assert isinstance(report.snapshots, list)


def test_memory_snapshot_creation():
    """Test MemorySnapshot dataclass"""
    from trainkeeper.gpu_profiler import MemorySnapshot
    
    snapshot = MemorySnapshot(
        timestamp=1234567890.0,
        allocated_mb=1024.5,
        reserved_mb=2048.0,
        max_allocated_mb=1500.0,
        device_id=0,
        tag="forward"
    )
    
    assert snapshot.timestamp == 1234567890.0
    assert snapshot.allocated_mb == 1024.5
    assert snapshot.tag == "forward"


def test_memory_report_summary():
    """Test MemoryReport summary generation"""
    from trainkeeper.gpu_profiler import MemoryReport
    
    report = MemoryReport(
        peak_allocated_mb=1024.0,
        peak_reserved_mb=2048.0,
        avg_allocated_mb=512.0,
        recommendations=["Test recommendation 1", "Test recommendation 2"]
    )
    
    summary = report.summary()
    
    assert "Peak Allocated: 1024.00 MB" in summary
    assert "Peak Reserved:  2048.00 MB" in summary
    assert "Avg Allocated:  512.00 MB" in summary
    assert "Test recommendation 1" in summary
    assert "Test recommendation 2" in summary


def test_memory_report_no_recommendations():
    """Test summary with no recommendations"""
    from trainkeeper.gpu_profiler import MemoryReport
    
    report = MemoryReport(
        peak_allocated_mb=100.0,
        peak_reserved_mb=200.0,
        avg_allocated_mb=50.0,
        recommendations=[]
    )
    
    summary = report.summary()
    assert "No optimization recommendations" in summary


def test_profiler_step_counter():
    """Test that profiler step counter works"""
    from trainkeeper.gpu_profiler import GPUProfiler
    import unittest.mock as mock
    import sys
    
    # Mock torch.cuda.is_available to return True
    with mock.patch('torch.cuda.is_available', return_value=True):
        with mock.patch('torch.cuda.set_device'):
            with mock.patch('torch.cuda.reset_peak_memory_stats'):
                with mock.patch('torch.cuda.memory_allocated', return_value=0):
                    with mock.patch('torch.cuda.memory_reserved', return_value=0):
                        with mock.patch('torch.cuda.max_memory_allocated', return_value=0):
                            profiler = GPUProfiler(device=0, check_interval=2)
                            profiler.start()
                            
                            assert profiler.step_count == 0
                            
                            profiler.step("step1")
                            assert profiler.step_count == 1
                            
                            profiler.step("step2")
                            assert profiler.step_count == 2
                            
                            profiler.step("step3")
                            assert profiler.step_count == 3


@pytest.mark.skipif(not pytest.importorskip("torch", None), reason="PyTorch not available")
def test_profiler_with_torch():
    """Test profiler with actual PyTorch"""
    import torch
    from trainkeeper.gpu_profiler import GPUProfiler
    
    profiler = GPUProfiler(device=0)
    
    # Even without CUDA, should handle gracefully
    profiler.start()
    profiler.step("test")
    report = profiler.stop()
    
    assert isinstance(report, object)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

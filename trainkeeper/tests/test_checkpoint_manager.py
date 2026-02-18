"""
Tests for checkpoint manager functionality
"""

import pytest
from pathlib import Path
import tempfile
import shutil


def test_checkpoint_metadata_creation():
    """Test CheckpointMetadata dataclass"""
    from trainkeeper.checkpoint_manager import CheckpointMetadata
    
    metadata = CheckpointMetadata(
        epoch=5,
        step=1000,
        timestamp=1234567890.0,
        metrics={"loss": 0.5, "acc": 0.95},
        size_mb=100.5,
        compressed=False,
        hash="abc123"
    )
    
    assert metadata.epoch == 5
    assert metadata.step == 1000
    assert metadata.metrics["acc"] == 0.95
    
    # Test to_dict
    data = metadata.to_dict()
    assert data["epoch"] == 5
    assert data["size_mb"] == 100.5


def test_checkpoint_manager_initialization():
    """Test CheckpointManager initialization"""
    from trainkeeper.checkpoint_manager import CheckpointManager
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(
            save_dir=tmpdir,
            keep_best=3,
            keep_last=2,
            metric="val_acc",
            mode="max"
        )
        
        assert manager.keep_best == 3
        assert manager.keep_last == 2
        assert manager.metric == "val_acc"
        assert manager.mode == "max"
        assert manager.save_dir.exists()


@pytest.mark.skipif(not pytest.importorskip("torch", None), reason="PyTorch not available")
def test_checkpoint_save():
    """Test basic checkpoint saving"""
    # Use mocks to avoid complex torch->transformers dependencies causing issues in test env
    class MockModel:
        def state_dict(self): return {"w": 1}
    
    class MockOptimizer:
        def state_dict(self): return {"lr": 0.1}
        
    from trainkeeper.checkpoint_manager import CheckpointManager
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(
            save_dir=tmpdir,
            keep_best=2,
            keep_last=1,
            metric="loss",
            mode="min"
        )
        
        model = MockModel()
        optimizer = MockOptimizer()
        
        path = manager.save(
            model=model,
            optimizer=optimizer,
            epoch=0,
            step=100,
            metrics={"loss": 0.5, "acc": 0.9}
        )
        
        assert path.exists()
        assert len(manager.checkpoints) == 1
        ckpt = manager.checkpoints[0]
        assert ckpt.epoch == 0
        assert ckpt.metrics["loss"] == 0.5


@pytest.mark.skipif(not pytest.importorskip("torch", None), reason="PyTorch not available")
def test_checkpoint_cleanup():
    """Test automatic checkpoint cleanup"""
    class MockModel:
        def state_dict(self): return {"w": 1}
    
    class MockOptimizer:
        def state_dict(self): return {"lr": 0.1}

    from trainkeeper.checkpoint_manager import CheckpointManager
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(
            save_dir=tmpdir,
            keep_best=2,
            keep_last=1,
            metric="acc",
            mode="max"
        )
        
        model = MockModel()
        optimizer = MockOptimizer()
        
        # Save multiple checkpoints
        for i in range(5):
            manager.save(
                model=model,
                optimizer=optimizer,
                epoch=i,
                step=i*100,
                metrics={"acc": 0.5 + i*0.1}
            )
        
        assert len(manager.checkpoints) <= 3


@pytest.mark.skipif(not pytest.importorskip("torch", None), reason="PyTorch not available")
def test_checkpoint_load_best():
    """Test loading best checkpoint"""
    class MockModel:
        def state_dict(self): return {"w": 1}
        def load_state_dict(self, state_dict): pass
    
    class MockOptimizer:
        def state_dict(self): return {"lr": 0.1}
        def load_state_dict(self, state_dict): pass

    from trainkeeper.checkpoint_manager import CheckpointManager
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(
            save_dir=tmpdir,
            keep_best=3,
            metric="acc",
            mode="max"
        )
        
        model = MockModel()
        optimizer = MockOptimizer()
        
        for i in range(3):
            manager.save(
                model=model,
                optimizer=optimizer,
                epoch=i,
                step=i*100,
                metrics={"acc": 0.7 + i*0.1}
            )
        
        # Checking logic without actual loading since torch.load might trigger the issue if file content is weird
        # But we need to call load_best to verify logic.
        # CheckpointManager.load_best calls torch.load.
        # We can mock torch.load too?
        # Or just trust that torch.load of a simple dict doesn't trigger transformers.
        # The crash happened at torch.optim.SGD.__init__.
        # So mocks should fix it.
        
        checkpoint = manager.load_best()
        
        assert checkpoint["epoch"] == 2
        assert checkpoint["metrics"]["acc"] == pytest.approx(0.9, 0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

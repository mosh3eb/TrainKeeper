"""
Tests for distributed training functionality
"""

import pytest
import os


def test_distributed_config_creation():
    """Test DistributedConfig dataclass"""
    from trainkeeper.distributed import DistributedConfig
    
    config = DistributedConfig(
        backend="nccl",
        world_size=4,
        rank=0
    )
    
    assert config.is_distributed == True
    assert config.is_main_process == True
    
    config2 = DistributedConfig(rank=1, world_size=4)
    assert config2.is_main_process == False


def test_distributed_config_single_process():
    """Test single process configuration"""
    from trainkeeper.distributed import DistributedConfig
    
    config = DistributedConfig()
    assert config.is_distributed == False
    assert config.is_main_process == True


def test_auto_detect_no_distributed():
    """Test auto-detection when not in distributed environment"""
    from trainkeeper.distributed import auto_detect_distributed
    
    # Clear any distributed env vars
    for key in ["RANK", "WORLD_SIZE", "LOCAL_RANK", "SLURM_PROCID"]:
        os.environ.pop(key, None)
    
    config = auto_detect_distributed()
    assert config.world_size == 1
    assert config.rank == 0


def test_auto_detect_torchrun():
    """Test auto-detection with torchrun environment"""
    from trainkeeper.distributed import auto_detect_distributed
    
    os.environ["RANK"] = "2"
    os.environ["WORLD_SIZE"] = "4"
    os.environ["LOCAL_RANK"] = "2"
    os.environ["MASTER_ADDR"] = "192.168.1.1"
    os.environ["MASTER_PORT"] = "29501"
    
    config = auto_detect_distributed()
    
    assert config.rank == 2
    assert config.world_size == 4
    assert config.local_rank == 2
    assert config.master_addr == "192.168.1.1"
    assert config.master_port == "29501"
    
    # Cleanup
    for key in ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]:
        os.environ.pop(key, None)


def test_print_dist():
    """Test distributed printing"""
    from trainkeeper.distributed import DistributedConfig, print_dist
    from io import StringIO
    import sys
    
    config = DistributedConfig(rank=0, world_size=2)
    
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    print_dist("Test message", config)
    
    output = sys.stdout.getvalue()
    sys.stdout = old_stdout
    
    assert "Test message" in output


def test_print_dist_non_main():
    """Test that non-main processes don't print"""
    from trainkeeper.distributed import DistributedConfig, print_dist
    from io import StringIO
    import sys
    
    config = DistributedConfig(rank=1, world_size=2)
    
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    print_dist("Test message", config)
    
    output = sys.stdout.getvalue()
    sys.stdout = old_stdout
    
    assert output == ""


def test_print_dist_force():
    """Test forced printing from non-main process"""
    from trainkeeper.distributed import DistributedConfig, print_dist
    from io import StringIO
    import sys
    
    config = DistributedConfig(rank=1, world_size=2)
    
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    print_dist("Test message", config, force=True)
    
    output = sys.stdout.getvalue()
    sys.stdout = old_stdout
    
    assert "Test message" in output
    assert "[Rank 1]" in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Distributed Training Support for TrainKeeper

Provides seamless integration with PyTorch DDP, FSDP, and DeepSpeed.
Makes distributed training reproducible and debuggable.
"""

import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import warnings
import functools


@dataclass
class DistributedConfig:
    """Configuration for distributed training"""
    backend: str = "nccl"  # nccl, gloo, mpi
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "29500"
    
    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1
    
    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def auto_detect_distributed() -> DistributedConfig:
    """
    Auto-detect distributed training configuration from environment.
    
    Supports:
    - torchrun / torch.distributed.launch
    - SLURM
    - Manual environment variables
    """
    config = DistributedConfig()
    
    # Try torchrun environment variables (recommended)
    if "RANK" in os.environ:
        config.rank = int(os.environ["RANK"])
        config.world_size = int(os.environ["WORLD_SIZE"])
        config.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        config.master_addr = os.environ.get("MASTER_ADDR", "localhost")
        config.master_port = os.environ.get("MASTER_PORT", "29500")
    
    # Try SLURM environment
    elif "SLURM_PROCID" in os.environ:
        config.rank = int(os.environ["SLURM_PROCID"])
        config.world_size = int(os.environ["SLURM_NTASKS"])
        config.local_rank = int(os.environ.get("SLURM_LOCALID", 0))
        
        # Get master node
        hostnames = os.environ["SLURM_JOB_NODELIST"]
        config.master_addr = hostnames.split(",")[0]
        config.master_port = os.environ.get("MASTER_PORT", "29500")
    
    # Legacy torch.distributed.launch
    elif "LOCAL_RANK" in os.environ:
        config.local_rank = int(os.environ["LOCAL_RANK"])
        config.rank = int(os.environ.get("RANK", config.local_rank))
        config.world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    return config


def setup_distributed(backend: str = "auto", timeout_minutes: int = 30) -> DistributedConfig:
    """
    Initialize distributed training with automatic configuration.
    
    Args:
        backend: Backend to use ('auto', 'nccl', 'gloo', 'mpi')
        timeout_minutes: Timeout for initialization
        
    Returns:
        DistributedConfig with current process information
        
    Example:
        >>> dist_config = setup_distributed()
        >>> if dist_config.is_main_process:
        >>>     print("I'm the main process!")
    """
    try:
        import torch
        import torch.distributed as dist
    except ImportError:
        warnings.warn("PyTorch not found. Distributed training requires PyTorch.")
        return DistributedConfig()
    
    config = auto_detect_distributed()
    
    if not config.is_distributed:
        return config
    
    # Auto-select backend
    if backend == "auto":
        if torch.cuda.is_available():
            backend = "nccl"
        else:
            backend = "gloo"
    
    config.backend = backend
    
    # Initialize process group if not already initialized
    if not dist.is_initialized():
        # Set environment variables
        os.environ["MASTER_ADDR"] = config.master_addr
        os.environ["MASTER_PORT"] = config.master_port
        
        dist.init_process_group(
            backend=backend,
            rank=config.rank,
            world_size=config.world_size,
            timeout=__import__("datetime").timedelta(minutes=timeout_minutes)
        )
        
        # Set device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(config.local_rank)
    
    return config


def cleanup_distributed():
    """Clean up distributed training"""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except ImportError:
        pass


def sync_across_processes(tensor, world_size: int):
    """Synchronize tensor across all processes"""
    try:
        import torch
        import torch.distributed as dist
        
        if not dist.is_initialized():
            return tensor
        
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size
        return tensor
    except ImportError:
        return tensor


def save_distributed_checkpoint(
    model,
    optimizer,
    epoch: int,
    save_dir: Path,
    dist_config: DistributedConfig,
    keep_last_n: int = 3
):
    """
    Save checkpoint in distributed training.
    Only main process saves to avoid race conditions.
    
    Args:
        model: PyTorch model (DDP/FSDP wrapped or unwrapped)
        optimizer: Optimizer
        epoch: Current epoch
        save_dir: Directory to save checkpoints
        dist_config: Distributed configuration
        keep_last_n: Number of recent checkpoints to keep
    """
    if not dist_config.is_main_process:
        return
    
    try:
        import torch
    except ImportError:
        return
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if FSDP
    is_fsdp = False
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import StateDictType, FullStateDictConfig
        is_fsdp = isinstance(model, FSDP)
    except ImportError:
        pass

    if is_fsdp:
        # FSDP saving logic
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            model_state = model.state_dict()
            # Optimizer saving for FSDP is complex (requires gathering). 
            # For simplicity in this version, we skip FSDP optimizer save or warn.
            # Real production usage needs FSDP.optim_state_dict(model, optimizer)
            # but that's quite heavy to implement fully generically here.
            # We'll save what we can.
            optimizer_state = FSDP.optim_state_dict(model, optimizer) if hasattr(FSDP, "optim_state_dict") else {}
    else:
        # DDP / Standard saving logic
        # Get underlying model (unwrap DDP)
        model_to_save = model
        if hasattr(model, "module"):
            model_to_save = model.module
        model_state = model_to_save.state_dict()
        optimizer_state = optimizer.state_dict()
    
    if not dist_config.is_main_process:
        return

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "dist_config": {
            "world_size": dist_config.world_size,
            "backend": dist_config.backend,
            "fsdp": is_fsdp
        }
    }
    
    checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Cleanup old checkpoints
    all_checkpoints = sorted(save_dir.glob("checkpoint_epoch_*.pt"))
    if len(all_checkpoints) > keep_last_n:
        for old_ckpt in all_checkpoints[:-keep_last_n]:
            old_ckpt.unlink()
    
    # Save latest symlink
    latest_path = save_dir / "checkpoint_latest.pt"
    if latest_path.exists():
        latest_path.unlink()
    latest_path.symlink_to(checkpoint_path.name)
    
    return checkpoint_path


def load_distributed_checkpoint(
    checkpoint_path: Path,
    model,
    optimizer=None,
    strict: bool = True
):
    """
    Load checkpoint in distributed setting.
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        strict: Whether to strictly enforce state dict keys match
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch required for checkpoint loading")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Get underlying model
    model_to_load = model
    if hasattr(model, "module"):
        model_to_load = model.module
    
    model_to_load.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint.get("epoch", 0)


def wrap_model_ddp(model, dist_config: DistributedConfig, **ddp_kwargs):
    """
    Wrap model with DistributedDataParallel.
    
    Args:
        model: PyTorch model
        dist_config: Distributed configuration
        **ddp_kwargs: Additional arguments for DDP
        
    Returns:
        DDP-wrapped model
    """
    try:
        import torch
        from torch.nn.parallel import DistributedDataParallel as DDP
    except ImportError:
        raise ImportError("PyTorch required for DDP")
    
    if not dist_config.is_distributed:
        return model
    
    # Move model to correct device
    if torch.cuda.is_available():
        model = model.to(f"cuda:{dist_config.local_rank}")
    
    # Default DDP kwargs
    default_kwargs = {
        "device_ids": [dist_config.local_rank] if torch.cuda.is_available() else None,
        "output_device": dist_config.local_rank if torch.cuda.is_available() else None,
        "find_unused_parameters": False,
    }
    default_kwargs.update(ddp_kwargs)
    
    return DDP(model, **default_kwargs)


def get_sampler_for_distributed(dataset, dist_config: DistributedConfig, shuffle: bool = True):
    """
    Get appropriate sampler for distributed training.
    
    Args:
        dataset: PyTorch dataset
        dist_config: Distributed configuration
        shuffle: Whether to shuffle data
        
    Returns:
        DistributedSampler if distributed, None otherwise
    """
    if not dist_config.is_distributed:
        return None
    
    try:
        from torch.utils.data.distributed import DistributedSampler
    except ImportError:
        raise ImportError("PyTorch required for DistributedSampler")
    
    return DistributedSampler(
        dataset,
        num_replicas=dist_config.world_size,
        rank=dist_config.rank,
        shuffle=shuffle
    )


# Context manager for easier distributed training
class distributed_training:
    """
    Context manager for distributed training setup and cleanup.
    
    Example:
        >>> with distributed_training() as dist_config:
        >>>     model = MyModel()
        >>>     model = wrap_model_ddp(model, dist_config)
        >>>     # Training code...
    """
    
    def __init__(self, backend: str = "auto", timeout_minutes: int = 30):
        self.backend = backend
        self.timeout_minutes = timeout_minutes
        self.config = None
    
    def __enter__(self) -> DistributedConfig:
        self.config = setup_distributed(self.backend, self.timeout_minutes)
        return self.config
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        cleanup_distributed()
        return False


def print_dist(message: str, dist_config: DistributedConfig, force: bool = False):
    """
    Print only from main process unless force=True.
    
    Args:
        message: Message to print
        dist_config: Distributed configuration
        force: If True, print from all processes with rank prefix
    """
    if force:
        print(f"[Rank {dist_config.rank}] {message}")
    elif dist_config.is_main_process:
        print(message)


def wrap_model_fsdp(model, dist_config: DistributedConfig, **fsdp_kwargs):
    """
    Wrap model with FullyShardedDataParallel (FSDP).
    
    Args:
        model: PyTorch model
        dist_config: Distributed configuration
        **fsdp_kwargs: Additional arguments for FSDP
        
    Returns:
        FSDP-wrapped model
    """
    try:
        import torch
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            BackwardPrefetch,
            ShardingStrategy,
            CPUOffload,
        )
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    except ImportError:
        warnings.warn("PyTorch FSDP not available. Returning unwrapped model.")
        return model
    
    if not dist_config.is_distributed:
        return model
    
    # Set default device_id
    torch.cuda.set_device(dist_config.local_rank)
    
    # Default FSDP kwargs
    # We use some sensible defaults for large model training
    default_kwargs = {
        "device_id": torch.cuda.current_device(),
        "cpu_offload": CPUOffload(offload_params=False),
        "mixed_precision": MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        ),
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
    }
    
    # Auto-wrap policy if not provided
    if "auto_wrap_policy" not in fsdp_kwargs and "auto_wrap_policy" not in default_kwargs:
        # Default to wrapping layers > 10M params
        default_kwargs["auto_wrap_policy"] = functools.partial(
            size_based_auto_wrap_policy, min_num_params=10_000_000
        )

    # Update with user provided kwargs, overriding defaults
    default_kwargs.update(fsdp_kwargs)
    
    return FSDP(model, **default_kwargs)


def get_model_state_dict(model):
    """
    Get state dict from model, handling DDP/FSDP unwrapping.
    For FSDP, this must be called under `FSDP.state_dict_type` context 
    config if full state dict is desired, but this helper just unwraps DDP.
    FSDP handling is done in save_distributed_checkpoint.
    """
    if hasattr(model, "module"):
        return model.module.state_dict()
    return model.state_dict()


import logging
import torch
import math
import itertools
from typing import Iterator, List, Optional, Any, Callable

logger = logging.getLogger(__name__)

class StreamingDataset(torch.utils.data.IterableDataset):
    """
    An iterable dataset that streams data efficiently from sharded sources.
    Optimized for multi-process loading and distributed training.
    """
    def __init__(
        self, 
        shards: List[Any], 
        loader_fn: Callable[[Any], Iterator[Any]],
        transform: Optional[Callable] = None,
        shuffle_shards: bool = True
    ):
        """
        Args:
            shards: List of shard identifiers (e.g., file paths).
            loader_fn: Function that takes a shard ID and returns an iterator of items.
            transform: Optional transformation function to apply to each item.
            shuffle_shards: Whether to shuffle the order of shards each epoch.
        """
        self.shards = shards
        self.loader_fn = loader_fn
        self.transform = transform
        self.shuffle_shards = shuffle_shards

    def _get_worker_shards(self) -> List[Any]:
        """Determine which shards this worker should process."""
        # 1. Start with all shards
        shards = list(self.shards)
        
        if self.shuffle_shards:
            # Deterministic shuffle based on epoch ideally, but here just random or fixed seed if needed
            # For true reproducibility, we'd need an epoch counter. 
            # PyTorch IterableDataset doesn't accept set_epoch easily.
            import random
            random.seed(42) # Fixed seed for now to ensure consistency across ranks if needed
            random.shuffle(shards)
            
        # 2. Shard for Distributed Training (DDP)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
             rank = torch.distributed.get_rank()
             world_size = torch.distributed.get_world_size()
             # Round robin assignment
             shards = shards[rank::world_size]

        # 3. Shard for DataLoader Workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            shards = shards[worker_id::num_workers]
            
        return shards

    def __iter__(self) -> Iterator[Any]:
        my_shards = self._get_worker_shards()
        
        for shard in my_shards:
            try:
                # Load data from shard
                for item in self.loader_fn(shard):
                    if self.transform:
                        yield self.transform(item)
                    else:
                        yield item
            except Exception as e:
                logger.error(f"Error reading shard {shard}: {e}")
                continue

def stream_from_files(file_paths: List[str], loader_fn: Callable) -> Iterator[Any]:
    """
    Helper to stream data from a list of files.
    """
    for file_path in file_paths:
        try:
            yield from loader_fn(file_path)
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            continue

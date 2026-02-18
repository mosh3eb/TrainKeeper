import pytest
import torch
from trainkeeper.streaming import StreamingDataset, stream_from_files

def mock_loader(file_path):
    # Simulate loading data from a file
    # file_path is like "shard_0", "shard_1"
    shard_id = int(file_path.split("_")[1])
    for i in range(3):
        yield shard_id * 10 + i

def test_streaming_dataset_basic():
    shards = ["shard_0", "shard_1"]
    dataset = StreamingDataset(
        shards=shards,
        loader_fn=mock_loader,
        shuffle_shards=False
    )
    
    data = list(dataset)
    # expect [0, 1, 2, 10, 11, 12] (order depends on default worker sharding, which is none here)
    expected = [0, 1, 2, 10, 11, 12]
    assert sorted(data) == sorted(expected)

def test_streaming_dataset_sharding():
    # Simulate multi-worker by manually checking _get_worker_shards logic
    # or just trust logic and test end-to-end if we could mock worker_info.
    # PyTorch allows setting worker_info in DataLoader.
    
    shards = ["shard_0", "shard_1", "shard_2", "shard_3"]
    dataset = StreamingDataset(shards=shards, loader_fn=mock_loader, shuffle_shards=False)
    
    # Create DataLoader with 2 workers
    loader = torch.utils.data.DataLoader(dataset, num_workers=2, batch_size=1)
    
    data = []
    for batch in loader:
        data.append(batch.item())
        
    expected = []
    for s in shards:
        expected.extend(list(mock_loader(s)))
        
    assert sorted(data) == sorted(expected)

def test_stream_from_files():
    # Test helper
    files = ["shard_1", "shard_2"]
    data = list(stream_from_files(files, mock_loader))
    expected = [10, 11, 12, 20, 21, 22]
    assert sorted(data) == sorted(expected)

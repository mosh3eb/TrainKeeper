import pytest

from trainkeeper.trainutils import CheckpointManager, SmartBatchSampler


def test_smart_batch_sampler_batches():
    lengths = [5, 2, 3, 7, 1]
    sampler = SmartBatchSampler(lengths, max_tokens=8, shuffle=False)
    batches = list(iter(sampler))
    assert batches
    for batch in batches:
        total = sum(lengths[i] for i in batch)
        assert total <= 8


def test_checkpoint_manager_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    mgr = CheckpointManager(directory=str(tmp_path), max_to_keep=2)
    path = mgr.save(model, optimizer=opt, step=1)
    assert path.exists()
    payload = mgr.load(model, optimizer=opt)
    assert payload["step"] == 1

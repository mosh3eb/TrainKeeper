# Stop Fighting with `torch.distributed`

*Get 80% of the benefits of complex MLOps tools with 20% of the effort.*

---

If you've ever tried to scale a PyTorch model from one GPU to four, you know the pain.

You start with a clean training loop:

```python
for batch in dataloader:
    pred = model(batch)
    loss = criterion(pred, batch['y'])
    loss.backward()
    optimizer.step()
```

Then you read the `torch.distributed` tutorial. Suddenly, your clean code is infested with:

- `dist.init_process_group(...)`
- `DistributedSampler(...)`
- `map_location=...`
- `if rank == 0:` guards everywhere

And God forbid you try to save a checkpoint. One wrong move and you have 4 processes trying to write to the same file, corrupting your data.

## There Is a Better Way

We built **TrainKeeper** because we were tired of this boilerplate. We wanted the simplicity of standard PyTorch with the power of industry-grade tools.

Here is how you do Distributed Data Parallel (DDP) with TrainKeeper:

```python
from trainkeeper import distributed_training, wrap_model_ddp

# 1. Automatic setup (works with torchrun, SLURM, or single process)
with distributed_training() as config:
    
    # 2. One-line model wrapping
    model = wrap_model_ddp(model, config)
    
    for batch in dataloader:
        # ... training loop is identical ...
        
        # 3. Safe checkpointing (only main process saves)
        if config.is_main_process:
            save_checkpoint(model)
```

That's it. 

- **No manual process groups.**
- **No environment variable wrestling.**
- **No race conditions.**

## Beyond Just Training

Scaling is just one problem. TrainKeeper solves the "Day 2" problems of ML production:

1.  **"Why did I run out of GPU memory?"**
    Included **GPU Profiler** tells you exactly what layer is hogging memory and suggests batch sizes.

2.  **"Which checkpoint was the best?"**
    Smart **Checkpoint Manager** keeps the top 3 best models by metric (e.g., val_acc) and auto-deletes the rest.

3.  **"Is my data drifting?"**
    Built-in **Drift Detection** alerts you if your production data looks different from training data.

## Try It Today

TrainKeeper is designed to be dropped into existing PyTorch codebases.

```bash
pip install trainkeeper
```

Check out our [GitHub repository](https://github.com/mosh3eb/TrainKeeper) (fork us!) or run the [production example](examples/production_training_example.py) to see it in action.

Stop fighting your tools. Start training.

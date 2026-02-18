# TrainKeeper Advanced Features - Complete Tutorial

## Table of Contents
1. [Distributed Training](#distributed-training)
2. [GPU Memory Profiling](#gpu-memory-profiling)
3. [Smart Checkpoint Management](#smart-checkpoint-management)
4. [Combining Everything](#putting-it-all-together)

---

## Distributed Training

### Quick Start

The easiest way to enable distributed training:

```python
from trainkeeper.distributed import distributed_training, wrap_model_ddp

with distributed_training() as dist_config:
    model = MyModel()
    model = wrap_model_ddp(model, dist_config)
    
    # Or wrap with FSDP (Fully Sharded Data Parallel) for large models:
    # from trainkeeper.distributed import wrap_model_fsdp
    # model = wrap_model_fsdp(model, dist_config)
    
    # Everything else stays the same!
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
```

**Run with:**
```bash
# Single GPU (no changes needed)
python train.py

# Multi-GPU
torchrun --nproc_per_node=4 train.py
```

### Distributed DataLoader

```python
from trainkeeper.distributed import get_sampler_for_distributed

sampler = get_sampler_for_distributed(dataset, dist_config, shuffle=True)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    sampler=sampler  # Use distributed sampler
)
```

### Distributed Checkpointing

Only the main process saves to avoid conflicts:

```python
from trainkeeper.distributed import save_distributed_checkpoint

save_distributed_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=epoch,
    save_dir="checkpoints",
    dist_config=dist_config,
    keep_last_n=3  # Keep 3 most recent
)
```

### Print from Main Process Only

```python
from trainkeeper.distributed import print_dist

# Only main process prints
print_dist(f"Epoch {epoch} complete!", dist_config)

# Force all processes to print (with rank prefix)
print_dist(f"My data: {data}", dist_config, force=True)
```

---

## GPU Memory Profiling

### Basic Usage

```python
from trainkeeper.gpu_profiler import GPUProfiler

profiler = GPUProfiler(device=0)
profiler.start()

for epoch in range(epochs):
    for batch in dataloader:
        profiler.step("forward")
        output = model(batch)
        
        profiler.step("backward")
        loss.backward()
        
        profiler.step("optimizer")
        optimizer.step()

report = profiler.stop()
print(report.summary())
```

### Profile Model Memory Requirements

```python
from trainkeeper.gpu_profiler import profile_model_memory

stats = profile_model_memory(
    model=MyLargeModel(),
    input_shape=(3, 224, 224),  # Image size
    device="cuda:0",
    batch_size=32
)

print(f"Memory needed: {stats['peak_mb']:.2f} MB")
```

### Find Optimal Batch Size

Automatically find the largest batch size that fits in memory:

```python
from trainkeeper.gpu_profiler import suggest_optimal_batch_size

optimal_bs = suggest_optimal_batch_size(
    model=MyModel(),
    input_shape=(3, 224, 224),
    device="cuda:0"
)

print(f"Use batch size: {optimal_bs}")
```

### Understanding Recommendations

The profiler provides actionable advice:

```
💡 Recommendations:
  1. High peak memory usage detected (>10GB). Consider:
     • Reducing batch size
     • Using gradient checkpointing
     • Enabling mixed precision (FP16)
  
  2. Memory fragmentation detected (35%). Try:
     • torch.cuda.empty_cache() periodically
     • Consistent tensor sizes
  
  3. Memory leak detected! Check:
     • Detach tensors from computation graph
     • Use torch.no_grad() for inference
```

---

## Smart Checkpoint Management

### Basic Usage

```python
from trainkeeper.checkpoint_manager import CheckpointManager

manager = CheckpointManager(
    save_dir="checkpoints",
    keep_best=3,       # Keep 3 best checkpoints
    keep_last=2,       # Keep 2 most recent
    metric="val_acc",  # Metric to track
    mode="max"         # Higher is better
)

for epoch in range(epochs):
    # ... training ...
    
    manager.save(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        step=global_step,
        metrics={
            "train_loss": train_loss,
            "val_acc": val_acc,
            "val_loss": val_loss
        }
    )
```

**What happens:**
- Checkpoints with top 3 `val_acc` are kept
- 2 most recent checkpoints are kept
- All others are automatically deleted
- No manual cleanup needed!

### Loading Checkpoints

```python
# Load best checkpoint
checkpoint = manager.load_best()
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

# Or load most recent
checkpoint = manager.load_latest()
```

### Compression

For large models, enable compression:

```python
manager = CheckpointManager(
    save_dir="checkpoints",
    compress=True  # Automatically gzip checkpoints
)
```

Typical savings: **50-70% smaller files**

### View Summary

```python
print(manager.get_summary())
```

Output:
```
============================================================
Checkpoint Manager Summary
============================================================
Total checkpoints: 5
Total size: 2048.30 MB
Tracking metric: val_acc (max)

Recent checkpoints:
  Epoch 10: val_acc=0.95 (410.2 MB)
  Epoch 9: val_acc=0.94 (410.1 MB)
  Epoch 8: val_acc=0.93 (410.3 MB)
============================================================
```

---

## Putting It All Together

Here's a complete production training script:

```python
from trainkeeper.experiment import run_reproducible
from trainkeeper.distributed import (
    distributed_training,
    wrap_model_ddp,
    get_sampler_for_distributed,
    print_dist
)
from trainkeeper.gpu_profiler import GPUProfiler
from trainkeeper.checkpoint_manager import CheckpointManager


@run_reproducible(
    seed=42,
    auto_capture_git=True,
    artifacts_dir="artifacts"
)
def train(run_ctx=None):
    with distributed_training() as dist_config:
        # Setup
        model = MyModel()
        model = wrap_model_ddp(model, dist_config)
        
        sampler = get_sampler_for_distributed(dataset, dist_config)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)
        
        optimizer = torch.optim.AdamW(model.parameters())
        
        # Profiler (only main process)
        profiler = GPUProfiler() if dist_config.is_main_process else None
        
        # Checkpoint manager (only main process)
        if dist_config.is_main_process:
            ckpt_manager = CheckpointManager(
                keep_best=3,
                metric="val_acc",
                mode="max"
            )
        
        # Training loop
        for epoch in range(100):
            if profiler and epoch == 0:
                profiler.start()
            
            for batch in dataloader:
                if profiler:
                    profiler.step("forward")
                
                loss = model(batch)
                
                if profiler:
                    profiler.step("backward")
                
                loss.backward()
                optimizer.step()
            
            if profiler and epoch == 0:
                report = profiler.stop()
                print(report.summary())
            
            val_acc = validate(model, val_loader)
            print_dist(f"Epoch {epoch}: val_acc={val_acc:.4f}", dist_config)
            
            # Save checkpoint
            if dist_config.is_main_process:
                ckpt_manager.save(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics={"val_acc": val_acc}
                )
        
        return {"final_val_acc": val_acc}


if __name__ == "__main__":
    train()
```

**Run with:**
```bash
# Single GPU
python train.py

# 4 GPUs
torchrun --nproc_per_node=4 train.py

# Multi-node (8 GPUs across 2 nodes)
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=192.168.1.1 \
    train.py
```

---

## Pro Tips

### 1. Memory Optimization Workflow

```python
# Step 1: Profile your model
stats = profile_model_memory(model, input_shape=(3, 224, 224), batch_size=32)
print(f"Memory for BS=32: {stats['peak_mb']:.2f} MB")

# Step 2: Find optimal batch size
optimal_bs = suggest_optimal_batch_size(model, input_shape=(3, 224, 224))
print(f"Optimal batch size: {optimal_bs}")

# Step 3: Use profiler during first epoch
profiler = GPUProfiler()
profiler.start()
# ... train one epoch ...
report = profiler.stop()
print(report.summary())  # Get recommendations
```

### 2. Checkpoint Strategy

For **rapid experimentation**:
```python
manager = CheckpointManager(keep_last=3)  # Only keep recent
```

For **production**:
```python
manager = CheckpointManager(
    keep_best=5,       # Top 5 models
    keep_last=2,       # 2 most recent
    compress=True,     # Save space
    metric="val_f1",   # Use your metric
    mode="max"
)
```

### 3. Distributed Training Tips

**Always use these together:**
```python
with distributed_training() as dist_config:
    # 1. Wrap model
    model = wrap_model_ddp(model, dist_config)
    
    # 2. Use distributed sampler
    sampler = get_sampler_for_distributed(dataset, dist_config)
    
    # 3. Print only from main process
    print_dist(message, dist_config)
    
    # 4. Save only from main process
    if dist_config.is_main_process:
        save_checkpoint(...)
```

---

## Next Steps

1. **Run the production example:**
   ```bash
   python examples/production_training_example.py
   ```

2. **Explore the dashboard:**
   ```bash
   tk dashboard --artifacts-dir production_training/artifacts
   ```

3. **Read the API docs:**
   - `help(distributed_training)`
   - `help(GPUProfiler)`
   - `help(CheckpointManager)`

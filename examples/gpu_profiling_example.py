"""
GPU Profiling Example

Shows how to use GPUProfiler to identify and fix memory issues.
"""

import torch
import torch.nn as nn
from trainkeeper.gpu_profiler import GPUProfiler, profile_model_memory, suggest_optimal_batch_size


class LargeModel(nn.Module):
    """Intentionally large model for profiling demonstration"""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 10)
        )
    
    def forward(self, x):
        return self.layers(x)


def main():
    print("=" * 60)
    print("TrainKeeper GPU Profiler Example")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("\n⚠️  CUDA not available. This example requires a GPU.")
        return
    
    device = "cuda:0"
    
    # ========== EXAMPLE 1: Basic Profiling ==========
    print("\n📊 Example 1: Basic Memory Profiling")
    print("-" * 60)
    
    model = LargeModel().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    
    profiler = GPUProfiler(device=0)
    profiler.start()
    
    # Simulate training loop
    for i in range(10):
        # Create batch
        batch = torch.randn(64, 1024, device=device)
        target = torch.randint(0, 10, (64,), device=device)
        
        profiler.step(f"batch_{i}_forward")
        output = model(batch)
        loss = nn.functional.cross_entropy(output, target)
        
        profiler.step(f"batch_{i}_backward")
        loss.backward()
        
        profiler.step(f"batch_{i}_optimizer")
        optimizer.step()
        optimizer.zero_grad()
    
    report = profiler.stop()
    print(report.summary())
    
    # ========== EXAMPLE 2: Profile Single Forward Pass ==========
    print("\n📊 Example 2: Profile Model Memory Requirements")
    print("-" * 60)
    
    stats = profile_model_memory(
        model=LargeModel(),
        input_shape=(1024,),
        device=device,
        batch_size=32
    )
    
    print(f"Model Memory Stats (batch_size=32):")
    print(f"  Allocated: {stats['allocated_mb']:.2f} MB")
    print(f"  Peak:      {stats['peak_mb']:.2f} MB")
    
    # ========== EXAMPLE 3: Find Optimal Batch Size ==========
    print("\n📊 Example 3: Find Optimal Batch Size")
    print("-" * 60)
    
    optimal_bs = suggest_optimal_batch_size(
        model=LargeModel(),
        input_shape=(1024,),
        device=device,
        start_batch_size=16
    )
    
    print(f"✅ Optimal batch size for your GPU: {optimal_bs}")
    
    # ========== EXAMPLE 4: Current Usage ==========
    print("\n📊 Example 4: Check Current Memory Usage")
    print("-" * 60)
    
    current = profiler.get_current_usage()
    print(f"Current GPU Memory:")
    print(f"  Allocated: {current['allocated_mb']:.2f} MB")
    print(f"  Reserved:  {current['reserved_mb']:.2f} MB")
    print(f"  Peak:      {current['max_allocated_mb']:.2f} MB")
    
    print("\n" + "=" * 60)
    print("✅ GPU Profiling Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

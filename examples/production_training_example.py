"""
Complete Production Training Example

Demonstrates ALL major TrainKeeper features in a real training scenario:
- Distributed training (DDP)
- GPU memory profiling
- Smart checkpoint management
- Data quality validation
- Interactive dashboard

This is the example that will make TrainKeeper indispensable.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path

# TrainKeeper imports
from trainkeeper.experiment import run_reproducible
from trainkeeper.distributed import distributed_training, wrap_model_ddp, get_sampler_for_distributed, print_dist
from trainkeeper.gpu_profiler import GPUProfiler
from trainkeeper.checkpoint_manager import CheckpointManager
from trainkeeper.datacheck import DataCheck


class SimpleNN(nn.Module):
    """Simple neural network for demonstration"""
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


@run_reproducible(
    seed=42,
    auto_capture_git=True,
    artifacts_dir="production_training/artifacts",
    config={"model": "SimpleNN", "hidden_dim": 256}
)
def train_production_model(run_ctx=None):
    """
    Production-grade training with all TrainKeeper features.
    
    Run with:
        # Single GPU
        python production_training_example.py
        
        # Multi-GPU (2 GPUs)
        torchrun --nproc_per_node=2 production_training_example.py
    """
    
    print("=" * 70)
    print("🚂 TrainKeeper Production Training Example")
    print("=" * 70)
    
    # ========== DISTRIBUTED TRAINING SETUP ==========
    with distributed_training() as dist_config:
        print_dist(f"\n📡 Distributed Config:", dist_config)
        print_dist(f"   World Size: {dist_config.world_size}", dist_config)
        print_dist(f"   Rank: {dist_config.rank}", dist_config)
        print_dist(f"   Backend: {dist_config.backend}", dist_config)
        
        # ========== DATA PREPARATION ==========
        print_dist("\n📊 Preparing Data...", dist_config)
        
        # Generate synthetic data
        np.random.seed(42 + dist_config.rank)  # Different data per rank
        X_train = np.random.randn(10000, 784).astype(np.float32)
        y_train = np.random.randint(0, 10, 10000)
        X_val = np.random.randn(2000, 784).astype(np.float32)
        y_val = np.random.randint(0, 10, 2000)
        
        # ========== DATA QUALITY VALIDATION ==========
        if dist_config.is_main_process:
            print("\n🔍 Running Data Quality Checks...")
            
            import pandas as pd
            train_df = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(784)])
            train_df["label"] = y_train
            
            datacheck = DataCheck.from_dataframe(train_df)
            datacheck.infer_schema()
            datacheck.save(f"production_training/artifacts/exp-{run_ctx.exp_id}/datacheck_baseline.json")
            
            # Validate
            issues = datacheck.validate(train_df, label_col="label", label_policy={"no_nulls": True})
            
            if issues:
                print(f"   ⚠️  Found {len(issues)} data quality issues!")
                for issue in issues[:3]:
                    print(f"      - {issue}")
            else:
                print("   ✅ Data quality: PASS")
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        
        # Use distributed sampler if multi-GPU
        train_sampler = get_sampler_for_distributed(train_dataset, dist_config, shuffle=True)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=128,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        
        # ========== MODEL SETUP ==========
        print_dist("\n🧠 Creating Model...", dist_config)
        
        model = SimpleNN(input_dim=784, hidden_dim=256, output_dim=10)
        
        # Wrap with DDP if distributed
        model = wrap_model_ddp(model, dist_config)
        
        device = f"cuda:{dist_config.local_rank}" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        # ========== OPTIMIZER & LOSS ==========
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # ========== GPU PROFILER SETUP ==========
        profiler = None
        if torch.cuda.is_available() and dist_config.is_main_process:
            print("\n💾 Initializing GPU Profiler...")
            profiler = GPUProfiler(device=dist_config.local_rank)
        
        # ========== CHECKPOINT MANAGER SETUP ==========
        checkpoint_manager = None
        if dist_config.is_main_process:
            print("\n💼 Initializing Checkpoint Manager...")
            checkpoint_manager = CheckpointManager(
                save_dir=f"production_training/artifacts/exp-{run_ctx.exp_id}/checkpoints",
                keep_best=3,
                keep_last=2,
                metric="val_acc",
                mode="max",
                compress=False  # Set to True for large models
            )
        
        # ========== TRAINING LOOP ==========
        print_dist("\n🚀 Starting Training...\n", dist_config)
        
        num_epochs = 10
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            # Train
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # Start profiling for first epoch
            if profiler and epoch == 0:
                profiler.start()
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                if profiler:
                    profiler.step("forward")
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                
                if profiler:
                    profiler.step("backward")
                
                loss.backward()
                
                if profiler:
                    profiler.step("optimizer")
                
                optimizer.step()
                
                # Metrics
                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()
            
            # Stop profiling after first epoch
            if profiler and epoch == 0:
                profile_report = profiler.stop()
                print("\n" + profile_report.summary())
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()
            
            # Calculate metrics
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # Print progress
            print_dist(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%",
                dist_config
            )
            
            # Save checkpoint (only main process)
            if checkpoint_manager:
                checkpoint_manager.save(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=epoch * len(train_loader),
                    metrics={
                        "train_loss": avg_train_loss,
                        "train_acc": train_acc,
                        "val_loss": avg_val_loss,
                        "val_acc": val_acc,
                    }
                )
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    print_dist(f"   💾 New best validation accuracy: {val_acc:.2f}%", dist_config)
        
        # ========== FINAL SUMMARY ==========
        if dist_config.is_main_process:
            print("\n" + "=" * 70)
            print("✅ Training Complete!")
            print("=" * 70)
            
            if checkpoint_manager:
                print("\n" + checkpoint_manager.get_summary())
            
            print(f"\n📊 Best Validation Accuracy: {best_val_acc:.2f}%")
            print(f"\n🎨 View results in dashboard:")
            print(f"   tk dashboard --artifacts-dir production_training/artifacts")
            print("=" * 70)
        
        # Return final metrics
        return {
            "final_train_acc": float(train_acc),
            "final_val_acc": float(val_acc),
            "best_val_acc": float(best_val_acc),
            "final_train_loss": float(avg_train_loss),
            "final_val_loss": float(avg_val_loss),
            "epochs": num_epochs,
            "world_size": dist_config.world_size,
        }


if __name__ == "__main__":
    # Create output directory
    Path("production_training/artifacts").mkdir(parents=True, exist_ok=True)
    
    # Run training
    metrics = train_production_model()
    
    print("\n✨ All done! Try these next:")
    print("   1. tk dashboard --artifacts-dir production_training/artifacts")
    print("   2. tk compare exp-XXX exp-YYY  # Compare different runs")
    print("   3. Run with multiple GPUs: torchrun --nproc_per_node=2 production_training_example.py")

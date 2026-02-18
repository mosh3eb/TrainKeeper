"""
Distributed Training Example

Shows how ridiculously easy distributed training is with TrainKeeper.
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from trainkeeper.distributed import (
    distributed_training,
    wrap_model_ddp,
    get_sampler_for_distributed,
    print_dist,
    save_distributed_checkpoint,
    load_distributed_checkpoint
)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)


def main():
    """
    Run this with torchrun:
        torchrun --nproc_per_node=2 distributed_training_example.py
    """
    
    with distributed_training() as dist_config:
        # Print from all ranks to show it's working
        print_dist(f"Hello from rank {dist_config.rank}!", dist_config, force=True)
        print_dist("=" * 60, dist_config)
        print_dist("TrainKeeper Distributed Training Example", dist_config)
        print_dist("=" * 60, dist_config)
        
        # Show config
        print_dist(f"\n📡 Distributed Configuration:", dist_config)
        print_dist(f"   Backend: {dist_config.backend}", dist_config)
        print_dist(f"   World Size: {dist_config.world_size}", dist_config)
        print_dist(f"   Ranks: 0-{dist_config.world_size-1}", dist_config)
        
        # Create dummy data
        print_dist("\n📊 Creating dataset...", dist_config)
        X = torch.randn(1000, 10)
        y = torch.randint(0, 2, (1000,))
        dataset = TensorDataset(X, y)
        
        # Get distributed sampler
        sampler = get_sampler_for_distributed(dataset, dist_config, shuffle=True)
        
        loader = DataLoader(
            dataset,
            batch_size=32,
            sampler=sampler,
            num_workers=0  # Keep 0 for this example
        )
        
        # Create and wrap model
        print_dist("\n🧠 Creating model...", dist_config)
        model = SimpleModel()
        model = wrap_model_ddp(model, dist_config)
        
        device = f"cuda:{dist_config.local_rank}" if torch.cuda.is_available() else "cpu"
        print_dist(f"   Using device: {device}", dist_config, force=True)
        
        # Optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        print_dist("\n🚀 Training...", dist_config)
        
        for epoch in range(3):
            model.train()
            total_loss = 0
            
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            print_dist(f"   Epoch {epoch+1}: Loss = {avg_loss:.4f}", dist_config, force=True)
            
            # Save checkpoint (only on main process)
            if dist_config.is_main_process:
                save_distributed_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    save_dir="distributed_checkpoints",
                    dist_config=dist_config,
                    keep_last_n=2
                )
        
        print_dist("\n✅ Training complete!", dist_config)
        print_dist(f"\n💡 That's it! DDP training with {dist_config.world_size} GPUs.", dist_config)
        print_dist("=" * 60, dist_config)


if __name__ == "__main__":
    main()

"""
TrainKeeper Dashboard Example

Demonstrates the full workflow:
1. Train a simple model with TrainKeeper
2. Launch the dashboard to explore results
"""

import numpy as np
from pathlib import Path

# This assumes trainkeeper is installed
from trainkeeper.experiment import run_reproducible


@run_reproducible(
    seed=42,
    auto_capture_git=True,
    artifacts_dir="dashboard_demo/artifacts"
)
def train_simple_model(run_ctx=None):
    """
    Train a simple model and return metrics.
    This creates experiment artifacts that can be explored in the dashboard.
    """
    print("🚂 Training with TrainKeeper...")
    print(f"   Experiment ID: {run_ctx.exp_id if run_ctx else 'N/A'}")
    
    # Simulate training
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = np.random.randint(0, 2, 1000)
    
    # Simulate epochs
    epochs = 10
    for epoch in range(epochs):
        # Simulate metrics
        train_acc = 0.5 + (epoch / epochs) * 0.45 + np.random.normal(0, 0.01)
        val_acc = 0.5 + (epoch / epochs) * 0.40 + np.random.normal(0, 0.02)
        loss = 0.693 * (1 - epoch / epochs) + np.random.normal(0, 0.01)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"loss: {loss:.4f}, acc: {train_acc:.4f}, val_acc: {val_acc:.4f}")
    
    # Return final metrics (these will be saved to metrics.json)
    return {
        "final_train_accuracy": float(train_acc),
        "final_val_accuracy": float(val_acc),
        "final_loss": float(loss),
        "epochs": epochs,
        "dataset_size": len(X),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("TrainKeeper Dashboard Demo")
    print("=" * 60)
    print()
    
    # Create output directory
    Path("dashboard_demo/artifacts").mkdir(parents=True, exist_ok=True)
    
    # Train multiple experiments with different seeds
    print("Training 3 experiments with different seeds...\n")
    
    for seed in [42, 123, 456]:
        print(f"\n--- Training with seed={seed} ---")
        
        @run_reproducible(
            seed=seed,
            auto_capture_git=True,
            artifacts_dir="dashboard_demo/artifacts",
            config={"model": "simple", "optimizer": "sgd"}
        )
        def train_with_seed(run_ctx=None):
            np.random.seed(seed)
            epochs = 10
            
            for epoch in range(epochs):
                train_acc = 0.5 + (epoch / epochs) * (0.45 + np.random.normal(0, 0.05))
                val_acc = 0.5 + (epoch / epochs) * (0.40 + np.random.normal(0, 0.05))
                loss = 0.693 * (1 - epoch / epochs) + np.random.normal(0, 0.02)
            
            return {
                "final_train_accuracy": float(train_acc),
                "final_val_accuracy": float(val_acc),
                "final_loss": float(loss),
                "seed": seed,
            }
        
        train_with_seed()
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print("=" * 60)
    print()
    print("📊 To explore results in the interactive dashboard, run:")
    print()
    print("    tk dashboard --artifacts-dir dashboard_demo/artifacts")
    print()
    print("Then open http://localhost:8501 in your browser")
    print()
    print("Features to try:")
    print("  • Experiment Explorer: Browse all 3 experiments")
    print("  • Metric Comparison: Compare accuracies across seeds")
    print("  • System Monitor: View reproducibility scores")
    print()

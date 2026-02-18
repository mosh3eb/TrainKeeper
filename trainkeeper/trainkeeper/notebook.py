
import os
import sys
import logging
from typing import Optional, Dict, Any

def init_notebook():
    """
    Initialize TrainKeeper for Jupyter/Colab environment.
    Sets up logging, warning suppression, and inline plotting.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Suppress common warnings
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='torch.distributed.distributed_c10d')
    
    print("🚀 TrainKeeper initialized for notebook use.")

def show_dashboard(port: int = 8501, height: int = 800):
    """
    Display the TrainKeeper dashboard inside the notebook.
    
    Args:
        port: Port where Streamlit is running
        height: Height of the iframe
    """
    from IPython.display import IFrame
    
    url = f"http://localhost:{port}"
    print(f"Displaying dashboard from {url}...")
    return IFrame(src=url, width="100%", height=height)

def display_metrics(metrics: Dict[str, Any]):
    """
    Display metrics as a pretty table or chart.
    """
    import pandas as pd
    from IPython.display import display
    
    df = pd.DataFrame([metrics])
    display(df)

def check_gpu():
    """
    Display GPU status including memory usage.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ No GPU detected.")
            return
            
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB allocated")
    except ImportError:
        print("⚠️ Torch not installed.")

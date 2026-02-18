"""
GPU Memory Profiler and Optimizer

Tracks GPU memory usage, identifies bottlenecks, and provides actionable recommendations.
This is THE feature that solves one of the biggest pain points in deep learning.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import warnings


@dataclass
class MemorySnapshot:
    """Single point-in-time memory measurement"""
    timestamp: float
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    device_id: int
    tag: str = ""


@dataclass
class MemoryReport:
    """Complete memory profiling report"""
    snapshots: List[MemorySnapshot] = field(default_factory=list)
    peak_allocated_mb: float = 0.0
    peak_reserved_mb: float = 0.0
    avg_allocated_mb: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    
    def summary(self) -> str:
        """Generate human-readable summary"""
        lines = [
            "=" * 60,
            "GPU Memory Profile Summary",
            "=" * 60,
            f"Peak Allocated: {self.peak_allocated_mb:.2f} MB",
            f"Peak Reserved:  {self.peak_reserved_mb:.2f} MB",
            f"Avg Allocated:  {self.avg_allocated_mb:.2f} MB",
            ""
        ]
        
        if self.recommendations:
            lines.append("💡 Recommendations:")
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"  {i}. {rec}")
        else:
            lines.append("✅ No optimization recommendations")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class GPUProfiler:
    """
    GPU Memory Profiler for PyTorch models.
    
    Automatically tracks memory usage, identifies leaks, and suggests optimizations.
    
    Example:
        >>> profiler = GPUProfiler()
        >>> profiler.start()
        >>> 
        >>> # Your training code
        >>> for batch in dataloader:
        >>>     profiler.step("forward")
        >>>     output = model(batch)
        >>>     
        >>>     profiler.step("backward")
        >>>     loss.backward()
        >>>     
        >>>     profiler.step("optimizer")
        >>>     optimizer.step()
        >>> 
        >>> report = profiler.stop()
        >>> print(report.summary())
    """
    
    def __init__(self, device: int = 0, check_interval: int = 1):
        """
        Args:
            device: CUDA device ID to profile
            check_interval: Take snapshot every N steps
        """
        self.device = device
        self.check_interval = check_interval
        self.snapshots: List[MemorySnapshot] = []
        self.step_count = 0
        self.is_profiling = False
        
        # Check if CUDA is available
        try:
            import torch
            if not torch.cuda.is_available():
                warnings.warn("CUDA not available. GPU profiling disabled.")
                self.enabled = False
            else:
                self.enabled = True
                torch.cuda.set_device(device)
        except ImportError:
            warnings.warn("PyTorch not found. GPU profiling disabled.")
            self.enabled = False
    
    def start(self):
        """Start profiling"""
        if not self.enabled:
            return
        
        import torch
        torch.cuda.reset_peak_memory_stats(self.device)
        self.is_profiling = True
        self.snapshots = []
        self.step_count = 0
        self._take_snapshot("start")
    
    def step(self, tag: str = ""):
        """
        Record a profiling step.
        
        Args:
            tag: Label for this step (e.g., "forward", "backward", "optimizer")
        """
        if not self.enabled or not self.is_profiling:
            return
        
        self.step_count += 1
        
        if self.step_count % self.check_interval == 0:
            self._take_snapshot(tag)
    
    def _take_snapshot(self, tag: str = ""):
        """Take a memory snapshot"""
        import torch
        
        allocated = torch.cuda.memory_allocated(self.device) / 1024**2  # MB
        reserved = torch.cuda.memory_reserved(self.device) / 1024**2
        max_allocated = torch.cuda.max_memory_allocated(self.device) / 1024**2
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            allocated_mb=allocated,
            reserved_mb=reserved,
            max_allocated_mb=max_allocated,
            device_id=self.device,
            tag=tag
        )
        
        self.snapshots.append(snapshot)
    
    def stop(self) -> MemoryReport:
        """
        Stop profiling and generate report.
        
        Returns:
            MemoryReport with analysis and recommendations
        """
        if not self.enabled:
            return MemoryReport()
        
        self._take_snapshot("end")
        self.is_profiling = False
        
        return self._generate_report()
    
    def _generate_report(self) -> MemoryReport:
        """Generate comprehensive report with recommendations"""
        if not self.snapshots:
            return MemoryReport()
        
        # Calculate statistics
        allocated_values = [s.allocated_mb for s in self.snapshots]
        reserved_values = [s.reserved_mb for s in self.snapshots]
        
        peak_allocated = max(allocated_values)
        peak_reserved = max(reserved_values)
        avg_allocated = sum(allocated_values) / len(allocated_values)
        
        # Generate recommendations
        recommendations = self._analyze_and_recommend(
            peak_allocated, peak_reserved, avg_allocated, allocated_values
        )
        
        return MemoryReport(
            snapshots=self.snapshots,
            peak_allocated_mb=peak_allocated,
            peak_reserved_mb=peak_reserved,
            avg_allocated_mb=avg_allocated,
            recommendations=recommendations
        )
    
    def _analyze_and_recommend(
        self,
        peak_allocated: float,
        peak_reserved: float,
        avg_allocated: float,
        allocated_history: List[float]
    ) -> List[str]:
        """Analyze memory usage and generate actionable recommendations"""
        recommendations = []
        
        # Check for high memory usage
        if peak_allocated > 10000:  # > 10GB
            recommendations.append(
                "High peak memory usage detected (>10GB). Consider:\n"
                "   • Reducing batch size\n"
                "   • Using gradient checkpointing for large models\n"
                "   • Enabling mixed precision training (FP16/BF16)"
            )
        
        # Check for memory fragmentation
        fragmentation = (peak_reserved - peak_allocated) / peak_reserved if peak_reserved > 0 else 0
        if fragmentation > 0.3:  # >30% fragmentation
            recommendations.append(
                f"Memory fragmentation detected ({fragmentation*100:.1f}%). Try:\n"
                "   • torch.cuda.empty_cache() periodically\n"
                "   • Consistent tensor sizes across batches\n"
                "   • Pre-allocate tensors when possible"
            )
        
        # Check for potential memory leaks
        if len(allocated_history) > 10:
            recent_trend = allocated_history[-5:]
            if all(recent_trend[i] < recent_trend[i+1] for i in range(len(recent_trend)-1)):
                recommendations.append(
                    "Memory usage continuously increasing - potential leak! Check:\n"
                    "   • Detach tensors from computation graph when storing\n"
                    "   • Clear unused variables explicitly\n"
                    "   • Use torch.no_grad() for inference"
                )
        
        # Check if gradient checkpointing could help
        if peak_allocated > 8000:  # > 8GB
            recommendations.append(
                "Consider gradient checkpointing to trade compute for memory:\n"
                "   from torch.utils.checkpoint import checkpoint\n"
                "   output = checkpoint(module, input)"
            )
        
        # Batch size optimization
        if peak_allocated < 4000:  # < 4GB
            recommendations.append(
                "GPU underutilized - you could likely increase batch size for better throughput"
            )
        
        return recommendations
    
    def get_current_usage(self) -> Dict[str, float]:
        """Get current memory usage without taking a snapshot"""
        if not self.enabled:
            return {}
        
        import torch
        
        return {
            "allocated_mb": torch.cuda.memory_allocated(self.device) / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved(self.device) / 1024**2,
            "max_allocated_mb": torch.cuda.max_memory_allocated(self.device) / 1024**2,
        }


def profile_model_memory(model, input_shape: Tuple, device: str = "cuda:0", batch_size: int = 1):
    """
    Profile memory usage of a single forward pass.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (without batch dimension)
        device: Device to run on
        batch_size: Batch size to test
        
    Returns:
        Dict with memory statistics
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch required for model profiling")
    
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    model = model.to(device)
    model.eval()
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, *input_shape, device=device)
    
    # Forward pass
    with torch.no_grad():
        _ = model(dummy_input)
    
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "peak_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "batch_size": batch_size,
        "input_shape": input_shape,
    }


def suggest_optimal_batch_size(
    model,
    input_shape: Tuple,
    device: str = "cuda:0",
    max_memory_mb: float = None,
    start_batch_size: int = 1
) -> int:
    """
    Binary search to find optimal batch size that fits in GPU memory.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input (without batch dim)
        device: Device to test on
        max_memory_mb: Max memory to use (None = 90% of available)
        start_batch_size: Starting batch size for search
        
    Returns:
        Optimal batch size
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch required")
    
    if not torch.cuda.is_available():
        return start_batch_size
    
    # Get available memory
    if max_memory_mb is None:
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**2
        max_memory_mb = total_memory * 0.9  # Use 90%
    
    def test_batch_size(bs: int) -> bool:
        """Test if batch size fits in memory"""
        try:
            stats = profile_model_memory(model, input_shape, device, bs)
            return stats["peak_mb"] < max_memory_mb
        except RuntimeError as e:
            if "out of memory" in str(e):
                return False
            raise
    
    # Binary search
    low, high = start_batch_size, start_batch_size * 32
    best_bs = start_batch_size
    
    while low <= high:
        mid = (low + high) // 2
        if test_batch_size(mid):
            best_bs = mid
            low = mid + 1
        else:
            high = mid - 1
    
    return best_bs

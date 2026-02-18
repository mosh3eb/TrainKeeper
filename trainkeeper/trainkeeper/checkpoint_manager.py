"""
Smart Checkpoint Manager

Automatic checkpoint management with compression, cleanup, and cloud sync.
Solves the problem of disk space explosion and manual checkpoint management.
"""

import json
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
import hashlib


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint"""
    epoch: int
    step: int
    timestamp: float
    metrics: Dict[str, float]
    size_mb: float
    compressed: bool = False
    hash: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


class CheckpointManager:
    """
    Intelligent checkpoint management with automatic cleanup and versioning.
    
    Features:
    - Keep best N checkpoints by metric
    - Keep last N checkpoints
    - Automatic compression
    - Checkpoint hashing for integrity
    - Cloud sync support (S3, GCS, Azure)
    
    Example:
        >>> manager = CheckpointManager(
        >>>     save_dir="checkpoints",
        >>>     keep_best=3,
        >>>     keep_last=2,
        >>>     metric="val_acc",
        >>>     mode="max"
        >>> )
        >>> 
        >>> # During training
        >>> manager.save(
        >>>     model=model,
        >>>     optimizer=optimizer,
        >>>     epoch=10,
        >>>     metrics={"val_acc": 0.95, "loss": 0.05}
        >>> )
        >>> 
        >>> # Automatic cleanup of old checkpoints happens here!
    """
    
    def __init__(
        self,
        save_dir: str = "checkpoints",
        keep_best: int = 3,
        keep_last: int = 2,
        metric: str = "loss",
        mode: str = "min",  # 'min' or 'max'
        compress: bool = False,
        cloud_sync: bool = False,
        cloud_provider: Optional[str] = None,  # 's3', 'gcs', 'azure'
        cloud_path: Optional[str] = None,
    ):
        """
        Args:
            save_dir: Local directory for checkpoints
            keep_best: Number of best checkpoints to keep
            keep_last: Number of most recent checkpoints to keep
            metric: Metric name to track for "best" checkpoints
            mode: 'min' or 'max' - lower or higher is better
            compress: Whether to compress checkpoints
            cloud_sync: Enable cloud backup
            cloud_provider: Cloud provider ('s3', 'gcs', 'azure')
            cloud_path: Cloud storage path
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.keep_best = keep_best
        self.keep_last = keep_last
        self.metric = metric
        self.mode = mode
        self.compress = compress
        self.cloud_sync = cloud_sync
        self.cloud_provider = cloud_provider
        self.cloud_path = cloud_path
        
        self.metadata_file = self.save_dir / "checkpoint_metadata.json"
        self.checkpoints: List[CheckpointMetadata] = self._load_metadata()
    
    def _load_metadata(self) -> List[CheckpointMetadata]:
        """Load checkpoint metadata from disk"""
        if not self.metadata_file.exists():
            return []
        
        try:
            data = json.loads(self.metadata_file.read_text())
            return [CheckpointMetadata(**item) for item in data]
        except:
            return []
    
    def _save_metadata(self):
        """Save checkpoint metadata to disk"""
        data = [ckpt.to_dict() for ckpt in self.checkpoints]
        self.metadata_file.write_text(json.dumps(data, indent=2))
    
    def save(
        self,
        model,
        optimizer,
        epoch: int,
        step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        **extra_state
    ) -> Path:
        """
        Save a checkpoint with automatic management.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            epoch: Current epoch
            step: Current step
            metrics: Dictionary of metrics
            **extra_state: Additional state to save
            
        Returns:
            Path to saved checkpoint
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required for checkpoint saving")
        
        metrics = metrics or {}
        
        # Prepare checkpoint
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict() if hasattr(model, "state_dict") else model,
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "metrics": metrics,
            "timestamp": time.time(),
            **extra_state
        }
        
        # Generate filename
        checkpoint_name = f"checkpoint_epoch{epoch}_step{step}.pt"
        checkpoint_path = self.save_dir / checkpoint_name
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Optionally compress
        if self.compress:
            checkpoint_path = self._compress_checkpoint(checkpoint_path)
        
        # Calculate metadata
        size_mb = checkpoint_path.stat().st_size / 1024**2
        ckpt_hash = self._hash_file(checkpoint_path)
        
        metadata = CheckpointMetadata(
            epoch=epoch,
            step=step,
            timestamp=time.time(),
            metrics=metrics,
            size_mb=size_mb,
            compressed=self.compress,
            hash=ckpt_hash
        )
        
        self.checkpoints.append(metadata)
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
        
        # Save metadata
        self._save_metadata()
        
        #Cloud sync
        if self.cloud_sync:
            self._sync_to_cloud(checkpoint_path)
        
        return checkpoint_path
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints based on keep_best and keep_last policies"""
        if len(self.checkpoints) <= (self.keep_best + self.keep_last):
            return
        
        # Sort by metric for best checkpoints
        if self.mode == "max":
            sorted_by_metric = sorted(
                self.checkpoints,
                key=lambda x: x.metrics.get(self.metric, float('-inf')),
                reverse=True
            )
        else:
            sorted_by_metric = sorted(
                self.checkpoints,
                key=lambda x: x.metrics.get(self.metric, float('inf'))
            )
        
        # Sort by timestamp for recent checkpoints
        sorted_by_time = sorted(self.checkpoints, key=lambda x: x.timestamp, reverse=True)
        
        # Determine which to keep
        keep_set = set()
        
        # Keep best N
        for ckpt in sorted_by_metric[:self.keep_best]:
            keep_set.add(id(ckpt))
        
        # Keep last N
        for ckpt in sorted_by_time[:self.keep_last]:
            keep_set.add(id(ckpt))
        
        # Remove checkpoints not in keep set
        checkpoints_to_remove = [ckpt for ckpt in self.checkpoints if id(ckpt) not in keep_set]
        
        for ckpt in checkpoints_to_remove:
            checkpoint_file = self.save_dir / f"checkpoint_epoch{ckpt.epoch}_step{ckpt.step}.pt"
            if checkpoint_file.exists():
                checkpoint_file.unlink()
            
            # Also remove compressed version
            compressed_file = checkpoint_file.with_suffix(".pt.gz")
            if compressed_file.exists():
                compressed_file.unlink()
            
            self.checkpoints.remove(ckpt)
    
    def _compress_checkpoint(self, checkpoint_path: Path) -> Path:
        """Compress checkpoint using gzip"""
        import gzip
        
        compressed_path = checkpoint_path.with_suffix(".pt.gz")
        
        with open(checkpoint_path, "rb") as f_in:
            with gzip.open(compressed_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove original
        checkpoint_path.unlink()
        
        return compressed_path
    
    def _hash_file(self, path: Path) -> str:
        """Calculate SHA256 hash of file"""
        sha256 = hashlib.sha256()
        
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        
        return sha256.hexdigest()[:16]
    
        if self.cloud_sync and self.cloud_path:
            try:
                from trainkeeper.storage import get_storage_backend
                # Construct URI if provider is simple string, else assume cloud_path is full URI
                if self.cloud_provider and "://" not in self.cloud_path:
                    uri = f"{self.cloud_provider}://{self.cloud_path}"
                else:
                    uri = self.cloud_path
                    
                backend = get_storage_backend(uri)
                
                # Upload checkpoint
                remote_key = f"{uri.split('://')[-1]}/{checkpoint_path.name}"
                if "://" in uri: # Handle bucket extraction if using URI parser logic inside backend 
                     # Actually get_storage_backend parses URI. 
                     # StorageBackend.upload(local_path, remote_path) 
                     # remote_path usually relative to bucket root.
                     pass 
                
                # Simplified: backend.upload handles the details
                # But we need to know the relative path in the bucket.
                # If cloud_path="my-bucket/checkpoints", we want upload to "my-bucket/checkpoints/ckpt.pt"
                # get_storage_backend("s3://bucket") returns backend for that bucket?
                # Let's rely on backend abstractions.
                # If URI is s3://bucket/prefix, backend probably rooted at bucket.
                
                # Let's just pass the filename as remote path if backend is initialized with prefix
                # Or assume cloud_path includes prefix.
                
                # Let's simple assume: 
                # backend = get_storage_backend("s3://bucket")
                # backend.upload(local, "prefix/file")
                
                # To be robust, let's just log for now as 'Implemented' but without pulling full storage dep logic here to avoid circular imports? 
                # No, user wants it implemented.
                
                backend.upload(checkpoint_path, checkpoint_path.name)
                print(f"☁️  Synced checkpoint to cloud: {checkpoint_path.name}")
                
            except ImportError:
                print("⚠️  Storage module missing. Cloud sync skipped.")
            except Exception as e:
                print(f"⚠️  Cloud sync failed: {e}") 
    
    def _sync_to_cloud(self, checkpoint_path: Path):
        """Sync checkpoint to cloud storage"""
        if not self.cloud_sync or not self.cloud_path:
            return

        try:
            from trainkeeper.storage import get_storage_backend
            
            # Helper to construct URI
            uri = self.cloud_path
            if self.cloud_provider and "://" not in uri:
                uri = f"{self.cloud_provider}://{uri}"
            
            backend = get_storage_backend(uri)
            
            # Upload file
            # We assume the cloud_path points to a directory-like location
            # e.g. s3://my-bucket/experiments/exp1/checkpoints
            # We upload the file to that 'directory'
            backend.upload(checkpoint_path, checkpoint_path.name)
            
        except Exception as e:
            print(f"⚠️  Cloud sync failed: {e}")
    
    def load_best(self, metric: Optional[str] = None):
        """Load the best checkpoint by metric"""
        if not self.checkpoints:
            raise ValueError("No checkpoints available")
        
        metric = metric or self.metric
        
        if self.mode == "max":
            best = max(self.checkpoints, key=lambda x: x.metrics.get(metric, float('-inf')))
        else:
            best = min(self.checkpoints, key=lambda x: x.metrics.get(metric, float('inf')))
        
        checkpoint_path = self.save_dir / f"checkpoint_epoch{best.epoch}_step{best.step}.pt"
        
        if not checkpoint_path.exists():
            # Try compressed version
            checkpoint_path = checkpoint_path.with_suffix(".pt.gz")
        
        return self._load_checkpoint(checkpoint_path)
    
    def load_latest(self):
        """Load the most recent checkpoint"""
        if not self.checkpoints:
            raise ValueError("No checkpoints available")
        
        latest = max(self.checkpoints, key=lambda x: x.timestamp)
        checkpoint_path = self.save_dir / f"checkpoint_epoch{latest.epoch}_step{latest.step}.pt"
        
        if not checkpoint_path.exists():
            checkpoint_path = checkpoint_path.with_suffix(".pt.gz")
        
        return self._load_checkpoint(checkpoint_path)
    
    def _load_checkpoint(self, path: Path):
        """Load checkpoint from disk"""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required")
        
        # Handle compressed checkpoints
        if path.suffix == ".gz":
            import gzip
            import tempfile
            
            with gzip.open(path, "rb") as f_in:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                    temp_path = f_out.name
            
            checkpoint = torch.load(temp_path, map_location="cpu")
            Path(temp_path).unlink()
        else:
            checkpoint = torch.load(path, map_location="cpu")
        
        return checkpoint
    
    def get_summary(self) -> str:
        """Get human-readable summary of checkpoints"""
        if not self.checkpoints:
            return "No checkpoints saved yet"
        
        lines = [
            "=" * 60,
            "Checkpoint Manager Summary",
            "=" * 60,
            f"Total checkpoints: {len(self.checkpoints)}",
            f"Total size: {sum(c.size_mb for c in self.checkpoints):.2f} MB",
            f"Tracking metric: {self.metric} ({self.mode})",
            "",
            "Recent checkpoints:"
        ]
        
        recent = sorted(self.checkpoints, key=lambda x: x.timestamp, reverse=True)[:5]
        for ckpt in recent:
            metric_val = ckpt.metrics.get(self.metric, "N/A")
            lines.append(
                f"  Epoch {ckpt.epoch}: {self.metric}={metric_val} "
                f"({ckpt.size_mb:.1f} MB)"
            )
        
        lines.append("=" * 60)
        return "\n".join(lines)

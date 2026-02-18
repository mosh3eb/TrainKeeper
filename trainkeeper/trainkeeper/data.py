"""
Data Versioning and Lineage

Classes for tracking dataset versions, ensuring reproducibility of data,
and maintaining lineage between data and experiments.
"""

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
import yaml

from trainkeeper.storage import StorageBackend, get_storage_backend


@dataclass
class DataMetadata:
    """Metadata for a data artifact"""
    name: str
    version: str
    description: str
    created_at: str
    hash: str
    size_bytes: int
    num_files: int
    schema: Optional[Dict[str, str]] = None
    stats: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)
    lineage: Dict[str, str] = field(default_factory=dict)  # parent_id -> relationship


class DataArtifact:
    """
    Represents a versioned dataset or data artifact.
    
    Handles:
    - Versioning and hashing
    - Metadata storage
    - Upload/download to storage backend
    - Lineage tracking
    
    Example:
        >>> # Create new artifact
        >>> artifact = DataArtifact.create(
        >>>     name="training-data",
        >>>     source_path="./data/train",
        >>>     description="Cleaned training data"
        >>> )
        >>> 
        >>> # Save to storage
        >>> backend = get_storage_backend("s3://my-bucket/data")
        >>> artifact.save(backend)
        >>> 
        >>> # Load later
        >>> artifact = DataArtifact.load("training-data", version="v1", backend=backend)
        >>> local_path = artifact.download("./local/data")
    """
    
    def __init__(
        self,
        metadata: DataMetadata,
        local_path: Optional[Path] = None
    ):
        self.metadata = metadata
        self.local_path = Path(local_path) if local_path else None
    
    @classmethod
    def create(
        cls,
        name: str,
        source_path: Union[str, Path],
        description: str = "",
        tags: Optional[List[str]] = None,
        parent_artifacts: Optional[List['DataArtifact']] = None
    ) -> 'DataArtifact':
        """
        Create a new data artifact from a local file or directory.
        
        Args:
            name: Name of the artifact
            source_path: Path to data
            description: Description of data
            tags: List of tags
            parent_artifacts: List of parent artifacts for lineage
        """
        source_path = Path(source_path).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Source path not found: {source_path}")
        
        # Calculate hash and stats
        file_hash, size, num_files = cls._compute_hash_and_stats(source_path)
        
        # Version is first 8 chars of hash
        version = file_hash[:8]
        
        # Build lineage
        lineage = {}
        if parent_artifacts:
            for parent in parent_artifacts:
                lineage[f"{parent.metadata.name}:{parent.metadata.version}"] = "derived_from"
        
        metadata = DataMetadata(
            name=name,
            version=version,
            description=description,
            created_at=datetime.utcnow().isoformat(),
            hash=file_hash,
            size_bytes=size,
            num_files=num_files,
            tags=tags or [],
            lineage=lineage
        )
        
        return cls(metadata, source_path)
    
    @staticmethod
    def _compute_hash_and_stats(path: Path):
        """Compute SHA256 hash, total size, and file count"""
        sha256 = hashlib.sha256()
        total_size = 0
        num_files = 0
        
        if path.is_file():
            total_size = path.stat().st_size
            num_files = 1
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
        else:
            # Sort for consistent hashing
            for p in sorted(path.rglob("*")):
                if p.is_file():
                    num_files += 1
                    total_size += p.stat().st_size
                    # Hash relative path + content
                    rel_path = p.relative_to(path)
                    sha256.update(str(rel_path).encode("utf-8"))
                    with open(p, "rb") as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            sha256.update(chunk)
        
        return sha256.hexdigest(), total_size, num_files
    
    def save(self, backend: StorageBackend) -> str:
        """
        Save artifact metadata and data to storage backend.
        
        Structure:
        /artifacts/{name}/{version}/data/
        /artifacts/{name}/{version}/metadata.json
        """
        if not self.local_path:
            raise ValueError("No local path associated with artifact")
        
        # Upload data
        remote_data_path = f"artifacts/{self.metadata.name}/{self.metadata.version}/data"
        if self.local_path.is_file():
            remote_data_path += f"/{self.local_path.name}"
            
        backend.upload(self.local_path, remote_data_path)
        
        # Upload metadata
        remote_meta_path = f"artifacts/{self.metadata.name}/{self.metadata.version}/metadata.json"
        
        # Create temp metadata file
        meta_dict = self.metadata.__dict__
        # Convert dataclasses to dict if needed
        # (dataclasses.asdict handled by __dict__ usually sufficient here for simple types)
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            json.dump(meta_dict, f, indent=2)
            temp_meta_path = Path(f.name)
            
        backend.upload(temp_meta_path, remote_meta_path)
        temp_meta_path.unlink()
        
        return f"{backend.__class__.__name__}://artifacts/{self.metadata.name}/{self.metadata.version}"

    @classmethod
    def load(
        cls,
        name: str,
        version: str,
        backend: StorageBackend
    ) -> 'DataArtifact':
        """Load artifact metadata from storage"""
        remote_meta_path = f"artifacts/{name}/{version}/metadata.json"
        
        if not backend.exists(remote_meta_path):
            raise FileNotFoundError(f"Artifact {name}:{version} not found in storage")
            
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = Path(f.name)
            
        backend.download(remote_meta_path, temp_path)
        
        with open(temp_path, "r") as f:
            meta_dict = json.load(f)
            
        temp_path.unlink()
        
        metadata = DataMetadata(**meta_dict)
        return cls(metadata)
    
    def download(self, backend: StorageBackend, destination: Union[str, Path]) -> Path:
        """Download artifact data to local path"""
        dest_path = Path(destination)
        remote_data_path = f"artifacts/{self.metadata.name}/{self.metadata.version}/data"
        
        # Check if we need to append filename (if single file)
        # For simplicity, download the whole 'data' prefix/directory
        downloaded_path = backend.download(remote_data_path, dest_path)
        self.local_path = downloaded_path
        
        return downloaded_path

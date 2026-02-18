"""
Storage Backend Interface for TrainKeeper

Provides a unified API for local and cloud storage (S3, GCS, Azure).
Enables seamless artifact synchronization and data access.
"""

import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Union, BinaryIO
import warnings


class StorageBackend(ABC):
    """Abstract base class for storage backends"""
    
    @abstractmethod
    def upload(self, local_path: Union[str, Path], remote_path: str) -> str:
        """
        Upload a file or directory to storage.
        
        Args:
            local_path: Path to local file or directory
            remote_path: Destination path in storage
            
        Returns:
            URI of the uploaded artifact
        """
        pass
    
    @abstractmethod
    def download(self, remote_path: str, local_path: Union[str, Path]) -> Path:
        """
        Download a file or directory from storage.
        
        Args:
            remote_path: Path in storage
            local_path: Destination local path
            
        Returns:
            Path to downloaded file/directory
        """
        pass
    
    @abstractmethod
    def exists(self, remote_path: str) -> bool:
        """Check if path exists in storage"""
        pass
    
    @abstractmethod
    def list_files(self, remote_path: str) -> List[str]:
        """List files in a directory"""
        pass
    
    @abstractmethod
    def delete(self, remote_path: str):
        """Delete a file or directory"""
        pass


class LocalStorage(StorageBackend):
    """Local filesystem storage backend"""
    
    def __init__(self, root_dir: Union[str, Path] = "."):
        self.root_dir = Path(root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_full_path(self, remote_path: str) -> Path:
        return self.root_dir / remote_path
    
    def upload(self, local_path: Union[str, Path], remote_path: str) -> str:
        local_path = Path(local_path)
        dest_path = self._get_full_path(remote_path)
        
        if local_path.is_dir():
            if dest_path.exists():
                shutil.rmtree(dest_path)
            shutil.copytree(local_path, dest_path)
        else:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, dest_path)
            
        return str(dest_path)
    
    def download(self, remote_path: str, local_path: Union[str, Path]) -> Path:
        src_path = self._get_full_path(remote_path)
        dest_path = Path(local_path)
        
        if not src_path.exists():
            raise FileNotFoundError(f"File not found in storage: {remote_path}")
            
        if src_path.is_dir():
            if dest_path.exists():
                shutil.rmtree(dest_path)
            shutil.copytree(src_path, dest_path)
        else:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dest_path)
            
        return dest_path
    
    def exists(self, remote_path: str) -> bool:
        return self._get_full_path(remote_path).exists()
    
    def list_files(self, remote_path: str) -> List[str]:
        path = self._get_full_path(remote_path)
        if not path.exists():
            return []
            
        if path.is_file():
            return [remote_path]
            
        files = []
        for p in path.rglob("*"):
            if p.is_file():
                rel_path = p.relative_to(self.root_dir)
                files.append(str(rel_path))
        return files
    
    def delete(self, remote_path: str):
        path = self._get_full_path(remote_path)
        if not path.exists():
            return
            
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


class S3Storage(StorageBackend):
    """AWS S3 storage backend"""
    
    def __init__(self, bucket: str, prefix: str = "", profile_name: Optional[str] = None):
        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError:
            raise ImportError("boto3 required for S3Storage. Install with: pip install boto3")
        
        self.bucket = bucket
        self.prefix = prefix
        self.session = boto3.Session(profile_name=profile_name)
        self.s3 = self.session.client("s3")
        
        # Verify bucket access
        try:
            self.s3.head_bucket(Bucket=bucket)
        except ClientError as e:
            raise ValueError(f"Could not access bucket {bucket}: {e}")

    def _get_key(self, remote_path: str) -> str:
        key = os.path.join(self.prefix, remote_path)
        return key.lstrip("/")  # S3 keys shouldn't start with /
    
    def upload(self, local_path: Union[str, Path], remote_path: str) -> str:
        local_path = Path(local_path)
        key = self._get_key(remote_path)
        
        if local_path.is_file():
            self.s3.upload_file(str(local_path), self.bucket, key)
        else:
            # Upload directory
            for root, _, files in os.walk(local_path):
                for file in files:
                    local_file = os.path.join(root, file)
                    rel_path = os.path.relpath(local_file, local_path)
                    s3_key = os.path.join(key, rel_path)
                    self.s3.upload_file(local_file, self.bucket, s3_key)
        
        return f"s3://{self.bucket}/{key}"
    
    def download(self, remote_path: str, local_path: Union[str, Path]) -> Path:
        key = self._get_key(remote_path)
        dest_path = Path(local_path)
        
        # Check if it looks like a directory or file
        # This is tricky in S3, simplified logic here:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            is_file = True
        except:
            is_file = False
            
        if is_file:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            self.s3.download_file(self.bucket, key, str(dest_path))
        else:
            # Download directory
            paginator = self.s3.get_paginator("list_objects_v2")
            for result in paginator.paginate(Bucket=self.bucket, Prefix=key):
                if "Contents" not in result:
                    continue
                for obj in result["Contents"]:
                    obj_key = obj["Key"]
                    rel_path = os.path.relpath(obj_key, key)
                    local_file = dest_path / rel_path
                    local_file.parent.mkdir(parents=True, exist_ok=True)
                    self.s3.download_file(self.bucket, obj_key, str(local_file))
                    
        return dest_path
    
    def exists(self, remote_path: str) -> bool:
        key = self._get_key(remote_path)
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except:
            # Check if it's a "directory" (prefix)
            resp = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=key, MaxKeys=1)
            return "Contents" in resp
    
    def list_files(self, remote_path: str) -> List[str]:
        key = self._get_key(remote_path)
        files = []
        paginator = self.s3.get_paginator("list_objects_v2")
        
        for result in paginator.paginate(Bucket=self.bucket, Prefix=key):
            if "Contents" in result:
                for obj in result["Contents"]:
                    # Return relative to prefix, not full key
                    rel_path = os.path.relpath(obj["Key"], self.prefix)
                    files.append(rel_path)
                    
        return files
    
    def delete(self, remote_path: str):
        key = self._get_key(remote_path)
        # Delete object or prefix
        self.s3.delete_object(Bucket=self.bucket, Key=key)
        
        # Also clean up "directory"
        paginator = self.s3.get_paginator("list_objects_v2")
        for result in paginator.paginate(Bucket=self.bucket, Prefix=key):
            if "Contents" in result:
                objects = [{"Key": obj["Key"]} for obj in result["Contents"]]
                self.s3.delete_objects(Bucket=self.bucket, Delete={"Objects": objects})


def get_storage_backend(uri: str) -> StorageBackend:
    """
    Factory function to get storage backend from URI.
    
    Examples:
        - "s3://my-bucket/prefix" -> S3Storage
        - "gs://my-bucket/prefix" -> GCSStorage (TODO)
        - "file://./local/path" -> LocalStorage
        - "./local/path" -> LocalStorage
    """
    if uri.startswith("s3://"):
        parts = uri[5:].split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        return S3Storage(bucket, prefix)
    
    elif uri.startswith("gs://"):
        raise NotImplementedError("GCS storage not yet implemented")
        
    elif uri.startswith("file://"):
        return LocalStorage(uri[7:])
        
    else:
        return LocalStorage(uri)

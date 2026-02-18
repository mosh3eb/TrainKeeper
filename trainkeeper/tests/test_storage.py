import os
import shutil
import tempfile
import pytest
from pathlib import Path
from trainkeeper.storage import LocalStorage, get_storage_backend


class TestLocalStorage:
    @pytest.fixture
    def storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield LocalStorage(tmpdir)

    def test_upload_file(self, storage):
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("test content")
            local_path = Path(f.name)
        
        try:
            remote_path = "subdir/test.txt"
            uri = storage.upload(local_path, remote_path)
            
            assert storage.exists(remote_path)
            assert str(storage._get_full_path(remote_path)).endswith("subdir/test.txt")
        finally:
            local_path.unlink()

    def test_upload_directory(self, storage):
        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir)
            (local_dir / "file1.txt").write_text("content1")
            (local_dir / "sub").mkdir()
            (local_dir / "sub" / "file2.txt").write_text("content2")
            
            remote_path = "uploaded_dir"
            storage.upload(local_dir, remote_path)
            
            assert storage.exists(remote_path)
            assert storage.exists("uploaded_dir/file1.txt")
            assert storage.exists("uploaded_dir/sub/file2.txt")

    def test_download_file(self, storage):
        # Setup
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("download me")
            src_path = Path(f.name)
        
        storage.upload(src_path, "remote.txt")
        src_path.unlink()
        
        # Test
        with tempfile.TemporaryDirectory() as tmpdir:
            dest_path = Path(tmpdir) / "downloaded.txt"
            storage.download("remote.txt", dest_path)
            
            assert dest_path.exists()
            assert dest_path.read_text() == "download me"

    def test_list_files(self, storage):
        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = Path(tmpdir)
            (local_dir / "a.txt").touch()
            (local_dir / "b.txt").touch()
            (local_dir / "sub").mkdir()
            (local_dir / "sub" / "c.txt").touch()
            
            storage.upload(local_dir, "list_test")
            
            files = storage.list_files("list_test")
            assert len(files) == 3
            assert "list_test/a.txt" in files
            assert "list_test/sub/c.txt" in files

    def test_delete(self, storage):
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("delete me")
            src_path = Path(f.name)
            
        storage.upload(src_path, "to_delete.txt")
        src_path.unlink()
        
        assert storage.exists("to_delete.txt")
        storage.delete("to_delete.txt")
        assert not storage.exists("to_delete.txt")


def test_get_storage_backend():
    assert isinstance(get_storage_backend("file://./test"), LocalStorage)
    assert isinstance(get_storage_backend("./test"), LocalStorage)
    # S3 test skipped as it requires boto3

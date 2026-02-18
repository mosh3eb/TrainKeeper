import json
import pytest
from pathlib import Path
import tempfile
from trainkeeper.data import DataArtifact, DataMetadata
from trainkeeper.storage import LocalStorage


class TestDataArtifact:
    @pytest.fixture
    def storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield LocalStorage(tmpdir)

    @pytest.fixture
    def sample_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            (data_dir / "train.csv").write_text("col1,col2\n1,2")
            (data_dir / "meta.json").write_text('{"key": "value"}')
            yield data_dir

    def test_create_artifact(self, sample_data):
        artifact = DataArtifact.create(
            name="test-dataset",
            source_path=sample_data,
            description="Test dataset",
            tags=["csv", "test"]
        )
        
        assert artifact.metadata.name == "test-dataset"
        assert artifact.metadata.version is not None
        assert artifact.metadata.num_files == 2
        assert "csv" in artifact.metadata.tags
        assert artifact.local_path.resolve() == sample_data.resolve()

    def test_hashing_consistency(self, sample_data):
        """Ensure hashing is deterministic"""
        artifact1 = DataArtifact.create("d1", sample_data)
        artifact2 = DataArtifact.create("d2", sample_data)
        
        assert artifact1.metadata.hash == artifact2.metadata.hash
        assert artifact1.metadata.version == artifact2.metadata.version

    def test_save_and_load(self, sample_data, storage):
        # Create and save
        original = DataArtifact.create("saved-data", sample_data)
        uri = original.save(storage)
        
        # Load metadata
        loaded = DataArtifact.load(
            name="saved-data",
            version=original.metadata.version,
            backend=storage
        )
        
        assert loaded.metadata.hash == original.metadata.hash
        assert loaded.metadata.size_bytes == original.metadata.size_bytes
        
        # Download data
        with tempfile.TemporaryDirectory() as tmpdir:
            download_path = Path(tmpdir) / "downloaded"
            loaded.download(storage, download_path)
            
            assert (download_path / "train.csv").exists()
            assert (download_path / "train.csv").read_text() == "col1,col2\n1,2"

    def test_lineage(self, sample_data):
        parent = DataArtifact.create("parent", sample_data)
        
        child_data = sample_data  # modifying slightly in real world
        child = DataArtifact.create(
            name="child", 
            source_path=child_data, 
            parent_artifacts=[parent]
        )
        
        parent_key = f"{parent.metadata.name}:{parent.metadata.version}"
        assert parent_key in child.metadata.lineage
        assert child.metadata.lineage[parent_key] == "derived_from"

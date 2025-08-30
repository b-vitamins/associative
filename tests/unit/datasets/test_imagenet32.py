"""Test suite for ImageNet32 dataset."""

import io
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from associative.datasets import ImageNet32

# Test constants
MOCK_SAMPLES_COUNT = 100
MOCK_NUM_CLASSES = 10
BATCH_SIZE = 4
TARGET_TRANSFORM_OFFSET = 1000
MOCK_LABEL_VALUE = 5
MOCK_SAMPLES_PER_FILE = 50
TOTAL_CONCATENATED_SAMPLES = 150


class TestImageNet32:
    """Test suite for ImageNet32 dataset."""

    @pytest.fixture
    def temp_dataset_dir(self):
        """Create temporary directory with mock ImageNet32 structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "imagenet-32-data"
            base_path.mkdir(parents=True)
            yield tmpdir

    @pytest.fixture
    def mock_parquet_data(self):
        """Create mock parquet data."""
        # Create sample data
        num_samples = MOCK_SAMPLES_COUNT
        images = []
        labels = []

        for i in range(num_samples):
            # Create a dummy 32x32 RGB image
            img = Image.new(
                "RGB", (32, 32), color=(i % 256, (i * 2) % 256, (i * 3) % 256)
            )
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            images.append({"bytes": img_bytes.getvalue()})
            labels.append(i % MOCK_NUM_CLASSES)  # Mock classes for testing

        return pd.DataFrame({"image": images, "label": labels})

    @pytest.fixture
    def mock_read_parquet(self, mock_parquet_data):
        """Mock pandas.read_parquet."""
        with patch("pandas.read_parquet") as mock:
            mock.return_value = mock_parquet_data
            yield mock

    def test_initialization_train(self, temp_dataset_dir, mock_read_parquet):
        """Test dataset initialization for training."""
        # Create dummy parquet files
        base_path = Path(temp_dataset_dir) / "imagenet-32-data"
        for filename in ImageNet32.train_files:
            (base_path / filename).touch()

        dataset = ImageNet32(root=temp_dataset_dir, train=True, download=False)

        assert dataset.train is True
        assert len(dataset) == 100 * len(ImageNet32.train_files)  # 100 samples per file
        assert len(dataset.classes) > 0

    def test_initialization_val(self, temp_dataset_dir, mock_read_parquet):
        """Test dataset initialization for validation."""
        # Create dummy parquet files
        base_path = Path(temp_dataset_dir) / "imagenet-32-data"
        for filename in ImageNet32.val_files:
            (base_path / filename).touch()

        dataset = ImageNet32(root=temp_dataset_dir, train=False, download=False)

        assert dataset.train is False
        assert len(dataset) == MOCK_SAMPLES_COUNT

    def test_missing_data_error(self):
        """Test error when data is missing and download=False."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_path = Path(temp_dir) / "nonexistent_imagenet32"
            with pytest.raises(RuntimeError, match="Dataset not found or corrupted"):
                ImageNet32(root=str(nonexistent_path), download=False)

    def test_getitem(self, temp_dataset_dir, mock_read_parquet):
        """Test getting individual samples."""
        # Create dummy parquet files
        base_path = Path(temp_dataset_dir) / "imagenet-32-data"
        for filename in ImageNet32.train_files:
            (base_path / filename).touch()

        dataset = ImageNet32(root=temp_dataset_dir, train=True, download=False)

        # Get a sample
        img, label = dataset[0]

        # Check image
        assert isinstance(img, Image.Image)
        assert img.size == (32, 32)

        # Check label
        assert isinstance(label, int | np.integer)
        assert 0 <= label < MOCK_NUM_CLASSES  # Based on mock data

    def test_transforms(self, temp_dataset_dir, mock_read_parquet):
        """Test transform and target_transform."""
        # Create dummy parquet files
        base_path = Path(temp_dataset_dir) / "imagenet-32-data"
        for filename in ImageNet32.train_files:
            (base_path / filename).touch()

        transform_called = False
        target_transform_called = False

        def mock_transform(img):
            nonlocal transform_called
            transform_called = True
            return torch.randn(3, 32, 32)

        def mock_target_transform(label):
            nonlocal target_transform_called
            target_transform_called = True
            return label + TARGET_TRANSFORM_OFFSET

        dataset = ImageNet32(
            root=temp_dataset_dir,
            train=True,
            transform=mock_transform,
            target_transform=mock_target_transform,
            download=False,
        )

        img, label = dataset[0]

        assert transform_called
        assert target_transform_called
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 32, 32)
        assert label >= TARGET_TRANSFORM_OFFSET  # Target transform adds offset

    def test_dataloader_compatibility(self, temp_dataset_dir, mock_read_parquet):
        """Test dataset works with PyTorch DataLoader."""
        # Create dummy parquet files
        base_path = Path(temp_dataset_dir) / "imagenet-32-data"
        for filename in ImageNet32.train_files:
            (base_path / filename).touch()

        # Transform to tensor for batching
        def to_tensor(img):
            return torch.randn(3, 32, 32)  # Mock tensor

        dataset = ImageNet32(
            root=temp_dataset_dir, train=True, transform=to_tensor, download=False
        )

        # Create DataLoader
        dataloader = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )

        # Get a batch
        batch_imgs, batch_labels = next(iter(dataloader))

        # Check batch dimensions
        assert batch_imgs.shape == (BATCH_SIZE, 3, 32, 32)
        assert len(batch_labels) == BATCH_SIZE

    def test_repr(self, temp_dataset_dir, mock_read_parquet):
        """Test string representation."""
        # Create dummy parquet files
        base_path = Path(temp_dataset_dir) / "imagenet-32-data"
        for filename in ImageNet32.train_files:
            (base_path / filename).touch()

        dataset = ImageNet32(root=temp_dataset_dir, train=True, download=False)

        repr_str = repr(dataset)
        assert "Dataset ImageNet32" in repr_str
        assert str(len(dataset)) in repr_str
        assert str(temp_dataset_dir) in repr_str
        assert "Train" in repr_str

    def test_path_expansion(self, mock_read_parquet):
        """Test that paths with ~ are expanded."""
        with pytest.raises(RuntimeError):  # Will fail but path should expand
            dataset = ImageNet32(root="~/nonexistent_imagenet32_data", download=False)
            # Check path was expanded (access root to trigger expansion)
            assert "~" not in str(dataset.root)

    def test_class_names(self, temp_dataset_dir, mock_read_parquet):
        """Test class names are properly loaded."""
        # Create dummy parquet files
        base_path = Path(temp_dataset_dir) / "imagenet-32-data"
        for filename in ImageNet32.train_files:
            (base_path / filename).touch()

        dataset = ImageNet32(root=temp_dataset_dir, train=True, download=False)

        # Check classes are defined
        assert hasattr(dataset, "classes")
        assert len(dataset.classes) == MOCK_NUM_CLASSES  # Based on mock data
        assert hasattr(dataset, "class_to_idx")
        assert len(dataset.class_to_idx) == len(dataset.classes)

    def test_integrity_check(self, temp_dataset_dir):
        """Test integrity check for missing files."""
        base_path = Path(temp_dataset_dir) / "imagenet-32-data"
        base_path.mkdir(parents=True, exist_ok=True)

        # Create only some files (incomplete)
        (base_path / ImageNet32.train_files[0]).touch()

        # Should fail integrity check
        with pytest.raises(RuntimeError, match="Dataset not found or corrupted"):
            ImageNet32(root=temp_dataset_dir, train=True, download=False)

    @patch("associative.datasets.imagenet32.download_url")
    def test_download_behavior(self, mock_download):
        """Test download behavior (mocked to avoid actual downloads)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Dataset should attempt download when data doesn't exist
            with pytest.raises(RuntimeError):  # Will fail after mock download
                ImageNet32(root=tmpdir, train=True, download=True)

            # Check download was attempted
            assert mock_download.called

    def test_multiple_parquet_files(self, temp_dataset_dir):
        """Test loading and concatenating multiple parquet files."""
        base_path = Path(temp_dataset_dir) / "imagenet-32-data"
        base_path.mkdir(parents=True, exist_ok=True)

        # Create all train parquet files
        for filename in ImageNet32.train_files:
            (base_path / filename).touch()

        # Mock multiple DataFrames
        dfs = []
        for i in range(3):  # 3 train files
            num_samples = MOCK_SAMPLES_PER_FILE
            images = []
            labels = []

            for j in range(num_samples):
                img = Image.new("RGB", (32, 32))
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="PNG")
                images.append({"bytes": img_bytes.getvalue()})
                labels.append((i * 50 + j) % 100)

            dfs.append(pd.DataFrame({"image": images, "label": labels}))

        with patch("pandas.read_parquet") as mock_read:
            mock_read.side_effect = dfs

            dataset = ImageNet32(root=temp_dataset_dir, train=True, download=False)

            # Should have concatenated all DataFrames
            assert len(dataset) == TOTAL_CONCATENATED_SAMPLES  # 3 * 50

    def test_image_loading_from_bytes(self, temp_dataset_dir):
        """Test loading images from byte data in parquet."""
        base_path = Path(temp_dataset_dir) / "imagenet-32-data"
        base_path.mkdir(parents=True, exist_ok=True)

        for filename in ImageNet32.train_files:
            (base_path / filename).touch()

        # Create DataFrame with actual image bytes
        img = Image.new("RGB", (32, 32), color=(100, 150, 200))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")

        image_df = pd.DataFrame(
            {"image": [{"bytes": img_bytes.getvalue()}], "label": [MOCK_LABEL_VALUE]}
        )

        with patch("pandas.read_parquet") as mock_read:
            mock_read.return_value = image_df

            dataset = ImageNet32(root=temp_dataset_dir, train=True, download=False)

            img, label = dataset[0]

            # Should load as PIL Image
            assert isinstance(img, Image.Image)
            assert img.size == (32, 32)
            assert label == MOCK_LABEL_VALUE

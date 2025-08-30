"""ImageNet-1K 32x32 Dataset.

A downsampled version of ImageNet with 32x32 pixel images,
similar to CIFAR-10/100 format. Based on the dataset from:
https://huggingface.co/datasets/benjamin-paine/imagenet-1k-32x32
"""

import io
import os
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def download_url(url: str, root: str, filename: str | None = None) -> None:
    """Download a file from a url and place it in root."""
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    if os.path.isfile(fpath):
        print(f"Using cached file {filename}")
        return

    print(f"Downloading {url} to {fpath}")

    # Strictly validate URL scheme for security
    parsed_url = urllib.parse.urlparse(url)
    if parsed_url.scheme not in ("http", "https"):
        raise ValueError(
            f"URL scheme '{parsed_url.scheme}' is not allowed. Only 'http' and 'https' are permitted."
        )

    # Additional validation: ensure URL starts with allowed prefixes
    if not url.startswith(("http://", "https://")):
        raise ValueError("URL must start with 'http://' or 'https://'")

    # At this point we've validated the URL is safe (http/https only)
    # S310 is about auditing URL schemes, which we've done above
    with urllib.request.urlopen(url) as response, open(fpath, "wb") as out_file:  # noqa: S310
        out_file.write(response.read())


class ImageNet32(Dataset):
    """ImageNet-1K 32x32 Dataset.

    Args:
        root: Root directory where dataset exists or will be saved
        train: If True, creates dataset from training set, otherwise from validation set
        transform: A function/transform that takes in a PIL image and returns a transformed version
        target_transform: A function/transform that takes in the target and transforms it
        download: If true, downloads the dataset from the internet
    """

    base_folder = "imagenet-32-data"
    base_url = "https://huggingface.co/datasets/benjamin-paine/imagenet-1k-32x32/resolve/main/data"

    # Training data is split into 3 parquet files
    train_files: ClassVar[list[str]] = [
        "train-00000-of-00003.parquet",
        "train-00001-of-00003.parquet",
        "train-00002-of-00003.parquet",
    ]

    # Validation data
    val_files: ClassVar[list[str]] = [
        "validation-00000-of-00001.parquet",
    ]

    # Test data
    test_files: ClassVar[list[str]] = [
        "test-00000-of-00001.parquet",
    ]

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it"
            )

        # Load data from parquet files
        self._load_data()
        self._load_meta()

    def _load_data(self) -> None:
        """Load data from parquet files."""
        files = self.train_files if self.train else self.val_files
        data_frames = []

        for filename in files:
            file_path = os.path.join(self.root, self.base_folder, filename)
            if os.path.exists(file_path):
                dataframe = pd.read_parquet(file_path)
                data_frames.append(dataframe)

        if not data_frames:
            raise RuntimeError("No data files found")

        # Combine all dataframes
        self.df = pd.concat(data_frames, ignore_index=True)

    def _load_meta(self) -> None:
        """Load metadata from the dataset."""
        # Use standard ImageNet-1K class names (1000 classes)
        # These correspond to the label indices in the parquet files
        self.classes = [
            "tench",
            "goldfish",
            "great white shark",
            "tiger shark",
            "hammerhead",
            # ... (we'll use simplified class names for now)
        ]

        # If we have data loaded, get actual class count from labels
        if hasattr(self, "df") and not self.df.empty:
            unique_labels = sorted(self.df["label"].unique())
            num_classes = len(unique_labels)

            # Create simplified class names
            self.classes = [f"class_{i}" for i in range(num_classes)]
        else:
            # Default to 1000 ImageNet classes
            self.classes = [f"class_{i}" for i in range(1000)]

        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """Get item at index."""
        row = self.df.iloc[index]

        # Get image from parquet data
        image_data = row["image"]
        target = int(row["label"])

        # Image is stored as a dict with 'bytes' key containing raw image bytes
        if isinstance(image_data, dict) and "bytes" in image_data:
            img_bytes = image_data["bytes"]
            img = Image.open(io.BytesIO(img_bytes))
        else:
            # Fallback if image is stored directly
            img = image_data

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.df)

    def _check_integrity(self) -> bool:
        """Check if dataset files exist."""
        root = Path(self.root)
        base_path = root / self.base_folder
        if not base_path.exists():
            return False

        # Check if we have the expected parquet files
        files = self.train_files if self.train else self.val_files
        return all((base_path / filename).exists() for filename in files)

    def download(self) -> None:
        """Download the dataset."""
        if self._check_integrity():
            return

        # Create data directory
        data_dir = Path(self.root) / self.base_folder
        data_dir.mkdir(parents=True, exist_ok=True)

        # Download parquet files
        files = self.train_files if self.train else self.val_files
        for filename in files:
            url = f"{self.base_url}/{filename}"
            download_url(url, str(data_dir), filename)

    def __repr__(self) -> str:
        """String representation."""
        head = "Dataset ImageNet32"
        body = [f"    Number of datapoints: {self.__len__()}"]
        body.append(f"    Root location: {self.root}")
        body.append(f"    Split: {'Train' if self.train else 'Val'}")
        lines = [head, *body]
        return "\n".join(lines)

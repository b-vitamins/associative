#!/usr/bin/env python3
"""Download the complete MovieChat-1K dataset."""

import os
import sys
from pathlib import Path
from typing import Literal

# Add project root to path
sys.path.append(str(Path(__file__).parents[3]))

from associative.datasets.moviechat import MovieChat1K


def main() -> None:
    """Download the full MovieChat-1K dataset."""
    # Check HF token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN environment variable is required.\n"
            "Please set it with: export HF_TOKEN='your_token_here'\n"
            "Get your token from: https://huggingface.co/settings/tokens"
        )

    splits: list[Literal["train", "test"]] = ["train", "test"]
    for split in splits:
        try:
            # Create dataset with download=True (uses default root=None for XDG cache)
            MovieChat1K(
                split=split,
                num_frames=512,  # Arbitrary, not used for download
                download=True,
            )
        except Exception as e:
            print(f"Error downloading {split} split: {e}")
            continue


if __name__ == "__main__":
    main()

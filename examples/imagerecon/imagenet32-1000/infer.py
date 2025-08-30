#!/usr/bin/env python3
"""Inference script for Image Reconstruction on ImageNet-32 - reconstruct individual images."""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parents[3]))

import argparse
from typing import cast

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms
from train import setup_model

from associative.datasets.utils import generate_mask_indices


def load_and_preprocess_image(image_path: str, mean: tuple, std: tuple) -> torch.Tensor:
    """Load and preprocess an image for inference."""
    # Load image
    if image_path == "random":
        # Generate random ImageNet-32 like image
        img_array = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        pil_image = Image.fromarray(img_array)
    else:
        pil_image = Image.open(image_path).convert("RGB")

    # Define transforms
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    # Transform returns a tensor, then unsqueeze adds batch dimension
    tensor_image = cast(torch.Tensor, transform(pil_image))
    return tensor_image.unsqueeze(0)


def reconstruct_image(
    model: torch.nn.Module,
    image: torch.Tensor,
    mask_ratio: float,
    num_patches: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruct a masked image."""
    model.eval()

    with torch.no_grad():
        # Generate mask
        batch_idx, mask_idx = generate_mask_indices(
            1, num_patches, mask_ratio, device=device
        )

        # Create masked patches for visualization
        patches = model.patch_embed.to_patches(image)  # type: ignore
        masked_patches = patches.clone()
        masked_patches[batch_idx, mask_idx] = 0.0
        masked_image = model.patch_embed.from_patches(masked_patches)  # type: ignore

        # Get reconstruction
        output = model(image, mask=(batch_idx, mask_idx), alpha=1.0)
        reconstruction = model.patch_embed.from_patches(output)  # type: ignore

    return image, masked_image, reconstruction


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main inference function."""
    # Parse additional arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image", type=str, default="random", help='Path to input image or "random"'
    )
    parser.add_argument(
        "--mask-ratio", type=float, default=None, help="Masking ratio (0-1)"
    )
    parser.add_argument(
        "--output", type=str, default="reconstruction.png", help="Output path"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = setup_model(cfg)
    model = model.to(device)

    # Load checkpoint
    checkpoint_path = Path(cfg.checkpoint_dir) / "best_model.pth"
    if not checkpoint_path.exists():
        checkpoint_path = Path(cfg.checkpoint_dir) / "latest.pth"

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded model from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return

    # Dataset stats for normalization (ImageNet stats)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # Load and preprocess image
    image = load_and_preprocess_image(args.image, mean, std).to(device)

    # Use provided mask ratio or default from config
    mask_ratio = args.mask_ratio if args.mask_ratio is not None else cfg.data.mask_ratio

    # Reconstruct
    original, masked, reconstructed = reconstruct_image(
        model, image, mask_ratio, cfg.model.num_patches, device
    )

    # Unnormalize for visualization
    def unnormalize(x):
        x = x.clone()
        for i, (m, s) in enumerate(zip(mean, std, strict=False)):
            x[:, i] = x[:, i] * s + m
        return x.clamp(0, 1)

    original = unnormalize(original.cpu())
    masked = unnormalize(masked.cpu())
    reconstructed = unnormalize(reconstructed.cpu())

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(original[0].permute(1, 2, 0))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(masked[0].permute(1, 2, 0))
    axes[1].set_title(f"Masked ({int(mask_ratio * 100)}%)")
    axes[1].axis("off")

    axes[2].imshow(reconstructed[0].permute(1, 2, 0))
    axes[2].set_title("Reconstructed")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved reconstruction to {args.output}")

    # Also save individual images
    output_dir = Path(args.output).parent
    transforms.ToPILImage()(original[0]).save(output_dir / "original.png")
    transforms.ToPILImage()(masked[0]).save(output_dir / "masked.png")
    transforms.ToPILImage()(reconstructed[0]).save(output_dir / "reconstructed.png")
    print(f"Saved individual images to {output_dir}")


if __name__ == "__main__":
    main()

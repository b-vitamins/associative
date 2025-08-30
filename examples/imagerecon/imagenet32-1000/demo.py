#!/usr/bin/env python3
"""Demo script showing Image Reconstruction with different masking ratios on ImageNet-32."""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parents[3]))

import matplotlib
import numpy as np
import torch
from torchvision import transforms

matplotlib.use("Agg")  # Non-interactive backend
import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from train import setup_model

from associative.datasets.imagenet32 import ImageNet32
from associative.datasets.utils import generate_mask_indices


def unnormalize(tensor, mean, std):
    """Unnormalize a tensor with given mean and std."""
    tensor = tensor.clone()
    for i, (m, s) in enumerate(zip(mean, std, strict=False)):
        tensor[:, i] = tensor[:, i] * s + m
    return tensor.clamp(0, 1)


def create_demo_grid(model, dataset, device, cfg, **kwargs):
    """Create a demo grid showing different masking ratios."""
    mean = kwargs.get("mean", (0.485, 0.456, 0.406))
    std = kwargs.get("std", (0.229, 0.224, 0.225))
    model.eval()

    # Define masking ratios to demonstrate
    mask_ratios = [0.0, 0.25, 0.5, 0.75, 0.85, 0.95]
    num_samples = 4

    # Select random images
    indices = torch.randperm(len(dataset))[:num_samples]

    # Create figure
    fig, axes = plt.subplots(num_samples, len(mask_ratios) + 1, figsize=(20, 12))

    with torch.no_grad():
        for row, idx in enumerate(indices):
            # Get image
            image, _ = dataset[idx]
            image = image.unsqueeze(0).to(device)

            # Show original
            orig = unnormalize(image.cpu(), mean, std)
            axes[row, 0].imshow(orig[0].permute(1, 2, 0).clamp(0, 1))
            axes[row, 0].axis("off")
            if row == 0:
                axes[row, 0].set_title("Original", fontsize=14, fontweight="bold")

            # Process with different mask ratios
            for col, mask_ratio in enumerate(mask_ratios):
                # Generate mask
                batch_idx, mask_idx = generate_mask_indices(
                    1,
                    cfg.model.num_patches,
                    mask_ratio,
                    device=device,
                )

                # Get reconstruction
                output = model(
                    image,
                    mask=(batch_idx, mask_idx),
                    alpha=cfg.model.step_size,
                )
                reconstruction = model.patch_embed.from_patches(output)

                # Unnormalize and display
                recon = unnormalize(reconstruction.cpu(), mean, std)
                axes[row, col + 1].imshow(recon[0].permute(1, 2, 0).clamp(0, 1))
                axes[row, col + 1].axis("off")
                if row == 0:
                    axes[row, col + 1].set_title(
                        f"{int(mask_ratio * 100)}% masked", fontsize=12
                    )

    plt.suptitle(
        "Image Reconstruction Reconstruction with Different Masking Ratios (ImageNet-32)",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    # Save figure
    viz_dir = Path(cfg.visualization_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)
    output_path = viz_dir / "imagerecon_demo_grid.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def create_progressive_masking_demo(model, dataset, device, cfg, **kwargs):
    """Create a figure showing progressive masking on a single image."""
    mean = kwargs.get("mean", (0.485, 0.456, 0.406))
    std = kwargs.get("std", (0.229, 0.224, 0.225))
    model.eval()

    # Get a good example image
    image_idx = 42  # Fixed seed for reproducibility
    image, _ = dataset[image_idx]
    image = image.unsqueeze(0).to(device)

    # Define fine-grained masking ratios
    mask_ratios = np.linspace(0, 0.95, 12)

    # Create figure
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    with torch.no_grad():
        for idx, mask_ratio in enumerate(mask_ratios):
            # Generate mask
            batch_idx, mask_idx = generate_mask_indices(
                1,
                cfg.model.num_patches,
                mask_ratio,
                device=device,
            )

            # Create masked version for visualization
            patches = model.patch_embed.to_patches(image)
            masked_patches = patches.clone()
            masked_patches[batch_idx, mask_idx] = 0.0
            masked_image = model.patch_embed.from_patches(masked_patches)

            # Get reconstruction
            output = model(
                image,
                mask=(batch_idx, mask_idx),
                alpha=cfg.model.step_size,
            )
            reconstruction = model.patch_embed.from_patches(output)

            # Create side-by-side view
            masked_vis = (
                unnormalize(masked_image.cpu(), mean, std)[0]
                .permute(1, 2, 0)
                .clamp(0, 1)
            )
            recon_vis = (
                unnormalize(reconstruction.cpu(), mean, std)[0]
                .permute(1, 2, 0)
                .clamp(0, 1)
            )

            # Concatenate masked and reconstructed
            combined = torch.cat([masked_vis, recon_vis], dim=1)

            axes[idx].imshow(combined)
            axes[idx].axis("off")
            axes[idx].set_title(f"{int(mask_ratio * 100)}% masked", fontsize=10)

    plt.suptitle(
        "Progressive Masking: Masked Input (left) vs Reconstruction (right) - ImageNet-32",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    # Save figure
    viz_dir = Path(cfg.visualization_dir)
    output_path = viz_dir / "imagerecon_progressive_masking.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    """Create Image Reconstruction demonstration figures."""
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
        print("No checkpoint found. Please train a model first.")
        return

    model.eval()

    # Load dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = ImageNet32(
        root=cfg.data.root,
        train=False,
        transform=transform,
        download=True,
    )

    print("\nGenerating Image Reconstruction demonstrations...")
    print("=" * 50)

    # ImageNet normalization stats
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # Create demo grid
    grid_path = create_demo_grid(model, dataset, device, cfg, mean=mean, std=std)
    print(f"✓ Demo grid saved to: {grid_path}")

    # Create progressive masking demo
    prog_path = create_progressive_masking_demo(
        model, dataset, device, cfg, mean=mean, std=std
    )
    print(f"✓ Progressive masking demo saved to: {prog_path}")

    print("\nDemo generation complete!")


if __name__ == "__main__":
    main()

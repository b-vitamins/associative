#!/usr/bin/env python3
"""Visualize Image Reconstruction reconstructions for ImageNet-32."""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parents[3]))

import hydra
import matplotlib
import torch
from omegaconf import DictConfig
from torchvision import transforms
from torchvision.utils import save_image

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from train import setup_model

from associative.datasets.imagenet32 import ImageNet32
from associative.datasets.utils import generate_mask_indices


def unnormalize(tensor, mean, std):
    """Unnormalize a tensor with given mean and std."""
    tensor = tensor.clone()
    for i, (m, s) in enumerate(zip(mean, std, strict=False)):
        tensor[:, i] = tensor[:, i] * s + m
    return tensor.clamp(0, 1)


def load_model_checkpoint(cfg, device):
    """Load model and checkpoint."""
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
        return None

    model.eval()
    return model


def generate_reconstructions(model, dataset, cfg, device, num_samples):
    """Generate reconstructions for visualization."""
    indices = torch.randperm(len(dataset))[:num_samples]

    images = []
    masked_images = []
    reconstructions = []

    with torch.no_grad():
        for idx in indices:
            # Get image
            image, _ = dataset[idx]
            image = image.unsqueeze(0).to(device)

            # Generate mask
            batch_idx, mask_idx = generate_mask_indices(
                1,
                cfg.model.num_patches,
                cfg.data.mask_ratio,
                device=device,
            )

            # Create masked patches for visualization
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

            # Collect for visualization
            images.append(image)
            masked_images.append(masked_image)
            reconstructions.append(reconstruction)

    return images, masked_images, reconstructions


def save_visualizations(images, masked_images, reconstructions, **kwargs):
    """Save visualization figures."""
    mean = kwargs["mean"]
    std = kwargs["std"]
    cfg = kwargs["cfg"]
    num_samples = kwargs["num_samples"]

    # Stack and unnormalize
    images = torch.cat(images, dim=0)
    masked_images = torch.cat(masked_images, dim=0)
    reconstructions = torch.cat(reconstructions, dim=0)

    images = unnormalize(images, mean, std)
    masked_images = unnormalize(masked_images, mean, std)
    reconstructions = unnormalize(reconstructions, mean, std)

    # Create grid visualization
    resize = transforms.Resize((128, 128), antialias=True)
    vis_tensor = torch.cat(
        [resize(images), resize(masked_images), resize(reconstructions)], dim=0
    )

    # Save grid
    viz_dir = Path(cfg.visualization_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)
    output_path = viz_dir / "imagerecon_reconstructions.png"

    save_image(
        vis_tensor,
        output_path,
        nrow=num_samples,
        normalize=True,
        value_range=(0, 1),
    )
    print(f"Saved visualization to {output_path}")

    # Create matplotlib figure
    create_comparison_figure(
        images,
        masked_images,
        reconstructions,
        cfg=cfg,
        num_samples=num_samples,
        viz_dir=viz_dir,
    )


def create_comparison_figure(images, masked_images, reconstructions, **kwargs):
    """Create comparison figure with matplotlib."""
    cfg = kwargs["cfg"]
    num_samples = kwargs["num_samples"]
    viz_dir = kwargs["viz_dir"]

    fig, axes = plt.subplots(3, num_samples, figsize=(2 * num_samples, 6))

    for i in range(num_samples):
        # Original
        axes[0, i].imshow(images[i].cpu().permute(1, 2, 0).clamp(0, 1))
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original", fontsize=12)

        # Masked
        axes[1, i].imshow(masked_images[i].cpu().permute(1, 2, 0).clamp(0, 1))
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title(
                f"Masked ({int(cfg.data.mask_ratio * 100)}%)", fontsize=12
            )

        # Reconstructed
        axes[2, i].imshow(reconstructions[i].cpu().permute(1, 2, 0).clamp(0, 1))
        axes[2, i].axis("off")
        if i == 0:
            axes[2, i].set_title("Reconstructed", fontsize=12)

    plt.suptitle(
        "Image Reconstruction Reconstructions - ImageNet-32",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    comparison_path = viz_dir / "imagerecon_comparison.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison figure to {comparison_path}")


def visualize_reconstructions(cfg: DictConfig) -> None:
    """Visualize Image Reconstruction reconstructions from a trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model_checkpoint(cfg, device)
    if model is None:
        return

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

    # Create visualizations
    num_samples = 8
    images, masked_images, reconstructions = generate_reconstructions(
        model, dataset, cfg, device, num_samples
    )

    # ImageNet normalization stats
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # Save visualizations
    save_visualizations(
        images,
        masked_images,
        reconstructions,
        mean=mean,
        std=std,
        cfg=cfg,
        num_samples=num_samples,
    )


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for visualization."""
    visualize_reconstructions(cfg)


if __name__ == "__main__":
    main()

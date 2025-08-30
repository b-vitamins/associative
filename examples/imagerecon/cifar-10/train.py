#!/usr/bin/env python3
"""Train Energy Transformer with Image Reconstruction on CIFAR-10."""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parents[3]))

import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.nn import functional
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from associative import EnergyTransformer, EnergyTransformerConfig
from associative.datasets.utils import generate_mask_indices
from associative.utils.training import calculate_reconstruction_metrics


def setup_model(cfg):
    """Create and initialize model."""
    config = EnergyTransformerConfig(
        patch_size=cfg.model.patch_size,
        num_patches=cfg.model.num_patches,
        embed_dim=cfg.model.embed_dim,
        num_heads=cfg.model.num_heads,
        qk_dim=cfg.model.qk_dim,
        mlp_ratio=cfg.model.mlp_ratio,
        num_layers=cfg.model.num_layers,
        num_time_steps=cfg.model.num_time_steps,
        step_size=cfg.model.step_size,
        norm_eps=cfg.model.norm_eps,
        attn_bias=cfg.model.attn_bias,
        mlp_bias=cfg.model.mlp_bias,
        attn_beta=cfg.model.attn_beta,
    )
    return EnergyTransformer(config)


def train_epoch(model, dataloader, optimizer, cfg, device, **kwargs):
    """Train for one epoch."""
    epoch = kwargs.get("epoch", 1)
    epoch_start = kwargs.get("epoch_start", time.time())
    model.train()

    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    total_batches = len(dataloader)

    # CIFAR-10 normalization stats
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    for batch_idx, (batch_images, _) in enumerate(dataloader):
        images = batch_images.to(device)
        batch_size = images.shape[0]

        # Generate mask indices
        batch_idx_mask, mask_idx = generate_mask_indices(
            batch_size,
            cfg.model.num_patches,
            cfg.data.mask_ratio,
            device=device,
        )
        mask = (batch_idx_mask, mask_idx)

        # Forward pass
        optimizer.zero_grad()
        output = model(
            images,
            mask=mask,
            alpha=cfg.model.step_size,
        )

        # Get target patches
        target_patches = model.patch_embed.to_patches(images)
        target_pixels = target_patches.reshape(batch_size, cfg.model.num_patches, -1)

        # Compute loss only on masked patches
        masked_output = output[batch_idx_mask, mask_idx]
        masked_target = target_pixels[batch_idx_mask, mask_idx]
        loss = functional.mse_loss(masked_output, masked_target)

        # Backward pass
        loss.backward()
        if cfg.train.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.gradient_clip)
        optimizer.step()

        # Calculate metrics every few batches
        if batch_idx % 10 == 0:
            with torch.no_grad():
                reconstructed = model.patch_embed.from_patches(output)
                metrics = calculate_reconstruction_metrics(
                    reconstructed, images, mean, std
                )
                if batch_idx == 0:
                    running_psnr = metrics["psnr"]
                    running_ssim = metrics["ssim"]
                else:
                    running_psnr = 0.9 * running_psnr + 0.1 * metrics["psnr"]
                    running_ssim = 0.9 * running_ssim + 0.1 * metrics["ssim"]

        # Update running loss
        if batch_idx == 0:
            running_loss = loss.detach()
        else:
            running_loss = 0.9 * running_loss + 0.1 * loss.detach()

        # Real-time progress display
        elapsed = time.time() - epoch_start
        lr = optimizer.param_groups[0]["lr"]
        sys.stdout.write(
            f"\rEpoch {epoch:3d}/{cfg.train.epochs} | Batch {batch_idx + 1:4d}/{total_batches} | "
            f"MSE: {running_loss:.6f} | PSNR: {running_psnr:.1f}dB | SSIM: {running_ssim:.4f} | "
            f"LR: {lr:.0e} | t: {elapsed:3.0f}s"
        )
        sys.stdout.flush()

    return running_loss


def validate(model, dataloader, cfg, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_samples = 0
    total_psnr = 0
    total_ssim = 0
    metric_samples = 0

    # CIFAR-10 normalization stats
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    with torch.no_grad():
        for _batch_idx, (batch_images, _) in enumerate(dataloader):
            images = batch_images.to(device)
            batch_size = images.shape[0]

            # Generate mask indices
            batch_idx_mask, mask_idx = generate_mask_indices(
                batch_size,
                cfg.model.num_patches,
                cfg.data.mask_ratio,
                device=device,
            )
            mask = (batch_idx_mask, mask_idx)

            # Forward pass
            output = model(
                images,
                mask=mask,
                alpha=cfg.model.step_size,
            )

            # Get target patches
            target_patches = model.patch_embed.to_patches(images)
            target_pixels = target_patches.reshape(
                batch_size, cfg.model.num_patches, -1
            )

            # Compute loss
            masked_output = output[batch_idx_mask, mask_idx]
            masked_target = target_pixels[batch_idx_mask, mask_idx]
            loss = functional.mse_loss(masked_output, masked_target)

            total_loss += loss.detach() * batch_size
            total_samples += batch_size

            # Calculate PSNR and SSIM on all validation batches (smaller dataset)
            reconstructed = model.patch_embed.from_patches(output)
            metrics = calculate_reconstruction_metrics(reconstructed, images, mean, std)

            total_psnr += metrics["psnr"] * batch_size
            total_ssim += metrics["ssim"] * batch_size
            metric_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_psnr = total_psnr / metric_samples if metric_samples > 0 else 0.0
    avg_ssim = total_ssim / metric_samples if metric_samples > 0 else 0.0

    return {"loss": avg_loss, "psnr": avg_psnr, "ssim": avg_ssim}


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training CIFAR-10 Image Reconstruction on {device}")

    # Set random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load data
    from torchvision import transforms
    from torchvision.datasets import CIFAR10

    from associative.datasets.utils import DATASET_STATS

    # Get normalization stats
    mean = DATASET_STATS["cifar10"]["mean"]
    std = DATASET_STATS["cifar10"]["std"]

    # Define transforms
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    # Create datasets
    train_dataset = CIFAR10(
        root=cfg.data.root,
        train=True,
        download=True,
        transform=train_transform,
    )

    val_dataset = CIFAR10(
        root=cfg.data.root,
        train=False,
        download=True,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    # Setup model
    model = setup_model(cfg).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(
        f"Training samples: {len(train_dataset):,} | Validation samples: {len(val_dataset):,}"
    )

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.optimizer.lr,
        betas=cfg.train.optimizer.betas,
        weight_decay=cfg.train.optimizer.weight_decay,
    )

    scheduler = None
    if cfg.train.scheduler.enabled:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cfg.train.epochs,
            eta_min=cfg.train.scheduler.eta_min,
        )

    # Setup checkpointing
    best_val_loss = float("inf")
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print("-" * 120)
    for epoch in range(1, cfg.train.epochs + 1):
        epoch_start = time.time()

        # Train
        train_epoch(
            model,
            train_loader,
            optimizer,
            cfg,
            device,
            epoch=epoch,
            epoch_start=epoch_start,
        )

        # Validate
        val_metrics = validate(model, val_loader, cfg, device)

        # Update scheduler
        if scheduler is not None:
            scheduler.step()

        # Print validation metrics
        print(
            f" | Val MSE: {val_metrics['loss']:.6f} | Val PSNR: {val_metrics['psnr']:.1f}dB | Val SSIM: {val_metrics['ssim']:.4f}"
        )

        # Save checkpoint
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "val_loss": val_metrics["loss"],
                "config": cfg,
            }
            torch.save(checkpoint, checkpoint_dir / "best_model.pth")

        # Regular checkpoint
        if epoch % cfg.save_interval == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "val_loss": val_metrics["loss"],
                "config": cfg,
            }
            torch.save(checkpoint, checkpoint_dir / f"checkpoint_epoch_{epoch}.pth")

    # Training complete
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()

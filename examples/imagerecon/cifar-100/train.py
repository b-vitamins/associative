#!/usr/bin/env python3
"""Train Energy Transformer with Image Reconstruction on CIFAR-100."""

import sys
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.append(str(Path(__file__).parents[3]))

import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from associative import EnergyTransformer, EnergyTransformerConfig
from associative.datasets.utils import generate_mask_indices
from associative.utils.training import calculate_reconstruction_metrics

# Constants
METRICS_CALCULATION_LIMIT = (
    20  # Calculate metrics for only first N batches to save time
)


def setup_data(cfg: DictConfig) -> tuple[DataLoader, DataLoader, int, int]:
    """Setup CIFAR-100 data loaders."""
    from torchvision import datasets, transforms

    # Define transforms
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
            ),
        ]
    )

    # Load datasets
    train_dataset = datasets.CIFAR100(
        root=cfg.data.root,
        train=True,
        download=True,
        transform=transform,
    )

    val_dataset = datasets.CIFAR100(
        root=cfg.data.root,
        train=False,
        download=True,
        transform=transform,
    )

    # Create data loaders
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

    return train_loader, val_loader, len(train_dataset), len(val_dataset)


def setup_model_and_optimizer(
    cfg: DictConfig, device: torch.device
) -> tuple[EnergyTransformer, torch.optim.Optimizer, Any]:
    """Setup model, optimizer, and scheduler."""
    # Create model config
    model_config = EnergyTransformerConfig(
        embed_dim=cfg.model.embed_dim,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        patch_size=cfg.model.patch_size,
        num_patches=cfg.model.num_patches,
    )

    # Create model
    model = EnergyTransformer(model_config).to(device)

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        betas=cfg.optimizer.betas,
        weight_decay=cfg.optimizer.weight_decay,
    )

    # Setup scheduler
    if cfg.scheduler.type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)
    elif cfg.scheduler.type == "cosine_warm_restarts":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.scheduler.T_0,
            T_mult=cfg.scheduler.T_mult,
            eta_min=cfg.scheduler.eta_min,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {cfg.scheduler.type}")

    return model, optimizer, scheduler


def setup_model(cfg: DictConfig) -> Any:
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


def train_epoch(
    model: Any,
    dataloader: Any,
    optimizer: Any,
    cfg: DictConfig,
    device: torch.device,
    **kwargs: Any,
) -> dict[str, float]:
    """Train for one epoch."""
    epoch = kwargs.get("epoch", 1)
    epoch_start = kwargs.get("epoch_start", time.time())
    model.train()

    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    total_batches = len(dataloader)

    # CIFAR-100 normalization stats - use exact values from DATASET_STATS
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    for batch_idx, (batch_data, _) in enumerate(dataloader):
        batch_size = batch_data.shape[0]
        batch_images = batch_data.to(device)

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
            batch_images,
            mask=mask,
            alpha=cfg.model.step_size,
        )

        # Get target patches
        target_patches = model.patch_embed.to_patches(batch_images)
        target_pixels = target_patches.reshape(batch_size, cfg.model.num_patches, -1)

        # Compute loss only on masked patches
        masked_output = output[batch_idx_mask, mask_idx]
        masked_target = target_pixels[batch_idx_mask, mask_idx]
        # Match second implementation's loss calculation
        from einops import reduce

        loss = reduce((masked_output - masked_target) ** 2, "b ... -> b", "mean").mean()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        if cfg.train.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.gradient_clip)
        optimizer.step()

        # Calculate metrics
        running_loss += loss.item()

        # Log every N batches
        if batch_idx % cfg.train.log_interval == 0:
            avg_loss = running_loss / (batch_idx + 1)
            batch_time = time.time() - epoch_start
            remaining_time = (
                batch_time / (batch_idx + 1) * (total_batches - batch_idx - 1)
            )
            remaining_str = f"{remaining_time // 60:.0f}m {remaining_time % 60:.0f}s"

            print(
                f"Epoch {epoch} [{batch_idx}/{total_batches}] "
                f"Loss: {avg_loss:.4f} | "
                f"Time: {batch_time:.1f}s | "
                f"ETA: {remaining_str}"
            )

        # Only calculate full metrics on subset of batches
        if batch_idx % 50 == 0:
            # Calculate metrics for a sample from the batch
            metrics = calculate_reconstruction_metrics(
                output[0:1], target_pixels[0:1], mean, std
            )
            running_psnr += metrics["psnr"]
            running_ssim += metrics["ssim"]

            print(
                f"  Sample metrics - PSNR: {metrics['psnr']:.2f} | "
                f"SSIM: {metrics['ssim']:.4f}"
            )

    return {
        "loss": running_loss / total_batches,
        "psnr": float(running_psnr / max((total_batches // 50), 1)),
        "ssim": float(running_ssim / max((total_batches // 50), 1)),
    }


def validate(
    model: Any, dataloader: Any, cfg: DictConfig, device: torch.device
) -> dict[str, float]:
    """Validate model."""
    model.eval()

    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = 0

    # CIFAR-100 normalization stats
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    with torch.no_grad():
        for batch_idx, (batch_data, _) in enumerate(dataloader):
            batch_size = batch_data.shape[0]
            batch_images = batch_data.to(device)

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
                batch_images,
                mask=mask,
                alpha=cfg.model.step_size,
            )

            # Get target patches
            target_patches = model.patch_embed.to_patches(batch_images)
            target_pixels = target_patches.reshape(
                batch_size, cfg.model.num_patches, -1
            )

            # Compute loss
            masked_output = output[batch_idx_mask, mask_idx]
            masked_target = target_pixels[batch_idx_mask, mask_idx]
            from einops import reduce

            loss = reduce(
                (masked_output - masked_target) ** 2, "b ... -> b", "mean"
            ).mean()

            total_loss += loss.item()

            # Calculate reconstruction metrics for first sample
            if (
                batch_idx < METRICS_CALCULATION_LIMIT
            ):  # Only calculate for subset to save time
                metrics = calculate_reconstruction_metrics(
                    output[0:1], target_pixels[0:1], mean, std
                )
                total_psnr += metrics["psnr"]
                total_ssim += metrics["ssim"]

            num_batches += 1

    return {
        "loss": float(total_loss / num_batches),
        "psnr": float(total_psnr / min(num_batches, 20)),
        "ssim": float(total_ssim / min(num_batches, 20)),
    }


@hydra.main(version_base=None, config_path=".", config_name="config")
def train(cfg: DictConfig):
    """Main training function."""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training CIFAR-100 Image Reconstruction on {device}")
    print(f"Config:\n{cfg}")

    # Set random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Setup data and model using helper functions
    train_loader, val_loader, train_size, val_size = setup_data(cfg)
    model, optimizer, scheduler = setup_model_and_optimizer(cfg, device)

    print(f"Train samples: {train_size}")
    print(f"Val samples: {val_size}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Training loop
    best_val_loss = float("inf")
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.train.epochs + 1):
        epoch_start = time.time()
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{cfg.train.epochs}")
        print(f"{'=' * 60}")

        # Train
        train_metrics = train_epoch(
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
        if scheduler:
            scheduler.step()

        # Log metrics
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val PSNR: {val_metrics['psnr']:.2f}")
        print(f"  Val SSIM: {val_metrics['ssim']:.4f}")

        epoch_time = time.time() - epoch_start
        print(f"  Epoch Time: {epoch_time:.1f}s")

        # Save checkpoint
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "best_val_loss": best_val_loss,
                "config": cfg,
            }
            torch.save(checkpoint, checkpoint_dir / "best_model.pt")
            print(f"  Saved best model (val_loss: {best_val_loss:.4f})")

        # Save periodic checkpoint
        if epoch % cfg.train.save_interval == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "config": cfg,
            }
            torch.save(checkpoint, checkpoint_dir / f"checkpoint_epoch_{epoch}.pt")

    print(f"\nTraining completed! Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()

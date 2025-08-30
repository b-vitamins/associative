#!/usr/bin/env python3
"""Train Energy Transformer with Image Reconstruction on ImageNet-32."""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parents[3]))

import gc
import time
from dataclasses import dataclass
from typing import Any

import hydra
import numpy as np
import torch
from einops import reduce
from omegaconf import DictConfig
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torchvision import transforms

from associative import EnergyTransformer, EnergyTransformerConfig
from associative.datasets.imagenet32 import ImageNet32
from associative.datasets.utils import generate_mask_indices
from associative.utils.training import calculate_reconstruction_metrics

# Constants
METRICS_CALCULATION_LIMIT = (
    10  # Calculate metrics for only first N batches to save time
)


def setup_training_context(cfg: DictConfig, device: torch.device) -> "TrainingContext":
    """Setup training context with model and optimizers."""
    # Setup data
    train_loader, val_loader = setup_data(cfg)

    # Setup model
    model = setup_model(cfg).to(device)

    # Optionally compile model for faster training
    compiled_model = model
    if cfg.train.compile and hasattr(torch, "compile"):
        print("Compiling model for faster training...")
        compiled_model = torch.compile(model)

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.optimizer.lr,
        weight_decay=cfg.train.optimizer.weight_decay,
        betas=cfg.train.optimizer.betas,
    )

    scheduler = None
    if cfg.train.scheduler.name == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cfg.train.epochs,
            eta_min=cfg.train.scheduler.min_lr,
        )
    elif cfg.train.scheduler.name == "cosine_restarts":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.train.scheduler.T_0,
            T_mult=cfg.train.scheduler.T_mult,
            eta_min=cfg.train.scheduler.min_lr,
        )

    return TrainingContext(
        cfg=cfg,
        device=device,
        model=model,
        compiled_model=compiled_model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
    )


def setup_model(cfg: DictConfig) -> EnergyTransformer:
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


@dataclass
class TrainingContext:
    """Context for training functions."""

    model: Any  # Can be Module or compiled model
    compiled_model: Any  # Compiled version of the model
    optimizer: torch.optim.Optimizer
    scheduler: Any  # Learning rate scheduler
    train_loader: DataLoader
    val_loader: DataLoader
    cfg: DictConfig
    device: torch.device
    epoch: int = 1
    global_step: int = 0
    best_val_loss: float = float("inf")


def train_step(
    batch_images: torch.Tensor, context: TrainingContext
) -> dict[str, float]:
    """Single training step."""
    batch_size = batch_images.shape[0]

    # Generate mask indices
    batch_idx_mask, mask_idx = generate_mask_indices(
        batch_size,
        context.cfg.model.num_patches,
        context.cfg.data.mask_ratio,
        device=context.device,
    )
    mask = (batch_idx_mask, mask_idx)

    # Forward pass
    output = context.model(
        batch_images,
        mask=mask,
        alpha=context.cfg.model.step_size,
    )

    # Get target patches
    if hasattr(context.model, "patch_embed"):
        target_patches = context.model.patch_embed.to_patches(batch_images)
    else:
        # Compiled model - access underlying module
        target_patches = context.model._orig_mod.patch_embed.to_patches(batch_images)  # type: ignore[attr-defined]
    target_pixels = target_patches.reshape(
        batch_size, context.cfg.model.num_patches, -1
    )

    # Compute loss only on masked patches
    masked_output = output[batch_idx_mask, mask_idx]
    masked_target = target_pixels[batch_idx_mask, mask_idx]
    loss = reduce((masked_output - masked_target) ** 2, "b ... -> b", "mean").mean()

    # Backward pass
    context.optimizer.zero_grad()
    loss.backward()
    if context.cfg.train.gradient_clip > 0:
        torch.nn.utils.clip_grad_norm_(
            context.model.parameters(), context.cfg.train.gradient_clip
        )
    context.optimizer.step()

    return {"loss": loss.item()}


def train_epoch(dataloader: DataLoader, context: TrainingContext) -> dict[str, float]:
    """Train for one epoch."""
    context.model.train()

    running_loss = 0.0
    num_batches = 0
    epoch_start = time.time()

    for batch_idx, (batch_data, _) in enumerate(dataloader):
        batch_images = batch_data.to(context.device)

        # Training step
        metrics = train_step(batch_images, context)
        running_loss += metrics["loss"]
        num_batches += 1
        context.global_step += 1

        # Log progress
        if batch_idx % context.cfg.train.log_interval == 0:
            avg_loss = running_loss / num_batches
            batch_time = time.time() - epoch_start
            batch_size = getattr(dataloader, "batch_size", None) or 1
            samples_per_sec = (batch_idx + 1) * batch_size / batch_time

            print(
                f"Epoch {context.epoch} [{batch_idx}/{len(dataloader)}] "
                f"Loss: {avg_loss:.4f} | "
                f"Speed: {samples_per_sec:.1f} samples/s"
            )

    return {"loss": running_loss / num_batches}


def validate(dataloader: DataLoader, context: TrainingContext) -> dict[str, float]:
    """Validate model."""
    context.model.eval()

    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = 0

    # ImageNet-32 normalization stats
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    with torch.no_grad():
        for batch_idx, (batch_data, _) in enumerate(dataloader):
            if batch_idx >= context.cfg.data.val_batches:
                break

            batch_size = batch_data.shape[0]
            batch_images = batch_data.to(context.device)

            # Generate mask indices
            batch_idx_mask, mask_idx = generate_mask_indices(
                batch_size,
                context.cfg.model.num_patches,
                context.cfg.data.mask_ratio,
                device=context.device,
            )
            mask = (batch_idx_mask, mask_idx)

            # Forward pass
            output = context.model(
                batch_images,
                mask=mask,
                alpha=context.cfg.model.step_size,
            )

            # Get target patches
            if hasattr(context.model, "patch_embed"):
                target_patches = context.model.patch_embed.to_patches(batch_images)
            else:
                # Compiled model - access underlying module
                target_patches = context.model._orig_mod.patch_embed.to_patches(
                    batch_images
                )  # type: ignore[attr-defined]
            target_pixels = target_patches.reshape(
                batch_size, context.cfg.model.num_patches, -1
            )

            # Compute loss
            masked_output = output[batch_idx_mask, mask_idx]
            masked_target = target_pixels[batch_idx_mask, mask_idx]
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
        "loss": total_loss / num_batches,
        "psnr": float(total_psnr / min(num_batches, 10)),
        "ssim": float(total_ssim / min(num_batches, 10)),
    }


def setup_data(cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
    """Setup data loaders."""
    # Define transforms
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    # Create datasets
    train_dataset = ImageNet32(
        root=cfg.data.root,
        train=True,
        download=cfg.data.download,
        transform=transform,
    )

    val_dataset = ImageNet32(
        root=cfg.data.root,
        train=False,
        download=cfg.data.download,
        transform=transform,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=cfg.data.num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=cfg.data.num_workers > 0,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    return train_loader, val_loader


def save_checkpoint(context: TrainingContext, val_metrics: dict, checkpoint_dir: Path):
    """Save model checkpoint."""
    if val_metrics["loss"] < context.best_val_loss:
        context.best_val_loss = val_metrics["loss"]
        checkpoint = {
            "epoch": context.epoch,
            "model_state_dict": context.model.state_dict(),
            "optimizer_state_dict": context.optimizer.state_dict(),
            "best_val_loss": context.best_val_loss,
            "config": context.cfg,
        }
        torch.save(checkpoint, checkpoint_dir / "best_model.pt")
        print(f"  Saved best model (val_loss: {context.best_val_loss:.4f})")

    # Save periodic checkpoint
    if context.epoch % context.cfg.train.save_interval == 0:
        checkpoint = {
            "epoch": context.epoch,
            "model_state_dict": context.model.state_dict(),
            "optimizer_state_dict": context.optimizer.state_dict(),
            "config": context.cfg,
        }
        torch.save(checkpoint, checkpoint_dir / f"checkpoint_epoch_{context.epoch}.pt")


@hydra.main(version_base=None, config_path=".", config_name="config")
def train(cfg: DictConfig):
    """Main training function."""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training ImageNet-32 Image Reconstruction on {device}")
    print(f"Config:\n{cfg}")

    # Set random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Setup training context
    context = setup_training_context(cfg, device)

    # Count parameters
    total_params = sum(p.numel() for p in context.model.parameters())
    trainable_params = sum(
        p.numel() for p in context.model.parameters() if p.requires_grad
    )
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Training components are now setup in context

    # Setup checkpointing
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    for epoch in range(1, cfg.train.epochs + 1):
        context.epoch = epoch
        epoch_start = time.time()

        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{cfg.train.epochs}")
        print(f"{'=' * 60}")

        # Train
        train_metrics = train_epoch(context.train_loader, context)

        # Validate
        val_metrics = validate(context.val_loader, context)

        # Update scheduler
        if context.scheduler:
            context.scheduler.step()

        # Log metrics
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val PSNR: {val_metrics['psnr']:.2f}")
        print(f"  Val SSIM: {val_metrics['ssim']:.4f}")
        print(f"  Epoch Time: {epoch_time:.1f}s")

        # Save checkpoint
        save_checkpoint(context, val_metrics, checkpoint_dir)

        # Garbage collection
        if epoch % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\nTraining completed! Best val loss: {context.best_val_loss:.4f}")


if __name__ == "__main__":
    train()

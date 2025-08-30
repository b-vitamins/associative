#!/usr/bin/env python3
"""Train Graph Energy Transformer on ZINC dataset."""

import sys
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.append(str(Path(__file__).parents[3]))

import logging

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn import functional
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader as GeometricDataLoader

from associative import EnergyTransformerConfig, GraphEnergyTransformer
from associative.utils.graph import prepare_graph_batch
from associative.utils.training import MetricTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class ZINCDataModule:
    """Data module for ZINC dataset."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.num_features = 1  # ZINC has 1D features
        self.num_classes = 1  # Regression task

    def setup(
        self,
    ) -> tuple[GeometricDataLoader, GeometricDataLoader, GeometricDataLoader]:
        """Setup data loaders for train, val, and test."""
        # Load ZINC dataset splits
        train_dataset = ZINC(root=self.cfg.data.root, subset=True, split="train")
        val_dataset = ZINC(root=self.cfg.data.root, subset=True, split="val")
        test_dataset = ZINC(root=self.cfg.data.root, subset=True, split="test")

        print("Dataset: ZINC")
        print(
            f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)} samples"
        )

        # Create loaders
        train_loader = GeometricDataLoader(
            train_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.get("pin_memory", True),
        )

        val_loader = GeometricDataLoader(
            val_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.get("pin_memory", True),
        )

        test_loader = GeometricDataLoader(
            test_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.get("pin_memory", True),
        )

        return train_loader, val_loader, test_loader


def setup_model(cfg: DictConfig) -> GraphEnergyTransformer:
    """Setup model from config."""
    config = EnergyTransformerConfig(
        embed_dim=cfg.model.d_model,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        mlp_ratio=cfg.model.d_ff / cfg.model.d_model if cfg.model.d_ff else 4.0,
        input_dim=1,  # ZINC features
        out_dim=1,  # Regression
    )

    return GraphEnergyTransformer(config)


def train_step(
    model: Any, batch: Any, device: torch.device, cfg: DictConfig
) -> tuple[float, float]:
    """Single training step."""
    # Prepare batch
    batch = batch.to(device)
    node_features, edge_index, edge_attr, pos_encodings, batch_idx = (
        prepare_graph_batch(batch)
    )

    # Forward pass
    outputs = model(
        node_features=node_features,
        edge_index=edge_index,
        batch=batch_idx,
    )

    # Compute loss (MAE for regression)
    loss = functional.l1_loss(outputs.squeeze(), batch.y)

    # L2 regularization
    if cfg.train.l2_reg > 0:
        l2_loss = sum(p.pow(2).sum() for p in model.parameters())
        loss = loss + cfg.train.l2_reg * l2_loss

    # Backward pass
    loss.backward()

    return loss.item(), 0.0  # Return dummy accuracy for regression


def evaluate(
    model: Any, loader: Any, device: torch.device, cfg: DictConfig
) -> tuple[float, float]:
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch_data in loader:
            batch = batch_data.to(device)
            node_features, edge_index, edge_attr, pos_encodings, batch_idx = (
                prepare_graph_batch(batch)
            )

            # Forward pass
            outputs = model(
                node_features=node_features,
                edge_index=edge_index,
                batch=batch_idx,
            )

            # Compute loss (MAE)
            loss = functional.l1_loss(outputs.squeeze(), batch.y)
            total_loss += loss.item() * batch.y.size(0)
            total_samples += batch.y.size(0)

    avg_loss = total_loss / total_samples
    model.train()
    return avg_loss, 0.0  # Return dummy accuracy for regression


def train_one_epoch(
    model: Any, train_loader: Any, optimizer: Any, device: torch.device, **kwargs: Any
) -> tuple[float, float]:
    """Train for one epoch."""
    model.train()
    cfg = kwargs["cfg"]
    kwargs.get("tracker")
    epoch = kwargs.get("epoch", 0)

    total_loss = 0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()

        loss, _ = train_step(model, batch, device, cfg)  # Unpack tuple

        # Gradient clipping
        if cfg.train.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.gradient_clip)

        optimizer.step()

        total_loss += loss
        num_batches += 1

        # Log progress
        if (batch_idx + 1) % cfg.train.log_interval == 0 or (batch_idx + 1) == len(
            train_loader
        ):
            avg_loss = total_loss / num_batches
            print(
                f"Train Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] "
                f"MAE Loss: {avg_loss:.4f}"
            )

    return total_loss / num_batches, 0.0  # Return dummy accuracy


def save_checkpoint(model: Any, optimizer: Any, scheduler: Any, **kwargs: Any) -> None:
    """Save model checkpoint."""
    epoch = kwargs.get("epoch", 0)
    checkpoint_dir = kwargs.get("checkpoint_dir")
    best_val_loss = kwargs.get("best_val_loss", float("inf"))
    val_loss = kwargs.get("val_loss", float("inf"))
    cfg = kwargs["cfg"]

    if val_loss < best_val_loss:
        # Save best model
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "best_val_loss": val_loss,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        if checkpoint_dir is not None:
            torch.save(checkpoint, checkpoint_dir / "best_model.pt")
        print(f"Saved best model with val MAE: {val_loss:.4f}")

    # Save latest model
    if epoch % cfg.train.save_interval == 0:
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        if checkpoint_dir is not None:
            torch.save(checkpoint, checkpoint_dir / f"checkpoint_epoch_{epoch}.pt")


def setup_training(
    cfg: DictConfig, device: torch.device
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Setup training components."""
    # Setup logging
    print(f"Using device: {device}")
    print(f"Configuration: {OmegaConf.to_yaml(cfg.model)}")

    # Set random seed
    seed = cfg.get("seed", 3407)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setup data
    data_module = ZINCDataModule(cfg)
    train_loader, val_loader, test_loader = data_module.setup()

    # Setup model
    model = setup_model(cfg).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.optimizer.lr,
        weight_decay=cfg.train.optimizer.weight_decay,
    )

    # Setup scheduler
    scheduler = None
    if cfg.train.scheduler.enabled:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.train.epochs, eta_min=cfg.train.scheduler.min_lr
        )

    return model, optimizer, scheduler, train_loader, val_loader, test_loader


@hydra.main(version_base=None, config_path=".", config_name="config")
def train(cfg: DictConfig) -> None:
    """Main training function."""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup training components
    model, optimizer, scheduler, train_loader, val_loader, test_loader = setup_training(
        cfg, device
    )

    # Setup tracking and checkpointing
    tracker = MetricTracker()
    best_val_loss = float("inf")
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.train.epochs + 1):
        tracker.start_epoch()

        # Training
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            cfg=cfg,
            tracker=tracker,
            epoch=epoch,
        )

        # Validation
        val_loss, _ = evaluate(model, val_loader, device, cfg)

        # Test
        test_loss, _ = evaluate(model, test_loader, device, cfg)

        # Update scheduler
        if scheduler:
            scheduler.step()

        # Log metrics
        print(
            f"\nEpoch {epoch}/{cfg.train.epochs} Summary:\n"
            f"  Train MAE: {train_loss:.4f}\n"
            f"  Val MAE: {val_loss:.4f}\n"
            f"  Test MAE: {test_loss:.4f}\n"
        )

        # Save checkpoint
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch=epoch,
            checkpoint_dir=checkpoint_dir,
            best_val_loss=best_val_loss,
            val_loss=val_loss,
            cfg=cfg,
        )

        best_val_loss = min(best_val_loss, val_loss)

    print(f"\nTraining completed! Best val MAE: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()

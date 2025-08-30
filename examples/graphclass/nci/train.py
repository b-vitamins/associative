#!/usr/bin/env python3
"""Train Graph Energy Transformer on NCI1 dataset."""

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
from torch.utils.data import Subset
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader as GeometricDataLoader

from associative import EnergyTransformerConfig, GraphEnergyTransformer
from associative.utils.graph import prepare_graph_batch
from associative.utils.training import MetricTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class NCIDataModule:
    """Data module for NCI1 dataset."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.num_features = 37  # NCI1 has 37D features
        self.num_classes = 2  # Binary classification

    def setup(
        self,
    ) -> tuple[GeometricDataLoader, GeometricDataLoader, GeometricDataLoader]:
        """Setup data loaders for train, val, and test."""
        # Load NCI1 dataset
        dataset = TUDataset(
            root=self.cfg.data.root,
            name="NCI1",
            use_node_attr=False,  # NCI1 has no node attributes
        )

        # Split dataset
        num_samples = len(dataset)
        indices = np.random.permutation(num_samples)

        train_size = int(num_samples * self.cfg.data.train_split)
        val_size = int(num_samples * self.cfg.data.val_split)

        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[train_size + val_size :]

        # Create data loaders
        train_dataset = Subset(dataset, train_indices.tolist())
        val_dataset = Subset(dataset, val_indices.tolist())
        test_dataset = Subset(dataset, test_indices.tolist())

        train_loader = GeometricDataLoader(
            train_dataset,  # type: ignore[arg-type]
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
        )

        val_loader = GeometricDataLoader(
            val_dataset,  # type: ignore[arg-type]
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
        )

        test_loader = GeometricDataLoader(
            test_dataset,  # type: ignore[arg-type]
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
        )

        print("NCI1 Dataset Statistics:")
        print(f"  Total samples: {num_samples}")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
        print(f"  Node features: {self.num_features}")
        print(f"  Classes: {self.num_classes}")

        return train_loader, val_loader, test_loader


def setup_model(cfg: DictConfig) -> GraphEnergyTransformer:
    """Setup model from config."""
    config = EnergyTransformerConfig(
        embed_dim=cfg.model.d_model,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        mlp_ratio=cfg.model.d_ff / cfg.model.d_model if cfg.model.d_ff else 4.0,
        input_dim=37,  # NCI1 features
        out_dim=2,  # Binary classification
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

    # Compute loss
    loss = functional.cross_entropy(outputs, batch.y)

    # Compute accuracy
    preds = outputs.argmax(dim=-1)
    correct = (preds == batch.y).float().sum()
    total = batch.y.size(0)
    accuracy = correct / total

    # L2 regularization
    if cfg.train.l2_reg > 0:
        l2_loss = sum(p.pow(2).sum() for p in model.parameters())
        loss = loss + cfg.train.l2_reg * l2_loss

    # Backward pass
    loss.backward()

    return loss.item(), accuracy.item()


def evaluate(
    model: Any, loader: Any, device: torch.device, cfg: DictConfig
) -> tuple[float, float]:
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    total_correct = 0
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

            # Compute loss
            loss = functional.cross_entropy(outputs, batch.y)
            total_loss += loss.item() * batch.y.size(0)

            # Compute accuracy
            preds = outputs.argmax(dim=-1)
            total_correct += (preds == batch.y).float().sum().item()
            total_samples += batch.y.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    model.train()
    return avg_loss, accuracy


def train_one_epoch(
    model: Any, train_loader: Any, optimizer: Any, device: torch.device, **kwargs: Any
) -> tuple[float, float]:
    """Train for one epoch."""
    model.train()
    cfg = kwargs["cfg"]
    kwargs.get("tracker")
    epoch = kwargs.get("epoch", 0)

    total_loss = 0
    total_acc = 0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()

        loss, acc = train_step(model, batch, device, cfg)

        # Gradient clipping
        if cfg.train.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.gradient_clip)

        optimizer.step()

        total_loss += loss
        total_acc += acc
        num_batches += 1

        # Log progress
        if (batch_idx + 1) % cfg.train.log_interval == 0 or (batch_idx + 1) == len(
            train_loader
        ):
            avg_loss = total_loss / num_batches
            avg_acc = total_acc / num_batches
            print(
                f"Train Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {avg_loss:.4f} Acc: {avg_acc:.4f}"
            )

    return total_loss / num_batches, total_acc / num_batches


def save_checkpoint(model: Any, optimizer: Any, scheduler: Any, **kwargs: Any) -> None:
    """Save model checkpoint."""
    epoch = kwargs.get("epoch", 0)
    checkpoint_dir = kwargs.get("checkpoint_dir")
    best_val_acc = kwargs.get("best_val_acc", 0)
    val_acc = kwargs.get("val_acc", 0)
    cfg = kwargs["cfg"]

    if val_acc > best_val_acc:
        # Save best model
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "best_val_acc": val_acc,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        if checkpoint_dir is not None:
            torch.save(checkpoint, checkpoint_dir / "best_model.pt")
        print(f"Saved best model with val acc: {val_acc:.4f}")

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
    data_module = NCIDataModule(cfg)
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
    best_val_acc = 0.0
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.train.epochs + 1):
        tracker.start_epoch()

        # Training
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            cfg=cfg,
            tracker=tracker,
            epoch=epoch,
        )

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, device, cfg)

        # Test
        test_loss, test_acc = evaluate(model, test_loader, device, cfg)

        # Update scheduler
        if scheduler:
            scheduler.step()

        # Log metrics
        print(
            f"\nEpoch {epoch}/{cfg.train.epochs} Summary:\n"
            f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}\n"
            f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}\n"
            f"  Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}\n"
        )

        # Save checkpoint
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch=epoch,
            checkpoint_dir=checkpoint_dir,
            best_val_acc=best_val_acc,
            val_acc=val_acc,
            cfg=cfg,
        )

        best_val_acc = max(best_val_acc, val_acc)

    print(f"\nTraining completed! Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    train()

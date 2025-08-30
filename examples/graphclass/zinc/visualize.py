#!/usr/bin/env python3
"""Visualize Graph Energy Transformer embeddings and predictions on ZINC."""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parents[3]))

import hydra
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader as GeometricDataLoader
from train import setup_model

from associative.utils.graph import prepare_graph_batch


def extract_embeddings(model, loader, cfg, max_samples=1000):
    """Extract embeddings and predictions from the model."""
    model.eval()

    embeddings = []
    predictions = []
    targets = []
    energies = []
    num_samples = 0

    with torch.no_grad():
        for batch in loader:
            if num_samples >= max_samples:
                break

            # Prepare batch
            x, adj, _, pos_enc, mask = prepare_graph_batch(
                batch,
                max_num_nodes=cfg.data.max_num_nodes,
                pos_encoding_k=cfg.model.pos_encoding_dim,
                pos_encoding_method=cfg.data.pos_encoding_method,
            )

            # Remove CLS token from input
            x = x[:, 1:]
            adj = adj[:, 1:, 1:]
            mask = mask[:, 1:] if mask is not None else None

            # Forward pass with energy computation
            output = model(
                x, adj, pos_enc, mask, alpha=cfg.model.step_size, return_energy=True
            )

            # Collect data
            embeddings.append(output["embeddings"][:, 0].cpu())  # CLS token embedding
            predictions.append(output["graph_embedding"].cpu())
            targets.append(batch.y.cpu())
            if "energy" in output:
                energies.append(output["energy"].cpu())

            num_samples += batch.num_graphs

    embeddings = torch.cat(embeddings, dim=0)[:max_samples]
    predictions = torch.cat(predictions, dim=0)[:max_samples]
    targets = torch.cat(targets, dim=0)[:max_samples]

    energies = torch.cat(energies, dim=0)[:max_samples] if energies else None

    return embeddings.numpy(), predictions.numpy(), targets.numpy(), energies


def visualize_embeddings(embeddings, targets, predictions, cfg):
    """Visualize embeddings using t-SNE and PCA."""
    output_dir = Path(cfg.visualization_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # t-SNE visualization
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d_tsne = tsne.fit_transform(embeddings)

    # PCA visualization
    print("Computing PCA...")
    pca = PCA(n_components=2)
    embeddings_2d_pca = pca.fit_transform(embeddings)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # t-SNE colored by target
    scatter = axes[0, 0].scatter(
        embeddings_2d_tsne[:, 0],
        embeddings_2d_tsne[:, 1],
        c=targets,
        cmap="viridis",
        alpha=0.6,
        s=10,
    )
    axes[0, 0].set_title("t-SNE Embeddings (colored by target)")
    axes[0, 0].set_xlabel("t-SNE 1")
    axes[0, 0].set_ylabel("t-SNE 2")
    plt.colorbar(scatter, ax=axes[0, 0])

    # t-SNE colored by prediction
    scatter = axes[0, 1].scatter(
        embeddings_2d_tsne[:, 0],
        embeddings_2d_tsne[:, 1],
        c=predictions.squeeze(),
        cmap="plasma",
        alpha=0.6,
        s=10,
    )
    axes[0, 1].set_title("t-SNE Embeddings (colored by prediction)")
    axes[0, 1].set_xlabel("t-SNE 1")
    axes[0, 1].set_ylabel("t-SNE 2")
    plt.colorbar(scatter, ax=axes[0, 1])

    # PCA colored by target
    scatter = axes[1, 0].scatter(
        embeddings_2d_pca[:, 0],
        embeddings_2d_pca[:, 1],
        c=targets,
        cmap="viridis",
        alpha=0.6,
        s=10,
    )
    axes[1, 0].set_title("PCA Embeddings (colored by target)")
    axes[1, 0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} var)")
    axes[1, 0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} var)")
    plt.colorbar(scatter, ax=axes[1, 0])

    # Prediction vs Target scatter
    axes[1, 1].scatter(targets, predictions.squeeze(), alpha=0.5, s=10)
    axes[1, 1].plot(
        [targets.min(), targets.max()], [targets.min(), targets.max()], "r--", lw=2
    )
    axes[1, 1].set_title("Predictions vs Targets")
    axes[1, 1].set_xlabel("True Value")
    axes[1, 1].set_ylabel("Predicted Value")
    axes[1, 1].set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_dir / "zinc_embeddings.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved embeddings visualization to {output_dir / 'zinc_embeddings.png'}")

    # Additional statistics
    mae = np.mean(np.abs(predictions.squeeze() - targets))
    print(f"\\nTest MAE: {mae:.4f}")

    # Save prediction histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(targets, bins=30, alpha=0.5, label="True", density=True)
    ax.hist(predictions.squeeze(), bins=30, alpha=0.5, label="Predicted", density=True)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of True vs Predicted Values")
    ax.legend()
    plt.savefig(output_dir / "zinc_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved distributions to {output_dir / 'zinc_distributions.png'}")


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main visualization function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = setup_model(cfg)
    model = model.to(device)

    # Load checkpoint
    checkpoint_path = Path(cfg.checkpoint_dir) / "best_model.pth"
    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    print(f"Loaded model from {checkpoint_path}")
    print(f"Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")

    # Load test dataset
    test_dataset = ZINC(root=cfg.data.root, subset=True, split="test")
    test_loader = GeometricDataLoader(
        test_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"\\nExtracting embeddings from {len(test_dataset)} test samples...")

    # Extract embeddings
    embeddings, predictions, targets, energies = extract_embeddings(
        model, test_loader, cfg, max_samples=2000
    )

    print(f"Extracted {len(embeddings)} embeddings")

    # Visualize
    visualize_embeddings(embeddings, targets, predictions, cfg)

    # Plot energy landscape if available
    if energies is not None:
        plt.figure(figsize=(8, 6))
        plt.scatter(targets, energies.numpy(), alpha=0.5, s=10)
        plt.xlabel("Target Value")
        plt.ylabel("Energy")
        plt.title("Energy Landscape")
        output_path = Path(cfg.visualization_dir) / "zinc_energy.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved energy landscape visualization to {output_path}")


if __name__ == "__main__":
    main()

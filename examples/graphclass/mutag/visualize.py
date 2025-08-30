#!/usr/bin/env python3
"""Visualize Graph Energy Transformer on MUTAG dataset."""

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

# Removed seaborn import - using matplotlib instead
from omegaconf import DictConfig
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader as GeometricDataLoader
from train import setup_model

from associative.utils.graph import prepare_graph_batch


def evaluate_and_visualize(model, loader, cfg, device):
    """Evaluate model and extract embeddings."""
    model.eval()

    embeddings = []
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            # Prepare batch
            x, adj, _, pos_enc, mask = prepare_graph_batch(
                batch,
                max_num_nodes=cfg.data.max_num_nodes,
                pos_encoding_k=cfg.model.pos_encoding_dim,
                pos_encoding_method=cfg.data.get("pos_encoding_method", "laplacian"),
            )

            # Remove CLS token from input
            x = x[:, 1:]
            adj = adj[:, 1:, 1:]
            mask = mask[:, 1:] if mask is not None else None

            # Forward pass
            output = model(x, adj, pos_enc, mask, alpha=cfg.model.step_size)
            logits = output["graph_embedding"]

            # Collect data
            embeddings.append(output["embeddings"][:, 0].cpu())  # CLS token
            predictions.append(logits.argmax(dim=-1).cpu())
            targets.append(batch.y.cpu())

    embeddings = torch.cat(embeddings, dim=0).numpy()
    predictions = torch.cat(predictions, dim=0).numpy()
    targets = torch.cat(targets, dim=0).numpy()

    return embeddings, predictions, targets


def create_confusion_matrix_plot(targets, predictions, output_dir):
    """Create confusion matrix visualization."""
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(8, 6))
    # Create heatmap using matplotlib
    im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im)

    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    # Set ticks and labels
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Non-mutagenic", "Mutagenic"])
    plt.yticks(tick_marks, ["Non-mutagenic", "Mutagenic"])
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()


def create_tsne_plot(embeddings, targets, output_dir):
    """Create t-SNE visualization."""
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=targets,
        cmap="coolwarm",
        alpha=0.7,
        s=50,
        edgecolors="black",
        linewidth=0.5,
    )
    plt.colorbar(scatter, ticks=[0, 1], label="Class")
    plt.title("t-SNE Visualization of Graph Embeddings")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    # Add class labels
    for class_idx in [0, 1]:
        class_points = embeddings_2d[targets == class_idx]
        if len(class_points) > 0:
            centroid = class_points.mean(axis=0)
            plt.text(
                centroid[0],
                centroid[1],
                f"Class {class_idx}",
                fontsize=12,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "alpha": 0.8},
            )

    plt.savefig(output_dir / "tsne_embeddings.png", dpi=150, bbox_inches="tight")
    plt.close()


def create_visualizations(embeddings, predictions, targets, cfg):
    """Create and save visualizations."""
    output_dir = Path(cfg.visualization_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set matplotlib style
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )

    # 1. Confusion Matrix
    create_confusion_matrix_plot(targets, predictions, output_dir)

    # 2. t-SNE visualization
    create_tsne_plot(embeddings, targets, output_dir)

    # 3. Classification Report
    report = classification_report(
        targets,
        predictions,
        target_names=["Non-mutagenic", "Mutagenic"],
        output_dict=True,
    )

    # Plot classification metrics
    metrics = ["precision", "recall", "f1-score"]
    classes = ["Non-mutagenic", "Mutagenic"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(classes))
    width = 0.25

    for i, metric in enumerate(metrics):
        values = [report[cls][metric] for cls in classes]  # type: ignore
        ax.bar(x + i * width, values, width, label=metric.capitalize())

    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title("Classification Metrics by Class")
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Add value labels on bars
    for i, metric in enumerate(metrics):
        values = [report[cls][metric] for cls in classes]  # type: ignore
        for j, v in enumerate(values):
            ax.text(j + i * width, v + 0.02, f"{v:.3f}", ha="center", va="bottom")  # type: ignore

    plt.savefig(output_dir / "classification_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Print results
    print("\\nClassification Report:")
    print("=" * 50)
    print(
        classification_report(
            targets, predictions, target_names=["Non-mutagenic", "Mutagenic"]
        )
    )

    accuracy = (predictions == targets).mean()
    print(f"\\nOverall Accuracy: {accuracy:.4f}")

    return report


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
    print(f"Best validation accuracy: {checkpoint.get('best_val_acc', 'N/A')}")

    # Load test dataset
    dataset = TUDataset(
        root=cfg.data.root,
        name=cfg.data.name,
        use_node_attr=cfg.data.use_node_attr,
        cleaned=cfg.data.cleaned,
    )

    # Split dataset (same as training)
    num_train = int(len(dataset) * cfg.data.train_split)
    num_val = int(len(dataset) * cfg.data.val_split)
    num_test = len(dataset) - num_train - num_val

    _, _, test_dataset = torch.utils.data.random_split(
        dataset,
        [num_train, num_val, num_test],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    test_loader = GeometricDataLoader(
        test_dataset,  # type: ignore
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"\\nEvaluating on {len(test_dataset)} test samples...")

    # Evaluate and visualize
    embeddings, predictions, targets = evaluate_and_visualize(
        model, test_loader, cfg, device
    )

    # Create visualizations
    create_visualizations(embeddings, predictions, targets, cfg)

    print("\\nSaved visualizations to visualizations/")


if __name__ == "__main__":
    main()

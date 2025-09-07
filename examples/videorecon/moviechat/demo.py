#!/usr/bin/env python3
"""Comparative experiment for video reconstruction using different basis functions.

This experiment evaluates the performance of Continuous Hopfield Networks with
various basis function types on video embedding reconstruction tasks using the
MovieChat dataset.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[3]))


from dataclasses import dataclass

import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import functional
from tqdm import tqdm

from associative.datasets.moviechat import create_moviechat_dataloader
from associative.nn.modules.config import (
    BasisConfig,
    CCCPConfig,
    ContinuousHopfieldConfig,
)
from associative.nn.modules.continuous import ContinuousHopfield
from associative.utils.masking import add_noise_to_embeddings, generate_random_mask
from associative.utils.video.embeddings import get_embedder

matplotlib.use("Agg")


@dataclass
class PlotConfig:
    """Configuration for plotting a metric."""

    x_pos: np.ndarray
    means: list
    stds: list
    colors: list
    ylabel: str
    title: str
    ylim: tuple | None = None
    log_scale: bool = False
    value_format: str = ".3f"
    y_offset: float = 0.02


class BasisFunctionExperiment:
    """Experiment runner for comparing basis functions in CHN models."""

    def __init__(self, cfg: DictConfig):
        """Initialize experiment with configuration.

        Args:
            cfg: Hydra configuration object
        """
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.basis_types = cfg.experiment.get(
            "basis_types",
            ["rectangular", "gaussian", "fourier", "polynomial"],
        )
        self.num_videos = cfg.experiment.get("num_videos", 10)
        self.results_dir = Path(cfg.experiment.get("results_dir", "outputs/results"))
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def create_chn_model(self, basis_type: str) -> ContinuousHopfield:
        """Create a CHN model with specified basis function type.

        Args:
            basis_type: Type of basis function to use

        Returns:
            Configured ContinuousHopfield model
        """
        basis_config = BasisConfig(
            num_basis=self.cfg.model.chn.basis.num_basis,
            basis_type=basis_type,
            domain=self.cfg.model.chn.basis.domain,
        )

        cccp_config = CCCPConfig(
            max_iterations=self.cfg.model.chn.cccp.max_iterations,
            tolerance=self.cfg.model.chn.cccp.tolerance,
            step_size=self.cfg.model.chn.cccp.get("step_size", 1.0),
            momentum=self.cfg.model.chn.cccp.get("momentum", 0.0),
            track_trajectory=False,
            use_line_search=self.cfg.model.chn.cccp.get("use_line_search", False),
        )

        chn_config = ContinuousHopfieldConfig(
            basis_config=basis_config,
            cccp_config=cccp_config,
            beta=self.cfg.model.chn.beta,
            regularization=self.cfg.model.chn.basis.get("regularization", 1e-3),
            integration_points=self.cfg.model.chn.get("num_integration_points", 500),
        )

        return ContinuousHopfield(chn_config).to(self.device)

    def compute_embeddings(self, frames: Tensor, embedder: torch.nn.Module) -> Tensor:
        """Compute embeddings for video frames in chunks.

        Args:
            frames: Video frames tensor of shape (num_frames, C, H, W)
            embedder: Embedding model

        Returns:
            Embeddings tensor of shape (num_frames, embed_dim)
        """
        chunk_size = self.cfg.experiment.get("embedding_chunk_size", 16)
        embedding_chunks = []

        with torch.no_grad():
            for i in range(0, frames.size(0), chunk_size):
                chunk = frames[i : min(i + chunk_size, frames.size(0))]
                embeddings = embedder(chunk)
                embedding_chunks.append(embeddings.cpu())

        return torch.cat(embedding_chunks, dim=0)

    def evaluate_reconstruction(
        self,
        model: ContinuousHopfield,
        embeddings: Tensor,
        mask_ratio: float,
    ) -> dict[str, float]:
        """Evaluate reconstruction performance for a single model.

        Args:
            model: ContinuousHopfield model to evaluate
            embeddings: Original embeddings of shape (batch_size, num_frames, embed_dim)
            mask_ratio: Ratio of frames to mask

        Returns:
            Dictionary of evaluation metrics
        """
        batch_size, num_frames, embed_dim = embeddings.shape

        use_noise = self.cfg.experiment.get("use_noise_masking", False)

        if use_noise:
            noise_std = self.cfg.experiment.get("noise_std", 5.0)
            torch.manual_seed(42)
            queries = add_noise_to_embeddings(embeddings, noise_std=noise_std)

            with torch.no_grad():
                reconstructed = model(memories=embeddings, queries=queries)

            original_flat = embeddings.view(-1, embed_dim)
            reconstructed_flat = reconstructed.view(-1, embed_dim)

        else:
            mask = generate_random_mask(
                batch_size=batch_size,
                sequence_length=num_frames,
                mask_ratio=mask_ratio,
                device=self.device,
            )

            masked_embeddings = embeddings.clone()
            masked_embeddings[mask] = 0.0

            with torch.no_grad():
                reconstructed = model(memories=embeddings, queries=masked_embeddings)

            mask_expanded = mask.unsqueeze(-1).expand_as(embeddings)
            original_flat = embeddings[mask_expanded].view(-1, embed_dim)
            reconstructed_flat = reconstructed[mask_expanded].view(-1, embed_dim)

        cosine_sim = functional.cosine_similarity(
            reconstructed_flat, original_flat, dim=-1
        ).mean()

        mse = functional.mse_loss(reconstructed_flat, original_flat)

        norm_error = torch.norm(reconstructed_flat - original_flat, dim=-1) / (
            torch.norm(original_flat, dim=-1) + 1e-8
        )

        return {
            "cosine_similarity": cosine_sim.item(),
            "mse": mse.item(),
            "normalized_error": norm_error.mean().item(),
        }

    def run_experiment(self) -> dict[str, list[dict]]:
        """Run the complete experiment.

        Returns:
            Dictionary mapping basis types to lists of metric results
        """
        embedder_cfg = dict(self.cfg.model.embedder)
        embedder_name = embedder_cfg.pop("name")
        embedder = get_embedder(embedder_name, **embedder_cfg).to(self.device)
        embedder.freeze()

        _, dataloader = create_moviechat_dataloader(
            split="test",
            batch_size=1,
            num_workers=self.cfg.data.num_workers,
            num_frames=self.cfg.data.num_frames,
            resolution=self.cfg.data.resolution,
        )

        results = {basis_type: [] for basis_type in self.basis_types}

        pbar = tqdm(dataloader, total=self.num_videos, desc="Processing videos")

        for videos_processed, batch in enumerate(pbar):
            if videos_processed >= self.num_videos:
                break

            frames = batch["frames"].squeeze(0).to(self.device)

            embeddings = self.compute_embeddings(frames, embedder)
            embeddings = embeddings.unsqueeze(0).to(self.device)

            for basis_type in self.basis_types:
                pbar.set_description(f"Video {videos_processed + 1}: {basis_type}")

                model = self.create_chn_model(basis_type)

                metrics = self.evaluate_reconstruction(
                    model, embeddings, self.cfg.data.mask_ratio
                )

                results[basis_type].append(metrics)

                del model
                torch.cuda.empty_cache()

        pbar.close()

        return results

    def _calculate_stats(self, results: dict[str, list[dict]]) -> dict:
        """Calculate statistics for each basis type."""
        stats = {}
        for basis_type in self.basis_types:
            metrics = results[basis_type]
            stats[basis_type] = {
                "cosine_similarity": {
                    "mean": np.mean([m["cosine_similarity"] for m in metrics]),
                    "std": np.std([m["cosine_similarity"] for m in metrics]),
                },
                "mse": {
                    "mean": np.mean([m["mse"] for m in metrics]),
                    "std": np.std([m["mse"] for m in metrics]),
                },
                "normalized_error": {
                    "mean": np.mean([m["normalized_error"] for m in metrics]),
                    "std": np.std([m["normalized_error"] for m in metrics]),
                },
            }
        return stats

    def _plot_metric(self, ax, config: PlotConfig):
        """Plot a single metric with error bars and labels."""
        ax.bar(
            config.x_pos,
            config.means,
            color=config.colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=1,
        )

        ax.errorbar(
            config.x_pos,
            config.means,
            yerr=config.stds,
            fmt="none",
            color="black",
            capsize=4,
            capthick=1.5,
            elinewidth=1.5,
            alpha=0.7,
        )

        for i, (mean, std) in enumerate(zip(config.means, config.stds, strict=False)):
            y_pos = mean + std * (1.2 if config.log_scale else 1.0) + config.y_offset
            ax.text(
                i,
                y_pos,
                f"{mean:{config.value_format}}",
                ha="center",
                va="bottom",
                fontsize=10 if config.value_format == ".3f" else 9,
                fontweight="bold",
            )

        ax.set_xticks(config.x_pos)
        ax.set_xticklabels(self.basis_types, rotation=30, ha="right")
        ax.set_xlabel("Basis Function Type", fontsize=11, fontweight="bold")
        ax.set_ylabel(config.ylabel, fontsize=11, fontweight="bold")
        ax.set_title(config.title, fontsize=12, fontweight="bold", pad=15)

        if config.ylim:
            ax.set_ylim(config.ylim)
        if config.log_scale:
            ax.set_yscale("log")

    def plot_results(self, results: dict[str, list[dict]]) -> None:
        """Create and save visualization plots.

        Args:
            results: Dictionary mapping basis types to lists of metric results
        """
        stats = self._calculate_stats(results)
        colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
            ax.set_axisbelow(True)

        x_pos = np.arange(len(self.basis_types))

        # Plot cosine similarity
        means = [stats[bt]["cosine_similarity"]["mean"] for bt in self.basis_types]
        stds = [stats[bt]["cosine_similarity"]["std"] for bt in self.basis_types]
        config = PlotConfig(
            x_pos=x_pos,
            means=means,
            stds=stds,
            colors=colors,
            ylabel="Cosine Similarity",
            title="Reconstruction Quality",
            ylim=(0, 1.05),
        )
        self._plot_metric(axes[0], config)

        # Plot MSE
        means = [stats[bt]["mse"]["mean"] for bt in self.basis_types]
        stds = [stats[bt]["mse"]["std"] for bt in self.basis_types]
        config = PlotConfig(
            x_pos=x_pos,
            means=means,
            stds=stds,
            colors=colors,
            ylabel="Mean Squared Error (log scale)",
            title="Reconstruction Error",
            log_scale=True,
            value_format=".2e",
        )
        self._plot_metric(axes[1], config)

        # Plot normalized error
        means = [stats[bt]["normalized_error"]["mean"] for bt in self.basis_types]
        stds = [stats[bt]["normalized_error"]["std"] for bt in self.basis_types]
        config = PlotConfig(
            x_pos=x_pos,
            means=means,
            stds=stds,
            colors=colors,
            ylabel="Normalized Error",
            title="Relative Error",
            y_offset=0.01,
        )
        self._plot_metric(axes[2], config)

        fig.suptitle(
            "CHN Basis Function Comparison on Video Reconstruction",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )

        plt.tight_layout()

        output_path = self.results_dir / "basis-comparison.png"
        plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Plots saved to {output_path}")
        plt.close()

    def print_summary(self, results: dict[str, list[dict]]) -> None:
        """Print experiment summary to console.

        Args:
            results: Dictionary mapping basis types to lists of metric results
        """
        stats = {}
        for basis_type in self.basis_types:
            metrics = results[basis_type]
            stats[basis_type] = {
                "cos_sim": (
                    np.mean([m["cosine_similarity"] for m in metrics]),
                    np.std([m["cosine_similarity"] for m in metrics]),
                ),
                "mse": (
                    np.mean([m["mse"] for m in metrics]),
                    np.std([m["mse"] for m in metrics]),
                ),
                "norm_err": (
                    np.mean([m["normalized_error"] for m in metrics]),
                    np.std([m["normalized_error"] for m in metrics]),
                ),
            }

        best_cos_sim = max(self.basis_types, key=lambda bt: stats[bt]["cos_sim"][0])
        best_mse = min(self.basis_types, key=lambda bt: stats[bt]["mse"][0])

        print("\n" + "=" * 62)
        print("              RECONSTRUCTION PERFORMANCE")
        print("=" * 62)
        print(" Basis Type    | Cos Sim ^ |    MSE v    | Norm Err v")
        print("---------------+-----------+-------------+---------------")

        for basis_type in self.basis_types:
            s = stats[basis_type]

            cos_marker = "*" if basis_type == best_cos_sim else " "

            print(
                f" {basis_type:13s} | "
                f"{s['cos_sim'][0]:.3f}+-{s['cos_sim'][1]:.3f}{cos_marker}| "
                f"{s['mse'][0]:.4e} | "
                f"{s['norm_err'][0]:.3f}+-{s['norm_err'][1]:.3f}"
            )

        print("=" * 62)
        print(f"\n* Best: {best_cos_sim} (similarity), {best_mse} (MSE)")
        print(
            f"Videos: {self.num_videos} | Frames: {self.cfg.data.num_frames} | "
            f"Mask: {self.cfg.data.mask_ratio:.0%}"
        )


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for the experiment.

    Args:
        cfg: Hydra configuration
    """
    device = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
    basis_types = cfg.experiment.get(
        "basis_types", ["rectangular", "gaussian", "fourier", "polynomial"]
    )

    print("\n" + "-" * 60)
    print("CHN BASIS FUNCTION COMPARISON")
    print("-" * 60)
    print(f"Device: {device}")
    print(
        f"Config: {cfg.experiment.get('num_videos', 10)} videos | {cfg.data.num_frames} frames | {cfg.data.mask_ratio:.0%} mask | {cfg.model.chn.basis.num_basis} basis"
    )
    print(f"Types:  {', '.join(basis_types)}")
    print("-" * 60)

    experiment = BasisFunctionExperiment(cfg)
    results = experiment.run_experiment()

    experiment.plot_results(results)
    experiment.print_summary(results)

    print("\n[DONE] Experiment complete")


if __name__ == "__main__":
    main()

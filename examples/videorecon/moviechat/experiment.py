#!/usr/bin/env python3
"""Video reconstruction experiment comparing basis functions and memory sizes."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[3]))

import gc
import time
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
from associative.utils.video.embeddings import get_embedder


@dataclass
class ExperimentConfig:
    """Experiment configuration parameters."""

    basis_types: list[str]
    memory_sizes: list[int]  # seq_len values
    basis_counts: list[int]  # N values
    num_videos: int
    device: torch.device
    beta: float = 1.0
    max_iterations: int = 10
    mask_ratio: float = 0.5


class DiscreteHopfield:
    """Modern discrete Hopfield network."""

    def __init__(self, beta: float = 1.0, max_iterations: int = 10):
        self.beta = beta
        self.max_iterations = max_iterations

    @torch.no_grad()
    def forward(self, memories: Tensor, queries: Tensor) -> Tensor:
        output = queries
        for _ in range(self.max_iterations):
            scores = torch.bmm(output, memories.transpose(1, 2)) * self.beta
            attention = functional.softmax(scores, dim=-1)
            output = torch.bmm(attention, memories)
        return output


class IncrementalStats:
    """Track mean and std incrementally to save memory."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, value: float):
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)

    @property
    def std(self):
        return np.sqrt(self.m2 / self.n) if self.n > 1 else 0.0


def evaluate_model(
    model, embeddings: Tensor, mask_ratio: float = 0.5, is_discrete: bool = False
) -> float:
    """Evaluate a single model on embeddings.

    Args:
        model: Model to evaluate (CHN or DiscreteHopfield)
        embeddings: Shape (1, num_frames, embed_dim)
        mask_ratio: Fraction of frames to mask
        is_discrete: Whether model is discrete Hopfield

    Returns:
        Cosine similarity score
    """
    batch_size, num_frames, embed_dim = embeddings.shape

    mask = torch.zeros(
        (batch_size, num_frames), dtype=torch.bool, device=embeddings.device
    )
    mask_start = int(num_frames * (1 - mask_ratio))
    mask[:, mask_start:] = True

    queries = embeddings.clone()
    queries[mask] = 0.0

    with torch.no_grad():
        if is_discrete:
            reconstructed = model.forward(embeddings, queries)
        else:
            reconstructed = model(memories=embeddings, queries=queries)

    mask_exp = mask.unsqueeze(-1).expand_as(embeddings)
    orig = embeddings[mask_exp].view(-1, embed_dim)
    pred = reconstructed[mask_exp].view(-1, embed_dim)

    return functional.cosine_similarity(orig, pred, dim=-1).mean().item()


def create_chn_model(
    basis_type: str, num_basis: int, beta: float, device: torch.device
) -> ContinuousHopfield:
    """Create a Continuous Hopfield Network model.

    Args:
        basis_type: Type of basis function
        num_basis: Number of basis functions (N)
        beta: Temperature parameter
        device: Device to place model on

    Returns:
        Configured ContinuousHopfield model
    """
    basis_cfg = BasisConfig(
        num_basis=num_basis,
        basis_type=basis_type,
        domain=(0.0, 1.0),
    )

    cccp_cfg = CCCPConfig(
        max_iterations=10,
        tolerance=1e-6,
    )

    chn_cfg = ContinuousHopfieldConfig(
        basis_config=basis_cfg,
        cccp_config=cccp_cfg,
        beta=beta,
        regularization=1e-3,
        integration_points=500,
    )

    return ContinuousHopfield(chn_cfg).to(device)


def process_single_configuration(
    embeddings: Tensor,
    model_type: str,
    memory_size: int,
    num_basis: int,
    config: ExperimentConfig,
) -> float:
    """Process a single (model_type, memory_size, num_basis) configuration.

    Args:
        embeddings: Full embeddings tensor (1, max_frames, embed_dim)
        model_type: One of basis types or "discrete_full" or "discrete_sub"
        memory_size: Memory size (number of frames to use)
        num_basis: Number of basis functions (or memory samples for discrete_sub)
        config: Experiment configuration

    Returns:
        Cosine similarity score
    """
    if embeddings.size(1) > memory_size:
        indices = torch.linspace(0, embeddings.size(1) - 1, memory_size).long()
        emb_memory = embeddings[:, indices].to(config.device)
    else:
        emb_memory = embeddings.to(config.device)

    if model_type == "discrete_full":
        model = DiscreteHopfield(beta=config.beta, max_iterations=config.max_iterations)
        score = evaluate_model(model, emb_memory, config.mask_ratio, is_discrete=True)

    elif model_type == "discrete_sub":
        indices_basis = torch.linspace(0, memory_size - 1, num_basis).long()
        emb_basis = emb_memory[:, indices_basis]
        model = DiscreteHopfield(beta=config.beta, max_iterations=config.max_iterations)
        score = evaluate_model(model, emb_basis, config.mask_ratio, is_discrete=True)
        del emb_basis

    else:
        model = create_chn_model(model_type, num_basis, config.beta, config.device)
        score = evaluate_model(model, emb_memory, config.mask_ratio)
        del model

    del emb_memory
    torch.cuda.empty_cache()

    return score


class VideoReconstructionExperiment:
    """Experiment runner with improved structure and monitoring."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results_dir = Path(cfg.experiment.get("results_dir", "outputs/results"))
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.config = ExperimentConfig(
            basis_types=["rectangular", "gaussian", "fourier", "polynomial"],
            memory_sizes=[512, 1024, 2048, 4096],
            basis_counts=[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
            num_videos=cfg.experiment.get("num_videos", 100),
            device=self.device,
            beta=cfg.model.chn.beta,
            mask_ratio=cfg.data.mask_ratio,
        )

        self.total_evaluations = self._calculate_total_evaluations()

        self._print_header()

    def _calculate_total_evaluations(self) -> int:
        """Calculate total number of model evaluations."""
        count = 0
        for memory_size in self.config.memory_sizes:
            for num_basis in self.config.basis_counts:
                if num_basis > memory_size:
                    continue
                count += 2 + len(self.config.basis_types)
        return count * self.config.num_videos

    def _print_header(self):
        """Print experiment header."""
        print("\n" + "-" * 60)
        print("VIDEO RECONSTRUCTION EXPERIMENT")
        print("-" * 60)
        print(f"Device: {self.device}")
        print(f"Videos: {self.config.num_videos}")
        print(f"Memory: {self.config.memory_sizes}")
        print(
            f"Basis:  {self.config.basis_counts[:3]}...{self.config.basis_counts[-2:]}"
        )
        print(f"Types:  {', '.join(self.config.basis_types)} + discrete")
        print(f"Total:  {self.total_evaluations:,} evaluations")
        print("-" * 60)

    def run_experiment(self) -> dict:
        """Run the full experiment with detailed progress tracking."""
        stats = self._initialize_stats()

        embedder = self._setup_embedder()

        dataset, dataloader = create_moviechat_dataloader(
            split="test",
            batch_size=1,
            num_workers=0,
            num_frames=max(self.config.memory_sizes),
            resolution=self.cfg.data.resolution,
        )

        print("\nProcessing videos...")
        pbar = tqdm(
            total=self.total_evaluations,
            desc="Progress",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        for video_idx, batch in enumerate(dataloader):
            if video_idx >= self.config.num_videos:
                break

            frames = batch["frames"].squeeze(0)
            embeddings = self._compute_embeddings(frames, embedder)

            self._process_video_all_configs(embeddings, stats, video_idx, pbar)

            del batch, frames, embeddings
            gc.collect()

            if (video_idx + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        pbar.close()

        del embedder
        torch.cuda.empty_cache()
        gc.collect()

        results = self._finalize_results(stats)
        self._print_summary(results)

        return results

    def _initialize_stats(self) -> dict:
        """Initialize statistics trackers for all configurations."""
        stats = {}
        for memory_size in self.config.memory_sizes:
            stats[memory_size] = {}

            for model_type in [
                *self.config.basis_types,
                "discrete_full",
                "discrete_sub",
            ]:
                stats[memory_size][model_type] = {}
                for num_basis in self.config.basis_counts:
                    if num_basis <= memory_size:
                        stats[memory_size][model_type][num_basis] = IncrementalStats()

        return stats

    def _setup_embedder(self):
        """Setup and return the embedder model."""
        embedder_cfg = dict(self.cfg.model.embedder)
        embedder_name = embedder_cfg.pop("name")
        embedder = get_embedder(embedder_name, **embedder_cfg).to(self.device)
        embedder.eval()
        return embedder

    def _compute_embeddings(self, frames: Tensor, embedder) -> Tensor:
        """Compute embeddings for video frames."""
        embeddings = []
        chunk_size = 16

        with torch.no_grad():
            for i in range(0, frames.size(0), chunk_size):
                chunk = frames[i : min(i + chunk_size, frames.size(0))].to(self.device)
                emb = embedder(chunk).cpu()
                embeddings.append(emb)
                del chunk

        return torch.cat(embeddings, dim=0).unsqueeze(0)  # (1, frames, dim)

    def _process_video_all_configs(
        self, embeddings: Tensor, stats: dict, video_idx: int, pbar: tqdm
    ):
        """Process all (seq_len, N, model_type) configurations for one video."""

        for memory_size in self.config.memory_sizes:
            if embeddings.size(1) < memory_size:
                continue

            for num_basis in self.config.basis_counts:
                if num_basis > memory_size:
                    continue

                # Update description with current configuration
                pbar.set_postfix_str(
                    f"Video {video_idx + 1}/{self.config.num_videos} | L={memory_size} N={num_basis}",
                    refresh=False,
                )

                score = process_single_configuration(
                    embeddings, "discrete_full", memory_size, num_basis, self.config
                )
                stats[memory_size]["discrete_full"][num_basis].update(score)
                pbar.update(1)

                score = process_single_configuration(
                    embeddings, "discrete_sub", memory_size, num_basis, self.config
                )
                stats[memory_size]["discrete_sub"][num_basis].update(score)
                pbar.update(1)

                for basis_type in self.config.basis_types:
                    score = process_single_configuration(
                        embeddings, basis_type, memory_size, num_basis, self.config
                    )
                    stats[memory_size][basis_type][num_basis].update(score)
                    pbar.update(1)

    def _finalize_results(self, stats: dict) -> dict:
        """Convert incremental stats to final results format."""
        results = {}
        for memory_size in self.config.memory_sizes:
            results[memory_size] = {}
            for model_type in stats[memory_size]:
                results[memory_size][model_type] = {}
                for num_basis in stats[memory_size][model_type]:
                    if stats[memory_size][model_type][num_basis].n > 0:
                        results[memory_size][model_type][num_basis] = (
                            stats[memory_size][model_type][num_basis].mean,
                            stats[memory_size][model_type][num_basis].std,
                        )
        return results

    def _print_summary(self, results: dict):
        """Print a summary of results."""
        print("\n" + "=" * 60)
        print("              RESULTS SUMMARY")
        print("=" * 60)

        for memory_size in self.config.memory_sizes:
            print(f"\nL = {memory_size}:")
            print("-" * 40)

            best_scores = {}
            for model_type in results[memory_size]:
                if results[memory_size][model_type]:
                    best_n = max(
                        results[memory_size][model_type].keys(),
                        key=lambda n: results[memory_size][model_type][n][0],
                    )
                    best_score = results[memory_size][model_type][best_n][0]
                    best_scores[model_type] = (best_n, best_score)

            sorted_types = sorted(
                best_scores.items(), key=lambda x: x[1][1], reverse=True
            )

            for model_type, (best_n, score) in sorted_types[:5]:
                marker = " *" if score == sorted_types[0][1][1] else ""
                print(f"  {model_type:15s} N={best_n:4d}: {score:.4f}{marker}")

        print("=" * 60)

    def _setup_plot_style(self):
        """Configure matplotlib plotting parameters."""
        plt.rcParams["font.size"] = 10
        plt.rcParams["axes.labelsize"] = 11
        plt.rcParams["axes.titlesize"] = 12
        plt.rcParams["xtick.labelsize"] = 10
        plt.rcParams["ytick.labelsize"] = 10
        plt.rcParams["legend.fontsize"] = 9
        plt.rcParams["figure.titlesize"] = 14

    def _get_plot_colors(self) -> dict:
        """Return color mapping for different basis types."""
        return {
            "rectangular": "#2E86AB",
            "gaussian": "#A23B72",
            "fourier": "#F18F01",
            "polynomial": "#C73E1D",
        }

    def _get_plot_markers(self) -> dict:
        """Return marker mapping for different basis types."""
        return {
            "rectangular": "o",
            "gaussian": "^",
            "fourier": "D",
            "polynomial": "v",
        }

    def _configure_axis(self, ax, seq_len: int):
        """Configure axis appearance and limits."""
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)
        ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)

        ax.set_xscale("log", base=2)
        ax.set_xlim(8, seq_len)

        x_ticks = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        x_ticks = [x for x in x_ticks if x <= seq_len]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(
            [str(x) if x in [8, 64, 256, 1024, 4096] else "" for x in x_ticks]
        )

        ax.set_ylim(0.15, 0.95)
        ax.set_yticks(np.arange(0.2, 1.0, 0.1))
        ax.set_xlabel("N (number of basis functions)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Cosine Similarity", fontsize=11, fontweight="bold")
        ax.set_title(
            f"Memory Size: seq_len = {seq_len:,} frames",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )

    def _plot_discrete_full(self, ax, results: dict, seq_len: int):
        """Plot discrete full memory baseline."""
        if (
            "discrete_full" not in results[seq_len]
            or not results[seq_len]["discrete_full"]
        ):
            return

        n_vals = sorted(results[seq_len]["discrete_full"].keys())
        if not n_vals:
            return

        mean_val = results[seq_len]["discrete_full"][n_vals[0]][0]
        std_val = (
            results[seq_len]["discrete_full"][n_vals[0]][1]
            if len(results[seq_len]["discrete_full"][n_vals[0]]) > 1
            else 0
        )

        ax.axhline(
            y=mean_val,
            color="#333333",
            linestyle=":",
            linewidth=2.5,
            label="Discrete HN (full memory)",
            alpha=0.8,
        )

        if std_val > 0:
            ax.fill_between(
                [8, seq_len],
                mean_val - std_val,
                mean_val + std_val,
                color="#333333",
                alpha=0.1,
            )

    def _plot_discrete_sub(self, ax, results: dict, seq_len: int):
        """Plot discrete subsampled results."""
        if (
            "discrete_sub" not in results[seq_len]
            or not results[seq_len]["discrete_sub"]
        ):
            return

        n_vals = sorted(results[seq_len]["discrete_sub"].keys())
        means = [results[seq_len]["discrete_sub"][n][0] for n in n_vals]
        stds = [
            results[seq_len]["discrete_sub"][n][1]
            if len(results[seq_len]["discrete_sub"][n]) > 1
            else 0
            for n in n_vals
        ]

        ax.plot(
            n_vals,
            means,
            color="#8B4789",
            linestyle="--",
            linewidth=2.5,
            marker="s",
            markersize=5,
            label="Discrete HN (subsampled)",
            alpha=0.9,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )

        if any(stds):
            means_arr = np.array(means)
            stds_arr = np.array(stds)
            ax.fill_between(
                n_vals,
                means_arr - stds_arr,
                means_arr + stds_arr,
                color="#8B4789",
                alpha=0.15,
            )

    def _plot_basis_results(
        self, ax, results: dict, seq_len: int, colors: dict, markers: dict
    ):
        """Plot results for basis function models."""
        for basis_type in self.config.basis_types:
            if basis_type not in results[seq_len] or not results[seq_len][basis_type]:
                continue

            n_vals = sorted(results[seq_len][basis_type].keys())
            means = [results[seq_len][basis_type][n][0] for n in n_vals]
            stds = [
                results[seq_len][basis_type][n][1]
                if len(results[seq_len][basis_type][n]) > 1
                else 0
                for n in n_vals
            ]

            ax.plot(
                n_vals,
                means,
                color=colors[basis_type],
                linewidth=2.5,
                marker=markers[basis_type],
                markersize=6,
                label=f"CHN ({basis_type})",
                alpha=0.9,
                markeredgecolor="white",
                markeredgewidth=0.5,
            )

            if any(stds):
                means_arr = np.array(means)
                stds_arr = np.array(stds)
                ax.fill_between(
                    n_vals,
                    means_arr - stds_arr,
                    means_arr + stds_arr,
                    color=colors[basis_type],
                    alpha=0.15,
                )

    def plot_results(self, results: dict):
        """Create visualization plots with professional aesthetics."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        axes = axes.flatten()

        self._setup_plot_style()
        colors = self._get_plot_colors()
        markers = self._get_plot_markers()

        for idx, seq_len in enumerate(self.config.memory_sizes):
            ax = axes[idx]
            self._configure_axis(ax, seq_len)
            self._plot_discrete_full(ax, results, seq_len)
            self._plot_discrete_sub(ax, results, seq_len)
            self._plot_basis_results(ax, results, seq_len, colors, markers)

            if idx == 0:
                legend = ax.legend(
                    loc="lower right",
                    fontsize=9,
                    framealpha=0.95,
                    edgecolor="gray",
                    fancybox=True,
                    shadow=True,
                    ncol=1,
                    columnspacing=1,
                    handlelength=2.5,
                    handletextpad=0.5,
                )
                legend.get_frame().set_linewidth(0.5)

        fig.suptitle(
            "Video Reconstruction Performance: Basis Function Comparison",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )
        fig.patch.set_facecolor("white")
        plt.tight_layout(rect=(0, 0.02, 1, 0.96))

        output_path = self.results_dir / "video-reconstruction-results.png"
        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        print(f"\nPlot saved to: {output_path}")
        plt.close()
        plt.rcParams.update(matplotlib.rcParamsDefault)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    """Main entry point."""
    start_time = time.time()

    experiment = VideoReconstructionExperiment(cfg)
    results = experiment.run_experiment()
    experiment.plot_results(results)

    elapsed = time.time() - start_time
    print(f"\n[DONE] Experiment completed in {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()

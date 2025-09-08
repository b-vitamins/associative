#!/usr/bin/env python3
"""Video reconstruction experiment comparing basis functions and memory sizes."""
# ruff: noqa: N806

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[3]))

# Suppress FFmpeg/libav warnings (like mmco errors)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;quiet"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

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

from associative.datasets.moviechat import MovieChat1K
from associative.nn.modules.config import (
    BasisConfig,
    ContinuousHopfieldConfig,
)
from associative.nn.modules.continuous import ContinuousHopfield
from associative.utils.masking import apply_spatial_mask

# Constants for memory management
MEMORY_THRESHOLD_GB = 8  # Threshold for chunking in GB
LARGE_BASIS_THRESHOLD = 1024  # Threshold for large basis cleanup
LARGE_MEMORY_THRESHOLD = 2048  # Threshold for large memory cleanup


@dataclass
class ExperimentConfig:
    """Experiment configuration parameters."""

    basis_types: list[str]
    memory_sizes: list[int]  # L values (number of frames)
    basis_counts: list[int]  # N values (number of basis functions)
    num_videos: int
    device: torch.device
    beta: float = 1.0
    num_iterations: int = 3  # CHM-Net paper uses 3
    mask_ratio: float = 0.5
    regularization: float = 0.5  # lambda from paper
    integration_points: int = 500
    resolution: int = 224


class DiscreteHopfield:
    """Modern discrete Hopfield network (matching paper implementation)."""

    def __init__(self, beta: float = 1.0, num_iterations: int = 3):
        self.beta = beta
        self.num_iterations = num_iterations

    @torch.no_grad()
    def forward(self, memories: Tensor, queries: Tensor) -> Tensor:
        """Forward pass matching CHM-Net paper.

        Args:
            memories: Shape (L_sub, D) - subsampled frames
            queries: Shape (L, D) - masked frames

        Returns:
            Reconstructed frames of shape (L, D)
        """
        output = queries
        for _ in range(self.num_iterations):
            scores = self.beta * output @ memories.T  # (L, L_sub)
            probs = functional.softmax(scores, dim=1)
            output = probs @ memories  # (L, D)
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


def create_chn_model(
    basis_type: str, num_basis: int, config: ExperimentConfig
) -> ContinuousHopfield:
    """Create a Continuous Hopfield Network model matching paper config.

    Args:
        basis_type: Type of basis function
        num_basis: Number of basis functions (N)
        config: Experiment configuration

    Returns:
        Configured ContinuousHopfield model
    """
    basis_cfg = BasisConfig(
        basis_type=basis_type,
        num_basis=num_basis,
        domain=(0.0, 1.0),
        overlap=0.0,  # Non-overlapping as per paper
    )

    chn_cfg = ContinuousHopfieldConfig(
        basis_config=basis_cfg,
        beta=config.beta,
        regularization=config.regularization,
        integration_points=config.integration_points,
        num_iterations=config.num_iterations,
    )

    return ContinuousHopfield(chn_cfg).to(config.device)


@torch.no_grad()
def _compute_cosine_similarity_chunked(
    output: Tensor, target: Tensor, chunk_size: int = 512
) -> float:
    """Compute cosine similarity in chunks to save memory."""
    L = output.size(0)
    cosine_sims = []

    for i in range(0, L, chunk_size):
        end_idx = min(i + chunk_size, L)
        output_chunk = output[i:end_idx]
        target_chunk = target[i:end_idx]

        # Compute norms for chunk
        pred_norm = output_chunk.norm(p=2, dim=1)
        target_norm = target_chunk.norm(p=2, dim=1)

        # Compute cosine similarity for chunk
        dot_product = (output_chunk * target_chunk).sum(dim=1)
        chunk_sim = dot_product / (pred_norm * target_norm + 1e-8)
        cosine_sims.append(chunk_sim.mean().item())

        del output_chunk, target_chunk, pred_norm, target_norm, dot_product, chunk_sim

    return sum(cosine_sims) / len(cosine_sims)


@torch.no_grad()
def _evaluate_discrete_hopfield(
    video_flat: Tensor, queries: Tensor, num_basis: int, config: ExperimentConfig
) -> float:
    """Evaluate discrete Hopfield network."""
    L = video_flat.size(0)

    # Select memories
    if num_basis < L:
        indices = torch.linspace(0, L - 1, num_basis).long()
        memories = video_flat[indices]
    else:
        memories = video_flat.clone()

    # Initialize variables for type checking
    output = queries.clone()
    scores = None
    probs = None

    # Iterative retrieval
    for _ in range(config.num_iterations):
        scores = config.beta * output @ memories.T
        probs = functional.softmax(scores, dim=1)
        output = probs @ memories

    # Compute similarity
    score = _compute_cosine_similarity_chunked(output, video_flat)

    # Cleanup
    del memories, output
    if scores is not None:
        del scores
    if probs is not None:
        del probs

    return score


@torch.no_grad()
def _evaluate_continuous_hopfield_chunked(
    video_flat: Tensor,
    queries: Tensor,
    model_type: str,
    num_basis: int,
    config: ExperimentConfig,
) -> float:
    """Evaluate continuous Hopfield with chunking for large inputs."""
    L = video_flat.size(0)
    max_chunk_size = max(64, min(512, L // 4))
    all_outputs = []

    for chunk_start in range(0, L, max_chunk_size):
        chunk_end = min(chunk_start + max_chunk_size, L)
        chunk_video = video_flat[chunk_start:chunk_end]
        chunk_queries = queries[chunk_start:chunk_end]

        # Create model for this chunk
        chunk_basis = min(num_basis, chunk_end - chunk_start, 256)
        model = create_chn_model(model_type, chunk_basis, config)

        # Add batch dimension
        video_batch = chunk_video.unsqueeze(0)
        queries_batch = chunk_queries.unsqueeze(0)

        try:
            # Forward pass
            output, _ = model(video_batch, queries_batch)
            output = output.squeeze(0)
            all_outputs.append(output.cpu())
        except torch.cuda.OutOfMemoryError:
            print(f"OOM on chunk {chunk_start}-{chunk_end}, using input as fallback")
            all_outputs.append(chunk_queries.cpu())
        finally:
            # Clear model memory
            if hasattr(model, "memory") and hasattr(model.memory, "coefficients"):
                model.memory.coefficients = torch.empty(0, device=config.device)
            if "model" in locals():
                del model
            if "video_batch" in locals():
                del video_batch
            if "queries_batch" in locals():
                del queries_batch
            if "output" in locals():
                del output  # type: ignore[possibly-undefined]
            torch.cuda.empty_cache()

    # Concatenate outputs
    if all_outputs:
        output = torch.cat(all_outputs, dim=0).to(config.device)
    else:
        output = queries.clone()
    del all_outputs

    # Compute similarity
    score = _compute_cosine_similarity_chunked(output, video_flat)

    # Cleanup
    if "output" in locals():
        del output

    return score


@torch.no_grad()
def _evaluate_continuous_hopfield_normal(
    video_flat: Tensor,
    queries: Tensor,
    model_type: str,
    num_basis: int,
    config: ExperimentConfig,
) -> float:
    """Evaluate continuous Hopfield without chunking."""
    model = create_chn_model(model_type, num_basis, config)

    # Add batch dimension
    video_batch = video_flat.unsqueeze(0)
    queries_batch = queries.unsqueeze(0)

    try:
        output, _ = model(video_batch, queries_batch)
        output = output.squeeze(0)
    except torch.cuda.OutOfMemoryError:
        print("OOM in normal processing, fallback to queries")
        output = queries.clone()
    finally:
        # Clear model memory
        if hasattr(model, "memory") and hasattr(model.memory, "coefficients"):
            model.memory.coefficients = torch.empty(0, device=config.device)
        if "model" in locals():
            del model
        if "video_batch" in locals():
            del video_batch
        if "queries_batch" in locals():
            del queries_batch
        torch.cuda.empty_cache()

    # Compute similarity
    score = _compute_cosine_similarity_chunked(output, video_flat)

    # Cleanup
    if "output" in locals():
        del output

    return score


@torch.no_grad()
def evaluate_single_config(
    video_cpu: Tensor,
    model_type: str,
    memory_size: int,
    num_basis: int,
    config: ExperimentConfig,
) -> float:
    """Evaluate ONE configuration and return score. Complete memory isolation.

    Args:
        video_cpu: Video on CPU [frames, C, H, W], range [0, 1]
        model_type: "discrete" or basis type
        memory_size: L - number of frames
        num_basis: N - number of basis functions
        config: Experiment configuration

    Returns:
        Cosine similarity score as Python float
    """
    try:
        # Preprocessing: clone, sample, normalize, mask
        video = video_cpu.clone()

        # Sample frames if needed
        if video.size(0) > memory_size:
            indices = torch.linspace(0, video.size(0) - 1, memory_size).long()
            video = video[indices]

        # Normalize to [-1, 1]
        video = (video - 0.5) / 0.5

        # Convert to [L, H, W, C] for masking
        video = video.permute(0, 2, 3, 1)

        # Apply mask (on CPU)
        video_masked, _ = apply_spatial_mask(video, config.mask_ratio, "lower")

        # Flatten and move to GPU
        L = video.size(0)
        video_flat = video.reshape(L, -1).to(config.device)
        queries = video_masked.reshape(L, -1).to(config.device)

        # Clean up CPU tensors
        del video, video_masked

        # Dispatch to appropriate evaluation function
        if model_type == "discrete":
            score = _evaluate_discrete_hopfield(video_flat, queries, num_basis, config)
        else:
            # Continuous Hopfield - check memory requirements
            total_elements = L * video_flat.shape[1]
            memory_estimate_gb = (total_elements * num_basis * 4) / (1024**3)

            if memory_estimate_gb > MEMORY_THRESHOLD_GB:
                score = _evaluate_continuous_hopfield_chunked(
                    video_flat, queries, model_type, num_basis, config
                )
            else:
                score = _evaluate_continuous_hopfield_normal(
                    video_flat, queries, model_type, num_basis, config
                )

        # Clean up GPU tensors
        del video_flat, queries

    finally:
        # ALWAYS clear GPU memory
        torch.cuda.empty_cache()

    return float(score)  # Ensure Python float


class VideoReconstructionExperiment:
    """Experiment runner matching CHM-Net paper setup."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results_dir = Path(cfg.experiment.get("results_dir", "outputs/results"))
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.config = ExperimentConfig(
            basis_types=["rectangular", "gaussian", "fourier", "polynomial"],
            memory_sizes=[128, 256, 512, 1024],  # L values
            basis_counts=[
                8,
                16,
                32,
                64,
                128,
                256,
                512,
                1024,
            ],  # N values (will be filtered per L)
            num_videos=cfg.experiment.get("num_videos", 100),
            device=self.device,
            beta=1.0,  # From paper
            num_iterations=3,  # From paper
            mask_ratio=0.5,  # From paper
            regularization=0.5,  # lambda from paper
            integration_points=500,  # From paper
            resolution=224,  # From paper
        )

        self.total_evaluations = self._calculate_total_evaluations()

        self._print_header()

    def _calculate_total_evaluations(self) -> int:
        """Calculate total number of model evaluations."""
        count = 0
        for memory_size in self.config.memory_sizes:
            # Get valid N values for this L (powers of 2 up to L)
            valid_n_values = [n for n in self.config.basis_counts if n <= memory_size]
            for _num_basis in valid_n_values:
                # One discrete + basis_types continuous models
                count += 1 + len(self.config.basis_types)
        return count * self.config.num_videos

    def _print_header(self):
        """Print experiment header."""
        print("-" * 60)
        print("VIDEO RECONSTRUCTION EXPERIMENT")
        print("-" * 60)
        print(f"Device: {self.device}")
        print(f"Videos: {self.config.num_videos}")
        print(f"Memory (L): {self.config.memory_sizes}")
        print("Basis (N) per L:")
        for L in self.config.memory_sizes:
            valid_n = [n for n in self.config.basis_counts if n <= L]
            print(f"  L={L:4d}: N={valid_n}")
        print(f"Types:  {', '.join(self.config.basis_types)} + discrete")
        print(f"Total:  {self.total_evaluations:,} evaluations")
        print("-" * 60)

    def run_experiment(self) -> dict:
        """Run the full experiment matching paper's approach."""
        stats = self._initialize_stats()

        # Load dataset using test split as per paper
        dataset = MovieChat1K(
            split="test",  # Use test split
            num_frames=max(self.config.memory_sizes),
            resolution=self.config.resolution,
            download=False,
            max_videos=self.config.num_videos,
        )

        print(f"Found {len(dataset)} videos in test split")
        print("Processing videos...")
        pbar = tqdm(
            total=self.total_evaluations,
            desc="Progress",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        for video_idx in range(min(len(dataset), self.config.num_videos)):
            sample = dataset[video_idx]
            video_cpu = sample["frames"].cpu()  # Keep on CPU: [L, C, H, W]
            video_id = sample["video_id"]

            # Process all configs for this video
            self._process_video_all_configs(video_cpu, stats, video_idx, video_id, pbar)

            # Clean up
            del sample, video_cpu

            # Periodic major cleanup
            if (video_idx + 1) % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        pbar.close()

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
                "discrete",  # Only one discrete baseline
            ]:
                stats[memory_size][model_type] = {}
                for num_basis in self.config.basis_counts:
                    if num_basis <= memory_size:
                        stats[memory_size][model_type][num_basis] = IncrementalStats()

        return stats

    # Removed _setup_embedder and _compute_embeddings methods as we work with raw frames

    def _process_video_all_configs(
        self, video_cpu: Tensor, stats: dict, video_idx: int, video_id: str, pbar: tqdm
    ):
        """Process all (L, N, model_type) configurations for one video.

        Args:
            video_cpu: Video tensor on CPU [frames, C, H, W]
        """
        for memory_size in self.config.memory_sizes:
            if video_cpu.size(0) < memory_size:
                continue

            for num_basis in self.config.basis_counts:
                if num_basis > memory_size:
                    continue

                # Update progress bar
                pbar.set_postfix_str(
                    f"V{video_idx + 1} L={memory_size} N={num_basis}",
                    refresh=False,
                )

                # Process discrete baseline with error handling
                try:
                    score = evaluate_single_config(
                        video_cpu, "discrete", memory_size, num_basis, self.config
                    )
                    stats[memory_size]["discrete"][num_basis].update(score)
                except Exception as e:
                    print(f"\nError in discrete L={memory_size} N={num_basis}: {e}")
                    stats[memory_size]["discrete"][num_basis].update(0.0)
                finally:
                    pbar.update(1)
                    torch.cuda.empty_cache()

                # Process each basis type with error handling
                for basis_type in self.config.basis_types:
                    try:
                        score = evaluate_single_config(
                            video_cpu, basis_type, memory_size, num_basis, self.config
                        )
                        stats[memory_size][basis_type][num_basis].update(score)
                    except Exception as e:
                        print(
                            f"\nError in {basis_type} L={memory_size} N={num_basis}: {e}"
                        )
                        stats[memory_size][basis_type][num_basis].update(0.0)
                    finally:
                        pbar.update(1)
                        torch.cuda.empty_cache()

                # Extra cleanup for large configurations
                if (
                    num_basis >= LARGE_BASIS_THRESHOLD
                    or memory_size >= LARGE_MEMORY_THRESHOLD
                ):
                    gc.collect()
                    torch.cuda.empty_cache()

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
        print("=" * 60)
        print("              RESULTS SUMMARY")
        print("=" * 60)

        for memory_size in self.config.memory_sizes:
            print(f"L = {memory_size}:")
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
        """Configure matplotlib plotting parameters for publication quality."""
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
        plt.rcParams["font.size"] = 11
        plt.rcParams["axes.labelsize"] = 13
        plt.rcParams["axes.titlesize"] = 14
        plt.rcParams["xtick.labelsize"] = 11
        plt.rcParams["ytick.labelsize"] = 11
        plt.rcParams["legend.fontsize"] = 10
        plt.rcParams["figure.titlesize"] = 16
        plt.rcParams["lines.linewidth"] = 2.5
        plt.rcParams["lines.markersize"] = 7
        plt.rcParams["axes.linewidth"] = 1.5
        plt.rcParams["grid.linewidth"] = 0.5
        plt.rcParams["grid.alpha"] = 0.3

    def _get_plot_colors(self) -> dict:
        """Return professional color mapping for different basis types."""
        return {
            "rectangular": "#1f77b4",  # Strong blue
            "gaussian": "#ff7f0e",  # Orange
            "fourier": "#2ca02c",  # Green
            "polynomial": "#d62728",  # Red
        }

    def _get_plot_markers(self) -> dict:
        """Return distinct marker mapping for different basis types."""
        return {
            "rectangular": "o",  # Circle
            "gaussian": "s",  # Square
            "fourier": "D",  # Diamond
            "polynomial": "^",  # Triangle up
        }

    def _configure_axis(self, ax, seq_len: int):
        """Configure axis appearance for publication quality."""
        # Remove top and right spines for cleaner look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)

        # Add subtle grid
        ax.grid(True, alpha=0.25, linestyle="-", linewidth=0.5, which="both")
        ax.set_axisbelow(True)

        # Set log scale for x-axis
        ax.set_xscale("log", base=2)
        ax.set_xlim(6, seq_len * 1.5)

        # Configure x-ticks based on L value
        x_ticks = [8, 16, 32, 64, 128, 256, 512, 1024]
        x_ticks = [x for x in x_ticks if x <= seq_len]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(x) for x in x_ticks])

        # Minor ticks for better readability
        ax.tick_params(axis="both", which="major", labelsize=11, length=6, width=1.5)
        ax.tick_params(axis="both", which="minor", length=3, width=1)

        # Set y-axis limits and ticks
        ax.set_ylim(0.4, 1.0)
        ax.set_yticks(np.arange(0.4, 1.05, 0.1))
        ax.set_yticklabels([f"{y:.1f}" for y in np.arange(0.4, 1.05, 0.1)])

        # Labels with proper formatting
        ax.set_xlabel("Basis Functions (N)", fontsize=13, fontweight="medium")
        ax.set_ylabel("Cosine Similarity", fontsize=13, fontweight="medium")

        # Title
        ax.set_title(
            f"L = {seq_len} frames",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )

    def _plot_discrete(self, ax, results: dict, seq_len: int):
        """Plot discrete Hopfield baseline with professional styling."""
        if "discrete" not in results[seq_len] or not results[seq_len]["discrete"]:
            return

        n_vals = sorted(results[seq_len]["discrete"].keys())
        means = [results[seq_len]["discrete"][n][0] for n in n_vals]
        stds = [
            results[seq_len]["discrete"][n][1]
            if len(results[seq_len]["discrete"][n]) > 1
            else 0
            for n in n_vals
        ]

        # Plot with distinctive style for baseline
        ax.plot(
            n_vals,
            means,
            color="#9467bd",  # Purple for baseline
            linestyle="--",
            linewidth=2.5,
            marker="v",
            markersize=7,
            label="Discrete Hopfield",
            alpha=0.85,
            markeredgecolor="white",
            markeredgewidth=1.0,
            zorder=5,  # Draw on top
        )

        # Add shaded error region
        if any(stds):
            means_arr = np.array(means)
            stds_arr = np.array(stds)
            ax.fill_between(
                n_vals,
                means_arr - stds_arr,
                means_arr + stds_arr,
                color="#9467bd",
                alpha=0.12,
                zorder=1,
            )

    def _plot_basis_results(
        self, ax, results: dict, seq_len: int, colors: dict, markers: dict
    ):
        """Plot results for basis function models with professional styling."""
        for i, basis_type in enumerate(self.config.basis_types):
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

            # Plot with solid lines for continuous models
            ax.plot(
                n_vals,
                means,
                color=colors[basis_type],
                linewidth=2.5,
                marker=markers[basis_type],
                markersize=7,
                label=f"CHN {basis_type.capitalize()}",
                alpha=0.9,
                markeredgecolor="white",
                markeredgewidth=1.0,
                zorder=10 + i,  # Layer properly
            )

            # Add shaded error regions
            if any(stds):
                means_arr = np.array(means)
                stds_arr = np.array(stds)
                ax.fill_between(
                    n_vals,
                    means_arr - stds_arr,
                    means_arr + stds_arr,
                    color=colors[basis_type],
                    alpha=0.12,
                    zorder=2,
                )

    def plot_results(self, results: dict):
        """Create publication-quality visualization plots."""
        # Create figure with optimal spacing
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(
            2, 2, hspace=0.25, wspace=0.2, left=0.08, right=0.95, top=0.92, bottom=0.08
        )

        self._setup_plot_style()
        colors = self._get_plot_colors()
        markers = self._get_plot_markers()

        for idx, seq_len in enumerate(self.config.memory_sizes):
            ax = fig.add_subplot(gs[idx // 2, idx % 2])
            self._configure_axis(ax, seq_len)

            # Plot discrete baseline first (so it appears behind)
            self._plot_discrete(ax, results, seq_len)

            # Plot continuous models on top
            self._plot_basis_results(ax, results, seq_len, colors, markers)

            # Add legend to first subplot
            if idx == 0:
                legend = ax.legend(
                    loc="lower right",
                    fontsize=10,
                    framealpha=0.95,
                    edgecolor="#cccccc",
                    fancybox=False,
                    shadow=False,
                    ncol=1,
                    columnspacing=1.0,
                    handlelength=2.5,
                    handletextpad=0.8,
                    borderpad=0.5,
                    frameon=True,
                )
                legend.get_frame().set_linewidth(1.0)
                legend.get_frame().set_facecolor("white")

            # Add subtle annotations for key points
            if seq_len in results and "discrete" in results[seq_len]:
                n_vals = sorted(results[seq_len]["discrete"].keys())
                if n_vals:
                    # Annotate compression ratio
                    ax.text(
                        0.05,
                        0.95,
                        f"Compression: {n_vals[0] / seq_len:.1%} - {n_vals[-1] / seq_len:.1%}",
                        transform=ax.transAxes,
                        fontsize=9,
                        verticalalignment="top",
                        bbox={
                            "boxstyle": "round,pad=0.3",
                            "facecolor": "white",
                            "edgecolor": "gray",
                            "alpha": 0.8,
                        },
                    )

        # Add main title
        fig.suptitle(
            "Video Reconstruction Results",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )
        fig.text(
            0.5,
            0.94,
            "MovieChat-1K Test Split",
            ha="center",
            fontsize=12,
            style="italic",
        )

        # Add common axis labels
        fig.text(
            0.5,
            0.02,
            "Basis Functions (N)",
            ha="center",
            fontsize=14,
            fontweight="medium",
        )
        fig.text(
            0.02,
            0.5,
            "Cosine Similarity",
            va="center",
            rotation="vertical",
            fontsize=14,
            fontweight="medium",
        )

        # Save with high quality
        output_path = self.results_dir / "video-reconstruction-results.png"
        plt.savefig(
            output_path,
            dpi=300,  # Higher DPI for publication
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            pad_inches=0.1,
        )

        # Also save as PDF for publications
        pdf_path = self.results_dir / "video-reconstruction-results.pdf"
        plt.savefig(
            pdf_path,
            format="pdf",
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            pad_inches=0.1,
        )

        print("Plots saved to:")
        print(f"  PNG: {output_path}")
        print(f"  PDF: {pdf_path}")
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
    print(f"[DONE] Experiment completed in {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()

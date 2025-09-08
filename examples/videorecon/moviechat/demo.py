#!/usr/bin/env python3
"""Demo comparing basis functions for video reconstruction matching CHM-Net paper.

This demo evaluates Continuous Hopfield Networks with various basis functions
on video reconstruction using MovieChat-1K test split with exact paper settings.
"""
# ruff: noqa: N803, N806, N812

import gc
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[3]))

# Suppress FFmpeg/libav warnings (like mmco errors)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;quiet"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm

from associative.datasets.moviechat import MovieChat1K
from associative.nn.modules.config import BasisConfig, ContinuousHopfieldConfig
from associative.nn.modules.continuous import ContinuousHopfield
from associative.utils.masking import apply_spatial_mask

matplotlib.use("Agg")


class VideoReconstructionDemo:
    """Demo runner for comparing basis functions on video reconstruction."""

    def __init__(self, N: int = 128, L: int = 1024):
        """Initialize demo with paper parameters.

        Args:
            N: Number of basis functions to use (default: 128 for good compression)
            L: Number of frames/memory size (default: 1024)
        """
        # Configuration from CHM-Net paper
        self.L = L  # Number of frames (memory size)
        self.N = N  # Fixed number of basis functions for comparison
        self.resolution = 224
        self.beta = 1.0
        self.num_iterations = 3
        self.mask_ratio = 0.5
        self.regularization = 0.5  # λ from paper
        self.integration_points = 500

        # Basis types to compare
        self.basis_types = ["rectangular", "gaussian", "fourier", "polynomial"]

        # Number of videos to average over
        self.num_videos = 100

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Results directory
        self.results_dir = Path("outputs/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def create_model(self, basis_type: str) -> ContinuousHopfield:
        """Create a Continuous Hopfield model with specified basis type.

        Args:
            basis_type: Type of basis function

        Returns:
            Configured ContinuousHopfield model
        """
        basis_config = BasisConfig(
            basis_type=basis_type,
            num_basis=self.N,  # Fixed N for all basis types
            domain=(0.0, 1.0),
            overlap=0.0,  # Non-overlapping as per paper
        )

        config = ContinuousHopfieldConfig(
            basis_config=basis_config,
            beta=self.beta,
            regularization=self.regularization,
            integration_points=self.integration_points,
            num_iterations=self.num_iterations,
        )

        return ContinuousHopfield(config).to(self.device)

    @torch.no_grad()
    def prepare_video(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare video for Hopfield network processing.

        Args:
            video: Video frames tensor of shape [L, C, H, W], range [0, 1]

        Returns:
            Tuple of (video_flat, queries) both of shape [L, D] on device
        """
        L = video.size(0)

        # Normalize to [-1, 1] as per CHM-Net paper
        video = (video - 0.5) / 0.5

        # Convert to [L, H, W, C] for masking
        video = video.permute(0, 2, 3, 1)

        # Apply spatial mask
        video_masked, _ = apply_spatial_mask(
            video,
            mask_ratio=self.mask_ratio,
            mask_type="lower",
        )

        # Reshape to [L, D] where D = H*W*C
        video_flat = video.reshape(L, -1).to(self.device)
        queries = video_masked.reshape(L, -1).to(self.device)

        # Clean up intermediate tensors
        del video_masked

        return video_flat, queries

    def process_video(self, video: torch.Tensor, basis_type: str) -> float:
        """Process a single video with specified basis type.

        Args:
            video: Video frames tensor of shape [L, C, H, W], range [0, 1]
            basis_type: Type of basis function

        Returns:
            Cosine similarity score
        """
        # Prepare video
        video_flat, queries = self.prepare_video(video)

        # Create model with specified basis type
        model = self.create_model(basis_type)

        # Add batch dimension for model forward pass
        video_batch = video_flat.unsqueeze(0)  # (1, L, D)
        queries_batch = queries.unsqueeze(0)  # (1, L, D)

        # Perform reconstruction
        with torch.no_grad():
            reconstructed, _ = model(video_batch, queries_batch)
            reconstructed = reconstructed.squeeze(0)  # Back to (L, D)

        # CRITICAL: Clear the model's internal coefficients to free memory
        if (
            hasattr(model, "memory")
            and hasattr(model.memory, "coefficients")
            and model.memory.coefficients is not None
        ):
            del model.memory.coefficients
            # Set to empty tensor instead of None to avoid type issues
            model.memory.coefficients = torch.empty(0, device=self.device)

        # Delete model completely
        del model

        # Delete batch tensors
        del video_batch, queries_batch
        torch.cuda.empty_cache()

        # Compute cosine similarity
        with torch.no_grad():
            video_norm = F.normalize(video_flat, p=2, dim=1)
            recon_norm = F.normalize(reconstructed, p=2, dim=1)
            cosine_sim = (video_norm * recon_norm).sum(dim=1).mean().item()

        # Clean up remaining tensors
        del reconstructed, video_norm, recon_norm
        del video_flat, queries
        torch.cuda.empty_cache()

        return cosine_sim

    def _print_configuration(self):
        """Print demo configuration."""
        print("=" * 60)
        print("VIDEO RECONSTRUCTION DEMO")
        print("=" * 60)
        print("Configuration:")
        print(f"  L (frames): {self.L}")
        print(f"  N (basis functions): {self.N}")
        print(f"  Compression ratio: {self.N / self.L:.1%}")
        print(f"  Resolution: {self.resolution}x{self.resolution}")
        print(f"  Beta: {self.beta}")
        print(f"  Iterations: {self.num_iterations}")
        print(f"  Mask ratio: {self.mask_ratio}")
        print(f"  Regularization (lambda): {self.regularization}")
        print(f"  Integration points: {self.integration_points}")
        print(f"  Videos to test: {self.num_videos}")
        print(f"  Basis types: {', '.join(self.basis_types)}")
        print("=" * 60)

    def _evaluate_discrete_baseline(self, video_flat, queries):
        """Evaluate discrete Hopfield baseline."""
        # Subsample frames for discrete baseline (use same N as continuous)
        if self.N < self.L:
            indices = torch.linspace(0, self.L - 1, self.N).long()
            X_sub = video_flat[indices]
        else:
            X_sub = video_flat

        # Run discrete Hopfield
        Q_disc = queries.clone()
        for _ in range(self.num_iterations):
            scores = self.beta * Q_disc @ X_sub.T
            probs = F.softmax(scores, dim=1)
            Q_disc = probs @ X_sub

        # Compute similarity
        video_norm = F.normalize(video_flat, p=2, dim=1)
        Q_disc_norm = F.normalize(Q_disc, p=2, dim=1)
        cosine_sim_disc = (video_norm * Q_disc_norm).sum(dim=1).mean().item()

        # Clean up
        del X_sub, Q_disc, Q_disc_norm, video_norm

        return cosine_sim_disc

    def run_demo(self):
        """Run the complete demo experiment."""
        self._print_configuration()

        # Load dataset (test split as per paper)
        dataset = MovieChat1K(
            split="test",
            num_frames=self.L,
            resolution=self.resolution,
            download=False,
            max_videos=self.num_videos,
        )

        print(f"Found {len(dataset)} videos in test split")
        print("Processing videos...")

        # Initialize results storage
        results = {basis_type: [] for basis_type in self.basis_types}

        # Also track discrete baseline
        results["discrete"] = []

        # Process each video
        for video_idx in tqdm(range(len(dataset)), desc="Videos"):
            sample = dataset[video_idx]
            video = sample["frames"]  # Shape: [L, C, H, W], range [0, 1]

            # Keep video on CPU initially
            video = video.cpu()

            # Test each basis type
            for basis_type in self.basis_types:
                cosine_sim = self.process_video(video, basis_type)
                results[basis_type].append(cosine_sim)
                # Clean up after each basis type to prevent accumulation
                torch.cuda.empty_cache()

            # Discrete baseline (subsampled frames)
            video_flat, queries = self.prepare_video(video)
            cosine_sim_disc = self._evaluate_discrete_baseline(video_flat, queries)
            results["discrete"].append(cosine_sim_disc)

            # Clean up
            del video, video_flat, queries
            del sample
            torch.cuda.empty_cache()

            # Extra cleanup every 10 videos
            if (video_idx + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        # Final cleanup before statistics
        torch.cuda.empty_cache()

        # Compute statistics
        stats = {}
        for model_type in [*list(self.basis_types), "discrete"]:
            similarities = results[model_type]
            stats[model_type] = {
                "mean": np.mean(similarities),
                "std": np.std(similarities) if len(similarities) > 1 else 0.0,
                "min": np.min(similarities) if similarities else 0.0,
                "max": np.max(similarities) if similarities else 0.0,
            }

        return stats

    def plot_results(self, stats):
        """Create bar plot visualization comparing basis functions."""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Prepare data for plotting
        model_types = [*self.basis_types, "discrete"]
        means = [stats[mt]["mean"] for mt in model_types]
        stds = [stats[mt]["std"] for mt in model_types]

        # Colors for different model types
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        # Create bar plot
        x_pos = np.arange(len(model_types))
        bars = ax.bar(
            x_pos, means, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5
        )

        # Add error bars
        ax.errorbar(
            x_pos,
            means,
            yerr=stds,
            fmt="none",
            color="black",
            capsize=5,
            capthick=2,
            elinewidth=2,
            alpha=0.7,
        )

        # Add value labels on bars
        for _i, (bar, mean, std) in enumerate(zip(bars, means, stds, strict=False)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + std + 0.01,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        # Customize plot
        ax.set_xticks(x_pos)
        ax.set_xticklabels([mt.capitalize() for mt in model_types], fontsize=12)
        ax.set_xlabel("Basis Function", fontsize=14, fontweight="medium")
        ax.set_ylabel("Cosine Similarity", fontsize=14, fontweight="medium")
        ax.set_title(
            f"Video Reconstruction (N={self.N}, L={self.L})",
            fontsize=15,
            fontweight="bold",
            pad=20,
        )

        # Set y-axis limits
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_locator(MultipleLocator(0.1))

        # Add grid
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add horizontal line at discrete baseline for reference
        discrete_mean = stats["discrete"]["mean"]
        ax.axhline(
            y=discrete_mean,
            color="gray",
            linestyle=":",
            alpha=0.5,
            label=f"Discrete baseline: {discrete_mean:.3f}",
        )
        ax.legend(loc="upper right", fontsize=11)

        # Save plot
        output_path = self.results_dir / "demo-basis-comparison.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Plot saved to: {output_path}")
        plt.close()

    def print_summary(self, stats):
        """Print summary of results."""
        print("=" * 70)
        print(
            f"           RESULTS SUMMARY (N={self.N}, compression={self.N / self.L:.1%})"
        )
        print("=" * 70)
        print("Model Type       |   Mean   |   Std    |   Min    |   Max")
        print("-" * 70)

        # Sort by mean performance
        sorted_models = sorted(
            stats.keys(), key=lambda x: stats[x]["mean"], reverse=True
        )

        for model_type in sorted_models:
            s = stats[model_type]
            is_best = model_type == sorted_models[0]
            marker = " *" if is_best else "  "

            # Different formatting for discrete baseline
            if model_type == "discrete":
                print("-" * 70)
                print(
                    f"Discrete         | {s['mean']:.4f} | {s['std']:.4f} | "
                    f"{s['min']:.4f} | {s['max']:.4f}{marker}"
                )
            else:
                print(
                    f"CHN {model_type:12s} | {s['mean']:.4f} | {s['std']:.4f} | "
                    f"{s['min']:.4f} | {s['max']:.4f}{marker}"
                )

        print("=" * 70)
        print("* = Best performing model")
        # Analysis
        best_continuous = max(
            (mt for mt in self.basis_types), key=lambda x: stats[x]["mean"]
        )
        discrete_mean = stats["discrete"]["mean"]
        best_cont_mean = stats[best_continuous]["mean"]

        print("Analysis:")
        print(f"  Best continuous: {best_continuous} ({best_cont_mean:.4f})")
        print(f"  Discrete baseline: {discrete_mean:.4f}")

        if best_cont_mean > discrete_mean:
            improvement = (best_cont_mean - discrete_mean) / discrete_mean * 100
            print(f"  Continuous beats discrete by {improvement:.1f}%")
        else:
            deficit = (discrete_mean - best_cont_mean) / discrete_mean * 100
            print(f"  Discrete beats continuous by {deficit:.1f}%")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare basis functions for video reconstruction"
    )
    parser.add_argument(
        "--N", type=int, default=128, help="Number of basis functions (default: 128)"
    )
    parser.add_argument(
        "--L", type=int, default=1024, help="Number of frames (default: 1024)"
    )
    parser.add_argument(
        "--num-videos",
        type=int,
        default=100,
        help="Number of videos to test (default: 100)",
    )
    args = parser.parse_args()

    demo = VideoReconstructionDemo(N=args.N, L=args.L)
    demo.num_videos = args.num_videos

    stats = demo.run_demo()
    demo.plot_results(stats)
    demo.print_summary(stats)
    print("[DONE] Demo complete")


if __name__ == "__main__":
    main()

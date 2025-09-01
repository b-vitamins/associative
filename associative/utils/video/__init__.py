"""Video processing utilities for associative memory models.

This module provides PyTorch-style video processing utilities for training
and evaluating associative memory models on video reconstruction tasks.
Follows PyTorch design patterns for familiarity and composability.

Usage:
    >>> import associative.utils.video as VU
    >>> import associative.utils.video.functional as VF
    >>> from associative.utils.video import transforms, loaders, embeddings, metrics

    # Functional API
    >>> frames = VF.load_video("path/to/video.mp4", num_frames=512)
    >>> masked_frames, mask = VF.apply_mask(frames, mask_ratio=0.5, mask_type="bottom_half")

    # Class API with transforms
    >>> transform = transforms.Compose([
    ...     transforms.UniformSample(num_frames=512),
    ...     transforms.Resize((224, 224)),
    ...     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ... ])

    # Data loading
    >>> loader = loaders.VideoLoader(dataset, transform=transform, batch_size=32)

    # Embedding extraction
    >>> embedder = embeddings.EVAClipEmbedder(model_name="eva_clip_vit_g_14")
    >>> embeds = embedder(frames)

    # Metrics calculation
    >>> calculator = metrics.ReconstructionMetrics()
    >>> similarity = calculator.cosine_similarity(pred_frames, target_frames)
"""

# Import main classes for convenience
from .embeddings import EmbeddingExtractor, EVAClipEmbedder, register_embedder
from .loaders import BatchProcessor, VideoDataLoader
from .metrics import CosineSimilarity, ReconstructionMetrics
from .transforms import ApplyMask, Compose, Normalize, Resize, UniformSample

# Version info
__version__ = "0.1.0"

# Main exports
__all__ = [
    "ApplyMask",
    "BatchProcessor",
    # Transforms
    "Compose",
    "CosineSimilarity",
    "EVAClipEmbedder",
    # Core classes
    "EmbeddingExtractor",
    "Normalize",
    "ReconstructionMetrics",
    "Resize",
    "UniformSample",
    "VideoDataLoader",
    # Registration
    "register_embedder",
]

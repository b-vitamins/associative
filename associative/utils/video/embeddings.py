"""Video embedding extraction modules.

Provides PyTorch modules for extracting embeddings from video frames
using pre-trained vision models with clean, consistent APIs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from transformers import (
    Blip2QFormerConfig,
    Blip2QFormerModel,
    CLIPVisionConfig,
    CLIPVisionModel,
)

from ._registry import EMBEDDER_REGISTRY, register_embedder

# Constants for tensor dimensions
VIDEO_DIMS_4D = 4  # (N, C, H, W)
VIDEO_DIMS_5D = 5  # (B, N, C, H, W)
RGB_CHANNELS = 3


@dataclass
class VisionModelConfig:
    """Configuration for vision transformer models."""

    image_size: int
    patch_size: int
    num_channels: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    layer_norm_eps: float


# Vision model configurations
MODEL_CONFIGS = {
    "eva_clip_vit_g_14": VisionModelConfig(
        image_size=224,
        patch_size=14,
        num_channels=3,
        hidden_size=1408,
        intermediate_size=6144,
        num_hidden_layers=40,
        num_attention_heads=16,
        layer_norm_eps=1e-6,
    ),
    "clip_vit_b_32": VisionModelConfig(
        image_size=224,
        patch_size=32,
        num_channels=3,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        layer_norm_eps=1e-5,
    ),
}


class EmbeddingExtractor(nn.Module, ABC):
    """Abstract base class for video embedding extractors.

    Provides a consistent interface for all embedding extractors with
    support for parameter freezing and training mode management.

    Args:
        freeze_weights: Whether to freeze model weights
        device: Device for computation
        dtype: Data type for parameters
    """

    def __init__(
        self,
        freeze_weights: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.freeze_weights = freeze_weights

        if device is not None:
            self.to(device)
        if dtype is not None:
            self.to(dtype)

    @abstractmethod
    def forward(self, frames: Tensor) -> Tensor:
        """Extract embeddings from video frames.

        Args:
            frames: Input frames of shape (N, C, H, W) or (B, N, C, H, W)
                   where N = num_frames, C = channels, H = height, W = width

        Returns:
            Embeddings of shape (N, embed_dim) or (B, N, embed_dim)

        Raises:
            ValueError: If input has invalid shape or format
        """
        pass

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Embedding dimension of the model."""
        pass

    @property
    @abstractmethod
    def expected_input_size(self) -> tuple[int, int]:
        """Expected input size as (height, width)."""
        pass

    def freeze(self) -> None:
        """Freeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = False
        self.freeze_weights = True

    def unfreeze(self) -> None:
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = True
        self.freeze_weights = False

    def train(self, mode: bool = True) -> "EmbeddingExtractor":
        """Set training mode, respecting freeze state."""
        if self.freeze_weights:
            # Stay in eval mode when frozen
            return super().train(False)
        return super().train(mode)


class EVAClipEmbedder(EmbeddingExtractor):
    """EVA-CLIP vision transformer with Q-Former for video embeddings.

    Uses EVA-CLIP vision encoder followed by Q-Former for spatial pooling,
    enabling efficient extraction of semantic video frame embeddings.

    Args:
        model_name: Name of EVA-CLIP model variant
        num_query_tokens: Number of learnable query tokens
        freeze_vit: Whether to freeze ViT backbone
        freeze_qformer: Whether to freeze Q-Former
        pooling_strategy: How to aggregate query tokens ("mean", "cls", "max")
        device: Device for computation
        dtype: Data type for parameters

    Example:
        >>> embedder = EVAClipEmbedder(num_query_tokens=32)
        >>> frames = torch.randn(100, 3, 224, 224)
        >>> embeddings = embedder(frames)  # Shape: (100, 768)
    """

    def __init__(
        self,
        model_name: str = "eva_clip_vit_g_14",
        num_query_tokens: int = 32,
        freeze_vit: bool = True,
        freeze_qformer: bool = True,
        pooling_strategy: str = "mean",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            freeze_weights=freeze_vit and freeze_qformer,
            device=device,
            dtype=dtype,
        )

        # Validate inputs
        if pooling_strategy not in ["mean", "cls", "max"]:
            raise ValueError(
                f"Invalid pooling_strategy: {pooling_strategy}. "
                f"Must be one of: mean, cls, max"
            )

        if model_name not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}"
            )

        # Store configuration
        self.model_name = model_name
        self.num_query_tokens = num_query_tokens
        self.freeze_vit = freeze_vit
        self.freeze_qformer = freeze_qformer
        self.pooling_strategy = pooling_strategy

        # Build model components
        self._build_model(MODEL_CONFIGS[model_name])
        self._apply_freezing()

    def _build_model(self, config: VisionModelConfig) -> None:
        """Build EVA-CLIP + Q-Former architecture."""
        # Vision encoder
        vision_config = CLIPVisionConfig(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            layer_norm_eps=config.layer_norm_eps,
        )
        self.visual_encoder = CLIPVisionModel(vision_config)
        self.ln_vision = nn.LayerNorm(config.hidden_size)

        # Q-Former for spatial pooling
        qformer_config = Blip2QFormerConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            cross_attention_frequency=1,
            encoder_hidden_size=config.hidden_size,
            layer_norm_eps=1e-5,
        )
        self.qformer = Blip2QFormerModel(qformer_config)

        # Learnable query tokens
        self.query_tokens = nn.Parameter(
            torch.zeros(1, self.num_query_tokens, qformer_config.hidden_size)
        )
        nn.init.normal_(self.query_tokens, std=qformer_config.initializer_range)

        # Store dimensions
        self._embed_dim = qformer_config.hidden_size
        self._input_size = (config.image_size, config.image_size)

    def _apply_freezing(self) -> None:
        """Apply parameter freezing based on configuration."""
        if self.freeze_vit:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
            for param in self.ln_vision.parameters():
                param.requires_grad = False

        if self.freeze_qformer:
            for param in self.qformer.parameters():
                param.requires_grad = False
            self.query_tokens.requires_grad = False

    def forward(self, frames: Tensor) -> Tensor:
        """Extract embeddings from video frames.

        Args:
            frames: Video frames of shape (N, 3, H, W) or (B, N, 3, H, W)

        Returns:
            Frame embeddings of shape (N, embed_dim) or (B, N, embed_dim)
        """
        # Validate and reshape input
        original_shape = frames.shape
        frames, is_batched = self._prepare_input(frames)

        # Extract visual features
        image_embeds = self.visual_encoder(pixel_values=frames).last_hidden_state
        image_embeds = self.ln_vision(image_embeds)

        # Apply Q-Former with learnable queries
        query_tokens = self.query_tokens.expand(frames.shape[0], -1, -1)
        attention_mask = torch.ones(
            image_embeds.shape[:-1],
            dtype=torch.long,
            device=image_embeds.device,
        )

        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=attention_mask,
        ).last_hidden_state

        # Pool query outputs
        embeddings = self._pool_outputs(query_outputs)

        # Restore original batch dimensions if needed
        if is_batched:
            batch_size = original_shape[0]
            num_frames = original_shape[1]
            embeddings = embeddings.view(batch_size, num_frames, -1)

        return embeddings

    def _prepare_input(self, frames: Tensor) -> tuple[Tensor, bool]:
        """Validate and prepare input frames."""
        if frames.dim() not in [VIDEO_DIMS_4D, VIDEO_DIMS_5D]:
            raise ValueError(
                f"Expected 4D or 5D tensor, got {frames.dim()}D. "
                f"Shape should be (N, C, H, W) or (B, N, C, H, W)"
            )

        is_batched = frames.dim() == VIDEO_DIMS_5D
        if is_batched:
            frames = frames.flatten(0, 1)

        # Validate channels
        if frames.shape[1] != RGB_CHANNELS:
            raise ValueError(f"Expected 3 channels, got {frames.shape[1]}")

        # Validate spatial dimensions
        h, w = frames.shape[2:]
        expected_h, expected_w = self.expected_input_size
        if h != expected_h or w != expected_w:
            raise ValueError(
                f"Expected input size {self.expected_input_size}, got ({h}, {w})"
            )

        return frames, is_batched

    def _pool_outputs(self, query_outputs: Tensor) -> Tensor:
        """Pool query outputs according to strategy."""
        if self.pooling_strategy == "mean":
            return query_outputs.mean(dim=1)
        if self.pooling_strategy == "cls":
            return query_outputs[:, 0, :]
        # max
        return query_outputs.max(dim=1).values

    @property
    def embed_dim(self) -> int:
        """Embedding dimension."""
        return self._embed_dim

    @property
    def expected_input_size(self) -> tuple[int, int]:
        """Expected input size as (height, width)."""
        return self._input_size

    def encode_batch(self, frames: Tensor, batch_size: int = 32) -> Tensor:
        """Encode frames in batches for memory efficiency.

        Args:
            frames: Input frames of shape (N, 3, H, W)
            batch_size: Number of frames per batch

        Returns:
            Embeddings of shape (N, embed_dim)
        """
        if frames.shape[0] <= batch_size:
            return self.forward(frames)

        embeddings = []
        for i in range(0, frames.shape[0], batch_size):
            batch = frames[i : i + batch_size]
            embeddings.append(self.forward(batch))

        return torch.cat(embeddings, dim=0)


class CLIPEmbedder(EmbeddingExtractor):
    """CLIP vision encoder for video frame embeddings.

    Uses CLIP's vision transformer to extract semantic embeddings
    from individual video frames.

    Args:
        model_name: CLIP model variant (currently supports "ViT-B/32")
        device: Device for computation
        dtype: Data type for parameters

    Example:
        >>> embedder = CLIPEmbedder()
        >>> frames = torch.randn(50, 3, 224, 224)
        >>> embeddings = embedder(frames)  # Shape: (50, 768)
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ):
        super().__init__(device=device, dtype=dtype)
        self.model_name = model_name

        # Map model name to config
        if model_name.lower() != "vit-b/32":
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Currently only 'ViT-B/32' is supported"
            )

        # Build CLIP vision model
        config = MODEL_CONFIGS["clip_vit_b_32"]
        vision_config = CLIPVisionConfig(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            layer_norm_eps=config.layer_norm_eps,
        )
        self.model = CLIPVisionModel(vision_config)

        # Store dimensions
        self._embed_dim = config.hidden_size
        self._input_size = (config.image_size, config.image_size)

    def forward(self, frames: Tensor) -> Tensor:
        """Extract CLIP embeddings from frames.

        Args:
            frames: Video frames of shape (N, 3, H, W) or (B, N, 3, H, W)

        Returns:
            Frame embeddings of shape (N, embed_dim) or (B, N, embed_dim)
        """
        # Validate and reshape input
        original_shape = frames.shape
        frames, is_batched = self._prepare_input(frames)

        # Extract CLIP features
        outputs = self.model(pixel_values=frames)
        embeddings = outputs.pooler_output

        # Restore original batch dimensions if needed
        if is_batched:
            batch_size = original_shape[0]
            num_frames = original_shape[1]
            embeddings = embeddings.view(batch_size, num_frames, -1)

        return embeddings

    def _prepare_input(self, frames: Tensor) -> tuple[Tensor, bool]:
        """Validate and prepare input frames."""
        if frames.dim() not in [VIDEO_DIMS_4D, VIDEO_DIMS_5D]:
            raise ValueError(
                f"Expected 4D or 5D tensor, got {frames.dim()}D. "
                f"Shape should be (N, C, H, W) or (B, N, C, H, W)"
            )

        is_batched = frames.dim() == VIDEO_DIMS_5D
        if is_batched:
            frames = frames.flatten(0, 1)

        # Validate channels
        if frames.shape[1] != RGB_CHANNELS:
            raise ValueError(f"Expected 3 channels, got {frames.shape[1]}")

        # Validate spatial dimensions
        h, w = frames.shape[2:]
        expected_h, expected_w = self.expected_input_size
        if h != expected_h or w != expected_w:
            raise ValueError(
                f"Expected input size {self.expected_input_size}, got ({h}, {w})"
            )

        return frames, is_batched

    @property
    def embed_dim(self) -> int:
        """Embedding dimension."""
        return self._embed_dim

    @property
    def expected_input_size(self) -> tuple[int, int]:
        """Expected input size as (height, width)."""
        return self._input_size


# Register default embedders
register_embedder("eva_clip_vit_g_14", EVAClipEmbedder)
register_embedder("clip_vit_b_32", CLIPEmbedder)


def get_embedder(name: str, **kwargs: Any) -> EmbeddingExtractor:
    """Get embedder by name from registry.

    Args:
        name: Registered embedder name
        **kwargs: Arguments for embedder constructor

    Returns:
        Initialized embedding extractor

    Raises:
        KeyError: If embedder not found

    Example:
        >>> embedder = get_embedder("eva_clip_vit_g_14", num_query_tokens=64)
    """
    if name not in EMBEDDER_REGISTRY:
        available = list(EMBEDDER_REGISTRY.keys())
        raise KeyError(f"Embedder '{name}' not found. Available: {available}")

    embedder_class = EMBEDDER_REGISTRY[name]
    return embedder_class(**kwargs)


def list_embedders() -> list[str]:
    """List available embedder names.

    Returns:
        List of registered embedder names
    """
    return list(EMBEDDER_REGISTRY.keys())

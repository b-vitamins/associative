# Revised embeddings.py
"""Video embedding extraction modules.

This module provides PyTorch modules for extracting embeddings from video
frames using various pre-trained models. Follows nn.Module patterns for
consistency with PyTorch ecosystem.
"""

from abc import ABC, abstractmethod
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

# Constants for tensor dimensions and channels
EXPECTED_VIDEO_DIMS = 4  # (N, C, H, W)
RGB_CHANNELS = 3
BATCH_VIDEO_DIMS = 5  # (B, N, C, H, W)

KNOWN_VISION_CONFIGS = {
    "eva_clip_vit_g_14": {
        "image_size": 224,
        "patch_size": 14,
        "num_channels": 3,
        "hidden_size": 1408,
        "intermediate_size": 6144,
        "num_hidden_layers": 40,
        "num_attention_heads": 16,
        "layer_norm_eps": 1e-6,
    },
    "clip_vit_b_32": {
        "image_size": 224,
        "patch_size": 32,
        "num_channels": 3,
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "layer_norm_eps": 1e-5,
    },
}


class EmbeddingExtractor(nn.Module, ABC):
    """Abstract base class for video embedding extractors.

    All embedding extractors should inherit from this class and implement
    the forward method. This follows PyTorch's nn.Module pattern.

    Args:
        freeze_weights: Whether to freeze pre-trained model weights
        device: Device to run model on
        dtype: Data type for computations
    """

    def __init__(
        self,
        freeze_weights: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialize embedding extractor.

        Args:
            freeze_weights: Whether to freeze weights during training
            device: Device for model computation
            dtype: Data type for model parameters
        """
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
            frames: Input video frames of shape (N, C, H, W) or (B, N, C, H, W)

        Returns:
            Frame embeddings of shape (N, embed_dim) or (B, N, embed_dim)

        Raises:
            RuntimeError: If model fails to process input
            ValueError: If input tensor has wrong format
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
        """Expected input image size as (height, width)."""
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
        """Set training mode, respecting freeze_weights setting."""
        if self.freeze_weights:
            # Keep in eval mode if frozen
            super().train(False)
        else:
            super().train(mode)
        return self


class EVAClipEmbedder(EmbeddingExtractor):
    """EVA-CLIP vision transformer for video frame embeddings.

    Uses EVA-CLIP ViT-G/14 model followed by Q-former for spatial pooling,
    matching the setup from Santos et al. (2025).

    Args:
        model_name: Name of EVA-CLIP model variant
        num_query_tokens: Number of query tokens for Q-former
        freeze_vit: Whether to freeze vision transformer weights
        freeze_qformer: Whether to freeze Q-former weights
        pooling_strategy: How to pool query tokens to single embedding
        device: Device for computation
        dtype: Data type for parameters

    Example:
        >>> embedder = EVAClipEmbedder("eva_clip_vit_g_14", num_query_tokens=32)
        >>> frames = torch.randn(100, 3, 224, 224)
        >>> embeddings = embedder(frames)
        >>> print(embeddings.shape)
        torch.Size([100, 768])
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
        """Initialize EVA-CLIP embedder.

        Args:
            model_name: EVA-CLIP model variant name
            num_query_tokens: Number of query tokens for spatial attention
            freeze_vit: Whether to freeze ViT backbone
            freeze_qformer: Whether to freeze Q-former
            pooling_strategy: How to pool tokens ("mean", "cls", "max")
            device: Device for model
            dtype: Parameter data type
            **kwargs: Additional arguments

        Raises:
            ValueError: If model_name not supported or invalid pooling strategy
            RuntimeError: If model fails to load
        """
        super().__init__(
            freeze_weights=freeze_vit and freeze_qformer, device=device, dtype=dtype
        )

        if pooling_strategy not in ["mean", "cls", "max"]:
            raise ValueError(f"Invalid pooling_strategy: {pooling_strategy}")

        if model_name not in KNOWN_VISION_CONFIGS:
            raise ValueError(
                f"Unknown model_name: {model_name}. Available: {list(KNOWN_VISION_CONFIGS.keys())}"
            )

        self.model_name = model_name
        self.num_query_tokens = num_query_tokens
        self.freeze_vit = freeze_vit
        self.freeze_qformer = freeze_qformer
        self.pooling_strategy = pooling_strategy
        self.vision_config = KNOWN_VISION_CONFIGS[model_name]

        # Model components will be initialized in _build_model()
        self.visual_encoder = None
        self.ln_vision = None
        self.qformer = None
        self.query_tokens = None

        self._build_model()
        self._setup_freezing()

    def _build_model(self) -> None:
        """Build the EVA-CLIP + Q-former architecture.

        Raises:
            RuntimeError: If model components fail to load
            ImportError: If required dependencies are missing
        """
        vision_config = CLIPVisionConfig(**self.vision_config)
        self.visual_encoder = CLIPVisionModel(vision_config)

        vision_dim = vision_config.hidden_size
        self.ln_vision = nn.LayerNorm(vision_dim)

        qformer_config = Blip2QFormerConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            cross_attention_frequency=1,
            encoder_hidden_size=vision_dim,
            layer_norm_eps=1e-5,
        )
        self.qformer = Blip2QFormerModel(qformer_config)

        self.query_tokens = nn.Parameter(
            torch.zeros(1, self.num_query_tokens, qformer_config.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=qformer_config.initializer_range)

        self._embed_dim = qformer_config.hidden_size
        self._input_size = (vision_config.image_size, vision_config.image_size)

    def _setup_freezing(self) -> None:
        """Setup parameter freezing according to configuration."""
        if self.freeze_vit and self.visual_encoder is not None:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
            if self.ln_vision is not None:
                for param in self.ln_vision.parameters():
                    param.requires_grad = False

        if self.freeze_qformer and self.qformer is not None:
            for param in self.qformer.parameters():
                param.requires_grad = False
            if self.query_tokens is not None:
                self.query_tokens.requires_grad = False

    def forward(self, frames: Tensor) -> Tensor:
        """Extract embeddings from video frames.

        Args:
            frames: Video frames of shape (N, 3, H, W) or (B, N, 3, H, W)
                   Expected to be normalized to [-1, 1] or [0, 1] range

        Returns:
            Frame embeddings of shape (N, embed_dim) or (B, N, embed_dim)

        Raises:
            RuntimeError: If model forward pass fails
            ValueError: If input tensor has wrong shape or range
        """
        # Input validation
        if frames.dim() not in [4, 5]:
            raise ValueError(f"Expected 4D or 5D tensor, got {frames.dim()}D")

        if frames.dim() == BATCH_VIDEO_DIMS:
            # Batch of videos: (B, N, 3, H, W) -> (B*N, 3, H, W)
            batch_size, num_frames = frames.shape[:2]
            frames = frames.reshape(-1, *frames.shape[2:])
            batched_input = True
        else:
            batched_input = False
            batch_size = 1  # Not used, but needed for type safety
            num_frames = frames.shape[0]

        if frames.shape[1] != RGB_CHANNELS:
            raise ValueError(f"Expected 3 channels, got {frames.shape[1]}")

        # Check input size
        expected_h, expected_w = self.expected_input_size
        if frames.shape[2] != expected_h or frames.shape[3] != expected_w:
            raise ValueError(
                f"Expected input size {self.expected_input_size}, "
                f"got {frames.shape[2:]}"
            )

        # Model forward pass - assert components are initialized
        assert self.visual_encoder is not None, "Visual encoder not initialized"
        assert self.ln_vision is not None, "Layer norm not initialized"
        assert self.query_tokens is not None, "Query tokens not initialized"
        assert self.qformer is not None, "Q-former not initialized"

        image_embeds = self.visual_encoder(pixel_values=frames).last_hidden_state
        image_embeds = self.ln_vision(image_embeds)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        image_attention_mask = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device
        )

        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
        ).last_hidden_state

        # Pooling
        if self.pooling_strategy == "mean":
            embeddings = query_outputs.mean(dim=1)
        elif self.pooling_strategy == "cls":
            embeddings = query_outputs[:, 0, :]
        elif self.pooling_strategy == "max":
            embeddings = query_outputs.max(dim=1).values
        else:
            # This should never happen due to validation in __init__, but for type safety
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        if batched_input:
            # Reshape back to (B, N, embed_dim)
            embeddings = embeddings.reshape(batch_size, num_frames, -1)

        return embeddings

    @property
    def embed_dim(self) -> int:
        """Embedding dimension."""
        return self._embed_dim

    @property
    def expected_input_size(self) -> tuple[int, int]:
        """Expected input image size."""
        return self._input_size

    def encode_batch(self, frames: Tensor, batch_size: int = 32) -> Tensor:
        """Encode frames in batches to manage memory.

        Args:
            frames: Input frames of shape (N, 3, H, W)
            batch_size: Batch size for processing

        Returns:
            Embeddings of shape (N, embed_dim)

        Example:
            >>> embedder = EVAClipEmbedder()
            >>> large_frames = torch.randn(1000, 3, 224, 224)
            >>> embeddings = embedder.encode_batch(large_frames, batch_size=64)
        """
        if frames.shape[0] <= batch_size:
            return self.forward(frames)

        embeddings_list = []
        for i in range(0, frames.shape[0], batch_size):
            batch = frames[i : i + batch_size]
            batch_embeddings = self.forward(batch)
            embeddings_list.append(batch_embeddings)

        return torch.cat(embeddings_list, dim=0)


class CLIPEmbedder(EmbeddingExtractor):
    """CLIP vision encoder for video frame embeddings.

    Args:
        model_name: CLIP model variant
        device: Device for computation
        dtype: Data type for parameters

    Example:
        >>> embedder = CLIPEmbedder("ViT-B/32")
        >>> frames = torch.randn(100, 3, 224, 224)
        >>> embeddings = embedder(frames)
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ):
        """Initialize CLIP embedder.

        Args:
            model_name: CLIP model name
            device: Device for model
            dtype: Parameter data type
            **kwargs: Additional arguments
        """
        super().__init__(device=device, dtype=dtype)
        self.model_name = model_name

        if model_name.lower() != "vit-b/32":
            raise ValueError(
                f"Unsupported model_name: {model_name}. Only 'ViT-B/32' is supported in this implementation."
            )

        config_key = "clip_vit_b_32"
        vision_config = CLIPVisionConfig(**KNOWN_VISION_CONFIGS[config_key])
        self.model = CLIPVisionModel(vision_config)

        self._embed_dim = vision_config.hidden_size
        self._input_size = (vision_config.image_size, vision_config.image_size)

    def forward(self, frames: Tensor) -> Tensor:
        """Extract CLIP embeddings."""
        # Input validation
        if frames.dim() not in [4, 5]:
            raise ValueError(f"Expected 4D or 5D tensor, got {frames.dim()}D")

        if frames.dim() == BATCH_VIDEO_DIMS:
            # Batch of videos: (B, N, 3, H, W) -> (B*N, 3, H, W)
            batch_size, num_frames = frames.shape[:2]
            frames = frames.reshape(-1, *frames.shape[2:])
            batched = True
        else:
            batched = False
            batch_size = 1  # Not used, but needed for type safety
            num_frames = frames.shape[0]

        if frames.shape[1] != RGB_CHANNELS:
            raise ValueError(f"Expected 3 channels, got {frames.shape[1]}")

        # Check input size
        expected_h, expected_w = self.expected_input_size
        if frames.shape[2] != expected_h or frames.shape[3] != expected_w:
            raise ValueError(
                f"Expected input size {self.expected_input_size}, "
                f"got {frames.shape[2:]}"
            )

        outputs = self.model(pixel_values=frames)
        embeddings = outputs.pooler_output

        if batched:
            embeddings = embeddings.reshape(batch_size, num_frames, -1)

        return embeddings

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def expected_input_size(self) -> tuple[int, int]:
        return self._input_size


# Register default embedders
register_embedder("eva_clip_vit_g_14", EVAClipEmbedder)
register_embedder("clip_vit_b_32", CLIPEmbedder)


def get_embedder(name: str, **kwargs: Any) -> EmbeddingExtractor:
    """Get embedder by name from registry.

    Args:
        name: Embedder name
        **kwargs: Arguments to pass to embedder constructor

    Returns:
        Initialized embedding extractor

    Raises:
        KeyError: If embedder name not found in registry

    Example:
        >>> embedder = get_embedder("eva_clip_vit_g_14", num_query_tokens=32)
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

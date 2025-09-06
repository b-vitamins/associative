"""Audio embedding extraction modules.

Provides efficient audio embedding extractors using pre-trained models
and learnable representations, following PyTorch conventions.
"""

from typing import Literal

import torch
import torchaudio.transforms
import torchvggish
from torch import Tensor, nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor


class VGGishEmbedding(nn.Module):
    """VGGish audio embedding model.

    Produces 128-dimensional embeddings from audio using a CNN trained on AudioSet.
    Processes audio in 0.96 second windows.

    Args:
        use_pca: Apply PCA whitening to embeddings
        aggregate: Aggregation method for frame embeddings (None, "mean", "max")
        freeze: Freeze pre-trained weights
        device: Device for computation
        dtype: Data type for computation
    """

    def __init__(
        self,
        use_pca: bool = True,
        aggregate: Literal[None, "mean", "max"] = None,
        freeze: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.use_pca = use_pca
        self.aggregate = aggregate
        self.sample_rate = 16000  # Fixed for VGGish
        self.dim = 128

        # Initialize model components
        self.model = torchvggish.vggish()
        if use_pca:
            self.pca = torchvggish.PCA()  # type: ignore[attr-defined]
        else:
            self.pca = None

        if freeze:
            for param in self.parameters():
                param.requires_grad = False
            self.eval()

        if device:
            self.to(device)
        if dtype:
            self.to(dtype)

    def forward(self, audio: Tensor) -> Tensor:
        """Extract VGGish embeddings.

        Args:
            audio: Audio tensor of shape (samples,) or (batch, samples)

        Returns:
            Embeddings of shape:
            - No aggregation: (num_frames, 128) or (batch, num_frames, 128)
            - With aggregation: (128,) or (batch, 128)
        """
        if audio.dim() not in [1, 2]:
            raise ValueError(f"Expected 1D or 2D audio tensor, got shape {audio.shape}")

        # Add batch dimension if needed
        squeeze_batch = audio.dim() == 1
        if squeeze_batch:
            audio = audio.unsqueeze(0)

        embeddings_list = self._extract_batch_embeddings(audio)

        # Handle single sample case
        if len(embeddings_list) == 1 and squeeze_batch:
            return embeddings_list[0]

        # Handle padding for frame-level embeddings
        if self.aggregate is None and len(embeddings_list) > 1:
            return self._pad_and_stack_embeddings(embeddings_list)

        return torch.stack(embeddings_list)

    def _extract_batch_embeddings(self, audio: Tensor) -> list[Tensor]:
        """Extract embeddings for each audio in batch."""
        embeddings_list = []
        batch_size = audio.shape[0]

        for i in range(batch_size):
            emb = self._extract_single_embedding(audio[i])
            embeddings_list.append(emb)

        return embeddings_list

    def _extract_single_embedding(self, audio: Tensor) -> Tensor:
        """Extract embedding for single audio sample."""
        with torch.set_grad_enabled(self.training):
            emb = self.model(audio)  # (num_frames, 128)

            if self.pca is not None:
                emb = self.pca(emb)

            if self.aggregate == "mean":
                return emb.mean(dim=0)  # (128,)
            if self.aggregate == "max":
                return emb.max(dim=0)[0]  # (128,)

            return emb

    def _pad_and_stack_embeddings(self, embeddings_list: list[Tensor]) -> Tensor:
        """Pad embeddings to same length and stack."""
        max_frames = max(e.shape[0] for e in embeddings_list)
        padded = []

        for emb in embeddings_list:
            if emb.shape[0] < max_frames:
                pad_size = max_frames - emb.shape[0]
                padding = torch.zeros(pad_size, 128, device=emb.device, dtype=emb.dtype)
                padded_emb = torch.cat([emb, padding], dim=0)
            else:
                padded_emb = emb
            padded.append(padded_emb)

        return torch.stack(padded)  # (batch, max_frames, 128)


class Wav2Vec2Embedding(nn.Module):
    """Wav2Vec 2.0 audio embedding model.

    Self-supervised model that learns from raw waveforms.
    Base model: 768-dim, Large model: 1024-dim.

    Args:
        model_size: "base" (768-dim) or "large" (1024-dim)
        aggregate: Aggregation method (None, "mean", "max", "first", "last")
        layer: Which transformer layer to use (-1 for last)
        freeze: Freeze pre-trained weights
        device: Device for computation
        dtype: Data type for computation
    """

    def __init__(  # noqa: PLR0913
        self,
        model_size: Literal["base", "large"] = "base",
        aggregate: Literal[None, "mean", "max", "first", "last"] = "mean",
        layer: int = -1,
        freeze: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        if model_size not in ["base", "large"]:
            raise ValueError(f"model_size must be 'base' or 'large', got {model_size}")

        valid_aggregations = [None, "mean", "max", "first", "last"]
        if aggregate not in valid_aggregations:
            raise ValueError(
                f"aggregate must be one of {valid_aggregations}, got {aggregate}"
            )

        self.model_size = model_size
        self.aggregate = aggregate
        self.layer = layer
        self.sample_rate = 16000  # Fixed for Wav2Vec2
        self.dim = 768 if model_size == "base" else 1024

        # Load model
        model_name = f"facebook/wav2vec2-{model_size}-960h"
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        if device:
            self.to(device)
        if dtype:
            self.to(dtype)

    def forward(self, audio: Tensor) -> Tensor:
        """Extract Wav2Vec2 embeddings.

        Args:
            audio: Audio tensor of shape (samples,) or (batch, samples)

        Returns:
            Embeddings of shape:
            - No aggregation: (seq_len, dim) or (batch, seq_len, dim)
            - With aggregation: (dim,) or (batch, dim)
        """
        if audio.dim() not in [1, 2]:
            raise ValueError(f"Expected 1D or 2D audio tensor, got shape {audio.shape}")

        # Add batch dimension if needed
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        # Process audio through processor
        # Note: processor expects numpy arrays on CPU
        audio_np = audio.detach().cpu().numpy()
        inputs = self.processor(  # type: ignore
            audio_np,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )

        # Move to correct device
        device = audio.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Extract features
        with torch.set_grad_enabled(self.training):
            outputs = self.model(**inputs, output_hidden_states=True)

            # Get features from specified layer
            if self.layer == -1:
                hidden_states = outputs.last_hidden_state
            else:
                hidden_states = outputs.hidden_states[self.layer]

            # Apply aggregation
            if self.aggregate == "mean":
                embeddings = hidden_states.mean(dim=1)  # (batch, dim)
            elif self.aggregate == "max":
                embeddings = hidden_states.max(dim=1)[0]  # (batch, dim)
            elif self.aggregate == "first":
                embeddings = hidden_states[:, 0, :]  # (batch, dim)
            elif self.aggregate == "last":
                embeddings = hidden_states[:, -1, :]  # (batch, dim)
            else:
                embeddings = hidden_states  # (batch, seq_len, dim)

        # Remove batch dimension if needed
        if squeeze_batch:
            embeddings = embeddings.squeeze(0)

        return embeddings


class MelSpectrogramEmbedding(nn.Module):
    """Mel-spectrogram based audio embeddings.

    Simple, efficient baseline without pre-training.
    Optionally includes learned projection layer.

    Args:
        n_mels: Number of mel frequency bins
        n_fft: FFT window size
        hop_length: Hop length for STFT
        projection_dim: Output dimension (None = n_mels)
        sample_rate: Audio sample rate
        aggregate: Aggregation method (None, "mean", "max")
        device: Device for computation
        dtype: Data type for computation
    """

    def __init__(  # noqa: PLR0913
        self,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 160,
        projection_dim: int | None = None,
        sample_rate: int = 16000,
        aggregate: Literal[None, "mean", "max"] = "mean",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.projection_dim = projection_dim
        self.sample_rate = sample_rate
        self.aggregate = aggregate
        self.dim = projection_dim or n_mels

        # Mel-spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            normalized=True,
        )

        # Optional projection layer
        if projection_dim is not None:
            self.projection = nn.Linear(n_mels, projection_dim)
        else:
            self.projection = None

        if device:
            self.to(device)
        if dtype:
            self.to(dtype)

    def forward(self, audio: Tensor) -> Tensor:
        """Extract mel-spectrogram embeddings.

        Args:
            audio: Audio tensor of shape (samples,) or (batch, samples)

        Returns:
            Embeddings of shape:
            - No aggregation: (n_mels, time) or (batch, n_mels, time)
            - With aggregation: (dim,) or (batch, dim)
        """
        if audio.dim() not in [1, 2]:
            raise ValueError(f"Expected 1D or 2D audio tensor, got shape {audio.shape}")

        # Add batch dimension if needed
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        # Extract mel-spectrogram
        mel_spec = self.mel_transform(audio)  # (batch, n_mels, time)

        # Apply aggregation
        if self.aggregate == "mean":
            embeddings = mel_spec.mean(dim=-1)  # (batch, n_mels)
        elif self.aggregate == "max":
            embeddings = mel_spec.max(dim=-1)[0]  # (batch, n_mels)
        else:
            embeddings = mel_spec  # (batch, n_mels, time)

        # Apply projection if specified and aggregated
        if self.projection is not None and self.aggregate is not None:
            embeddings = self.projection(embeddings)  # (batch, projection_dim)

        # Remove batch dimension if needed
        if squeeze_batch:
            if self.aggregate is None:
                embeddings = embeddings.squeeze(0)  # (n_mels, time)
            else:
                embeddings = embeddings.squeeze(0)  # (dim,)

        return embeddings


# Registry of available embedders
AUDIO_EMBEDDERS = {
    "vggish": VGGishEmbedding,
    "wav2vec2": Wav2Vec2Embedding,
    "mel_spectrogram": MelSpectrogramEmbedding,
}


def get_audio_embedder(name: str, **kwargs) -> nn.Module:
    """Get audio embedder by name.

    Args:
        name: Embedder name ("vggish", "wav2vec2", "mel_spectrogram")
        **kwargs: Arguments for embedder constructor

    Returns:
        Initialized embedder

    Raises:
        KeyError: If embedder not found
    """
    if name not in AUDIO_EMBEDDERS:
        available = list(AUDIO_EMBEDDERS.keys())
        raise KeyError(f"Unknown embedder '{name}'. Available: {available}")

    embedder_class = AUDIO_EMBEDDERS[name]
    return embedder_class(**kwargs)

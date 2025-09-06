"""Audio processing utilities for multimodal learning.

This module provides audio-specific utilities for MET training including:
- Audio metrics (PESQ, STOI, SDR)
- Audio transforms (masking, corruption)
- Audio feature extraction
- Audio embedding models (VGGish, Wav2Vec)
"""

from .embeddings import (
    MelSpectrogramEmbedding,
    VGGishEmbedding,
    Wav2Vec2Embedding,
    get_audio_embedder,
)
from .metrics import PESQ, SDR, STOI, AudioReconstructionMetrics, get_audio_metric
from .transforms import (
    AddAudioNoise,
    ApplyAudioMask,
    AudioTransform,
    BandpassFilter,
    MelSpectrogram,
)

__all__ = [
    "PESQ",
    "SDR",
    "STOI",
    "AddAudioNoise",
    "ApplyAudioMask",
    "AudioReconstructionMetrics",
    "AudioTransform",
    "BandpassFilter",
    "MelSpectrogram",
    "MelSpectrogramEmbedding",
    "VGGishEmbedding",
    "Wav2Vec2Embedding",
    "get_audio_embedder",
    "get_audio_metric",
]

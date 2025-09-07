"""Decoder modules for multimodal reconstruction.

Clean, modular decoders for reconstructing signals from embeddings.
Supports video, audio, and cross-modal translation tasks.
"""

from typing import Any, Literal

import torch
from torch import Tensor, nn

# Constants for tensor dimension checks
DIM_2D = 2
DIM_3D = 3
DIM_4D = 4


class VideoDecoder(nn.Module):
    """Decoder for reconstructing video frames from embeddings.

    Simple, effective architecture using transposed convolutions
    for spatial upsampling and temporal expansion.

    Args:
        input_dim: Dimension of input embeddings
        num_frames: Number of frames to reconstruct
        height: Frame height in pixels
        width: Frame width in pixels
        channels: Number of channels (default 3 for RGB)
        hidden_dims: Dimensions for hidden layers
        device: Device for computation
        dtype: Data type for parameters
    """

    def __init__(  # noqa: PLR0913
        self,
        input_dim: int,
        num_frames: int,
        height: int,
        width: int,
        channels: int = 3,
        hidden_dims: list[int] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.channels = channels
        self.hidden_dims = hidden_dims or [512, 256, 128]

        # Build decoder architecture
        self._build_layers()

        if device is not None:
            self.to(device)
        if dtype is not None:
            self.to(dtype)

    def _build_layers(self):
        """Build decoder layers with progressive upsampling architecture.

        Creates an initial embedding projection layer followed by transposed
        convolutional layers for spatial upsampling to reconstruct video frames.
        """
        # Calculate initial spatial size for progressive upsampling
        num_upsample_layers = len(self.hidden_dims)
        self.initial_size = max(4, self.height // (2**num_upsample_layers))

        # Initial projection from embedding to spatial features
        self.embed_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dims[0] * self.initial_size**2),
            nn.ReLU(),
        )

        # Progressive upsampling layers
        layers = []
        in_channels = self.hidden_dims[0]

        for i, out_channels in enumerate(self.hidden_dims[1:] + [self.channels]):
            is_final = i == len(self.hidden_dims) - 1

            # Transposed convolution for upsampling
            layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )

            if not is_final:
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Tanh())  # Output in [-1, 1]

            in_channels = out_channels

        self.decoder = nn.Sequential(*layers)

    def forward(self, embeddings: Tensor) -> Tensor:
        """Reconstruct video frames from embeddings.

        Args:
            embeddings: Input embeddings of shape:
                - (batch, num_frames, input_dim) for frame embeddings
                - (batch, input_dim) for global embeddings

        Returns:
            Reconstructed frames of shape (batch, num_frames, channels, H, W)
        """
        batch_size = embeddings.shape[0]

        # Handle different input shapes
        if embeddings.dim() == DIM_2D:
            # Global embedding: expand to all frames
            embeddings = embeddings.unsqueeze(1).expand(-1, self.num_frames, -1)

        # Process each frame
        frames = []
        for t in range(self.num_frames):
            # Get embedding for this frame
            frame_embed = embeddings[:, t]  # (batch, input_dim)

            # Project to spatial features
            x = self.embed_proj(frame_embed)
            x = x.view(
                batch_size, self.hidden_dims[0], self.initial_size, self.initial_size
            )

            # Decode to image
            x = self.decoder(x)

            # Ensure correct output size
            if x.shape[-2:] != (self.height, self.width):
                x = nn.functional.interpolate(
                    x,
                    size=(self.height, self.width),
                    mode="bilinear",
                    align_corners=False,
                )

            frames.append(x)

        # Stack frames: (batch, num_frames, channels, H, W)
        return torch.stack(frames, dim=1)

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AudioDecoder(nn.Module):
    """Decoder for reconstructing audio waveforms from embeddings.

    Uses 1D transposed convolutions for efficient waveform generation.

    Args:
        input_dim: Dimension of input embeddings
        sample_rate: Target sample rate in Hz
        duration: Duration in seconds
        channels: Number of audio channels (1=mono, 2=stereo)
        hidden_dims: Hidden layer dimensions
        device: Device for computation
        dtype: Data type for parameters
    """

    def __init__(  # noqa: PLR0913
        self,
        input_dim: int,
        sample_rate: int = 16000,
        duration: float = 5.0,
        channels: int = 1,
        hidden_dims: list[int] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.sample_rate = sample_rate
        self.duration = duration
        self.channels = channels
        self.num_samples = int(sample_rate * duration)
        self.hidden_dims = hidden_dims or [256, 512, 256]

        self._build_layers()

        if device is not None:
            self.to(device)
        if dtype is not None:
            self.to(dtype)

    def _build_layers(self):
        """Build audio decoder layers with 1D progressive upsampling.

        Creates an initial embedding projection followed by 1D transposed
        convolutional layers for temporal upsampling to generate audio waveforms.
        """
        # Calculate initial length for progressive upsampling
        num_upsample = len(self.hidden_dims)
        self.initial_length = max(64, self.num_samples // (2**num_upsample))

        # Initial projection
        self.embed_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dims[0] * self.initial_length),
            nn.ReLU(),
        )

        # Progressive upsampling with 1D convolutions
        layers = []
        in_channels = self.hidden_dims[0]

        for i, out_channels in enumerate(self.hidden_dims[1:] + [self.channels]):
            is_final = i == len(self.hidden_dims) - 1

            # 1D transposed convolution
            layers.append(
                nn.ConvTranspose1d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )

            if not is_final:
                layers.append(nn.BatchNorm1d(out_channels))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Tanh())  # Output in [-1, 1]

            in_channels = out_channels

        self.decoder = nn.Sequential(*layers)

    def forward(self, embeddings: Tensor) -> Tensor:
        """Reconstruct audio waveform from embeddings.

        Args:
            embeddings: Input embeddings of shape:
                - (batch, input_dim) for global embeddings
                - (batch, num_frames, input_dim) for frame embeddings (averaged)

        Returns:
            Reconstructed waveform of shape:
                - (batch, num_samples) for mono
                - (batch, channels, num_samples) for multi-channel
        """
        # Handle frame embeddings by averaging
        if embeddings.dim() == DIM_3D:
            embeddings = embeddings.mean(dim=1)

        batch_size = embeddings.shape[0]

        # Project to initial waveform features
        x = self.embed_proj(embeddings)
        x = x.view(batch_size, self.hidden_dims[0], self.initial_length)

        # Decode to waveform
        x = self.decoder(x)

        # Ensure correct output length
        if x.shape[-1] != self.num_samples:
            x = nn.functional.interpolate(
                x, size=self.num_samples, mode="linear", align_corners=False
            )

        # Handle channel formatting
        if self.channels == 1:
            x = x.squeeze(1)  # Remove channel dim for mono

        return x

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CrossModalDecoder(nn.Module):
    """Decoder for cross-modal reconstruction.

    Translates embeddings from one modality to another using
    a bridge representation and modality-specific decoders.

    Args:
        input_dim: Dimension of source modality embeddings
        output_shape: Shape of target modality output
        source_modality: Type of input modality
        target_modality: Type of output modality
        bridge_dim: Dimension of bridging representation
        device: Device for computation
        dtype: Data type for parameters
    """

    def __init__(  # noqa: PLR0913
        self,
        input_dim: int,
        output_shape: tuple[int, ...],
        source_modality: Literal["video", "audio", "text"],
        target_modality: Literal["video", "audio", "text"],
        bridge_dim: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        if source_modality == target_modality:
            raise ValueError("Source and target modalities must be different")

        self.input_dim = input_dim
        self.output_shape = output_shape
        self.source_modality = source_modality
        self.target_modality = target_modality
        self.bridge_dim = bridge_dim

        self._build_layers()

        if device is not None:
            self.to(device)
        if dtype is not None:
            self.to(dtype)

    def _build_layers(self):
        """Build cross-modal translation layers.

        Creates a source encoder that maps input embeddings to a bridge
        representation, and a target decoder that generates the output
        modality from the bridge representation.
        """
        # Source encoder: map to bridge representation
        self.source_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.bridge_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.bridge_dim * 2, self.bridge_dim),
            nn.LayerNorm(self.bridge_dim),
        )

        # Calculate output size
        output_size = 1
        for dim in self.output_shape:
            output_size *= dim

        # Target decoder: generate from bridge representation
        self.target_decoder = nn.Sequential(
            nn.Linear(self.bridge_dim, self.bridge_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.bridge_dim * 2, self.bridge_dim * 4),
            nn.ReLU(),
            nn.Linear(self.bridge_dim * 4, output_size),
        )

        # Output activation based on target modality
        if self.target_modality in ["video", "audio"]:
            self.output_activation = nn.Tanh()
        else:  # text
            self.output_activation = nn.Identity()

    def forward(self, embeddings: Tensor) -> Tensor:
        """Perform cross-modal reconstruction.

        Args:
            embeddings: Embeddings from source modality

        Returns:
            Reconstructed target modality signal
        """
        batch_size = embeddings.shape[0]

        # Encode to bridge representation
        bridge = self.source_encoder(embeddings)

        # Decode to target modality
        output = self.target_decoder(bridge)
        output = self.output_activation(output)

        # Reshape to target shape
        if len(self.output_shape) == 1:
            output = output.view(batch_size, self.output_shape[0])
        else:
            output = output.view(batch_size, *self.output_shape)

        return output

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HierarchicalVideoDecoder(nn.Module):
    """Hierarchical decoder for progressive video reconstruction.

    Generates video at multiple resolutions, combining outputs
    for high-quality reconstruction.

    Args:
        input_dim: Input embedding dimension
        num_frames: Number of frames to generate
        height: Final frame height
        width: Final frame width
        channels: Number of channels
        num_levels: Number of hierarchy levels
        scale_factors: Upsampling factor per level
        fusion_method: How to combine level outputs
        device: Device for computation
        dtype: Data type for parameters
    """

    def __init__(  # noqa: PLR0913
        self,
        input_dim: int,
        num_frames: int,
        height: int,
        width: int,
        channels: int = 3,
        num_levels: int = 3,
        scale_factors: list[int] | None = None,
        fusion_method: Literal["add", "concat", "attention"] = "add",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.channels = channels
        self.num_levels = num_levels
        self.scale_factors = scale_factors or [2] * num_levels
        self.fusion_method = fusion_method

        self._build_layers()

        if device is not None:
            self.to(device)
        if dtype is not None:
            self.to(dtype)

    def _build_layers(self):
        """Build hierarchical decoder levels at multiple resolutions.

        Creates separate VideoDecoder instances for each hierarchy level,
        with progressively finer resolutions and reduced hidden dimensions
        for coarser levels.
        """
        self.level_decoders = nn.ModuleList()

        # Create decoder for each level
        for level in range(self.num_levels):
            # Calculate dimensions for this level
            scale = 2 ** (self.num_levels - 1 - level)
            level_h = self.height // scale
            level_w = self.width // scale

            # Reduce hidden dims for coarser levels
            hidden_dims = [
                max(64, 512 // (2**level)),
                max(32, 256 // (2**level)),
                max(16, 128 // (2**level)),
            ]

            # Create level decoder
            decoder = VideoDecoder(
                input_dim=self.input_dim,
                num_frames=self.num_frames,
                height=level_h,
                width=level_w,
                channels=self.channels,
                hidden_dims=hidden_dims,
            )

            self.level_decoders.append(decoder)

        # Fusion layers for combining levels
        if self.fusion_method == "concat":
            self.fusion_conv = nn.Conv3d(
                self.channels * 2,  # Concatenated channels
                self.channels,
                kernel_size=3,
                padding=1,
            )
        elif self.fusion_method == "attention":
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim=self.channels,
                num_heads=1,  # Single head for simplicity
                batch_first=True,
            )

    def forward(
        self, embeddings: Tensor, return_all_levels: bool = False
    ) -> Tensor | list[Tensor]:
        """Hierarchical video reconstruction.

        Args:
            embeddings: Input embeddings
            return_all_levels: Return outputs from all levels

        Returns:
            Final reconstruction or list of all level outputs
        """
        batch_size = embeddings.shape[0]
        all_outputs = []
        combined_output = None

        for level in range(self.num_levels):
            # Generate at this level
            level_output = self.level_decoders[level](embeddings)

            # Combine with previous level if not first
            if level > 0 and combined_output is not None:
                # Upsample previous output to match current resolution
                prev_upsampled = nn.functional.interpolate(
                    combined_output.flatten(0, 1),  # Merge batch and time
                    size=level_output.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).view(
                    batch_size, self.num_frames, self.channels, *level_output.shape[-2:]
                )

                # Fuse outputs
                if self.fusion_method == "add":
                    combined_output = level_output + prev_upsampled
                elif self.fusion_method == "concat":
                    # Concatenate and convolve
                    concat = torch.cat([level_output, prev_upsampled], dim=2)
                    # Reshape for 3D conv: (batch, channels, frames, H, W)
                    concat = concat.permute(0, 2, 1, 3, 4)
                    combined_output = self.fusion_conv(concat)
                    # Reshape back: (batch, frames, channels, H, W)
                    combined_output = combined_output.permute(0, 2, 1, 3, 4)
                elif self.fusion_method == "attention":
                    # Flatten spatial dims for attention
                    b, t, c, h, w = level_output.shape
                    level_flat = level_output.view(b * t, c, h * w).permute(0, 2, 1)
                    prev_flat = prev_upsampled.view(b * t, c, h * w).permute(0, 2, 1)

                    # Apply attention
                    attended, _ = self.fusion_attention(
                        level_flat, prev_flat, prev_flat
                    )
                    combined_output = attended.permute(0, 2, 1).view(b, t, c, h, w)
            else:
                combined_output = level_output

            all_outputs.append(combined_output)

        if return_all_levels:
            return all_outputs
        assert combined_output is not None, (
            "combined_output should be defined after loop"
        )
        return combined_output

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_decoder(
    decoder_type: str,
    input_dim: int,
    output_shape: tuple[int, ...],
    **kwargs: Any,
) -> nn.Module:
    """Factory function for creating decoders.

    Args:
        decoder_type: Type of decoder ("video", "audio", "cross_modal", "hierarchical_video")
        input_dim: Input embedding dimension
        output_shape: Output shape for reconstruction
        **kwargs: Additional arguments for specific decoder

    Returns:
        Initialized decoder

    Raises:
        ValueError: If decoder type is unknown
    """
    if decoder_type == "video":
        # Parse output shape: (num_frames, channels, height, width)
        if len(output_shape) == DIM_4D:
            num_frames, channels, height, width = output_shape
        else:
            raise ValueError(
                f"Video decoder expects 4D output shape, got {len(output_shape)}D"
            )

        return VideoDecoder(
            input_dim=input_dim,
            num_frames=num_frames,
            height=height,
            width=width,
            channels=channels,
            **kwargs,
        )

    if decoder_type == "audio":
        # Parse output shape: (num_samples,) or (channels, num_samples)
        if len(output_shape) == 1:
            num_samples = output_shape[0]
            channels = 1
        elif len(output_shape) == DIM_2D:
            channels, num_samples = output_shape
        else:
            raise ValueError(
                f"Audio decoder expects 1D or 2D output shape, got {len(output_shape)}D"
            )

        # Infer sample rate and duration
        sample_rate = kwargs.pop("sample_rate", 16000)
        duration = num_samples / sample_rate

        return AudioDecoder(
            input_dim=input_dim,
            sample_rate=sample_rate,
            duration=duration,
            channels=channels,
            **kwargs,
        )

    if decoder_type == "cross_modal":
        return CrossModalDecoder(
            input_dim=input_dim,
            output_shape=output_shape,
            **kwargs,
        )

    if decoder_type == "hierarchical_video":
        # Parse output shape
        if len(output_shape) == DIM_4D:
            num_frames, channels, height, width = output_shape
        else:
            raise ValueError(
                f"Hierarchical video decoder expects 4D output shape, got {len(output_shape)}D"
            )

        return HierarchicalVideoDecoder(
            input_dim=input_dim,
            num_frames=num_frames,
            height=height,
            width=width,
            channels=channels,
            **kwargs,
        )

    raise ValueError(f"Unknown decoder type: {decoder_type}")

"""Associative memory transformer models."""

import logging
import math
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint

from .attention import EnergyAttention, GraphEnergyAttention
from .config import (
    EnergyAttentionConfig,
    EnergyBlockConfig,
    EnergyTransformerConfig,
    HopfieldConfig,
)
from .hopfield import Hopfield
from .normalization import EnergyLayerNorm
from .vision import PatchEmbed

logger = logging.getLogger(__name__)


class EnergyTransformerBlock(nn.Module):
    """Associative memory transformer block.

    Combines energy-based attention and Hopfield network for associative memory.

    Mixed Precision Training:
        block = EnergyTransformerBlock(dim, attn_config, hopfield_config, enable_amp=True)
        scaler = torch.cuda.amp.GradScaler()

        with torch.cuda.amp.autocast():
            energy = block(features)
            loss = energy.mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    """

    def __init__(
        self,
        dim: int,
        attention_config: EnergyAttentionConfig,
        hopfield_config: HopfieldConfig,
        use_gradient_checkpointing: bool = False,
        enable_amp: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.enable_amp = enable_amp
        self.attn = EnergyAttention(attention_config, **factory_kwargs)
        self.mlp = Hopfield(dim, config=hopfield_config, **factory_kwargs)

    def energy(
        self, hidden_states: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        """Compute block energy."""
        # Note: norm is applied outside in the evolve loop, not here
        device_type = "cuda" if hidden_states.is_cuda else "cpu"

        with torch.autocast(device_type=device_type, enabled=self.enable_amp):
            attn_energy = self.attn(hidden_states, attention_mask)
            mlp_energy = self.mlp(hidden_states)

        # Energy computation should remain in fp32 for numerical stability
        with torch.autocast(device_type=device_type, enabled=False):
            # Convert to fp32 if needed and add
            if self.enable_amp:
                attn_energy = attn_energy.float()
                mlp_energy = mlp_energy.float()
            return attn_energy + mlp_energy

    def checkpointed_forward(
        self, hidden_states: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        """Forward pass with gradient checkpointing."""
        if self.training and self.use_gradient_checkpointing:
            result = checkpoint(
                self.energy,
                hidden_states,
                attention_mask,
                use_reentrant=True,
                preserve_rng_state=False,
            )
            return cast(Tensor, result)
        return self.energy(hidden_states, attention_mask)

    def forward(
        self, hidden_states: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        """Forward pass returns energy."""
        return self.energy(hidden_states, attention_mask)


class EnergyTransformer(nn.Module):
    """Associative memory model for vision tasks.

    This model uses gradient-based dynamics to evolve hidden states
    by minimizing an energy function, implementing associative memory patterns.
    """

    def __init__(
        self,
        config: EnergyTransformerConfig,
        use_gradient_checkpointing: bool = False,
        enable_amp: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.config = config
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.enable_amp = enable_amp

        # Compute image size from num_patches and patch_size
        self.img_size = int(math.sqrt(config.num_patches) * config.patch_size)

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            **factory_kwargs,
        )

        # patch_dim for output projection
        patch_dim = config.patch_dim or (config.patch_size**2 * 3)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(
                config.embed_dim,
                config.out_dim or patch_dim,
                **factory_kwargs,
            ),
        )

        # Positional encoding
        self.pos_embed = nn.Parameter(
            torch.empty(1, config.num_patches + 1, config.embed_dim, **factory_kwargs)
        )

        # Class token
        self.cls_token = nn.Parameter(
            torch.empty(1, 1, config.embed_dim, **factory_kwargs)
        )

        # Mask token for image reconstruction
        self.mask_token = nn.Parameter(
            torch.empty(1, 1, config.embed_dim, **factory_kwargs)
        )

        # Transformer blocks
        # These are guaranteed to be non-None by config.__post_init__
        assert config.attention_config is not None
        assert config.hopfield_config is not None
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        EnergyLayerNorm(config.embed_dim, **factory_kwargs),
                        EnergyTransformerBlock(
                            config.embed_dim,
                            config.attention_config,
                            config.hopfield_config,
                            use_gradient_checkpointing=self.use_gradient_checkpointing,
                            enable_amp=self.enable_amp,
                            **factory_kwargs,
                        ),
                    ]
                )
                for _ in range(config.num_layers)
            ]
        )

        self._init_weights()

    def enable_gradient_checkpointing(self, enable: bool = True) -> None:
        """Enable or disable gradient checkpointing for all blocks."""
        self.use_gradient_checkpointing = enable
        for block_pair in self.blocks:
            block_pair = cast(nn.ModuleList, block_pair)
            block = block_pair[1]  # block is at index 1, norm at index 0
            if hasattr(block, "use_gradient_checkpointing"):
                block.use_gradient_checkpointing = enable  # type: ignore[attr-defined]

    def _init_weights(self) -> None:
        """Initialize special parameters."""
        nn.init.normal_(self.pos_embed, std=0.002)
        nn.init.ones_(self.cls_token)
        nn.init.normal_(self.mask_token, std=0.002)

    def no_weight_decay(self) -> list[str]:
        """List of parameters to exclude from weight decay."""
        return ["pos_embed", "cls_token", "mask_token"]

    def visualize(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        alpha: float = 1.0,
        *,
        attn_mask: list[Tensor] | None = None,
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Visualize energy evolution during forward pass."""
        # Patch embedding (includes projection)
        x = self.patch_embed(x)

        # Apply mask if provided
        if mask is not None:
            batch_idx, mask_idx = mask
            # Use advanced indexing to set mask tokens
            for b, m in zip(batch_idx, mask_idx, strict=False):
                x[b, m] = self.mask_token.to(dtype=x.dtype)

        # Add cls token and positional encoding
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.repeat(batch_size, 1, 1).to(dtype=x.dtype)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed.to(dtype=x.dtype)

        energies = []
        embeddings = [self.patch_embed.from_patches(self.output_proj(x)[:, 1:])]

        # Evolve through blocks
        for i in range(len(self.blocks)):
            block_pair = cast(nn.ModuleList, self.blocks[i])
            norm, block = block_pair[0], block_pair[1]
            for _ in range(self.config.num_time_steps):
                g = norm(x)
                grad, energy = torch.func.grad_and_value(block)(g, attn_mask)

                x = x - alpha * grad

                energies.append(energy)
                embeddings.append(
                    self.patch_embed.from_patches(self.output_proj(x)[:, 1:])
                )

        # Final energy (without adding embedding)
        if len(self.blocks) > 0:
            block_pair = cast(nn.ModuleList, self.blocks[-1])
            norm, block = block_pair[0], block_pair[1]
            g = norm(x)
            energies.append(block(g, attn_mask))

        return energies, embeddings

    def evolve(
        self,
        x: Tensor,
        alpha: float,
        *,
        attn_mask: list[Tensor] | None = None,
        return_energy: bool = False,
    ) -> tuple[Tensor, list[Tensor] | None]:
        """Evolve hidden states through gradient dynamics."""
        energies = [] if return_energy else None

        # Prepare for stochastic gradient descent if enabled
        use_noise = self.config.use_noise and self.training
        alpha_sqrt = math.sqrt(alpha) if use_noise else 0.0

        time_step = 0
        for i in range(len(self.blocks)):
            block_pair = cast(nn.ModuleList, self.blocks[i])
            norm, block = block_pair[0], block_pair[1]
            for _t in range(self.config.num_time_steps):
                g = norm(x)
                # Use checkpointed forward if available and enabled
                if (
                    hasattr(block, "checkpointed_forward")
                    and self.use_gradient_checkpointing
                ):

                    def energy_fn(x: Tensor, _block=block) -> Tensor:
                        return cast(Tensor, _block.checkpointed_forward(x, attn_mask))  # type: ignore[attr-defined]

                    grad, energy = torch.func.grad_and_value(energy_fn)(g)
                else:

                    def energy_fn(x: Tensor, _block=block) -> Tensor:
                        return cast(Tensor, _block.forward(x, attn_mask))

                    grad, energy = torch.func.grad_and_value(energy_fn)(g)

                # Update with gradient
                x = x - alpha * grad

                # Add noise if enabled (Langevin dynamics)
                if use_noise:
                    # Compute noise std (with optional decay)
                    if self.config.noise_decay:
                        noise_std = (
                            self.config.noise_std
                            * self.config.noise_gamma
                            / ((1 + time_step) ** self.config.noise_gamma)
                        )
                    else:
                        noise_std = self.config.noise_std

                    # Add noise term
                    noise = torch.randn_like(grad) * noise_std
                    x = x + alpha_sqrt * noise

                time_step += 1

                if return_energy and energies is not None:
                    energies.append(energy)

        if return_energy and energies is not None and len(self.blocks) > 0:
            # Add final energy
            block_pair = cast(nn.ModuleList, self.blocks[-1])
            norm, block = block_pair[0], block_pair[1]
            g = norm(x)
            energy = block(g, attn_mask)
            energies.append(energy)

        return x, energies

    def forward(
        self,
        x: Tensor,
        mask: tuple[Tensor, Tensor] | None = None,
        attn_mask: list[Tensor] | None = None,
        *,
        alpha: float = 1.0,
        return_energy: bool = False,
        use_cls: bool = False,
        use_amp: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """
        Args:
            x: Input images [B, C, H, W]
            mask: Optional (batch_idx, mask_idx) for masked reconstruction
            attn_mask: Optional attention mask
            alpha: Step size for gradient dynamics
            return_energy: Return energy values
            use_cls: Use class token for output
            use_amp: Enable automatic mixed precision

        Returns:
            Reconstructed patches or (patches, energies)

        Mixed Precision Training:
            model = EnergyTransformer(config, enable_amp=True)
            scaler = torch.cuda.amp.GradScaler()

            with torch.cuda.amp.autocast():
                output = model(inputs, use_amp=True)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        """
        batch_size = x.shape[0]

        # Determine device type for autocast
        device_type = "cuda" if x.is_cuda else "cpu"
        amp_enabled = use_amp or self.enable_amp

        with torch.autocast(device_type=device_type, enabled=amp_enabled):
            # Patch embedding (includes projection)
            x = self.patch_embed(x)

            # Apply mask if provided
            if mask is not None:
                batch_idx, mask_idx = mask
                # Match original implementation mask indexing
                x[batch_idx, mask_idx] = self.mask_token.to(dtype=x.dtype)

            # Add cls token and positional encoding
            # Convert all to same dtype at once for efficiency
            target_dtype = x.dtype
            cls_tokens = self.cls_token.repeat(batch_size, 1, 1).to(dtype=target_dtype)
            pos_embed_cast = self.pos_embed.to(dtype=target_dtype)
            x = torch.cat([cls_tokens, x], dim=1)
            x = x + pos_embed_cast

            # Evolve through gradient dynamics
            x, energies = self.evolve(
                x, alpha, attn_mask=attn_mask, return_energy=return_energy
            )

            # Output projection
            x = self.output_proj(x)
            yh = x[:, :1] if use_cls else x[:, 1:]

        if return_energy and energies is not None:
            return yh, energies
        return yh


class GraphEnergyBlock(nn.Module):
    """Graph-aware energy block combining attention and Hopfield layers."""

    def __init__(
        self,
        config: EnergyBlockConfig,
        use_gradient_checkpointing: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Attention with graph support
        attn_config = EnergyAttentionConfig(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            qk_dim=config.qk_dim,
            bias=config.attn_bias,
            beta=config.attn_beta,
        )
        self.attention = GraphEnergyAttention(attn_config, device=device, dtype=dtype)

        # Hopfield layer
        hopfield_config = HopfieldConfig(
            hidden_dim_ratio=config.mlp_ratio, bias=config.mlp_bias
        )
        self.hopfield = Hopfield(
            config.embed_dim, config=hopfield_config, device=device, dtype=dtype
        )

    def forward(
        self,
        x: Tensor,
        adjacency: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            adjacency: Optional adjacency matrix [batch, seq_len, seq_len, edge_dim]
            attention_mask: Optional attention mask [batch, seq_len, seq_len]

        Returns:
            grad: Gradient of energy w.r.t. input
            energy: Combined energy from attention and Hopfield
        """

        def energy_fn(x: Tensor) -> Tensor:
            if self.training and self.use_gradient_checkpointing:
                result = checkpoint(
                    self._energy_impl,
                    x,
                    adjacency,
                    attention_mask,
                    use_reentrant=True,
                    preserve_rng_state=False,
                )
                return cast(Tensor, result)
            return self._energy_impl(x, adjacency, attention_mask)

        return torch.func.grad_and_value(energy_fn)(x)

    def _energy_impl(
        self,
        x: Tensor,
        adjacency: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Implementation of energy computation."""
        attn_energy = self.attention(x, adjacency, attention_mask)
        hopfield_energy = self.hopfield(x)
        return attn_energy + hopfield_energy


class GraphEnergyTransformer(nn.Module):
    """Graph-based associative memory model with adjacency matrix support."""

    def __init__(
        self,
        config: EnergyTransformerConfig,
        compute_correlation: bool = True,
        use_gradient_checkpointing: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.config = config
        self.compute_correlation = compute_correlation
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Node encoder
        input_dim = config.input_dim or config.patch_dim
        assert input_dim is not None, "input_dim or patch_dim must be specified"
        self.node_encoder = nn.Linear(input_dim, config.embed_dim, **factory_kwargs)

        # CLS token
        self.cls_token = nn.Parameter(
            torch.empty((1, 1, config.embed_dim), **factory_kwargs)
        )

        # Positional encoding projection
        self.pos_encoder = nn.Linear(
            config.pos_encoding_dim or 10,  # Default k=10 for graph pos encoding
            config.embed_dim,
            bias=False,
            **factory_kwargs,
        )

        # Adjacency projection
        self.adj_proj = nn.Linear(1, config.num_heads, bias=False, **factory_kwargs)

        # Correlation computation (optional)
        if self.compute_correlation:
            self.corr_conv = nn.Conv2d(
                1,
                config.num_heads,
                kernel_size=3,
                padding=1,
                bias=False,
                **factory_kwargs,
            )

        # Energy blocks
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()

        block_config = EnergyBlockConfig(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            qk_dim=config.qk_dim,
            mlp_ratio=config.mlp_ratio,
            attn_bias=config.attn_bias,
            mlp_bias=config.mlp_bias,
            attn_beta=config.attn_beta,
        )

        for _ in range(config.num_layers):
            self.norms.append(
                EnergyLayerNorm(config.embed_dim, eps=config.norm_eps, **factory_kwargs)
            )
            self.blocks.append(
                GraphEnergyBlock(
                    block_config,
                    use_gradient_checkpointing=self.use_gradient_checkpointing,
                    device=device,
                    dtype=dtype,
                )
            )

        # Output decoder
        assert config.out_dim is not None, "out_dim should be set by config post_init"
        self.decoder = nn.Linear(config.embed_dim, config.out_dim, **factory_kwargs)

        self._init_weights()

    def enable_gradient_checkpointing(self, enable: bool = True) -> None:
        """Enable or disable gradient checkpointing for all blocks."""
        self.use_gradient_checkpointing = enable
        for block in self.blocks:
            if hasattr(block, "use_gradient_checkpointing"):
                block.use_gradient_checkpointing = enable  # type: ignore[attr-defined]
            if hasattr(block, "enable_amp") and hasattr(self, "enable_amp"):
                block.enable_amp = getattr(self, "enable_amp", False)  # type: ignore[attr-defined]

    def _init_weights(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.node_encoder.weight, std=0.02)
        if self.node_encoder.bias is not None:
            nn.init.zeros_(self.node_encoder.bias)
        nn.init.normal_(self.decoder.weight, std=0.02)
        if self.decoder.bias is not None:
            nn.init.zeros_(self.decoder.bias)

    def compute_adjacency_correlation(self, x: Tensor, adjacency: Tensor) -> Tensor:
        """Compute correlation-based adjacency refinement.

        Args:
            x: Node features [batch, num_nodes, embed_dim]
            adjacency: Input adjacency [batch, num_nodes, num_nodes, edge_dim]

        Returns:
            Refined adjacency matrix
        """
        # Project adjacency to num_heads dimensions
        adj_proj = self.adj_proj(adjacency)  # [batch, nodes, nodes, heads]

        if not self.compute_correlation:
            return adj_proj.permute(0, 3, 1, 2)  # [batch, heads, nodes, nodes]

        # Compute node correlations
        x_norm = functional.normalize(x, p=2, dim=-1)
        corr = torch.bmm(x_norm, x_norm.transpose(1, 2))  # [batch, nodes, nodes]
        corr = corr.unsqueeze(1)  # [batch, 1, nodes, nodes]

        # Apply convolution
        corr_refined = self.corr_conv(corr)  # [batch, heads, nodes, nodes]

        # Combine with adjacency
        return corr_refined * adj_proj.permute(0, 3, 1, 2)

    def prepare_tokens(
        self,
        x: Tensor,
        pos_encoding: Tensor,
        adjacency: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Prepare tokens with CLS and positional encoding.

        Args:
            x: Node features [batch, num_nodes, feat_dim]
            pos_encoding: Positional encodings [batch, num_nodes+1, pos_dim]
            adjacency: Adjacency matrix [batch, num_nodes, num_nodes, edge_dim]
            mask: Valid node mask [batch, num_nodes]

        Returns:
            Prepared tokens, updated adjacency, and updated mask
        """
        batch_size = x.shape[0]

        # Encode nodes
        x = self.node_encoder(x)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1).to(dtype=x.dtype)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional encoding
        pos_embed = self.pos_encoder(pos_encoding).to(dtype=x.dtype)
        x = x + pos_embed

        # Update adjacency to include CLS token (connects to all nodes)
        # Pad adjacency for CLS token
        adjacency = functional.pad(
            adjacency, (0, 0, 1, 0, 1, 0), value=1.0
        )  # [batch, num_nodes+1, num_nodes+1, edge_dim]

        # Update mask if provided
        if mask is not None:
            cls_mask = torch.ones(batch_size, 1, device=mask.device, dtype=mask.dtype)
            mask = torch.cat([cls_mask, mask], dim=1)

        return x, adjacency, mask

    def evolve(
        self,
        x: Tensor,
        adjacency: Tensor,
        alpha: float,
        mask: Tensor | None = None,
        return_energy: bool = False,
    ) -> tuple[Tensor, list[Tensor] | None]:
        """Evolve embeddings using gradient dynamics.

        Args:
            x: Input embeddings [batch, seq_len, embed_dim]
            adjacency: Adjacency matrix [batch, heads, seq_len, seq_len]
            alpha: Step size
            mask: Valid node mask [batch, seq_len]
            return_energy: Whether to return energy values

        Returns:
            Updated embeddings and optional energy values
        """
        energies = [] if return_energy else None

        # Prepare for stochastic gradient descent if enabled
        use_noise = self.config.use_noise and self.training
        alpha_sqrt = math.sqrt(alpha) if use_noise else 0.0

        # Create attention mask from node mask
        attention_mask = None
        if mask is not None:
            # Create pairwise mask
            attention_mask = mask.unsqueeze(1) * mask.unsqueeze(2)  # [batch, seq, seq]

        # Prepare adjacency for attention
        adj_for_attn = adjacency.permute(0, 2, 3, 1)  # [batch, seq, seq, heads]

        time_step = 0
        for norm, block in zip(self.norms, self.blocks, strict=False):
            for _t in range(self.config.num_time_steps):
                g = norm(x)
                grad, energy = block(g, adj_for_attn, attention_mask)

                if return_energy and energies is not None:
                    energies.append(energy)

                # Update with gradient step
                x = x - alpha * grad

                # Add noise if enabled (Langevin dynamics)
                if use_noise:
                    # Compute noise std (with optional decay)
                    if self.config.noise_decay:
                        noise_std = (
                            self.config.noise_std
                            * self.config.noise_gamma
                            / ((1 + time_step) ** self.config.noise_gamma)
                        )
                    else:
                        noise_std = self.config.noise_std

                    # Add noise term
                    noise = torch.randn_like(grad) * noise_std
                    x = x + alpha_sqrt * noise

                time_step += 1

                # Zero out gradients for padded positions
                if mask is not None:
                    x = x * mask.unsqueeze(-1)

        return x, energies

    def forward(
        self,
        node_features: Tensor,
        adjacency: Tensor,
        pos_encoding: Tensor,
        mask: Tensor | None = None,
        alpha: float = 1.0,
        return_energy: bool = False,
    ) -> dict[str, Tensor | list[Tensor] | None]:
        """Forward pass through graph-based associative memory model.

        Args:
            node_features: Node features [batch, num_nodes, feat_dim]
            adjacency: Adjacency matrix [batch, num_nodes, num_nodes, edge_dim]
            pos_encoding: Positional encodings [batch, num_nodes, pos_dim]
            mask: Valid node mask [batch, num_nodes]
            alpha: Evolution step size
            return_energy: Whether to return energy values

        Returns:
            Dictionary with:
                - 'graph_embedding': CLS token embedding [batch, embed_dim]
                - 'node_embeddings': Node embeddings [batch, num_nodes, embed_dim]
                - 'energies': Optional list of energy values
        """
        # Prepare tokens
        x, adjacency, mask = self.prepare_tokens(
            node_features, pos_encoding, adjacency, mask
        )

        # Compute refined adjacency
        adj_refined = self.compute_adjacency_correlation(x, adjacency)

        # Evolve embeddings
        x, energies = self.evolve(x, adj_refined, alpha, mask, return_energy)

        # Decode outputs
        x = self.decoder(x)

        # Separate CLS and node embeddings
        graph_embedding = x[:, 0]  # [batch, out_dim]
        node_embeddings = x[:, 1:]  # [batch, num_nodes, out_dim]

        return {
            "graph_embedding": graph_embedding,
            "node_embeddings": node_embeddings,
            "energies": energies,
        }

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, **kwargs):
        """Load pretrained graph-based associative memory model."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = EnergyTransformerConfig(**checkpoint["config"])
        model = cls(config, **kwargs)
        model.load_state_dict(checkpoint["model"])
        return model


class METBlock(nn.Module):
    """Multimodal Energy Transformer block.

    Implements multimodal energy-based transformer: combines energy-based attention
    and cross-modal Hopfield memory for multimodal associative processing.

    Follows the same pattern as EnergyTransformerBlock but handles multiple modalities.
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        attention_config: dict,
        hopfield_config: dict,
        use_gradient_checkpointing: bool = False,
        enable_amp: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize MET block following Algorithm 1 specification.

        Args:
            modality_dims: Dict mapping modality names to feature dimensions
            attention_config: Configuration for multimodal attention
            hopfield_config: Configuration for cross-modal Hopfield memory
            use_gradient_checkpointing: Enable gradient checkpointing
            enable_amp: Enable automatic mixed precision
            device: Device for parameters
            dtype: Data type for parameters
        """
        if not modality_dims:
            raise ValueError("modality_dims cannot be empty")
        if not all(isinstance(d, int) and d > 0 for d in modality_dims.values()):
            raise ValueError("All modality dimensions must be positive")

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.modality_dims = modality_dims
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.enable_amp = enable_amp

        # Layer normalization per modality (Algorithm 1, Step 1)
        self.norms = nn.ModuleDict(
            {
                modality: nn.LayerNorm(dim, **factory_kwargs)
                for modality, dim in modality_dims.items()
            }
        )

        # Import multimodal modules
        from .attention import MultimodalEnergyAttention
        from .hopfield import CrossModalHopfield

        # Multimodal attention (Algorithm 1, Steps 2-3)
        modality_configs = {
            modality: {
                "embed_dim": dim,
                "compression_dim": attention_config.get("compression_dims", {}).get(
                    modality, 100
                ),
                "num_heads": attention_config.get("num_heads", 8),
                "qk_dim": attention_config.get("qk_dim", 64),
                "basis_type": attention_config.get("basis_type", "rectangular"),
                "regularization": attention_config.get("regularization", 0.01),
            }
            for modality, dim in modality_dims.items()
        }

        self.attention = MultimodalEnergyAttention(
            modality_configs=modality_configs,
            cross_modal_pairs=None,  # All pairs
            num_integration_points=attention_config.get("integration_points", 50),
            **factory_kwargs,
        )

        # Cross-modal Hopfield memory (Algorithm 1, Step 4)
        num_prototypes_raw = hopfield_config.get("num_prototypes", 256)
        if isinstance(num_prototypes_raw, int):
            # Use same value for all modalities
            num_prototypes = {m: num_prototypes_raw for m in modality_dims}
        elif isinstance(num_prototypes_raw, dict):
            # If dict provided, use values for matching modalities, default for others
            default_value = 256
            # Try to get a default from the dict values
            if num_prototypes_raw:
                default_value = next(iter(num_prototypes_raw.values()))
            num_prototypes = {
                m: num_prototypes_raw.get(m, default_value) for m in modality_dims
            }
        else:
            num_prototypes = {m: 256 for m in modality_dims}

        self.hopfield = CrossModalHopfield(
            modality_dims=modality_dims,
            num_prototypes=num_prototypes,
            cross_weight=hopfield_config.get("cross_modal_weight", 0.3),
            temporal_window=hopfield_config.get("temporal_window", 3),
            activation_type=hopfield_config.get("activation", "softplus"),
            **factory_kwargs,
        )

    def energy(self, features: dict[str, Tensor]) -> Tensor:
        """Compute energy following Equation (2): E = E^cross + Σ_m [E^intra_m + E^HN_m].

        Args:
            features: Dict of features per modality (will be normalized internally)

        Returns:
            Total energy (scalar)
        """
        # Validate inputs
        if not isinstance(features, dict):
            raise ValueError("Features must be a dictionary")

        missing = set(self.modality_dims) - set(features)
        if missing:
            raise ValueError(f"Missing required modality: {missing}")

        # Check dimensions and batch consistency
        batch_sizes = []
        for modality, tensor in features.items():
            if modality in self.modality_dims:
                expected_dim = self.modality_dims[modality]
                if tensor.shape[-1] != expected_dim:
                    raise ValueError(
                        f"Expected dimension {expected_dim} for modality '{modality}', "
                        f"got {tensor.shape[-1]}"
                    )
                batch_sizes.append(tensor.shape[0])

        if not all(bs == batch_sizes[0] for bs in batch_sizes):
            raise ValueError("Batch size mismatch between modalities")

        # Apply layer normalization (Algorithm 1, Step 1)
        normalized = {
            modality: self.norms[modality](tensor)
            for modality, tensor in features.items()
        }

        # Compute energy with optional checkpointing
        if self.training and self.use_gradient_checkpointing:
            result = checkpoint(self._compute_energy, normalized, use_reentrant=False)
            return cast(Tensor, result)
        result = self._compute_energy(normalized)
        return cast(Tensor, result)

    def _compute_energy(self, normalized_features: dict[str, Tensor]) -> Tensor:
        """Compute energy from normalized features."""
        device_type = (
            "cuda" if next(iter(normalized_features.values())).is_cuda else "cpu"
        )

        with torch.autocast(device_type=device_type, enabled=self.enable_amp):
            # Multimodal attention energies (Steps 2-3)
            attention_energies = self.attention(
                normalized_features, return_breakdown=True
            )

            # Cross-modal Hopfield energy (Step 4)
            hopfield_energy = self.hopfield(normalized_features)

        # Sum components in fp32 for stability (Step 5)
        with torch.autocast(device_type=device_type, enabled=False):
            device = next(iter(normalized_features.values())).device
            total = torch.tensor(0.0, device=device, dtype=torch.float32)

            # Add attention energies
            if isinstance(attention_energies, dict):
                for energy_val in attention_energies.values():
                    if self.enable_amp and energy_val.dtype != torch.float32:
                        energy_converted = energy_val.float()
                        total = total + energy_converted
                    else:
                        total = total + energy_val
            else:
                if self.enable_amp and attention_energies.dtype != torch.float32:
                    attention_energies = attention_energies.float()
                total = total + attention_energies

            # Add Hopfield energy
            if isinstance(hopfield_energy, dict):
                for energy_val in hopfield_energy.values():
                    if self.enable_amp and energy_val.dtype != torch.float32:
                        energy_converted = energy_val.float()
                        total = total + energy_converted
                    else:
                        total = total + energy_val
            else:
                if self.enable_amp and hopfield_energy.dtype != torch.float32:
                    hopfield_energy = hopfield_energy.float()
                total = total + hopfield_energy

        return total

    def forward(self, features: dict[str, Tensor]) -> Tensor:
        """Forward pass returns energy value.

        Args:
            features: Dict of modality features

        Returns:
            Energy value (scalar)
        """
        return self.energy(features)

    def extra_repr(self) -> str:
        """String representation of module configuration."""
        return f"modalities={list(self.modality_dims.keys())}"


class MultimodalEnergyTransformer(nn.Module):
    """Multimodal Energy Transformer for multiple modality processing.

    Processes multiple modalities through energy minimization like EnergyTransformer,
    but handles heterogeneous inputs through the MET architecture.
    """

    def __init__(
        self,
        config,  # METConfig
        use_gradient_checkpointing: bool = False,
        enable_amp: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize MultimodalEnergyTransformer.

        Args:
            config: METConfig with model specification
            use_gradient_checkpointing: Enable gradient checkpointing
            enable_amp: Enable automatic mixed precision
            device: Device for parameters
            dtype: Data type for parameters
        """
        if not config.modality_configs:
            raise ValueError("modality_configs cannot be empty")

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.config = config
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.enable_amp = enable_amp

        # Extract modality dimensions after projection
        self.modality_dims = {
            modality: config.embed_dim for modality in config.modality_configs
        }

        # Input projections per modality
        self.input_projs = nn.ModuleDict(
            {
                modality: nn.Linear(
                    cfg["input_dim"], config.embed_dim, **factory_kwargs
                )
                for modality, cfg in config.modality_configs.items()
            }
        )

        # Stack of MET blocks
        self.blocks = nn.ModuleList()
        self.block_norms = nn.ModuleList()  # Pre-block norms like EnergyTransformer

        for _ in range(config.num_blocks):
            # Pre-block norms per modality
            self.block_norms.append(
                nn.ModuleDict(
                    {
                        modality: nn.LayerNorm(config.embed_dim, **factory_kwargs)
                        for modality in config.modality_configs
                    }
                )
            )

            # MET block
            attention_config = {
                "num_heads": config.num_heads,
                "compression_dims": config.compression_dims or {},
                "basis_type": "rectangular",
                "regularization": 0.01,
                "integration_points": config.integration_points,
            }

            hopfield_config = {
                "num_prototypes": config.num_prototypes,
                "activation": config.hopfield_activation,
                "cross_modal_weight": config.cross_modal_weight,
                "temporal_window": config.temporal_window,
            }

            self.blocks.append(
                METBlock(
                    modality_dims=self.modality_dims,
                    attention_config=attention_config,
                    hopfield_config=hopfield_config,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    enable_amp=enable_amp,
                    **factory_kwargs,
                )
            )

        # Output layer norms
        self.output_norms = nn.ModuleDict(
            {
                modality: nn.LayerNorm(config.embed_dim, **factory_kwargs)
                for modality in config.modality_configs
            }
        )

        # Output projections
        self.output_projs = nn.ModuleDict(
            {
                modality: nn.Linear(
                    config.embed_dim,
                    cfg.get("output_dim", cfg["input_dim"]),
                    **factory_kwargs,
                )
                for modality, cfg in config.modality_configs.items()
            }
        )

    def enable_gradient_checkpointing(self, enable: bool = True) -> None:
        """Enable or disable gradient checkpointing for all blocks."""
        self.use_gradient_checkpointing = enable
        for block in self.blocks:
            block.use_gradient_checkpointing = enable  # type: ignore[attr-defined]

    def no_weight_decay(self) -> list[str]:
        """List of parameters to exclude from weight decay."""
        no_decay = []
        for name, _ in self.named_parameters():
            if "norm" in name or "bias" in name:
                no_decay.append(name)
        return no_decay

    def evolve(
        self,
        features: dict[str, Tensor],
        step_size: float | None = None,
        return_energy: bool = False,
    ) -> tuple[dict[str, Tensor], list[Tensor] | None]:
        """Evolve features through gradient dynamics.

        Args:
            features: Input features per modality
            step_size: Step size η for gradient descent (uses config.step_size if None)
            return_energy: Whether to return energy values

        Returns:
            Evolved features and optional energy trajectory
        """
        # Use config step_size if not provided
        if step_size is None:
            step_size = getattr(self.config, "step_size", 0.001)

        energies = [] if return_energy else None
        x = {modality: tensor.clone() for modality, tensor in features.items()}

        # Evolve through blocks
        for _i, (norms, block) in enumerate(
            zip(self.block_norms, self.blocks, strict=False)
        ):
            for _ in range(self.config.num_time_steps):
                # Normalize per modality
                g = {
                    modality: cast(nn.ModuleDict, norms)[modality](x[modality])
                    for modality in x
                }

                # Compute gradient and energy
                def energy_fn(features: dict[str, Tensor], _block=block) -> Tensor:
                    return _block.energy(features)  # type: ignore[no-any-return]

                grad, energy = torch.func.grad_and_value(energy_fn)(g)

                # Update with gradient descent: x = x - η∇E
                for modality, tensor in x.items():
                    x[modality] = tensor - step_size * grad[modality]

                if return_energy and energies is not None:
                    energies.append(energy)

        return x, energies

    def _validate_input_completeness(self, inputs: dict[str, Tensor]) -> None:
        """Validate that all required modalities are provided."""
        expected = set(self.config.modality_configs.keys())
        provided = set(inputs.keys()) - {"pos_encodings"}
        missing = expected - provided
        if missing:
            raise ValueError(f"Missing required modality: {missing}")

    def _validate_input_dimensions(self, inputs: dict[str, Tensor]) -> None:
        """Validate input dimensions and consistency."""
        batch_sizes = []
        seq_lengths = []
        for modality, tensor in inputs.items():
            if modality == "pos_encodings":
                continue
            if modality in self.config.modality_configs:
                expected_dim = self.config.modality_configs[modality]["input_dim"]
                if tensor.shape[-1] != expected_dim:
                    raise ValueError(
                        f"Expected dimension {expected_dim} for modality '{modality}', "
                        f"got {tensor.shape[-1]}"
                    )
                batch_sizes.append(tensor.shape[0])
                seq_lengths.append(tensor.shape[1])

        if batch_sizes and not all(bs == batch_sizes[0] for bs in batch_sizes):
            raise ValueError("Batch size mismatch between modalities")
        if seq_lengths and not all(sl == seq_lengths[0] for sl in seq_lengths):
            raise ValueError("Sequence length mismatch between modalities")

    def _add_positional_encodings(
        self, projected: dict[str, Tensor], inputs: dict[str, Tensor]
    ) -> None:
        """Add positional encodings if provided."""
        if "pos_encodings" not in inputs:
            return

        pos_encodings_raw = inputs["pos_encodings"]
        pos_encodings = cast(dict[str, Tensor], pos_encodings_raw)
        for modality, tensor in projected.items():
            if isinstance(pos_encodings, dict) and modality in pos_encodings:
                projected[modality] = tensor + pos_encodings[modality]

    def project_inputs(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Project inputs to embedding dimension.

        Args:
            inputs: Dict of raw inputs per modality

        Returns:
            Dict of projected features
        """
        # Validate inputs
        self._validate_input_completeness(inputs)
        self._validate_input_dimensions(inputs)

        # Project inputs
        projected = {
            modality: self.input_projs[modality](tensor)
            for modality, tensor in inputs.items()
            if modality != "pos_encodings"
        }

        # Add positional encodings if provided
        self._add_positional_encodings(projected, inputs)
        return projected

    def forward(
        self,
        inputs: dict[str, Tensor],
        step_size: float | None = None,
        return_energies: bool = False,
        use_amp: bool = False,
    ) -> dict[str, Tensor] | tuple[dict[str, Tensor], list[Tensor]]:
        """Forward pass through MET.

        Args:
            inputs: Dict of inputs per modality
            step_size: Step size η for gradient descent (uses config.step_size if None)
            return_energies: Whether to return energy trajectory
            use_amp: Enable automatic mixed precision

        Returns:
            Output features dict, or (outputs, energies) if return_energies=True
        """
        device_type = "cuda" if next(iter(inputs.values())).is_cuda else "cpu"
        amp_enabled = use_amp or self.enable_amp

        with torch.autocast(device_type=device_type, enabled=amp_enabled):
            # Project inputs
            x = self.project_inputs(inputs)

            # Evolve through gradient dynamics
            x, energies = self.evolve(
                x, step_size=step_size, return_energy=return_energies
            )

            # Apply output norms and projections
            outputs = {}
            for modality in x:
                # Apply output norm
                if modality in self.output_norms:
                    normed = self.output_norms[modality](x[modality])
                else:
                    normed = x[modality]

                # Apply output projection if exists
                if modality in self.output_projs:
                    outputs[modality] = self.output_projs[modality](normed)
                else:
                    outputs[modality] = normed

        if return_energies and energies is not None:
            return cast(tuple[dict[str, Tensor], list[Tensor]], (outputs, energies))
        return cast(dict[str, Tensor], outputs)

    def visualize(
        self,
        inputs: dict[str, Tensor],
        step_size: float | None = None,
    ) -> tuple[list[Tensor], dict[str, list[Tensor]]]:
        """Visualize energy evolution during forward pass.

        Args:
            inputs: Input features dict
            step_size: Step size η for gradient descent (uses config.step_size if None)

        Returns:
            Tuple of (energies, embeddings_dict)
        """
        # Use config step_size if not provided
        if step_size is None:
            step_size = getattr(self.config, "step_size", 0.001)

        # Project inputs
        x = self.project_inputs(inputs)

        energies = []
        embeddings_dict = {modality: [x[modality].clone()] for modality in x}

        # Evolve through blocks and track states
        for norms, block in zip(self.block_norms, self.blocks, strict=False):
            for _ in range(self.config.num_time_steps):
                # Normalize
                g = {
                    modality: cast(nn.ModuleDict, norms)[modality](x[modality])
                    for modality in x
                }

                # Compute gradient and energy
                def energy_fn(features: dict[str, Tensor], _block=block) -> Tensor:
                    return _block.energy(features)  # type: ignore[no-any-return]

                grad, energy = torch.func.grad_and_value(energy_fn)(g)

                # Update with gradient descent: x = x - η∇E
                for modality, tensor in x.items():
                    x[modality] = tensor - step_size * grad[modality]
                    embeddings_dict[modality].append(x[modality].clone())

                energies.append(energy)

        return energies, embeddings_dict

    def extra_repr(self) -> str:
        """String representation of module configuration."""
        return (
            f"num_blocks={len(self.blocks)}, modalities={list(self.input_projs.keys())}"
        )

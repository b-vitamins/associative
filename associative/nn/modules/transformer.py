"""Associative memory transformer models."""

import logging
import math
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional

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
    """

    def __init__(
        self,
        dim: int,
        attention_config: EnergyAttentionConfig,
        hopfield_config: HopfieldConfig,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.attn = EnergyAttention(attention_config, **factory_kwargs)
        self.mlp = Hopfield(dim, config=hopfield_config, **factory_kwargs)

    def energy(
        self, hidden_states: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        """Compute block energy."""
        # Note: norm is applied outside in the evolve loop, not here
        attn_energy = self.attn(hidden_states, attention_mask)
        mlp_energy = self.mlp(hidden_states)
        return attn_energy + mlp_energy

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
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.config = config

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
                            **factory_kwargs,
                        ),
                    ]
                )
                for _ in range(config.num_layers)
            ]
        )

        self._init_weights()

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
                grad, energy = torch.func.grad_and_value(block)(g, attn_mask)

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
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """
        Args:
            x: Input images [B, C, H, W]
            mask: Optional (batch_idx, mask_idx) for masked reconstruction
            attn_mask: Optional attention mask
            alpha: Step size for gradient dynamics
            return_energy: Return energy values
            use_cls: Use class token for output

        Returns:
            Reconstructed patches or (patches, energies)
        """
        batch_size = x.shape[0]

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
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config

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
            attn_energy = self.attention(x, adjacency, attention_mask)
            hopfield_energy = self.hopfield(x)
            return attn_energy + hopfield_energy

        return torch.func.grad_and_value(energy_fn)(x)


class GraphEnergyTransformer(nn.Module):
    """Graph-based associative memory model with adjacency matrix support."""

    def __init__(
        self,
        config: EnergyTransformerConfig,
        compute_correlation: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.config = config
        self.compute_correlation = compute_correlation

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
                GraphEnergyBlock(block_config, device=device, dtype=dtype)
            )

        # Output decoder
        assert config.out_dim is not None, "out_dim should be set by config post_init"
        self.decoder = nn.Linear(config.embed_dim, config.out_dim, **factory_kwargs)

        self._init_weights()

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

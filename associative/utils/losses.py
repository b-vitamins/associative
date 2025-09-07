"""Loss functions for multimodal energy transformer training.

Clean, focused implementations of reconstruction, contrastive, triplet,
and composite losses for multi-objective training.
"""

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

# Constants for dimension checks
DIM_IMAGE = 4
CHANNELS_RGB = 3
MIN_IMAGE_SIZE = 32


class ReconstructionLoss(nn.Module):
    """Reconstruction loss for autoencoding tasks.

    Supports multiple pixel-wise loss types with optional masking
    and perceptual loss for improved visual quality.

    Args:
        loss_type: Type of pixel loss ("l1", "l2", "smooth_l1")
        reduction: How to reduce loss ("mean", "sum", "none")
        use_perceptual: Whether to add perceptual loss
        perceptual_weight: Weight for perceptual loss component
    """

    def __init__(
        self,
        loss_type: Literal["l1", "l2", "smooth_l1"] = "l2",
        reduction: Literal["mean", "sum", "none"] = "mean",
        use_perceptual: bool = False,
        perceptual_weight: float = 0.1,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        self.use_perceptual = use_perceptual
        self.perceptual_weight = perceptual_weight

        # Lazy initialization of perceptual loss
        self._perceptual_net: nn.Module | None = None

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Compute reconstruction loss.

        Args:
            pred: Predicted reconstruction
            target: Ground truth target
            mask: Optional binary mask (1 = include, 0 = exclude)

        Returns:
            Scalar loss value
        """
        # Compute base reconstruction loss
        if mask is not None:
            loss = self._compute_masked_loss(pred, target, mask)
        else:
            loss = self._compute_base_loss(pred, target)

        # Add perceptual loss if enabled and applicable
        if self.use_perceptual and self._is_image_tensor(pred):
            perceptual_loss = self._compute_perceptual_loss(pred, target)
            loss = loss + self.perceptual_weight * perceptual_loss

        return loss

    def _compute_base_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute base reconstruction loss without masking.

        Args:
            pred: Predicted reconstruction tensor.
            target: Ground truth target tensor.

        Returns:
            Computed loss based on the configured loss type.

        Raises:
            ValueError: If loss type is unknown.
        """
        if self.loss_type == "l1":
            return F.l1_loss(pred, target, reduction=self.reduction)
        if self.loss_type == "l2":
            return F.mse_loss(pred, target, reduction=self.reduction)
        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred, target, reduction=self.reduction)
        raise ValueError(f"Unknown loss type: {self.loss_type}")

    def _compute_masked_loss(
        self, pred: Tensor, target: Tensor, mask: Tensor
    ) -> Tensor:
        """Compute loss only on masked regions.

        Args:
            pred: Predicted reconstruction tensor.
            target: Ground truth target tensor.
            mask: Binary mask tensor (1 = include, 0 = exclude).

        Returns:
            Loss computed only over masked (valid) regions.
        """
        mask_sum = mask.sum()
        if mask_sum == 0:
            # All masked out - return zero loss
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # Apply mask to predictions and targets
        pred_masked = pred * mask
        target_masked = target * mask

        # Compute element-wise loss
        diff = self._compute_element_loss(pred_masked, target_masked)

        # Apply reduction accounting for mask
        return self._apply_reduction(diff, mask_sum)

    def _compute_element_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute element-wise loss based on loss type."""
        if self.loss_type == "l1":
            return (pred - target).abs()
        if self.loss_type == "l2":
            return (pred - target).pow(2)
        if self.loss_type == "smooth_l1":
            # Manual smooth L1 to handle masking properly
            abs_diff = (pred - target).abs()
            return torch.where(abs_diff < 1.0, 0.5 * abs_diff.pow(2), abs_diff - 0.5)
        raise ValueError(f"Unknown loss type: {self.loss_type}")

    def _apply_reduction(self, diff: Tensor, mask_sum: Tensor | None = None) -> Tensor:
        """Apply reduction to loss tensor."""
        if self.reduction == "mean":
            if mask_sum is not None:
                return diff.sum() / mask_sum
            return diff.mean()
        if self.reduction == "sum":
            return diff.sum()
        if self.reduction == "none":
            return diff
        raise ValueError(f"Unknown reduction: {self.reduction}")

    def _is_image_tensor(self, tensor: Tensor) -> bool:
        """Check if tensor is image-like (B, C, H, W) with RGB channels."""
        return (
            tensor.dim() == DIM_IMAGE
            and tensor.shape[1] == CHANNELS_RGB
            and tensor.shape[-1] >= MIN_IMAGE_SIZE
        )

    def _compute_perceptual_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute perceptual loss using pretrained features."""
        if self._perceptual_net is None:
            self._init_perceptual_net(pred.device)

        # Extract features
        assert self._perceptual_net is not None, "Perceptual net should be initialized"
        pred_features = self._perceptual_net(pred)
        target_features = self._perceptual_net(target)

        # Compare features
        return F.mse_loss(pred_features, target_features)

    def _init_perceptual_net(self, device: torch.device):
        """Initialize perceptual loss network using VGG16 features.

        Args:
            device: Device to move the network to.
        """
        try:
            import torchvision
        except ImportError:
            # Fallback to pixel loss only
            self.use_perceptual = False
            return

        # Use VGG16 features for perceptual loss
        vgg = torchvision.models.vgg16(
            weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1
        )
        # Extract early convolutional features
        # vgg.features is a Sequential module, cast for indexing
        features = vgg.features  # type: ignore[attr-defined]
        self._perceptual_net = nn.Sequential(*list(features[:8])).to(device)  # type: ignore[index]
        self._perceptual_net.eval()

        # Freeze weights
        for param in self._perceptual_net.parameters():
            param.requires_grad = False


class ContrastiveLoss(nn.Module):
    """Contrastive loss for cross-modal alignment.

    Implements InfoNCE loss that treats batch samples as negatives
    to learn aligned representations across modalities.

    Args:
        temperature: Temperature for softmax distribution
        normalize: Whether to L2-normalize embeddings
        symmetric: Whether to compute symmetric loss
    """

    def __init__(
        self,
        temperature: float = 0.07,
        normalize: bool = True,
        symmetric: bool = True,
    ):
        super().__init__()
        self.temperature = max(temperature, 1e-8)  # Numerical stability
        self.normalize = normalize
        self.symmetric = symmetric

    def forward(self, embeddings_a: Tensor, embeddings_b: Tensor) -> Tensor:
        """Compute contrastive loss between embedding pairs.

        Args:
            embeddings_a: Embeddings from modality A (batch, dim)
            embeddings_b: Embeddings from modality B (batch, dim)

        Returns:
            InfoNCE loss value
        """
        # Normalize if requested
        if self.normalize:
            embeddings_a = F.normalize(embeddings_a, p=2, dim=-1)
            embeddings_b = F.normalize(embeddings_b, p=2, dim=-1)

        # Compute similarity matrix
        batch_size = embeddings_a.shape[0]
        similarity = torch.matmul(embeddings_a, embeddings_b.T) / self.temperature

        # Labels: positive pairs are on diagonal
        labels = torch.arange(batch_size, device=embeddings_a.device)

        # Compute loss
        loss_a_to_b = F.cross_entropy(similarity, labels)

        if self.symmetric:
            # Compute reverse direction
            loss_b_to_a = F.cross_entropy(similarity.T, labels)
            return (loss_a_to_b + loss_b_to_a) / 2

        return loss_a_to_b


class TripletLoss(nn.Module):
    """Triplet margin loss for metric learning.

    Enforces that anchor is closer to positive than negative
    by at least a margin in the embedding space.

    Args:
        margin: Minimum distance margin between positive and negative
        distance: Distance metric ("euclidean" or "cosine")
        reduction: How to reduce batch dimension
        swap: Whether to use harder negative (swap if needed)
    """

    def __init__(
        self,
        margin: float = 0.2,
        distance: Literal["euclidean", "cosine"] = "euclidean",
        reduction: Literal["mean", "sum", "none"] = "mean",
        swap: bool = False,
    ):
        super().__init__()
        self.margin = margin
        self.distance = distance
        self.reduction = reduction
        self.swap = swap

    def forward(
        self,
        anchor: Tensor,
        positive: Tensor,
        negative: Tensor,
    ) -> Tensor:
        """Compute triplet loss.

        Args:
            anchor: Anchor embeddings (batch, dim)
            positive: Positive embeddings (batch, dim)
            negative: Negative embeddings (batch, dim)

        Returns:
            Triplet loss value
        """
        if self.distance == "euclidean":
            # Euclidean distance
            dist_pos = F.pairwise_distance(anchor, positive, p=2)
            dist_neg = F.pairwise_distance(anchor, negative, p=2)

            if self.swap:
                # Also compute distance from positive to negative
                dist_swap = F.pairwise_distance(positive, negative, p=2)
                # Use harder negative
                dist_neg = torch.minimum(dist_neg, dist_swap)

        elif self.distance == "cosine":
            # Cosine distance = 1 - cosine_similarity
            dist_pos = 1 - F.cosine_similarity(anchor, positive, dim=-1)
            dist_neg = 1 - F.cosine_similarity(anchor, negative, dim=-1)

            if self.swap:
                dist_swap = 1 - F.cosine_similarity(positive, negative, dim=-1)
                dist_neg = torch.minimum(dist_neg, dist_swap)

        else:
            raise ValueError(f"Unknown distance: {self.distance}")

        # Compute loss: max(0, margin + dist_pos - dist_neg)
        losses = F.relu(self.margin + dist_pos - dist_neg)

        # Apply reduction
        if self.reduction == "mean":
            return losses.mean()
        if self.reduction == "sum":
            return losses.sum()
        if self.reduction == "none":
            return losses
        raise ValueError(f"Unknown reduction: {self.reduction}")


class CompositeLoss(nn.Module):
    """Composite loss combining multiple objectives.

    Manages weighted combination of reconstruction, contrastive,
    and triplet losses for multi-objective training.

    Args:
        reconstruction_weight: Weight for reconstruction loss
        contrastive_weight: Weight for contrastive loss
        triplet_weight: Weight for triplet loss
        reconstruction_config: Config for reconstruction loss
        contrastive_config: Config for contrastive loss
        triplet_config: Config for triplet loss
    """

    def __init__(  # noqa: PLR0913
        self,
        reconstruction_weight: float = 1.0,
        contrastive_weight: float = 0.0,
        triplet_weight: float = 0.0,
        reconstruction_config: dict | None = None,
        contrastive_config: dict | None = None,
        triplet_config: dict | None = None,
    ):
        super().__init__()

        self.reconstruction_weight = reconstruction_weight
        self.contrastive_weight = contrastive_weight
        self.triplet_weight = triplet_weight

        # Initialize component losses
        self.reconstruction_loss = None
        self.contrastive_loss = None
        self.triplet_loss = None

        if reconstruction_weight > 0:
            config = reconstruction_config or {}
            self.reconstruction_loss = ReconstructionLoss(**config)

        if contrastive_weight > 0:
            config = contrastive_config or {}
            self.contrastive_loss = ContrastiveLoss(**config)

        if triplet_weight > 0:
            config = triplet_config or {}
            self.triplet_loss = TripletLoss(**config)

    def forward(
        self,
        reconstruction: tuple[Tensor, Tensor] | None = None,
        contrastive: tuple[Tensor, Tensor] | None = None,
        triplet: tuple[Tensor, Tensor, Tensor] | None = None,
        return_dict: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        """Compute composite loss.

        Args:
            reconstruction: Tuple of (predictions, targets)
            contrastive: Tuple of (embeddings_a, embeddings_b)
            triplet: Tuple of (anchor, positive, negative)
            return_dict: Whether to return loss component dict

        Returns:
            Total loss or dict with loss components
        """
        losses = {}
        total_loss = torch.tensor(0.0, requires_grad=True)

        # Reconstruction loss
        if reconstruction is not None and self.reconstruction_loss is not None:
            pred, target = reconstruction
            loss = self.reconstruction_loss(pred, target)
            losses["reconstruction"] = loss
            total_loss = total_loss + self.reconstruction_weight * loss

        # Contrastive loss
        if contrastive is not None and self.contrastive_loss is not None:
            emb_a, emb_b = contrastive
            loss = self.contrastive_loss(emb_a, emb_b)
            losses["contrastive"] = loss
            total_loss = total_loss + self.contrastive_weight * loss

        # Triplet loss
        if triplet is not None and self.triplet_loss is not None:
            anchor, positive, negative = triplet
            loss = self.triplet_loss(anchor, positive, negative)
            losses["triplet"] = loss
            total_loss = total_loss + self.triplet_weight * loss

        # Ensure we're on the right device
        if losses:
            device = next(iter(losses.values())).device
            if total_loss.device != device:
                total_loss = total_loss.to(device)

        losses["total"] = total_loss

        if return_dict:
            return losses
        return total_loss

    def update_weights(
        self,
        reconstruction: float | None = None,
        contrastive: float | None = None,
        triplet: float | None = None,
    ) -> None:
        """Update loss component weights dynamically.

        Args:
            reconstruction: New reconstruction weight
            contrastive: New contrastive weight
            triplet: New triplet weight
        """
        if reconstruction is not None:
            self.reconstruction_weight = reconstruction
        if contrastive is not None:
            self.contrastive_weight = contrastive
        if triplet is not None:
            self.triplet_weight = triplet

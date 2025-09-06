"""Tests for loss functions in associative.utils.losses.

Tests focus on behavioral contracts and public API, verifying that losses:
1. Produce mathematically correct results
2. Maintain proper gradient flow
3. Handle edge cases gracefully
4. Compose well for multi-objective training
"""

import pytest
import torch

from associative.utils.losses import (
    CompositeLoss,
    ContrastiveLoss,
    ReconstructionLoss,
    TripletLoss,
)


class TestReconstructionLoss:
    """Test reconstruction loss behavior."""

    def test_l1_loss_computation(self):
        """L1 loss should compute mean absolute error."""
        loss_fn = ReconstructionLoss(loss_type="l1")
        pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        target = torch.tensor([[1.5, 2.5], [2.5, 3.5]])

        loss = loss_fn(pred, target)

        # L1 loss = mean(|pred - target|) = mean([0.5, 0.5, 0.5, 0.5]) = 0.5
        assert torch.allclose(loss, torch.tensor(0.5), atol=1e-6)
        assert loss.requires_grad

    def test_l2_loss_computation(self):
        """L2 loss should compute mean squared error."""
        loss_fn = ReconstructionLoss(loss_type="l2")
        pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        target = torch.tensor([[1.5, 2.5], [2.5, 3.5]])

        loss = loss_fn(pred, target)

        # L2 loss = mean((pred - target)^2) = mean([0.25, 0.25, 0.25, 0.25]) = 0.25
        assert torch.allclose(loss, torch.tensor(0.25), atol=1e-6)
        assert loss.requires_grad

    def test_smooth_l1_loss_computation(self):
        """Smooth L1 should behave like L2 for small errors, L1 for large."""
        loss_fn = ReconstructionLoss(loss_type="smooth_l1")

        # Small errors (< 1.0) - behaves like L2
        pred_small = torch.tensor([0.0], requires_grad=True)
        target_small = torch.tensor([0.5])
        loss_small = loss_fn(pred_small, target_small)
        # For |x| < 1: loss = 0.5 * x^2 = 0.5 * 0.25 = 0.125
        assert torch.allclose(loss_small, torch.tensor(0.125), atol=1e-6)

        # Large errors (>= 1.0) - behaves like L1 - 0.5
        pred_large = torch.tensor([0.0], requires_grad=True)
        target_large = torch.tensor([2.0])
        loss_large = loss_fn(pred_large, target_large)
        # For |x| >= 1: loss = |x| - 0.5 = 2.0 - 0.5 = 1.5
        assert torch.allclose(loss_large, torch.tensor(1.5), atol=1e-6)

    def test_mask_application(self):
        """Mask should zero out loss in masked regions."""
        loss_fn = ReconstructionLoss(loss_type="l2")
        # Create predictions where masked regions have different values
        pred = torch.ones(2, 3, 4, 4, requires_grad=True)
        target = torch.zeros(2, 3, 4, 4)

        # Set different values in regions we'll mask
        pred_varied = pred.clone()
        pred_varied[:, :, :2, :] = 10.0  # Large error in top half

        # Create mask that masks out top half (high error region)
        mask = torch.ones(2, 3, 4, 4)
        mask[:, :, :2, :] = 0  # Mask out high error region

        loss_with_mask = loss_fn(pred_varied, target, mask=mask)
        loss_without_mask = loss_fn(pred_varied, target)

        # With mask excluding high error regions, loss should be much smaller
        assert loss_with_mask < loss_without_mask
        assert loss_with_mask.requires_grad

    def test_reduction_modes(self):
        """Test different reduction modes."""
        pred = torch.randn(2, 3, 4, 4, requires_grad=True)
        target = torch.randn(2, 3, 4, 4)

        # Mean reduction (default)
        loss_mean = ReconstructionLoss(reduction="mean")(pred, target)
        assert loss_mean.shape == ()

        # Sum reduction
        loss_sum = ReconstructionLoss(reduction="sum")(pred, target)
        assert loss_sum.shape == ()
        assert loss_sum > loss_mean  # Sum should be larger

        # No reduction
        loss_none = ReconstructionLoss(reduction="none")(pred, target)
        assert loss_none.shape == (2, 3, 4, 4)

    def test_perceptual_loss_integration(self):
        """Test perceptual loss adds to reconstruction loss."""
        loss_fn = ReconstructionLoss(
            loss_type="l2", use_perceptual=True, perceptual_weight=0.1
        )

        # Use properly sized RGB images for perceptual loss
        pred = torch.rand(2, 3, 64, 64, requires_grad=True)
        target = torch.rand(2, 3, 64, 64)

        loss_with_perceptual = loss_fn(pred, target)

        # Compare with pixel-only loss
        loss_pixel_only = ReconstructionLoss(loss_type="l2")(pred, target)

        # Perceptual loss should add to total
        assert loss_with_perceptual > loss_pixel_only
        assert loss_with_perceptual.requires_grad

    def test_handles_different_tensor_shapes(self):
        """Loss should work with various input shapes."""
        loss_fn = ReconstructionLoss()

        # 2D tensors (batch of vectors)
        pred_2d = torch.randn(32, 768, requires_grad=True)
        target_2d = torch.randn(32, 768)
        loss_2d = loss_fn(pred_2d, target_2d)
        assert loss_2d.shape == ()

        # 4D tensors (images)
        pred_4d = torch.randn(8, 3, 32, 32, requires_grad=True)
        target_4d = torch.randn(8, 3, 32, 32)
        loss_4d = loss_fn(pred_4d, target_4d)
        assert loss_4d.shape == ()

        # 5D tensors (video)
        pred_5d = torch.randn(4, 8, 3, 32, 32, requires_grad=True)
        target_5d = torch.randn(4, 8, 3, 32, 32)
        loss_5d = loss_fn(pred_5d, target_5d)
        assert loss_5d.shape == ()

    def test_gradient_flow(self):
        """Verify gradients flow through the loss."""
        loss_fn = ReconstructionLoss()
        pred = torch.randn(2, 3, 16, 16, requires_grad=True)
        target = torch.randn(2, 3, 16, 16)

        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert not torch.allclose(pred.grad, torch.zeros_like(pred.grad))

    def test_non_negative_loss(self):
        """Reconstruction loss should always be non-negative."""
        loss_fn = ReconstructionLoss()

        for _ in range(10):
            pred = torch.randn(4, 3, 32, 32, requires_grad=True)
            target = torch.randn(4, 3, 32, 32)
            loss = loss_fn(pred, target)
            assert loss >= 0


class TestContrastiveLoss:
    """Test contrastive loss behavior."""

    def test_infonce_loss_computation(self):
        """InfoNCE loss should treat batch samples as negatives."""
        loss_fn = ContrastiveLoss(temperature=0.1)

        # Create simple embeddings where first two are similar
        embeddings_a = torch.tensor(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], requires_grad=True
        )
        embeddings_b = torch.tensor(
            [
                [1.0, 0.1],  # Similar to a[0]
                [0.8, 0.2],  # Similar to a[1]
                [0.1, 1.0],  # Similar to a[2]
                [0.0, 0.9],  # Similar to a[3]
            ],
            requires_grad=True,
        )

        loss = loss_fn(embeddings_a, embeddings_b)

        assert loss > 0  # Contrastive loss is positive
        assert loss.requires_grad

    def test_temperature_effect(self):
        """Lower temperature should create sharper distributions."""
        embeddings_a = torch.randn(16, 256, requires_grad=True)
        embeddings_b = torch.randn(16, 256, requires_grad=True)

        loss_high_temp = ContrastiveLoss(temperature=1.0)(embeddings_a, embeddings_b)
        loss_low_temp = ContrastiveLoss(temperature=0.01)(embeddings_a, embeddings_b)

        # Lower temperature typically leads to higher loss (sharper distribution)
        assert loss_low_temp > loss_high_temp

    def test_normalized_embeddings(self):
        """Normalization should be applied when requested."""
        loss_fn = ContrastiveLoss(normalize=True)

        # Create embeddings with different norms
        embeddings_a = torch.randn(8, 128) * 10
        embeddings_b = torch.randn(8, 128) * 0.1
        embeddings_a.requires_grad_()
        embeddings_b.requires_grad_()

        loss = loss_fn(embeddings_a, embeddings_b)

        # Loss should be computed on normalized embeddings
        assert loss > 0
        assert loss.requires_grad

    def test_symmetric_loss(self):
        """Symmetric loss should average both directions."""
        embeddings_a = torch.randn(8, 128, requires_grad=True)
        embeddings_b = torch.randn(8, 128, requires_grad=True)

        loss_symmetric = ContrastiveLoss(symmetric=True)(embeddings_a, embeddings_b)
        loss_asymmetric = ContrastiveLoss(symmetric=False)(embeddings_a, embeddings_b)

        # Symmetric and asymmetric should generally differ
        assert loss_symmetric != loss_asymmetric

    def test_batch_size_scaling(self):
        """Loss should handle different batch sizes."""
        loss_fn = ContrastiveLoss()

        for batch_size in [2, 8, 32]:
            embeddings_a = torch.randn(batch_size, 256, requires_grad=True)
            embeddings_b = torch.randn(batch_size, 256, requires_grad=True)

            loss = loss_fn(embeddings_a, embeddings_b)
            assert loss.shape == ()
            assert loss > 0

    def test_gradient_flow(self):
        """Verify gradients flow through both embeddings."""
        loss_fn = ContrastiveLoss()
        embeddings_a = torch.randn(8, 128, requires_grad=True)
        embeddings_b = torch.randn(8, 128, requires_grad=True)

        loss = loss_fn(embeddings_a, embeddings_b)
        loss.backward()

        assert embeddings_a.grad is not None
        assert embeddings_b.grad is not None
        assert not torch.allclose(
            embeddings_a.grad, torch.zeros_like(embeddings_a.grad)
        )


class TestTripletLoss:
    """Test triplet loss behavior."""

    def test_triplet_margin_computation(self):
        """Triplet loss should enforce margin between positive and negative."""
        loss_fn = TripletLoss(margin=1.0)

        # Create embeddings where anchor is closer to positive
        anchor = torch.tensor([[1.0, 0.0]], requires_grad=True)
        positive = torch.tensor([[0.9, 0.1]], requires_grad=True)  # Close to anchor
        negative = torch.tensor([[0.0, 1.0]], requires_grad=True)  # Far from anchor

        loss = loss_fn(anchor, positive, negative)

        # Loss should be max(0, d(a,p) - d(a,n) + margin)
        # d(a,p) ≈ 0.14, d(a,n) ≈ 1.41, loss = max(0, 0.14 - 1.41 + 1.0) = 0
        assert loss >= 0
        assert loss.requires_grad

    def test_margin_violation(self):
        """Loss should be positive when margin is violated."""
        loss_fn = TripletLoss(margin=2.0)  # Large margin

        anchor = torch.tensor([[1.0, 0.0]], requires_grad=True)
        positive = torch.tensor([[0.8, 0.2]], requires_grad=True)
        negative = torch.tensor([[0.5, 0.5]], requires_grad=True)

        loss = loss_fn(anchor, positive, negative)

        # With large margin, loss should be positive
        assert loss > 0

    def test_no_loss_when_well_separated(self):
        """Loss should be zero when triplets are well separated."""
        loss_fn = TripletLoss(margin=0.1)  # Small margin

        anchor = torch.tensor([[1.0, 0.0]], requires_grad=True)
        positive = torch.tensor([[1.0, 0.0]], requires_grad=True)  # Same as anchor
        negative = torch.tensor([[-1.0, 0.0]], requires_grad=True)  # Opposite

        loss = loss_fn(anchor, positive, negative)

        # d(a,p) = 0, d(a,n) = 2, loss = max(0, 0 - 2 + 0.1) = 0
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_distance_type_options(self):
        """Test different distance metrics."""
        anchor = torch.randn(8, 128, requires_grad=True)
        positive = torch.randn(8, 128, requires_grad=True)
        negative = torch.randn(8, 128, requires_grad=True)

        # Euclidean distance
        loss_euclidean = TripletLoss(distance="euclidean")(anchor, positive, negative)
        assert loss_euclidean >= 0

        # Cosine distance
        loss_cosine = TripletLoss(distance="cosine")(anchor, positive, negative)
        assert loss_cosine >= 0

    def test_reduction_modes(self):
        """Test different reduction modes."""
        anchor = torch.randn(8, 128, requires_grad=True)
        positive = torch.randn(8, 128, requires_grad=True)
        negative = torch.randn(8, 128, requires_grad=True)

        loss_mean = TripletLoss(reduction="mean")(anchor, positive, negative)
        loss_sum = TripletLoss(reduction="sum")(anchor, positive, negative)

        assert loss_mean.shape == ()
        assert loss_sum.shape == ()
        # Sum should be larger for batch > 1
        assert loss_sum >= loss_mean

    def test_swap_option(self):
        """Test swap option for harder negatives."""
        loss_fn = TripletLoss(margin=0.2, swap=True)

        anchor = torch.randn(8, 128, requires_grad=True)
        positive = torch.randn(8, 128, requires_grad=True)
        negative = torch.randn(8, 128, requires_grad=True)

        loss = loss_fn(anchor, positive, negative)

        # Should compute loss with harder negative (swap if needed)
        assert loss >= 0
        assert loss.requires_grad


class TestCompositeLoss:
    """Test composite loss behavior."""

    def test_single_component(self):
        """Composite with single component should match that component."""
        pred = torch.randn(4, 3, 32, 32, requires_grad=True)
        target = torch.randn(4, 3, 32, 32)

        # Composite with only reconstruction
        composite = CompositeLoss(reconstruction_weight=2.0)
        composite_loss = composite(reconstruction=(pred, target))

        # Direct reconstruction loss
        recon_loss = ReconstructionLoss()(pred, target)

        # Should be scaled by weight
        assert torch.allclose(composite_loss, 2.0 * recon_loss, atol=1e-6)

    def test_multiple_components(self):
        """Multiple components should be weighted and summed."""
        composite = CompositeLoss(
            reconstruction_weight=1.0,
            contrastive_weight=0.5,
            triplet_weight=0.0,  # Disabled
        )

        # Prepare inputs
        pred = torch.randn(4, 3, 32, 32, requires_grad=True)
        target = torch.randn(4, 3, 32, 32)
        embeddings_a = torch.randn(8, 256, requires_grad=True)
        embeddings_b = torch.randn(8, 256, requires_grad=True)

        loss = composite(
            reconstruction=(pred, target), contrastive=(embeddings_a, embeddings_b)
        )

        assert loss > 0
        assert loss.requires_grad

    def test_component_dict_return(self):
        """Test returning component breakdown."""
        composite = CompositeLoss(reconstruction_weight=1.0, contrastive_weight=0.5)

        pred = torch.randn(4, 3, 32, 32, requires_grad=True)
        target = torch.randn(4, 3, 32, 32)
        embeddings_a = torch.randn(8, 256, requires_grad=True)
        embeddings_b = torch.randn(8, 256, requires_grad=True)

        result = composite(
            reconstruction=(pred, target),
            contrastive=(embeddings_a, embeddings_b),
            return_dict=True,
        )

        assert isinstance(result, dict)
        assert "total" in result
        assert "reconstruction" in result
        assert "contrastive" in result

        # Total should be weighted sum
        expected_total = result["reconstruction"] * 1.0 + result["contrastive"] * 0.5
        assert torch.allclose(result["total"], expected_total, atol=1e-6)

    def test_disabled_components_not_computed(self):
        """Components with zero weight should not be computed."""
        composite = CompositeLoss(
            reconstruction_weight=1.0,
            contrastive_weight=0.0,  # Disabled
            triplet_weight=0.0,  # Disabled
        )

        pred = torch.randn(4, 3, 32, 32, requires_grad=True)
        target = torch.randn(4, 3, 32, 32)

        # Only provide reconstruction
        loss = composite(reconstruction=(pred, target))

        assert loss > 0
        assert loss.requires_grad

    def test_none_inputs_skipped(self):
        """None inputs should be skipped gracefully."""
        composite = CompositeLoss(reconstruction_weight=1.0, contrastive_weight=1.0)

        pred = torch.randn(4, 3, 32, 32, requires_grad=True)
        target = torch.randn(4, 3, 32, 32)

        # Only provide reconstruction, contrastive is None
        loss = composite(reconstruction=(pred, target), contrastive=None)

        assert loss > 0
        assert loss.requires_grad

    def test_weight_update(self):
        """Test dynamic weight updates."""
        composite = CompositeLoss(reconstruction_weight=1.0)

        pred = torch.randn(4, 3, 32, 32, requires_grad=True)
        target = torch.randn(4, 3, 32, 32)

        loss_before = composite(reconstruction=(pred, target))

        # Update weight
        composite.update_weights(reconstruction=2.0)

        loss_after = composite(reconstruction=(pred, target))

        # Should be doubled
        assert torch.allclose(loss_after, 2.0 * loss_before, atol=1e-6)

    def test_gradient_flow_through_all_components(self):
        """Gradients should flow through all active components."""
        composite = CompositeLoss(reconstruction_weight=1.0, contrastive_weight=0.5)

        pred = torch.randn(4, 3, 32, 32, requires_grad=True)
        target = torch.randn(4, 3, 32, 32)
        embeddings_a = torch.randn(8, 256, requires_grad=True)
        embeddings_b = torch.randn(8, 256, requires_grad=True)

        loss = composite(
            reconstruction=(pred, target), contrastive=(embeddings_a, embeddings_b)
        )

        loss.backward()

        assert pred.grad is not None
        assert embeddings_a.grad is not None
        assert embeddings_b.grad is not None

    def test_configuration_passing(self):
        """Test that configurations are passed to components."""
        composite = CompositeLoss(
            reconstruction_weight=1.0,
            reconstruction_config={"loss_type": "l1", "reduction": "sum"},
            contrastive_weight=1.0,
            contrastive_config={"temperature": 0.5, "normalize": False},
        )

        # Verify configs were applied
        assert composite.reconstruction_loss is not None
        assert composite.reconstruction_loss.loss_type == "l1"
        assert composite.reconstruction_loss.reduction == "sum"
        assert composite.contrastive_loss is not None
        assert composite.contrastive_loss.temperature == 0.5
        assert composite.contrastive_loss.normalize is False


class TestLossNumericalStability:
    """Test numerical stability of loss functions."""

    def test_reconstruction_handles_extreme_values(self):
        """Reconstruction loss should handle extreme values."""
        loss_fn = ReconstructionLoss()

        pred = torch.tensor([[1e6, -1e6]], requires_grad=True)
        target = torch.tensor([[0.0, 0.0]])

        loss = loss_fn(pred, target)

        assert torch.isfinite(loss)
        assert loss.requires_grad

    def test_contrastive_handles_small_temperature(self):
        """Contrastive loss should handle very small temperature."""
        loss_fn = ContrastiveLoss(temperature=1e-8)

        embeddings_a = torch.randn(4, 32, requires_grad=True)
        embeddings_b = torch.randn(4, 32, requires_grad=True)

        loss = loss_fn(embeddings_a, embeddings_b)

        assert torch.isfinite(loss)
        assert loss.requires_grad

    def test_triplet_handles_identical_embeddings(self):
        """Triplet loss should handle identical embeddings."""
        loss_fn = TripletLoss()

        embeddings = torch.randn(4, 128, requires_grad=True)

        # All identical
        loss = loss_fn(embeddings, embeddings, embeddings)

        assert torch.isfinite(loss)
        assert loss >= 0

    def test_composite_prevents_nan_propagation(self):
        """Composite loss should prevent NaN propagation."""
        composite = CompositeLoss(reconstruction_weight=1.0, contrastive_weight=1.0)

        # Large values that might cause issues
        pred = torch.randn(2, 3, 8, 8, requires_grad=True) * 1e3
        target = torch.randn(2, 3, 8, 8)
        embeddings_a = torch.randn(4, 16, requires_grad=True) * 1e-3
        embeddings_b = torch.randn(4, 16, requires_grad=True) * 1e3

        loss = composite(
            reconstruction=(pred, target), contrastive=(embeddings_a, embeddings_b)
        )

        assert torch.isfinite(loss)
        assert loss.requires_grad


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
class TestLossDeviceCompatibility:
    """Test losses work on different devices."""

    def test_reconstruction_device(self, device):
        """Reconstruction loss should work on device."""
        loss_fn = ReconstructionLoss().to(device)
        pred = torch.randn(2, 3, 16, 16, requires_grad=True, device=device)
        target = torch.randn(2, 3, 16, 16, device=device)

        loss = loss_fn(pred, target)

        assert loss.device.type == device
        assert loss.requires_grad

    def test_contrastive_device(self, device):
        """Contrastive loss should work on device."""
        loss_fn = ContrastiveLoss().to(device)
        embeddings_a = torch.randn(8, 64, requires_grad=True, device=device)
        embeddings_b = torch.randn(8, 64, requires_grad=True, device=device)

        loss = loss_fn(embeddings_a, embeddings_b)

        assert loss.device.type == device
        assert loss.requires_grad

    def test_composite_device(self, device):
        """Composite loss should maintain device consistency."""
        loss_fn = CompositeLoss(reconstruction_weight=1.0, contrastive_weight=1.0).to(
            device
        )

        pred = torch.randn(2, 3, 8, 8, requires_grad=True, device=device)
        target = torch.randn(2, 3, 8, 8, device=device)
        embeddings_a = torch.randn(4, 32, requires_grad=True, device=device)
        embeddings_b = torch.randn(4, 32, requires_grad=True, device=device)

        loss = loss_fn(
            reconstruction=(pred, target), contrastive=(embeddings_a, embeddings_b)
        )

        assert loss.device.type == device
        assert loss.requires_grad


class TestLossIntegration:
    """Integration tests for loss functions in training scenarios."""

    def test_training_loop_workflow(self):
        """Test losses in a training loop."""
        loss_fn = CompositeLoss(reconstruction_weight=1.0, contrastive_weight=0.1)

        # Simulate optimizer
        pred = torch.randn(4, 3, 32, 32, requires_grad=True)
        embeddings_a = torch.randn(4, 256, requires_grad=True)
        optimizer = torch.optim.SGD([pred, embeddings_a], lr=0.01)

        initial_pred = pred.clone()

        # Training step
        for _ in range(3):
            optimizer.zero_grad()

            target = torch.randn(4, 3, 32, 32)
            embeddings_b = torch.randn(4, 256)

            loss = loss_fn(
                reconstruction=(pred, target), contrastive=(embeddings_a, embeddings_b)
            )

            loss.backward()
            optimizer.step()

        # Parameters should have changed
        assert not torch.allclose(pred, initial_pred)

    def test_loss_scheduling(self):
        """Test dynamic loss weight scheduling."""
        loss_fn = CompositeLoss(reconstruction_weight=1.0, contrastive_weight=0.1)

        pred = torch.randn(4, 3, 32, 32, requires_grad=True)
        target = torch.randn(4, 3, 32, 32)
        embeddings_a = torch.randn(8, 256, requires_grad=True)
        embeddings_b = torch.randn(8, 256, requires_grad=True)

        losses = []

        # Simulate weight scheduling
        for epoch in range(3):
            # Increase contrastive weight over time
            loss_fn.update_weights(contrastive=0.1 * (epoch + 1))

            loss = loss_fn(
                reconstruction=(pred, target), contrastive=(embeddings_a, embeddings_b)
            )
            losses.append(loss.item())

        # Losses should change as weights change
        assert losses[0] != losses[1] != losses[2]

    def test_mixed_precision_compatibility(self):
        """Test losses work with automatic mixed precision."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for mixed precision")

        loss_fn = ReconstructionLoss()

        pred = torch.randn(2, 3, 32, 32, requires_grad=True, device="cuda")
        target = torch.randn(2, 3, 32, 32, device="cuda")

        # Use autocast for mixed precision
        with torch.cuda.amp.autocast():
            loss = loss_fn(pred, target)

        assert loss.dtype in (torch.float16, torch.bfloat16)
        assert torch.isfinite(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

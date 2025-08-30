"""Integration tests for training workflows."""

import pytest
import torch
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset

from associative.nn.modules import (
    EnergyTransformer,
    EnergyTransformerConfig,
    GraphEnergyTransformer,
)


class TestVisionTraining:
    """Test vision model training scenarios."""

    @pytest.fixture
    def model(self):
        """Create a small model for testing."""
        config = EnergyTransformerConfig(
            patch_size=4,
            num_patches=16,  # 16x16 image
            embed_dim=32,
            num_heads=2,
            qk_dim=16,
            mlp_ratio=2.0,
            num_layers=1,
            num_time_steps=2,
        )
        return EnergyTransformer(config)

    @pytest.fixture
    def dataset(self):
        """Create synthetic dataset."""
        # Small 16x16 images
        images = torch.randn(10, 3, 16, 16)
        dataset = TensorDataset(images)
        return DataLoader(dataset, batch_size=2)

    def test_mae_training_step(self, model, dataset):
        """Test masked autoencoding training step."""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for batch in dataset:
            images = batch[0]

            # Create random mask
            num_patches = model.config.num_patches
            num_masked = num_patches // 2

            batch_size = images.shape[0]
            mask_idx = torch.stack(
                [torch.randperm(num_patches)[:num_masked] for _ in range(batch_size)]
            )
            batch_idx = torch.arange(batch_size).unsqueeze(1).expand(-1, num_masked)

            # Forward pass
            optimizer.zero_grad()
            output = model(images, mask=(batch_idx, mask_idx))

            # Compute reconstruction loss on masked patches
            target = model.patch_embed.to_patches(images)
            # Target should be patch pixels, not encoded
            target_pixels = target.reshape(batch_size, num_patches, -1)

            # Simple MSE loss on all patches
            loss = functional.mse_loss(output, target_pixels)

            # Backward
            loss.backward()
            optimizer.step()

            # Check gradients flowed - some params may not get gradients
            params_with_grad = sum(
                1 for p in model.parameters() if p.requires_grad and p.grad is not None
            )
            assert (
                params_with_grad > 0
            )  # At least some parameters should have gradients

            break  # Just test one batch

    def test_energy_minimization_training(self, model):
        """Test training with energy minimization objective."""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Single batch
        images = torch.randn(2, 3, 16, 16)

        # Forward with energy tracking
        optimizer.zero_grad()
        output, energies = model(images, return_energy=True)

        # Energy-based loss (encourage lower final energy)
        energy_loss = energies[-1]  # Final energy

        energy_loss.backward()
        optimizer.step()

        # Verify optimization - some params may not get gradients (e.g., unused normalization)
        params_with_grad = sum(
            1 for p in model.parameters() if p.requires_grad and p.grad is not None
        )
        total_params = sum(1 for p in model.parameters() if p.requires_grad)
        assert params_with_grad > 0  # At least some parameters should have gradients
        assert (
            params_with_grad / total_params > 0.3  # noqa: PLR2004
        )  # At least 30% of parameters should have gradients

    def test_training_with_different_alpha(self, model):
        """Test training with different evolution step sizes."""
        images = torch.randn(2, 3, 16, 16)

        # Try different alpha values
        alphas = [0.1, 0.5, 1.0, 2.0]
        outputs = []

        for alpha in alphas:
            out = model(images, alpha=alpha)
            outputs.append(out)

        # Outputs should be different with different alphas
        for i in range(1, len(outputs)):
            assert not torch.allclose(outputs[0], outputs[i])


class TestGraphTraining:
    """Test graph model training scenarios."""

    @pytest.fixture
    def model(self):
        """Create a small graph model."""
        config = EnergyTransformerConfig(
            input_dim=16,
            embed_dim=32,
            out_dim=3,  # 3-class classification
            num_heads=2,
            qk_dim=16,
            num_layers=1,
            num_time_steps=2,
        )
        return GraphEnergyTransformer(config)

    def test_node_classification_training(self, model):
        """Test node classification training."""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Create synthetic graph
        num_nodes = 20
        node_features = torch.randn(1, num_nodes, 16)
        adjacency = (torch.rand(1, num_nodes, num_nodes, 1) > 0.7).float()  # noqa: PLR2004
        adjacency = adjacency * adjacency.transpose(1, 2)  # Symmetric
        pos_encoding = torch.randn(1, num_nodes + 1, 10)  # +1 for CLS

        # Random labels
        labels = torch.randint(0, 3, (num_nodes,))

        # Training step
        optimizer.zero_grad()
        output = model(node_features, adjacency, pos_encoding)

        # Classification loss on node embeddings
        logits = output["node_embeddings"][0]  # [num_nodes, 3]
        loss = functional.cross_entropy(logits, labels)

        loss.backward()
        optimizer.step()

        # Verify gradients
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad)

    def test_graph_classification_training(self, model):
        """Test graph-level classification training."""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Batch of graphs
        batch_size = 4
        num_nodes = 15

        node_features = torch.randn(batch_size, num_nodes, 16)
        adjacency = torch.ones(batch_size, num_nodes, num_nodes, 1)
        pos_encoding = torch.randn(batch_size, num_nodes + 1, 10)  # +1 for CLS

        # Graph-level labels
        labels = torch.randint(0, 3, (batch_size,))

        # Training step
        optimizer.zero_grad()
        output = model(node_features, adjacency, pos_encoding)

        # Use graph embedding for classification
        graph_embeds = output["graph_embedding"]  # [batch_size, out_dim]
        loss = functional.cross_entropy(graph_embeds, labels)

        loss.backward()
        optimizer.step()

        assert loss.item() > 0

    def test_training_with_node_masking(self, model):
        """Test training with variable number of nodes."""
        model.train()

        # Graph with padding
        max_nodes = 20
        actual_nodes = 15

        node_features = torch.randn(1, max_nodes, 16)
        adjacency = torch.zeros(1, max_nodes, max_nodes, 1)
        adjacency[:, :actual_nodes, :actual_nodes, :] = (
            1.0  # Only actual nodes connected
        )
        pos_encoding = torch.randn(1, max_nodes + 1, 10)  # +1 for CLS

        # Mask for valid nodes
        mask = torch.zeros(1, max_nodes)
        mask[0, :actual_nodes] = 1.0

        # Forward pass
        output = model(node_features, adjacency, pos_encoding, mask=mask)

        # Check masked nodes have zero embeddings
        node_embeds = output["node_embeddings"]
        assert torch.allclose(
            node_embeds[0, actual_nodes:],
            torch.zeros_like(node_embeds[0, actual_nodes:]),
        )


class TestTrainingStability:
    """Test training stability and convergence."""

    def test_gradient_clipping(self):
        """Test gradient clipping for stability."""
        config = EnergyTransformerConfig(
            patch_size=4,
            num_patches=16,
            embed_dim=32,
            num_heads=2,
            qk_dim=16,
            num_layers=2,
            num_time_steps=3,
        )
        model = EnergyTransformer(config)

        # Large input to potentially cause large gradients
        x = torch.randn(2, 3, 16, 16) * 10

        # Forward and backward
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check gradients before clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))

        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Check gradients after clipping
        total_norm_after = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float("inf")
        )

        assert total_norm_after <= 1.0 + 1e-6

    def test_loss_convergence(self):
        """Test that loss decreases over iterations."""
        config = EnergyTransformerConfig(
            patch_size=4,
            num_patches=16,
            embed_dim=32,
            num_heads=2,
            qk_dim=16,
            num_layers=1,
            num_time_steps=2,
        )
        model = EnergyTransformer(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        # Fixed input for overfitting test
        x = torch.randn(1, 3, 16, 16)
        target = torch.randn(1, 16, 48)  # Target output

        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            output = model(x)
            loss = functional.mse_loss(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease
        assert losses[-1] < losses[0]

        # Check monotonic decrease (with some tolerance)
        decreasing_steps = sum(
            losses[i] > losses[i + 1] for i in range(len(losses) - 1)
        )
        assert decreasing_steps >= len(losses) // 2  # At least half should decrease

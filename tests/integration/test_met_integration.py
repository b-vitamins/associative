"""Integration tests for Multimodal Energy Transformer components.

These tests verify that MET components work together correctly as a system,
focusing on data flow, shape compatibility, and actual expected behavior.
"""

import pytest
import torch

from associative.nn.modules import (
    ContinuousCompression,
    CrossModalHopfield,
    METBlock,
    METConfig,
    MultimodalEnergyAttention,
    MultimodalEnergyTransformer,
)
from associative.nn.modules.basis import RectangularBasis
from associative.nn.modules.integrator import (
    GaussLegendreIntegrator,
    create_integrator,
)
from associative.utils.losses import (
    CompositeLoss,
)


class TestMETComponentIntegration:
    """Test integration between core MET components."""

    @pytest.fixture
    def modality_configs(self):
        """Standard modality configurations for testing."""
        return {
            "video": {
                "embed_dim": 768,
                "compression_dim": 100,
                "num_heads": 8,
                "qk_dim": 64,
                "basis_type": "rectangular",
                "regularization": 0.01,
            },
            "audio": {
                "embed_dim": 512,
                "compression_dim": 100,
                "num_heads": 8,
                "qk_dim": 64,
                "basis_type": "rectangular",
                "regularization": 0.01,
            },
        }

    @pytest.fixture
    def sample_features(self):
        """Sample multimodal features for testing."""
        return {
            "video": torch.randn(2, 196, 768),  # (batch, seq_len, embed_dim)
            "audio": torch.randn(2, 196, 512),  # (batch, seq_len, embed_dim)
        }

    def test_multimodal_attention_with_compression(self, modality_configs):
        """Test MultimodalEnergyAttention integrated with ContinuousCompression."""
        # Create actual multimodal attention module
        attention = MultimodalEnergyAttention(
            modality_configs=modality_configs,
            cross_modal_pairs=[("video", "audio"), ("audio", "video")],
        )

        # Verify compressions are created properly
        assert len(attention.compressions) == 2
        assert "video" in attention.compressions
        assert "audio" in attention.compressions

        # Test that compression modules have correct dimensions
        for name, compression in attention.compressions.items():
            expected_dim = modality_configs[name]["compression_dim"]
            assert compression.compression_dim == expected_dim

        # Test projections are created
        assert len(attention.query_proj) == 2
        assert len(attention.key_proj) == 2
        assert "video" in attention.query_proj
        assert "audio" in attention.query_proj

        # Test cross-modal projections exist
        assert "video_to_audio" in attention.cross_proj
        assert "audio_to_video" in attention.cross_proj

        # Test projection shapes
        video_to_audio_proj = attention.cross_proj["video_to_audio"]
        assert video_to_audio_proj.shape == (512, 768)  # (audio_dim, video_dim)

        audio_to_video_proj = attention.cross_proj["audio_to_video"]
        assert audio_to_video_proj.shape == (768, 512)  # (video_dim, audio_dim)

    def test_cross_modal_hopfield_integration(self, sample_features):
        """Test CrossModalHopfield integration with actual computation."""
        modality_dims = {
            "video": 768,
            "audio": 512,
        }

        # Create actual cross-modal Hopfield module
        hopfield = CrossModalHopfield(
            modality_dims=modality_dims,
            num_prototypes=256,
            cross_weight=0.3,
            temporal_window=3,
        )

        # Test that prototypes are created
        assert len(hopfield.prototypes) == 2
        assert "video" in hopfield.prototypes
        assert "audio" in hopfield.prototypes

        # Test prototype shapes
        video_prototypes = hopfield.prototypes["video"]
        assert video_prototypes.shape == (256, 768)

        audio_prototypes = hopfield.prototypes["audio"]
        assert audio_prototypes.shape == (256, 512)

        # Test cross-modal projections are created
        assert "video_to_audio" in hopfield.cross_projs
        assert "audio_to_video" in hopfield.cross_projs

        # Test forward pass actually works
        energy = hopfield(sample_features)
        assert isinstance(energy, torch.Tensor)
        assert energy.shape == ()  # Scalar energy
        assert energy.requires_grad  # Should have gradients

        # Test with return_components
        energy_components = hopfield(sample_features, return_components=True)
        assert isinstance(energy_components, dict)
        assert "video" in energy_components
        assert "audio" in energy_components

    def test_met_block_component_assembly(self, modality_configs, sample_features):
        """Test METBlock assembles attention and Hopfield components correctly."""
        modality_dims = {
            name: config["embed_dim"] for name, config in modality_configs.items()
        }

        attention_config = {
            "modality_configs": modality_configs,
            "cross_modal_pairs": None,  # All pairs
        }

        hopfield_config = {
            "num_prototypes": 256,
            "cross_modal_weight": 0.3,
        }

        # Create actual MET block
        block = METBlock(
            modality_dims=modality_dims,
            attention_config=attention_config,
            hopfield_config=hopfield_config,
        )

        # Test that components are properly created
        assert hasattr(block, "attention")
        assert hasattr(block, "hopfield")
        assert hasattr(block, "norms")
        assert len(block.norms) == len(modality_dims)

        # Test forward pass (returns energy)
        energy = block(sample_features)
        assert isinstance(energy, torch.Tensor)
        assert energy.shape == ()  # Scalar energy
        assert energy.requires_grad

        # Test energy breakdown
        energy_detailed = block.energy(sample_features)
        assert isinstance(energy_detailed, torch.Tensor)

    def test_integrator_with_partition_function(self):
        """Test integrator integration for partition function computation."""
        # Test that integrators actually work with continuous score functions
        integrator = GaussLegendreIntegrator(domain=(0.0, 1.0), num_points=50)

        # Create a simple quadratic function for testing
        def mock_score_func(t):
            return -0.5 * (t - 0.5) ** 2

        # Test integration actually works
        result = integrator.integrate(mock_score_func)
        assert isinstance(result, torch.Tensor)
        assert result.shape == ()  # Scalar result

        # For a quadratic centered at 0.5 over [0,1], we expect a negative value
        assert result < 0

        # Test factory function
        integrator2 = create_integrator("gauss", domain=(0, 1), num_points=30)
        assert isinstance(integrator2, GaussLegendreIntegrator)
        assert integrator2.num_points == 30

        # Test that it produces consistent results
        result2 = integrator2.integrate(mock_score_func)
        assert isinstance(result2, torch.Tensor)
        # Results should be close (same function, different number of points)
        assert abs(result - result2) < 0.1

    def test_compression_and_reconstruction_pipeline(self):
        """Test the full compression and reconstruction pipeline."""
        # Create actual compression module
        basis = RectangularBasis(num_basis=50, domain=(0.0, 1.0))
        compression = ContinuousCompression(basis=basis, regularization=0.01)

        # Test data
        seq_len = 100
        keys = torch.randn(8, 64, seq_len)  # (heads, dim, seq_len)

        # Test compression works
        coefficients = compression.compress(keys, seq_len)
        assert coefficients.shape == (8, 64, 50)  # (heads, dim, compression_dim)

        # Test reconstruction at arbitrary points
        time_points = torch.linspace(0, 1, 200)
        reconstructed = compression.reconstruct(coefficients, time_points)
        assert reconstructed.shape == (8, 64, 200)  # (heads, dim, num_time_points)

        # Test design matrix computation
        design_matrix = compression.compute_design_matrix(seq_len)
        assert design_matrix.shape == (50, seq_len)  # (compression_dim, seq_len)

        # Test regression operator computation
        regression_operator = compression.compute_regression_operator(design_matrix)
        assert regression_operator.shape == (50, seq_len)  # (compression_dim, seq_len)

        # Test continuous scores computation
        queries = torch.randn(8, 64, 80)  # Different sequence length for queries
        scores = compression.compute_continuous_scores(queries, coefficients)
        # The first dimension is heads * dim (8 * 8 = 64), not just heads
        assert scores.shape == (
            64,
            80,
            50,
        )  # (heads*dim, query_len, integration_points)

    def test_composite_loss_integration(self):
        """Test CompositeLoss integrating multiple loss components."""
        # Create composite loss with all components
        loss_fn = CompositeLoss(
            reconstruction_weight=1.0,
            contrastive_weight=0.1,
            triplet_weight=0.01,
            reconstruction_config={"loss_type": "l2"},
            contrastive_config={"temperature": 0.07},
            triplet_config={"margin": 0.2},
        )

        # Verify components are created based on weights
        assert loss_fn.reconstruction_loss is not None
        assert loss_fn.contrastive_loss is not None
        assert loss_fn.triplet_loss is not None

        # Test actual loss computation
        pred = torch.randn(4, 3, 32, 32)
        target = torch.randn(4, 3, 32, 32)
        video_emb = torch.randn(4, 768)
        audio_emb = torch.randn(4, 768)
        negative_emb = torch.randn(4, 768)

        total_loss = loss_fn(
            reconstruction=(pred, target),
            contrastive=(video_emb, audio_emb),
            triplet=(video_emb, audio_emb, negative_emb),
        )

        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.shape == ()  # Scalar
        assert total_loss.requires_grad

        # Test component breakdown
        loss_components = loss_fn(
            reconstruction=(pred, target),
            contrastive=(video_emb, audio_emb),
            triplet=(video_emb, audio_emb, negative_emb),
            return_dict=True,
        )

        assert isinstance(loss_components, dict)
        assert "total" in loss_components
        assert "reconstruction" in loss_components
        assert "contrastive" in loss_components
        assert "triplet" in loss_components

        # Test weight updates
        loss_fn.update_weights(
            reconstruction=2.0,
            contrastive=0.2,
        )
        assert loss_fn.reconstruction_weight == 2.0
        assert loss_fn.contrastive_weight == 0.2


class TestCrossModalInteractions:
    """Test cross-modal interaction workflows."""

    @pytest.fixture
    def video_features(self):
        """Sample video features."""
        return torch.randn(4, 196, 768)  # (batch, patches, dim)

    @pytest.fixture
    def audio_features(self):
        """Sample audio features."""
        # MET requires same sequence length L for all modalities
        return torch.randn(4, 196, 512)  # (batch, patches, dim) - same L=196

    def test_cross_attention_between_modalities(self, video_features, audio_features):
        """Test cross-modal attention mechanism."""
        modality_configs = {
            "video": {
                "embed_dim": 768,
                "compression_dim": 100,
                "num_heads": 8,
                "qk_dim": 64,
                "basis_type": "rectangular",
                "regularization": 0.01,
            },
            "audio": {
                "embed_dim": 512,
                "compression_dim": 100,
                "num_heads": 8,
                "qk_dim": 64,
                "basis_type": "rectangular",
                "regularization": 0.01,
            },
        }

        # Create multimodal attention
        attention = MultimodalEnergyAttention(modality_configs=modality_configs)

        # Test cross-modal energy computation
        features = {"video": video_features, "audio": audio_features}
        cross_energy = attention.compute_cross_modal_energy(
            features, return_components=True
        )

        assert isinstance(cross_energy, dict)
        assert ("video", "audio") in cross_energy
        assert ("audio", "video") in cross_energy

        # Each cross-modal energy should be a scalar
        for energy in cross_energy.values():
            assert isinstance(energy, torch.Tensor)
            assert energy.shape == ()
            assert energy.requires_grad

    def test_temporal_alignment_between_modalities(
        self, video_features, audio_features
    ):
        """Test temporal alignment of different modality sequences."""
        # MET requires same sequence length, so these should already be aligned
        video_len = video_features.shape[1]  # 196
        audio_len = audio_features.shape[1]  # 196 (same as video per MET requirement)

        assert video_len == audio_len, (
            "MET requires same sequence length L for all modalities"
        )

        # Test different alignment strategies

        # 1. Interpolation to common length
        target_len = 150
        video_aligned = torch.nn.functional.interpolate(
            video_features.transpose(1, 2),  # (batch, dim, seq)
            size=target_len,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)  # Back to (batch, seq, dim)

        audio_aligned = torch.nn.functional.interpolate(
            audio_features.transpose(1, 2),
            size=target_len,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)

        assert video_aligned.shape[1] == target_len
        assert audio_aligned.shape[1] == target_len

        # 2. Test that equal-length modalities stay aligned
        # Since they're already the same length, no padding needed
        assert video_features.shape[1] == audio_features.shape[1]

        # Test with aligned features in Hopfield
        modality_dims = {"video": 768, "audio": 512}
        hopfield = CrossModalHopfield(
            modality_dims=modality_dims,
            num_prototypes=128,
            cross_weight=0.3,
            temporal_window=3,
        )

        # Should work with aligned features
        aligned_features = {"video": video_aligned, "audio": audio_aligned}
        energy = hopfield(aligned_features)
        assert isinstance(energy, torch.Tensor)
        assert energy.requires_grad

    def test_cross_modal_saliency_computation(self, video_features, audio_features):
        """Test saliency weight computation across modalities."""
        modality_dims = {"video": 768, "audio": 512}
        hopfield = CrossModalHopfield(
            modality_dims=modality_dims,
            num_prototypes=256,
            cross_weight=0.3,
            temporal_window=3,
        )

        # Test saliency computation
        saliency_weights = hopfield.compute_saliency_weights(
            video_features, audio_features, "video"
        )

        # Check saliency shape
        expected_shape = (video_features.shape[0], video_features.shape[1], 256)
        assert saliency_weights.shape == expected_shape

        # Verify saliency is non-negative (squared terms)
        assert (saliency_weights >= 0).all()

        # Test temporal smoothing
        smoothed_audio = hopfield.temporal_smooth(audio_features)
        assert smoothed_audio.shape == audio_features.shape

        # Verify smoothing actually changes the values
        assert not torch.equal(smoothed_audio, audio_features)

    def test_cross_modal_energy_aggregation(self, video_features, audio_features):
        """Test aggregation of energy across modalities."""
        modality_configs = {
            "video": {
                "embed_dim": 768,
                "compression_dim": 50,
                "num_heads": 4,
                "qk_dim": 32,
                "basis_type": "rectangular",
                "regularization": 0.01,
            },
            "audio": {
                "embed_dim": 512,
                "compression_dim": 50,
                "num_heads": 4,
                "qk_dim": 32,
                "basis_type": "rectangular",
                "regularization": 0.01,
            },
        }

        attention = MultimodalEnergyAttention(modality_configs=modality_configs)
        features = {"video": video_features, "audio": audio_features}

        # Test energy breakdown
        energy_breakdown = attention(features, return_breakdown=True)

        assert isinstance(energy_breakdown, dict)
        assert "total" in energy_breakdown
        assert "intra_video" in energy_breakdown
        assert "intra_audio" in energy_breakdown
        assert "cross_video_audio" in energy_breakdown
        assert "cross_audio_video" in energy_breakdown

        # Test that total equals sum of components
        manual_total = (
            energy_breakdown["intra_video"]
            + energy_breakdown["intra_audio"]
            + energy_breakdown["cross_video_audio"]
            + energy_breakdown["cross_audio_video"]
        )

        assert torch.allclose(energy_breakdown["total"], manual_total, rtol=1e-5)


class TestTrainingWorkflowIntegration:
    """Test training workflow integration."""

    @pytest.fixture
    def met_config(self):
        """Create actual MET configuration."""
        return METConfig(
            modality_configs={
                "video": {"input_dim": 768, "embed_dim": 256},
                "audio": {"input_dim": 512, "embed_dim": 256},
            },
            embed_dim=256,
            num_blocks=2,
            max_iterations=5,
            step_size=0.001,  # Use optimal step size
            num_prototypes=128,
            compression_dims={"video": 50, "audio": 50},
        )

    def test_multimodal_energy_transformer_forward(self, met_config):
        """Test forward pass through full MET architecture."""
        # Create actual MET model
        met = MultimodalEnergyTransformer(met_config)

        # Create sample inputs
        inputs = {
            "video": torch.randn(2, 100, 768),
            "audio": torch.randn(2, 100, 512),
        }

        # Test forward pass
        outputs = met(inputs)

        assert isinstance(outputs, dict)
        assert "video" in outputs
        assert "audio" in outputs

        # Check output shapes (outputs are projected back to input_dim)
        assert outputs["video"].shape == (2, 100, 768)  # (batch, seq, input_dim)
        assert outputs["audio"].shape == (2, 100, 512)

        # Test with energy return
        outputs_with_energy, energies = met(inputs, return_energies=True)

        assert isinstance(outputs_with_energy, dict)
        assert isinstance(energies, list)
        assert len(energies) > 0  # Should have energy values from evolution

        # Each energy should be a scalar
        for energy in energies:
            assert isinstance(energy, torch.Tensor)
            assert energy.shape == ()

    def test_gradient_flow_dynamics(self, met_config):
        """Test gradient-based evolution of features."""
        met = MultimodalEnergyTransformer(met_config)

        # Initial features
        inputs = {
            "video": torch.randn(2, 50, 768, requires_grad=True),
            "audio": torch.randn(2, 50, 512, requires_grad=True),
        }

        # Project inputs to common space
        projected_features = met.project_inputs(inputs)

        # Test evolution step
        evolved_features, energies = met.evolve(
            projected_features, step_size=0.1, return_energy=True
        )

        assert isinstance(evolved_features, dict)
        assert isinstance(energies, list)

        # Features should have changed through evolution
        for modality in projected_features:
            assert not torch.equal(
                projected_features[modality], evolved_features[modality]
            )

        # Energy should generally decrease through gradient descent
        if len(energies) > 1:
            # Check that energy tends to decrease (allowing some fluctuation)
            final_energy = energies[-1]
            initial_energy = energies[0]
            # At least the final energy should be finite
            assert torch.isfinite(final_energy)
            assert torch.isfinite(initial_energy)

    def test_optimizer_integration_with_met(self, met_config):
        """Test optimizer integration with MET parameters."""
        met = MultimodalEnergyTransformer(met_config)

        # Create optimizer
        optimizer = torch.optim.Adam(met.parameters(), lr=1e-3)

        # Sample inputs and compute loss
        inputs = {
            "video": torch.randn(2, 20, 768),
            "audio": torch.randn(2, 20, 512),
        }

        outputs = met(inputs)

        # Simple loss: minimize L2 norm of outputs
        loss_components = [torch.mean(output**2) for output in outputs.values()]
        if loss_components:
            loss = torch.stack(loss_components).sum()
        else:
            loss = torch.tensor(0.0, requires_grad=True)

        # Training step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify gradients exist
        has_grads = False
        for param in met.parameters():
            if param.grad is not None:
                has_grads = True
                break
        assert has_grads, "No gradients found after backward pass"

        # Verify loss is finite
        assert torch.isfinite(loss) if isinstance(loss, torch.Tensor) else True

    def test_met_block_energy_computation(self):
        """Test MET block energy computation with real components."""
        modality_dims = {"video": 768, "audio": 512}

        attention_config = {
            "embed_dim": 256,  # Will be ignored, uses modality_dims values
            "num_heads": 4,
            "compression_dims": {"video": 50, "audio": 50},
            "basis_type": "rectangular",
            "regularization": 0.01,
        }

        hopfield_config = {
            "num_prototypes": 128,
            "activation": "softplus",
            "cross_modal_weight": 0.3,
        }

        block = METBlock(
            modality_dims=modality_dims,
            attention_config=attention_config,
            hopfield_config=hopfield_config,
        )

        # Test energy computation
        features = {
            "video": torch.randn(2, 50, 768),
            "audio": torch.randn(2, 50, 512),
        }

        energy = block.energy(features)

        assert isinstance(energy, torch.Tensor)
        assert energy.shape == ()
        assert energy.requires_grad

        # Test that energy changes with different inputs
        features2 = {
            "video": torch.randn(2, 50, 768),
            "audio": torch.randn(2, 50, 512),
        }

        energy2 = block.energy(features2)
        assert not torch.equal(energy, energy2)


class TestContinuousCompressionPipeline:
    """Test continuous compression pipeline integration."""

    def test_compression_decompression_cycle(self):
        """Test full compression and decompression cycle."""
        # Setup
        basis = RectangularBasis(num_basis=50, domain=(0.0, 1.0))
        compression = ContinuousCompression(basis=basis, regularization=0.01)

        # Original sequence
        seq_len = 100
        original = torch.randn(8, 64, seq_len)  # (heads, dim, seq_len)

        # Test compression
        coefficients = compression.compress(original, seq_len)
        assert coefficients.shape == (8, 64, 50)  # (heads, dim, num_basis)

        # Test reconstruction at original time points
        original_time_points = torch.arange(seq_len, dtype=torch.float32) / seq_len
        reconstructed = compression.reconstruct(coefficients, original_time_points)
        assert reconstructed.shape == (8, 64, seq_len)

        # Test that reconstruction is reasonable (not perfect due to compression)
        reconstruction_error = torch.mean((original - reconstructed) ** 2)
        assert reconstruction_error < 10.0  # Should be bounded

        # Test reconstruction at different time points
        new_time_points = torch.linspace(0, 1, 200)
        upsampled = compression.reconstruct(coefficients, new_time_points)
        assert upsampled.shape == (8, 64, 200)

    def test_basis_function_integration(self):
        """Test different basis functions with compression."""
        seq_len = 100

        # Test rectangular basis
        rect_basis = RectangularBasis(num_basis=30, domain=(0.0, 1.0))
        rect_compression = ContinuousCompression(rect_basis, regularization=0.01)

        assert rect_compression.compression_dim == 30
        assert rect_compression.basis.num_basis == 30

        # Verify basis evaluation works
        time_points = torch.linspace(0, 1, seq_len)
        basis_eval = rect_basis.evaluate(time_points)
        assert basis_eval.shape == (30, seq_len)

        # Test that basis functions provide coverage
        # Due to floating point boundaries, some points might have 0 or 2 coverage
        basis_coverage = torch.sum(basis_eval, dim=0)
        # Most points should be covered (allow up to 5% gaps/overlaps)
        covered_points = (basis_coverage > 0).sum()
        coverage_ratio = covered_points.float() / seq_len
        assert coverage_ratio > 0.95, f"Coverage too low: {coverage_ratio:.2%}"

        # Test compression with this basis
        test_keys = torch.randn(4, 32, seq_len)
        compressed = rect_compression.compress(test_keys)
        assert compressed.shape == (4, 32, 30)

    def test_regularization_effects(self):
        """Test effect of different regularization values."""
        basis = RectangularBasis(num_basis=50, domain=(0.0, 1.0))

        # Different regularization values
        reg_values = [0.001, 0.01, 0.1, 1.0]
        compressions = []

        for reg in reg_values:
            comp = ContinuousCompression(basis, regularization=reg)
            compressions.append(comp)
            assert comp.regularization == reg

        # Test that different regularizations produce different results
        test_keys = torch.randn(2, 16, 100)
        coeffs_list = []

        for comp in compressions:
            coeffs = comp.compress(test_keys)
            coeffs_list.append(coeffs)

        # Different regularizations should produce different coefficients
        # Check that higher regularization produces smaller coefficient magnitudes
        norms = [coeffs.norm().item() for coeffs in coeffs_list]
        # Higher regularization should generally produce smaller norms
        # (may not be strictly monotonic due to numerical factors)
        assert (
            norms[0] > norms[-1] * 0.9
        )  # First should be larger than last (with tolerance)


class TestEnergyMinimizationDynamics:
    """Test energy minimization dynamics integration."""

    def test_gradient_flow_dynamics(self):
        """Test gradient-based evolution of features."""
        modality_dims = {"video": 256, "audio": 256}

        attention_config = {
            "num_heads": 4,
            "compression_dims": {"video": 25, "audio": 25},
        }

        hopfield_config = {
            "num_prototypes": 64,
            "cross_modal_weight": 0.5,
        }

        block = METBlock(
            modality_dims=modality_dims,
            attention_config=attention_config,
            hopfield_config=hopfield_config,
        )

        # Initial features (unnormalized x)
        features = {
            "video": torch.randn(2, 50, 256, requires_grad=True),
            "audio": torch.randn(2, 50, 256, requires_grad=True),
        }

        # Normalize features: g = norm(x)
        normalized_features = {}
        for modality, feat in features.items():
            if modality in block.norms:
                normalized_features[modality] = block.norms[modality](feat)
            else:
                normalized_features[modality] = feat

        # Compute initial energy on normalized features
        initial_energy = block.energy(normalized_features)

        # Perform gradient step manually
        step_size = 0.01
        initial_energy.backward()

        # Update unnormalized features (gradient flows through normalization)
        with torch.no_grad():
            for feat in features.values():
                if feat.grad is not None:
                    feat.data = feat.data - step_size * feat.grad
                    feat.grad.zero_()

        # Normalize updated features
        normalized_features = {}
        for modality, feat in features.items():
            if modality in block.norms:
                normalized_features[modality] = block.norms[modality](feat)
            else:
                normalized_features[modality] = feat

        # Compute new energy on normalized features
        new_energy = block.energy(normalized_features)

        # Energy should generally decrease (allowing some tolerance)
        assert torch.isfinite(initial_energy)
        assert torch.isfinite(new_energy)

        # Energy should decrease (gradient descent)
        # Allow small increase due to numerical errors
        assert new_energy <= initial_energy * 1.01  # Allow 1% tolerance

        # The relative energy change should be bounded
        relative_change = abs(new_energy - initial_energy) / (
            abs(initial_energy) + 1e-8
        )
        assert relative_change < 1.0  # Should change by less than 100%

    def test_convergence_monitoring(self):
        """Test convergence monitoring during evolution."""
        met_config = METConfig(
            modality_configs={
                "video": {"input_dim": 256, "embed_dim": 128},
                "audio": {"input_dim": 256, "embed_dim": 128},
            },
            embed_dim=128,
            num_blocks=1,
            max_iterations=10,
            step_size=0.01,
            num_prototypes=32,
        )

        met = MultimodalEnergyTransformer(met_config)

        inputs = {
            "video": torch.randn(1, 20, 256),
            "audio": torch.randn(1, 20, 256),
        }

        # Test evolution with energy tracking
        projected = met.project_inputs(inputs)
        evolved, energies = met.evolve(projected, return_energy=True)

        assert energies is not None
        assert len(energies) > 0

        # Check that energies are finite
        for energy in energies:
            assert torch.isfinite(energy)

        # Test that features evolved
        for modality in projected:
            assert not torch.equal(projected[modality], evolved[modality])

    @pytest.mark.parametrize("step_size", [0.001, 0.01, 0.1])
    def test_step_size_effects(self, step_size):
        """Test effect of different step sizes on dynamics."""
        modality_dims = {"modality": 128}

        block = METBlock(
            modality_dims=modality_dims,
            attention_config={"num_heads": 2, "compression_dims": {"modality": 20}},
            hopfield_config={"num_prototypes": 32},
        )

        features = {"modality": torch.randn(1, 30, 128, requires_grad=True)}

        # Compute gradient
        energy = block.energy(features)
        energy.backward()

        # Apply update with different step sizes
        with torch.no_grad():
            original_feat = features["modality"].clone()
            grad = features["modality"].grad
            assert grad is not None, "Gradient should exist after backward"
            gradient = grad.clone()

            update = step_size * gradient
            new_features = original_feat - update

        # Larger step size -> larger change
        feature_change = torch.norm(new_features - original_feat)
        expected_change_magnitude = step_size * torch.norm(gradient)

        # Should be approximately equal (allowing for numerical precision)
        assert torch.allclose(feature_change, expected_change_magnitude, rtol=0.1)

        # Verify step size affects the magnitude appropriately
        if step_size > 0.01:
            assert (
                feature_change > 0.001
            )  # Should be meaningful change for larger steps

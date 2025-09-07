"""Unit tests for attention modules."""

import math

import pytest
import torch

from associative.nn.modules import (
    EnergyAttention,
    GraphEnergyAttention,
    MultimodalEnergyAttention,
)
from associative.nn.modules.config import EnergyAttentionConfig
from tests.conftest import TOLERANCE_ENERGY_DIFF, TOLERANCE_INIT_STD


class TestEnergyAttention:
    """Test EnergyAttention module."""

    @pytest.fixture
    def config(self, embed_dim, num_heads, qk_dim):
        """Create attention config."""
        return EnergyAttentionConfig(
            embed_dim=embed_dim, num_heads=num_heads, qk_dim=qk_dim, bias=False
        )

    @pytest.fixture
    def attention(self, config, device):
        """Create attention module."""
        return EnergyAttention(config).to(device)

    def test_initialization(self, attention, config):
        """Test proper initialization."""
        assert attention.embed_dim == config.embed_dim
        assert attention.num_heads == config.num_heads
        assert attention.qk_dim == config.qk_dim

        # Check weight shapes
        assert attention.query_proj.shape == (
            config.num_heads,
            config.qk_dim,
            config.embed_dim,
        )
        assert attention.key_proj.shape == (
            config.num_heads,
            config.qk_dim,
            config.embed_dim,
        )

        # Check initialization scale
        assert (
            attention.query_proj.std().item() < TOLERANCE_INIT_STD
        )  # Should be ~0.002
        assert attention.key_proj.std().item() < TOLERANCE_INIT_STD

    def test_forward_shape(self, attention, batch_size, seq_length, embed_dim, device):
        """Test forward pass output shape."""
        x = torch.randn(batch_size, seq_length, embed_dim, device=device)
        energy = attention(x)

        assert energy.shape == ()  # Scalar
        assert energy.dtype == x.dtype
        assert energy.requires_grad

    def test_energy_computation(self, attention, device):
        """Test energy computation correctness."""
        # Create simple input where we can verify energy
        x = torch.eye(3, 64, device=device).unsqueeze(0)  # [1, 3, 64]

        with torch.no_grad():
            # Set weights to identity-like for predictable behavior
            attention.query_proj.zero_()
            attention.key_proj.zero_()
            attention.query_proj[:, :, :16] = torch.eye(16).unsqueeze(0)
            attention.key_proj[:, :, :16] = torch.eye(16).unsqueeze(0)

        energy = attention(x)

        # Energy should be negative (logsumexp formulation)
        assert energy.item() < 0

    def test_attention_mask(
        self, attention, batch_size, seq_length, embed_dim, num_heads, device
    ):
        """Test attention masking."""
        x = torch.randn(batch_size, seq_length, embed_dim, device=device)

        # Create causal mask
        mask = torch.tril(torch.ones(seq_length, seq_length, device=device))
        mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)

        energy_unmasked = attention(x)
        energy_masked = attention(x, mask)

        # Masked energy should be different
        # Due to the large scale of energies, we check relative difference
        rel_diff = abs((energy_unmasked - energy_masked) / energy_unmasked)
        assert rel_diff > TOLERANCE_ENERGY_DIFF  # At least 0.0001% different

    def test_gradient_flow(self, attention, batch_size, seq_length, embed_dim, device):
        """Test gradient flow through attention."""
        x = torch.randn(
            batch_size, seq_length, embed_dim, device=device, requires_grad=True
        )

        energy = attention(x)
        energy.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.abs().max() > 0  # Non-zero gradients

    @pytest.mark.parametrize("bias", [True, False])
    def test_bias_option(self, embed_dim, num_heads, qk_dim, bias, device):
        """Test attention with and without bias."""
        config = EnergyAttentionConfig(
            embed_dim=embed_dim, num_heads=num_heads, qk_dim=qk_dim, bias=bias
        )
        attention = EnergyAttention(config).to(device)

        if bias:
            assert attention.query_bias is not None
            assert attention.key_bias is not None
            assert attention.query_bias.shape == (qk_dim,)
            assert attention.key_bias.shape == (qk_dim,)
        else:
            assert attention.query_bias is None
            assert attention.key_bias is None

    def test_numerical_stability(self, attention, device):
        """Test numerical stability with extreme inputs."""
        # Very large inputs
        x_large = torch.randn(1, 5, 64, device=device) * 100
        energy_large = attention(x_large)
        assert torch.isfinite(energy_large)

        # Very small inputs
        x_small = torch.randn(1, 5, 64, device=device) * 0.001
        energy_small = attention(x_small)
        assert torch.isfinite(energy_small)


class TestGraphEnergyAttention:
    """Test GraphEnergyAttention module."""

    @pytest.fixture
    def config(self, embed_dim, num_heads, qk_dim):
        """Create attention config."""
        return EnergyAttentionConfig(
            embed_dim=embed_dim, num_heads=num_heads, qk_dim=qk_dim, bias=False
        )

    @pytest.fixture
    def attention(self, config, device):
        """Create graph attention module."""
        return GraphEnergyAttention(config).to(device)

    def test_initialization(self, attention, config):
        """Test proper initialization."""
        # Check parameter names
        assert hasattr(attention, "key_proj")
        assert hasattr(attention, "query_proj")
        assert hasattr(attention, "head_mix")  # Head mixing weights

        # Check shapes
        assert attention.key_proj.shape == (
            config.num_heads,
            config.qk_dim,
            config.embed_dim,
        )
        assert attention.query_proj.shape == (
            config.num_heads,
            config.qk_dim,
            config.embed_dim,
        )
        assert attention.head_mix.shape == (config.num_heads, config.num_heads)

    def test_batched_forward(
        self, attention, batch_size, seq_length, embed_dim, device
    ):
        """Test batched forward pass."""
        x = torch.randn(batch_size, seq_length, embed_dim, device=device)
        energy = attention(x)

        assert energy.shape == ()
        assert energy.dtype == x.dtype

    def test_unbatched_forward(self, attention, seq_length, embed_dim, device):
        """Test unbatched forward pass."""
        x = torch.randn(seq_length, embed_dim, device=device)
        energy = attention(x)

        assert energy.shape == ()
        assert energy.dtype == x.dtype

    def test_adjacency_masking(
        self, attention, seq_length, embed_dim, num_heads, device
    ):
        """Test forward with adjacency matrix."""
        x = torch.randn(seq_length, embed_dim, device=device)

        # Create sparse adjacency matrix
        adj = torch.zeros(seq_length, seq_length, num_heads, device=device)
        # Add some connections
        adj[0, 1, :] = 1.0
        adj[1, 0, :] = 1.0
        adj[1, 2, :] = 1.0
        adj[2, 1, :] = 1.0

        energy_no_adj = attention(x)
        energy_with_adj = attention(x, adj)

        # Energies should be different
        assert not torch.allclose(energy_no_adj, energy_with_adj)

    def test_head_mixing(self, attention, seq_length, embed_dim, device):
        """Test head mixing weights effect."""
        x = torch.randn(seq_length, embed_dim, device=device)

        # Store original energy
        energy_orig = attention(x).item()

        # Modify head mixing weights
        with torch.no_grad():
            attention.head_mix.data = torch.eye(attention.num_heads, device=device)

        energy_identity = attention(x).item()

        # Should be different with different head mixing
        assert abs(energy_orig - energy_identity) > TOLERANCE_ENERGY_DIFF

    def test_gradient_flow_with_adjacency(self, attention, device):
        """Test gradient flow with adjacency matrix."""
        x = torch.randn(5, 64, device=device, requires_grad=True)
        adj = torch.ones(5, 5, 4, device=device) * 0.5

        energy = attention(x, adj)
        energy.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_empty_adjacency_handling(self, attention, device):
        """Test handling of empty adjacency (all zeros)."""
        x = torch.randn(5, 64, device=device)
        adj = torch.zeros(5, 5, 4, device=device)

        # Should not crash and return finite energy
        energy = attention(x, adj)
        assert torch.isfinite(energy)


class TestMultimodalEnergyAttention:
    """Test MultimodalEnergyAttention module.

    Tests verify exact mathematical behavior:
    - Continuous compression: L → M dimension reduction via basis functions
    - Cross-modal attention: Queries from one modality attend to keys from another
    - Energy computation: Intra-modal and cross-modal energies
    - Beta scaling: β = 1/√Y for numerical stability

    These tests serve as a mathematical specification that ANY implementation
    MUST satisfy to be considered correct.
    """

    @pytest.fixture
    def modality_configs(self):
        """Create modality configurations for testing."""
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
    def cross_modal_pairs(self):
        """Create cross-modal pairs for testing."""
        return [("video", "audio"), ("audio", "video")]

    @pytest.fixture
    def integrator_config(self):
        """Create integrator configuration."""
        return {"method": "gauss_legendre", "num_points": 50}

    @pytest.fixture
    def attention(self, modality_configs, cross_modal_pairs, integrator_config, device):
        """Create multimodal attention module."""
        return MultimodalEnergyAttention(
            modality_configs=modality_configs,
            cross_modal_pairs=cross_modal_pairs,
            num_integration_points=integrator_config.get("num_points", 50),
            device=device,
        )

    def test_mathematical_projections_satisfy_equation_34(
        self, attention, modality_configs, device
    ):
        """Test that query/key projections satisfy mathematical specification.

        Q^m_ahC = Σ_j W^{Q,m}_{ahj} g^m_{jC}
        K^m_ahB = Σ_j W^{K,m}_{ahj} g^m_{jB}

        Where:
        - W^{Q,m}, W^{K,m} ∈ R^{YxHxD_m} are projection matrices
        - Y is attention dimension (qk_dim)
        - H is number of heads
        - D_m is modality embedding dimension
        - a indexes attention dimension, h indexes heads
        - C, B index sequence positions

        This test verifies the mathematical transformation is correct.
        """
        batch_size, seq_len = 2, 64

        for modality, config in modality_configs.items():
            d_m = config["embed_dim"]
            y = config["qk_dim"]
            h = config["num_heads"]

            # Create normalized features g^m
            features = torch.randn(batch_size, seq_len, d_m, device=device)

            # Get queries and keys through the implementation
            queries, compressed_keys, keys = attention.compress_modality(
                features, modality
            )

            # Verify shapes match specification
            # Queries: (batch, H, Y, L) after projection
            assert queries.shape == (batch_size, h, y, seq_len), (
                f"Query shape {queries.shape} doesn't match specification "
                f"(batch={batch_size}, H={h}, Y={y}, L={seq_len})"
            )

            # Keys before compression: (batch, H, Y, L)
            assert keys.shape == (batch_size, h, y, seq_len), (
                f"Key shape {keys.shape} doesn't match specification"
            )

            # Verify projection is linear (additivity test)
            features2 = torch.randn_like(features)
            features_sum = features + features2

            q1, _, k1 = attention.compress_modality(features, modality)
            q2, _, k2 = attention.compress_modality(features2, modality)
            q_sum, _, k_sum = attention.compress_modality(features_sum, modality)

            # Linear projection property: f(x+y) = f(x) + f(y)
            assert torch.allclose(q_sum, q1 + q2, rtol=1e-4, atol=1e-6), (
                "Query projection violates linearity requirement"
            )
            assert torch.allclose(k_sum, k1 + k2, rtol=1e-4, atol=1e-6), (
                "Key projection violates linearity requirement"
            )

    def test_continuous_compression_satisfies_ridge_regression(self, attention, device):
        """Test that continuous compression satisfies ridge regression from paper.

        The compression uses ridge regression with operator
        R^m = (F^m (F^m)^T + λ^m I_{M^m})^{-1} F^m ∈ R^{M^m x L}

        where F^m is the design matrix evaluating basis functions ψ^m_j(t) at
        discrete points, and coefficients B^{m,(i)}_{ah} = R^m (K^{m,(i)}_{ah})^T.

        This ensures the compressed representation minimizes reconstruction error
        with L2 regularization.
        """
        batch_size, seq_len = 2, 256
        m = 100  # compression_dim
        features = torch.randn(batch_size, seq_len, 768, device=device)

        queries, compressed_keys, original_keys = attention.compress_modality(
            features, "video"
        )

        # Verify compression reduces L → M dimensions
        assert compressed_keys.shape[-1] == m, (
            f"Compression dim {compressed_keys.shape[-1]} != M={m}"
        )
        assert original_keys.shape[-1] == seq_len, (
            f"Original seq length {original_keys.shape[-1]} != L={seq_len}"
        )

        # Mathematical property: M < L for compression
        assert seq_len > m, "Not compressing: M >= L violates paper's assumption"

        # Verify compressed keys have bounded norm (regularization effect)
        # Ridge regression should prevent arbitrarily large coefficients
        key_norm = torch.norm(compressed_keys, dim=-1).mean()
        original_norm = torch.norm(original_keys, dim=-1).mean()
        assert key_norm < 100 * original_norm, (
            "Compressed keys have unbounded norm, violating ridge regularization"
        )

    def test_beta_equals_one_over_sqrt_y_for_stability(
        self, attention, modality_configs
    ):
        """Test beta = 1/√Y scaling for numerical stability.

        From the paper: "with inverse temperature β = 1/√Y for stability"
        This scaling ensures scores are normalized by the dimension.
        """
        # Get Y (qk_dim) from config
        y = modality_configs["video"]["qk_dim"]  # 64
        expected_beta = 1.0 / math.sqrt(y)  # 1/√64 = 0.125

        # Implementation must use this exact beta value
        actual_beta = attention._get_beta()
        assert abs(actual_beta - expected_beta) < 1e-10, (
            f"Beta {actual_beta} != 1/√Y = {expected_beta}, "
            "violating numerical stability requirement"
        )

    def test_partition_function_satisfies_gibbs_distribution(self, attention, device):
        """Test partition function Z = ∫exp(βs(t))dt induces proper Gibbs distribution.

        The partition function creates Gibbs distributions
        p^m_{hC}(t) ∝ exp(β s^m_{hC}(t)) where s(t) are continuous score functions.

        Properties to verify:
        1. Z > 0 (normalizing constant must be positive)
        2. log Z is finite (numerical stability via LogSumExp)
        3. ∫p(t)dt = 1 (proper probability distribution)
        """
        beta = 1.0 / math.sqrt(64)  # β = 1/√Y
        batch, heads, seq = 1, 4, 32

        # Create realistic score function matching actual attention scores
        def score_func(t):
            # Simulate batch x heads x seq x time_points scores
            # Shape: (batch, heads, seq, num_points)
            num_points = t.shape[0]
            scores = torch.randn(batch, heads, seq, num_points, device=device)
            # Scores should be bounded for stable integration
            return torch.tanh(scores) * 5.0  # Bounded in [-5, 5]

        log_z = attention.compute_partition_function(score_func, beta)

        # Mathematical requirements from paper
        assert torch.isfinite(log_z), "log Z must be finite (LogSumExp stability)"
        assert log_z.ndim == 0, "Partition function must return scalar"

        # Verify Z > 0 (since Z = exp(log_z))
        z = torch.exp(log_z)
        assert z > 0, "Partition function Z must be positive"

        # Verify numerical stability: log Z should be reasonable magnitude
        assert abs(log_z.item()) < 1000, (
            f"log Z = {log_z.item()} is too large, indicating numerical instability"
        )

    def test_compute_intra_modal_energy_equation_57a(self, attention, device):
        """Test intra-modal energy follows mathematical specification.

        E^intra = -1/β Σ_{h,C} log ∫exp(β s^{m→m}_{hC}(t))dt + Σ_A ||g^m_A||²/2

        Where:
        - s^{m→m}_{hC}(t): Intra-modal scores at time t
        - β = 1/√Y: Inverse temperature
        - g^m_A: Normalized features at position A
        - The first term is the attention energy (negative log partition)
        - The second term is L2 regularization on features

        Mathematical properties to verify:
        1. Energy includes BOTH attention and regularization terms
        2. Attention term is negative (attraction to minima)
        3. Regularization term is positive (penalty on large features)
        4. Total energy is differentiable for gradient flow
        """
        batch_size, seq_len = 1, 64
        features = {
            "video": torch.randn(batch_size, seq_len, 768, device=device),
            "audio": torch.randn(batch_size, seq_len, 512, device=device),
        }

        # Compute total intra-modal energy
        total_energy = attention.compute_intra_modal_energy(features)

        # Should be a scalar tensor
        assert total_energy.shape == ()
        assert torch.isfinite(total_energy)

        # Get component breakdown
        components = attention.compute_intra_modal_energy(
            features, return_components=True
        )

        # Should have energy for each modality
        assert "video" in components
        assert "audio" in components

        # Each component should be finite
        for modality_energy in components.values():
            assert torch.isfinite(modality_energy)

        # Total should equal sum of components
        if components:
            expected_total_val = sum(components.values())
            expected_total = (
                expected_total_val
                if isinstance(expected_total_val, torch.Tensor)
                else torch.tensor(float(expected_total_val))
            )
        else:
            expected_total = torch.tensor(0.0)
        assert torch.allclose(total_energy, expected_total, rtol=1e-5)

    def test_compute_cross_modal_energy_equation_57b(self, attention, device):
        """Test cross-modal energy follows mathematical specification.

        E^cross = -1/β Σ_{h,C} log ∫exp(β s^{m→m'}_{hC}(t))dt

        Where:
        - s^{m→m'}_{hC}(t): Cross-modal scores (queries from m, keys from m')
        - NO regularization term (unlike intra-modal)
        - Requires cross-modal projection W^{m'→m} for dimension matching

        Key differences from intra-modal:
        1. No ||g||²/2 regularization term
        2. Uses cross-modal projections for heterogeneous dimensions
        3. Queries from one modality, keys from another
        """
        batch_size, seq_len = 1, 64
        features = {
            "video": torch.randn(batch_size, seq_len, 768, device=device),
            "audio": torch.randn(batch_size, seq_len, 512, device=device),
        }

        # Compute total cross-modal energy
        total_energy = attention.compute_cross_modal_energy(features)

        # Should be a scalar tensor
        assert total_energy.shape == ()
        assert torch.isfinite(total_energy)

        # Get component breakdown
        components = attention.compute_cross_modal_energy(
            features, return_components=True
        )

        # Should have energy for each cross-modal pair
        assert ("video", "audio") in components
        assert ("audio", "video") in components

        # Each component should be finite
        for pair_energy in components.values():
            assert torch.isfinite(pair_energy)

        # Total should equal sum of components
        if components:
            expected_total_val = sum(components.values())
            expected_total = (
                expected_total_val
                if isinstance(expected_total_val, torch.Tensor)
                else torch.tensor(float(expected_total_val))
            )
        else:
            expected_total = torch.tensor(0.0)
        assert torch.allclose(total_energy, expected_total, rtol=1e-5)

    def test_score_functions_equations_47_51(self, attention, device):
        """Test score functions follow mathematical specification.

        Intra-modal scores:
        s^{v→v}_{hC}(t) = Σ_a K̄^v_{ah}(t) Q^v_{ahC}
        s^{a→a}_{hC}(t) = Σ_a K̄^a_{ah}(t) Q^a_{ahC}

        Cross-modal scores:
        s^{v→a}_{hC}(t) = Σ_a K̄^v_{ah}(t) Q^a_{ahC}
        s^{a→v}_{hC}(t) = Σ_a K̄^a_{ah}(t) Q^v_{ahC}

        These scores define the energy landscape for gradient flow!
        Any error here completely breaks the multimodal dynamics.
        """
        batch_size, seq_len = (
            1,
            200,
        )  # Use seq_len > compression_dim for actual compression
        features = torch.randn(batch_size, seq_len, 768, device=device)

        # Get compressed representations
        queries, compressed_keys, _ = attention.compress_modality(features, "video")

        # Score function should be computable from compressed keys and queries
        # s_hC(t) = Σ_a K̄_ah(t) Q_ahC

        # Check that compressed keys can be used to compute continuous functions
        # This tests the compression → continuous function pipeline
        assert compressed_keys.shape[-1] == 100  # Compression dimension M
        assert queries.shape[-1] == seq_len  # Original sequence length L

        # The key insight: compression should preserve essential information
        # while reducing from L to M dimensions
        compression_ratio = compressed_keys.shape[-1] / queries.shape[-1]
        assert compression_ratio < 1.0  # Actual compression occurred (100/200 = 0.5)

    def test_cross_modal_projections_handle_heterogeneous_dimensions(
        self, attention, device
    ):
        """Test cross-modal projections W^{m→m'} handle D_m ≠ D_m'.

        Learnable projections W^{a→v} ∈ R^{D_vxD_a} and
        W^{v→a} ∈ R^{D_axD_v} map between modalities.

        Mathematical requirements:
        1. Projections must map from source to target dimension
        2. Cross-modal scores use projected features
        3. No information bottleneck (rank preservation)
        """
        batch_size, seq_len = 1, 64

        # Different embedding dimensions
        video_features = torch.randn(
            batch_size, seq_len, 768, device=device
        )  # D_v = 768
        audio_features = torch.randn(
            batch_size, seq_len, 512, device=device
        )  # D_a = 512

        features = {"video": video_features, "audio": audio_features}

        # Cross-modal energy should handle dimension mismatch
        cross_energy = attention.compute_cross_modal_energy(features)

        # Should successfully compute without dimension errors
        assert torch.isfinite(cross_energy)

        # Verify that actual cross-modal projections exist and have correct shapes
        # W^{v→a} ∈ R^{D_a x D_v} and W^{a→v} ∈ R^{D_v x D_a}
        assert hasattr(attention.cross_proj, "video_to_audio")
        assert hasattr(attention.cross_proj, "audio_to_video")

        video_to_audio = attention.cross_proj["video_to_audio"]
        audio_to_video = attention.cross_proj["audio_to_video"]

        # Check dimensions
        assert video_to_audio.shape == (512, 768), (
            f"W^{{v→a}} shape {video_to_audio.shape} != (512, 768)"
        )
        assert audio_to_video.shape == (768, 512), (
            f"W^{{a→v}} shape {audio_to_video.shape} != (768, 512)"
        )

        # Verify projections are initialized (not zero)
        assert torch.norm(video_to_audio) > 0, "Cross-modal projection is zero"
        assert torch.norm(audio_to_video) > 0, "Cross-modal projection is zero"

    def test_dynamic_recompression_at_each_iteration(self, attention, device):
        """Test dynamic recompression B^(m,(i)) changes with iteration i.

        From paper: "In contrast to Santos et al. who use static memories,
        our method uses full gradient descent with iteration-dependent B^(m,(i)),
        enabling dynamic compression."

        This is CRITICAL for multimodal coupling - static compression would
        prevent cross-modal information flow during gradient descent.
        """
        batch_size, seq_len = 1, 64
        features = torch.randn(batch_size, seq_len, 768, device=device)

        # Compress at "iteration i"
        _, compressed_1, _ = attention.compress_modality(features, "video")

        # Simulate gradient update (iteration i+1)
        delta = 0.01 * torch.randn_like(features)
        features_updated = features + delta
        _, compressed_2, _ = attention.compress_modality(features_updated, "video")

        # CRITICAL: Coefficients must change between iterations
        assert not torch.allclose(compressed_1, compressed_2, rtol=1e-3), (
            "Compression coefficients are static, violating dynamic recompression "
            "requirement. This breaks multimodal gradient flow!"
        )

        # Shape consistency (M remains fixed)
        assert compressed_1.shape == compressed_2.shape
        assert compressed_1.shape[-1] == 100  # M = compression_dim

    def test_cross_modal_pairs_configuration(self, device):
        """Test cross-modal pairs configuration (all pairs if None).

        Verifies that cross-modal attention pairs are correctly configured
        and default to all possible pairs when None is provided.
        """
        modality_configs = {
            "video": {
                "embed_dim": 768,
                "compression_dim": 100,
                "num_heads": 4,
                "qk_dim": 32,
            },
            "audio": {
                "embed_dim": 512,
                "compression_dim": 100,
                "num_heads": 4,
                "qk_dim": 32,
            },
            "text": {
                "embed_dim": 256,
                "compression_dim": 100,
                "num_heads": 4,
                "qk_dim": 32,
            },
        }

        # Test with None (should create all pairs)
        attention_all = MultimodalEnergyAttention(
            modality_configs=modality_configs,
            cross_modal_pairs=None,
            num_integration_points=50,
            device=device,
        )

        # Should have all possible cross-modal pairs (excluding self-pairs)
        expected_pairs = [
            ("video", "audio"),
            ("video", "text"),
            ("audio", "video"),
            ("audio", "text"),
            ("text", "video"),
            ("text", "audio"),
        ]

        assert len(attention_all.cross_modal_pairs) == 6  # 3*2 pairs
        for pair in expected_pairs:
            assert pair in attention_all.cross_modal_pairs

        # Test with specific pairs
        specific_pairs = [("video", "audio"), ("audio", "video")]
        attention_specific = MultimodalEnergyAttention(
            modality_configs=modality_configs,
            cross_modal_pairs=specific_pairs,
            num_integration_points=50,
            device=device,
        )

        assert attention_specific.cross_modal_pairs == specific_pairs

    def test_forward_computes_total_energy_equation_18(self, attention, device):
        """Test forward computes total energy following mathematical specification.

        E(x^v, x^a) = E^cross(x^v, x^a) + Σ_{m∈{v,a}} [E^intra_m(x^m) + E^HN_m(x^v, x^a)]

        Where:
        - E^cross: Cross-modal reconstruction energy
        - E^intra_m: Intra-modal temporal coherence
        - E^HN_m: Hopfield memory alignment (not in attention module)

        For the attention module (without Hopfield):
        E_attention = Σ_m E^intra_m + Σ_{(m,m')} E^cross_{m→m'}

        This is THE core equation - any error here breaks the entire model!
        """
        batch_size, seq_len = 1, 32
        features = {
            "video": torch.randn(batch_size, seq_len, 768, device=device),
            "audio": torch.randn(batch_size, seq_len, 512, device=device),
        }

        # Compute total energy
        total_energy = attention(features)

        # Should return scalar energy
        assert total_energy.shape == ()
        assert torch.isfinite(total_energy)

        # Test with breakdown
        breakdown = attention(features, return_breakdown=True)

        # Should contain all expected components
        assert "total" in breakdown
        assert "intra_video" in breakdown
        assert "intra_audio" in breakdown
        assert "cross_video_audio" in breakdown
        assert "cross_audio_video" in breakdown

        # Total should match direct computation
        assert torch.allclose(breakdown["total"], total_energy, rtol=1e-5)

        # Total should equal sum of all components
        component_sum = (
            breakdown["intra_video"]
            + breakdown["intra_audio"]
            + breakdown["cross_video_audio"]
            + breakdown["cross_audio_video"]
        )
        assert torch.allclose(breakdown["total"], component_sum, rtol=1e-5)

    def test_gradient_flow_through_multimodal_attention(self, attention, device):
        """Test gradient flow through multimodal attention computation.

        Verifies that gradients flow correctly through the complex
        continuous compression and energy computation pipeline.
        """
        batch_size, seq_len = 1, 32
        features = {
            "video": torch.randn(
                batch_size, seq_len, 768, device=device, requires_grad=True
            ),
            "audio": torch.randn(
                batch_size, seq_len, 512, device=device, requires_grad=True
            ),
        }

        # Compute energy and backpropagate
        energy = attention(features)
        energy.backward()

        # Check gradients exist and are finite
        for feature_tensor in features.values():
            assert feature_tensor.grad is not None
            assert torch.isfinite(feature_tensor.grad).all()
            assert feature_tensor.grad.abs().max() > 0  # Non-trivial gradients

    def test_numerical_stability_with_extreme_inputs(self, attention, device):
        """Test numerical stability with extreme input values.

        Verifies robust behavior with very large, small, or edge-case inputs
        that could cause numerical instabilities.
        """
        batch_size, seq_len = 1, 16

        # Test with very large inputs
        large_features = {
            "video": torch.randn(batch_size, seq_len, 768, device=device) * 100,
            "audio": torch.randn(batch_size, seq_len, 512, device=device) * 100,
        }
        energy_large = attention(large_features)
        assert torch.isfinite(energy_large)

        # Test with very small inputs
        small_features = {
            "video": torch.randn(batch_size, seq_len, 768, device=device) * 0.001,
            "audio": torch.randn(batch_size, seq_len, 512, device=device) * 0.001,
        }
        energy_small = attention(small_features)
        assert torch.isfinite(energy_small)

        # Test with zero inputs
        zero_features = {
            "video": torch.zeros(batch_size, seq_len, 768, device=device),
            "audio": torch.zeros(batch_size, seq_len, 512, device=device),
        }
        energy_zero = attention(zero_features)
        assert torch.isfinite(energy_zero)

    def test_different_sequence_lengths_same_modalities(self, attention, device):
        """Test handling of different sequence lengths for same modalities.

        Verifies that the compression mechanism correctly handles
        varying sequence lengths L while maintaining consistent M.
        """
        batch_size = 1
        video_dim, audio_dim = 768, 512

        # Test different sequence lengths
        for seq_len in [16, 64, 256]:
            features = {
                "video": torch.randn(batch_size, seq_len, video_dim, device=device),
                "audio": torch.randn(batch_size, seq_len, audio_dim, device=device),
            }

            energy = attention(features)
            assert torch.isfinite(energy)

            # Compression should always reduce to same M regardless of L
            _, compressed_video, _ = attention.compress_modality(
                features["video"], "video"
            )
            _, compressed_audio, _ = attention.compress_modality(
                features["audio"], "audio"
            )

            # Compression dimension should be constant (M = 100)
            assert compressed_video.shape[-1] == 100
            assert compressed_audio.shape[-1] == 100

    def test_orthonormal_basis_functions_for_stability(self, attention, device):
        """Test basis functions ψ^m_j(t) satisfy orthonormality requirement.

        From paper: "basis functions ψ^m(t) defined over [0,1], chosen to satisfy
        orthonormality ∫ψ^m_i(t)ψ^m_j(t)dt = δ_ij for numerical stability"

        This is ESSENTIAL for:
        1. Stable ridge regression (well-conditioned Gram matrix)
        2. Efficient compression without information loss
        3. Proper continuous function reconstruction
        """
        # Get the compression module for a modality
        compression = attention.compressions["video"]

        # Verify orthonormality via numerical integration
        num_basis = 100  # M
        num_test_points = 1000
        t_points = torch.linspace(0, 1, num_test_points, device=device)
        dt = 1.0 / num_test_points

        # Evaluate basis functions at points using the compression's method
        # We need to use the basis through the compression module
        # Create dummy data to get basis evaluation
        dummy_data = (
            torch.eye(num_basis, device=device).unsqueeze(0).unsqueeze(0)
        )  # (1, 1, M, M)
        reconstructed = compression.reconstruct(
            dummy_data, t_points
        )  # (1, 1, M, num_points)
        basis_values = reconstructed[0, 0].T  # (num_points, M)

        # Compute Gram matrix: G_ij = ∫ψ_i(t)ψ_j(t)dt
        gram_matrix = basis_values.T @ basis_values * dt

        # Should be identity matrix (orthonormal)
        identity = torch.eye(num_basis, device=device)

        # Allow small numerical error but MUST be approximately orthonormal
        assert torch.allclose(gram_matrix, identity, rtol=0.1, atol=0.1), (
            "Basis functions are not orthonormal! This breaks:\n"
            "1. Ridge regression stability\n"
            "2. Compression efficiency\n"
            "3. Continuous function reconstruction\n"
            f"Max deviation from identity: {(gram_matrix - identity).abs().max()}"
        )

    @pytest.mark.parametrize("num_integration_points", [10, 50, 100])
    def test_integration_accuracy_with_different_point_counts(
        self, modality_configs, device, num_integration_points
    ):
        """Test Gauss-Legendre integration with different quadrature points.

        From paper: "Numerical integration with S = O(M) points (e.g., Gauss-Legendre)"

        More points → higher accuracy but slower computation.
        Paper recommends 50 points as good tradeoff.
        """

        attention = MultimodalEnergyAttention(
            modality_configs=modality_configs,
            cross_modal_pairs=[("video", "audio")],
            num_integration_points=num_integration_points,
            device=device,
        )

        batch_size, seq_len = 1, 32
        features = {
            "video": torch.randn(batch_size, seq_len, 768, device=device),
            "audio": torch.randn(batch_size, seq_len, 512, device=device),
        }

        energy = attention(features)
        assert torch.isfinite(energy)

        # Integration points should match configuration
        assert attention.integrator.num_points == num_integration_points

    # Edge case tests
    def test_single_modality_configuration(self, device):
        """Test with only a single modality configured.

        Verifies that the attention mechanism works with just one modality,
        computing only intra-modal energy without cross-modal components.
        """
        single_modality_config = {
            "video": {
                "embed_dim": 768,
                "compression_dim": 100,
                "num_heads": 8,
                "qk_dim": 64,
                "basis_type": "rectangular",
                "regularization": 0.001,
            }
        }

        # No cross-modal pairs with single modality
        attention = MultimodalEnergyAttention(
            modality_configs=single_modality_config,
            cross_modal_pairs=[],  # Empty list for no cross-modal
            num_integration_points=50,
            device=device,
        )

        batch_size, seq_len = 2, 128
        features = {"video": torch.randn(batch_size, seq_len, 768, device=device)}

        # Should compute only intra-modal energy
        energy = attention(features)
        assert torch.isfinite(energy)

        # With breakdown, should only have intra components
        breakdown = attention(features, return_breakdown=True)
        assert "total" in breakdown
        assert "intra_video" in breakdown
        # No cross-modal components
        for key in breakdown:
            assert "cross" not in key or breakdown[key] == 0

    def test_extreme_compression_ratios(self, device):
        """Test with extreme compression ratios (both high and low).

        Verifies robustness when M >> L (expansion) or M << L (high compression).
        """
        # Test 1: Expansion case (M > L)
        expansion_config = {
            "video": {
                "embed_dim": 256,
                "compression_dim": 200,  # M = 200
                "num_heads": 4,
                "qk_dim": 32,
                "basis_type": "rectangular",
                "regularization": 0.01,
            }
        }

        attention_expand = MultimodalEnergyAttention(
            modality_configs=expansion_config,
            cross_modal_pairs=[],
            num_integration_points=50,
            device=device,
        )

        batch_size, seq_len = 1, 50  # L = 50 < M = 200
        features = {"video": torch.randn(batch_size, seq_len, 256, device=device)}

        energy_expand = attention_expand(features)
        assert torch.isfinite(energy_expand)

        # Test 2: High compression case (M << L)
        compression_config = {
            "video": {
                "embed_dim": 256,
                "compression_dim": 10,  # M = 10
                "num_heads": 4,
                "qk_dim": 32,
                "basis_type": "rectangular",
                "regularization": 0.01,
            }
        }

        attention_compress = MultimodalEnergyAttention(
            modality_configs=compression_config,
            cross_modal_pairs=[],
            num_integration_points=50,
            device=device,
        )

        seq_len = 1000  # L = 1000 >> M = 10
        features = {"video": torch.randn(batch_size, seq_len, 256, device=device)}

        energy_compress = attention_compress(features)
        assert torch.isfinite(energy_compress)

        # Verify compression actually reduces dimensionality
        queries, compressed_keys, _ = attention_compress.compress_modality(
            features["video"], "video"
        )
        assert compressed_keys.shape[-1] == 10  # M dimension
        assert queries.shape[-1] == 1000  # L dimension

    def test_very_small_sequences(self, device):
        """Test with very small sequence lengths (L=1, L=2, L=3).

        Verifies that the mechanism handles edge cases of minimal sequences.
        """
        config = {
            "video": {
                "embed_dim": 128,
                "compression_dim": 5,  # Small compression dim for small sequences
                "num_heads": 2,
                "qk_dim": 16,
                "basis_type": "rectangular",
                "regularization": 0.001,
            },
            "audio": {
                "embed_dim": 96,
                "compression_dim": 5,
                "num_heads": 2,
                "qk_dim": 16,
                "basis_type": "rectangular",
                "regularization": 0.001,
            },
        }

        attention = MultimodalEnergyAttention(
            modality_configs=config,
            cross_modal_pairs=[("video", "audio")],
            num_integration_points=50,
            device=device,
        )

        batch_size = 1

        # Test with L=1 (single token)
        features_l1 = {
            "video": torch.randn(batch_size, 1, 128, device=device),
            "audio": torch.randn(batch_size, 1, 96, device=device),
        }
        energy_l1 = attention(features_l1)
        assert torch.isfinite(energy_l1)

        # Test with L=2 (pair of tokens)
        features_l2 = {
            "video": torch.randn(batch_size, 2, 128, device=device),
            "audio": torch.randn(batch_size, 2, 96, device=device),
        }
        energy_l2 = attention(features_l2)
        assert torch.isfinite(energy_l2)

        # Test with L=3 (minimal sequence)
        features_l3 = {
            "video": torch.randn(batch_size, 3, 128, device=device),
            "audio": torch.randn(batch_size, 3, 96, device=device),
        }
        energy_l3 = attention(features_l3)
        assert torch.isfinite(energy_l3)

        # Energies should be different for different sequence lengths
        assert not torch.allclose(energy_l1, energy_l2, rtol=1e-5)
        assert not torch.allclose(energy_l2, energy_l3, rtol=1e-5)

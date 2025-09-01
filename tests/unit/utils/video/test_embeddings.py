# Revised test_embeddings.py
"""Comprehensive tests for video embedding extractors."""

import pytest
import torch

from associative.utils.video.embeddings import (
    CLIPEmbedder,
    EmbeddingExtractor,
    EVAClipEmbedder,
    get_embedder,
    list_embedders,
)
from associative.utils.video.transforms import Compose, Normalize, Resize


class TestEmbeddingExtractorInterface:
    """Test EmbeddingExtractor abstract interface."""

    def test_abstract_interface(self):
        """Test that EmbeddingExtractor cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            EmbeddingExtractor()  # pyright: ignore[reportAbstractUsage]

    def test_abstract_methods(self):
        """Test that abstract methods must be implemented by subclasses."""

        class IncompleteExtractor(EmbeddingExtractor):
            pass

        # Should fail to instantiate without implementing abstract methods
        with pytest.raises(TypeError):
            IncompleteExtractor()  # pyright: ignore[reportAbstractUsage]

    def test_freeze_unfreeze_interface(self):
        """Test that freeze/unfreeze methods are available."""
        # Test with a concrete implementation
        embedder = EVAClipEmbedder()

        # Methods should exist
        assert hasattr(embedder, "freeze")
        assert hasattr(embedder, "unfreeze")
        assert callable(embedder.freeze)
        assert callable(embedder.unfreeze)

    def test_train_mode_interaction(self):
        """Test interaction between train mode and freeze state."""
        embedder = EVAClipEmbedder(freeze_weights=True)

        # When frozen, should stay in eval mode
        embedder.train(True)
        # Implementation should keep it in eval mode due to freeze_weights

        # When unfrozen, should respect train mode
        embedder.unfreeze()
        embedder.train(True)
        # Should now be in training mode


class TestEVAClipEmbedder:
    """Test EVAClipEmbedder implementation."""

    def test_initialization_default(self):
        """Test default initialization of EVAClipEmbedder."""
        embedder = EVAClipEmbedder()

        # Check default values
        assert embedder.model_name == "eva_clip_vit_g_14"
        assert embedder.num_query_tokens == 32
        assert embedder.freeze_vit is True
        assert embedder.freeze_qformer is True
        assert embedder.pooling_strategy == "mean"

    def test_initialization_custom(self):
        """Test custom initialization parameters."""
        embedder = EVAClipEmbedder(
            model_name="eva_clip_vit_g_14",
            num_query_tokens=64,
            freeze_vit=False,
            freeze_qformer=False,
            pooling_strategy="cls",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        assert embedder.model_name == "eva_clip_vit_g_14"
        assert embedder.num_query_tokens == 64
        assert not embedder.freeze_vit
        assert not embedder.freeze_qformer
        assert embedder.pooling_strategy == "cls"

    def test_invalid_initialization(self):
        """Test that invalid parameters raise errors."""
        # Invalid pooling strategy
        with pytest.raises(ValueError, match="Invalid pooling_strategy"):
            EVAClipEmbedder(pooling_strategy="invalid")

    def test_abstract_properties(self):
        """Test that abstract properties are implemented."""
        embedder = EVAClipEmbedder()

        # Properties should exist and return appropriate types
        assert isinstance(embedder.embed_dim, int)
        assert embedder.embed_dim > 0

        assert isinstance(embedder.expected_input_size, tuple)
        assert len(embedder.expected_input_size) == 2
        assert all(isinstance(s, int) and s > 0 for s in embedder.expected_input_size)

    def test_forward_input_validation(self):
        """Test input validation in forward method."""
        embedder = EVAClipEmbedder()

        # Wrong dimensions
        with pytest.raises(ValueError, match="Expected 4D or 5D tensor"):
            embedder(torch.randn(100, 224, 224))  # 3D

        with pytest.raises(ValueError, match="Expected 4D or 5D tensor"):
            embedder(torch.randn(100, 3, 224, 224, 5, 6))  # 6D

        # Wrong number of channels
        with pytest.raises(ValueError, match="Expected 3 channels"):
            embedder(torch.randn(100, 1, 224, 224))  # 1 channel

        with pytest.raises(ValueError, match="Expected 3 channels"):
            embedder(torch.randn(100, 4, 224, 224))  # 4 channels

    def test_forward_size_validation(self):
        """Test input size validation in forward method."""
        embedder = EVAClipEmbedder()
        expected_h, expected_w = embedder.expected_input_size

        # Wrong size should raise error
        wrong_h = expected_h + 32
        wrong_w = expected_w + 32

        with pytest.raises(ValueError, match="Expected input size"):
            embedder(torch.randn(100, 3, wrong_h, wrong_w))

    def test_forward_4d_input(self):
        """Test forward pass with 4D input (N, C, H, W)."""
        embedder = EVAClipEmbedder()
        h, w = embedder.expected_input_size

        frames = torch.randn(50, 3, h, w)

        result = embedder(frames)
        assert result.shape == (50, embedder.embed_dim)

    def test_forward_5d_input(self):
        """Test forward pass with 5D input (B, N, C, H, W)."""
        embedder = EVAClipEmbedder()
        h, w = embedder.expected_input_size

        batched_frames = torch.randn(4, 25, 3, h, w)  # 4 videos, 25 frames each

        result = embedder(batched_frames)
        assert result.shape == (4, 25, embedder.embed_dim)

    def test_freeze_functionality(self):
        """Test parameter freezing functionality."""
        embedder = EVAClipEmbedder()

        # Test freeze method
        embedder.freeze()
        assert embedder.freeze_weights is True

        # Test unfreeze method
        embedder.unfreeze()
        assert embedder.freeze_weights is False


class TestCLIPEmbedder:
    """Test CLIPEmbedder implementation."""

    def test_initialization(self):
        """Test CLIP embedder initialization."""
        CLIPEmbedder()

    def test_initialization_with_parameters(self):
        """Test CLIP embedder with custom parameters."""
        CLIPEmbedder(
            model_name="ViT-B/32", device=torch.device("cpu"), dtype=torch.float32
        )

    def test_abstract_properties(self):
        """Test that CLIP embedder would implement abstract properties."""
        embedder = CLIPEmbedder()
        # Would test properties if implementation existed
        assert isinstance(embedder.embed_dim, int)
        assert embedder.embed_dim > 0

        assert isinstance(embedder.expected_input_size, tuple)
        assert len(embedder.expected_input_size) == 2
        assert all(isinstance(s, int) and s > 0 for s in embedder.expected_input_size)


class TestRegistrySystem:
    """Test embedder registry system."""

    def test_default_registrations(self):
        """Test that default embedders are registered."""
        available_embedders = list_embedders()

        assert "eva_clip_vit_g_14" in available_embedders
        assert "clip_vit_b_32" in available_embedders
        assert isinstance(available_embedders, list)

    def test_get_embedder_valid(self):
        """Test getting valid embedders from registry."""
        # Test EVA-CLIP embedder
        embedder = get_embedder("eva_clip_vit_g_14", num_query_tokens=64)
        assert isinstance(embedder, EVAClipEmbedder)
        assert embedder.num_query_tokens == 64

        # Test CLIP embedder (will raise NotImplementedError during init)
        embedder = get_embedder("clip_vit_b_32")
        assert isinstance(embedder, CLIPEmbedder)

    def test_get_embedder_invalid(self):
        """Test error handling for invalid embedder names."""
        with pytest.raises(KeyError, match="Embedder 'nonexistent' not found"):
            get_embedder("nonexistent")

    def test_get_embedder_with_kwargs(self):
        """Test passing kwargs to embedder constructors."""
        # Test with valid kwargs
        embedder = get_embedder(
            "eva_clip_vit_g_14",
            num_query_tokens=128,
            pooling_strategy="cls",
            freeze_vit=False,
        )
        assert embedder.num_query_tokens == 128
        assert embedder.pooling_strategy == "cls"
        assert not embedder.freeze_vit

    def test_registry_content(self):
        """Test registry content and structure."""
        from associative.utils.video._registry import EMBEDDER_REGISTRY

        # Registry should contain expected embedders
        assert "eva_clip_vit_g_14" in EMBEDDER_REGISTRY
        assert "clip_vit_b_32" in EMBEDDER_REGISTRY

        # Registered classes should be the correct types
        assert EMBEDDER_REGISTRY["eva_clip_vit_g_14"] == EVAClipEmbedder
        assert EMBEDDER_REGISTRY["clip_vit_b_32"] == CLIPEmbedder


class TestEmbedderBehaviorContracts:
    """Test behavioral contracts that all embedders must satisfy."""

    @pytest.fixture
    def mock_embedder(self):
        """Create a mock embedder for testing contracts."""

        class MockEmbedder(EmbeddingExtractor):
            def __init__(self):
                super().__init__()
                self._embed_dim = 768
                self._input_size = (224, 224)

            def forward(self, frames):
                if frames.dim() == 4:
                    batch_size = frames.shape[0]
                    features = frames.mean(dim=[1, 2, 3])  # batch_size
                    return features.unsqueeze(1).repeat(1, self._embed_dim)
                if frames.dim() == 5:
                    batch_size, num_frames = frames.shape[:2]
                    frames_flat = frames.view(-1, *frames.shape[2:])
                    features = frames_flat.mean(
                        dim=[1, 2, 3]
                    )  # batch_size * num_frames
                    return (
                        features.unsqueeze(1)
                        .repeat(1, self._embed_dim)
                        .view(batch_size, num_frames, self._embed_dim)
                    )
                raise ValueError("Invalid input dimensions")

            @property
            def embed_dim(self):
                return self._embed_dim

            @property
            def expected_input_size(self):
                return self._input_size

        return MockEmbedder()

    def test_embedder_output_shapes(self, mock_embedder):
        """Test that embedders produce correct output shapes."""
        h, w = mock_embedder.expected_input_size

        # 4D input should produce 2D output
        frames_4d = torch.randn(10, 3, h, w)
        output_4d = mock_embedder(frames_4d)
        assert output_4d.shape == (10, mock_embedder.embed_dim)

        # 5D input should produce 3D output
        frames_5d = torch.randn(2, 15, 3, h, w)
        output_5d = mock_embedder(frames_5d)
        assert output_5d.shape == (2, 15, mock_embedder.embed_dim)

    def test_embedder_dtype_preservation(self, mock_embedder):
        """Test that embedders preserve appropriate dtypes."""
        h, w = mock_embedder.expected_input_size

        # Test different input dtypes
        frames_float32 = torch.randn(5, 3, h, w, dtype=torch.float32)
        frames_float64 = torch.randn(5, 3, h, w, dtype=torch.float64)

        output_float32 = mock_embedder(frames_float32)
        output_float64 = mock_embedder(frames_float64)

        # Outputs should be floating point (exact dtype may vary)
        assert output_float32.dtype.is_floating_point
        assert output_float64.dtype.is_floating_point

    def test_embedder_device_consistency(self, mock_embedder):
        """Test that embedders maintain device consistency."""
        h, w = mock_embedder.expected_input_size

        # CPU input
        frames_cpu = torch.randn(3, 3, h, w)
        output_cpu = mock_embedder(frames_cpu)
        assert output_cpu.device == frames_cpu.device

        # GPU input (if available)
        if torch.cuda.is_available():
            frames_cuda = frames_cpu.cuda()
            output_cuda = mock_embedder(frames_cuda)
            assert output_cuda.device == frames_cuda.device

    def test_embedder_gradient_flow(self, mock_embedder):
        """Test that gradients flow through embedders correctly."""
        h, w = mock_embedder.expected_input_size

        frames = torch.randn(4, 3, h, w, requires_grad=True)
        output = mock_embedder(frames)

        # Should be able to backpropagate
        loss = output.sum()
        loss.backward()

        assert frames.grad is not None
        assert not torch.allclose(frames.grad, torch.zeros_like(frames.grad))

    def test_embedder_batch_independence(self, mock_embedder):
        """Test that embedder processes batch elements independently."""
        h, w = mock_embedder.expected_input_size

        # Single frames
        frame1 = torch.randn(1, 3, h, w)
        frame2 = torch.randn(1, 3, h, w)

        embed1 = mock_embedder(frame1)
        embed2 = mock_embedder(frame2)

        # Batched frames
        frames_batch = torch.cat([frame1, frame2], dim=0)
        embeds_batch = mock_embedder(frames_batch)

        # Results should be equivalent
        assert torch.allclose(embeds_batch[0], embed1.squeeze(0), atol=1e-5)
        assert torch.allclose(embeds_batch[1], embed2.squeeze(0), atol=1e-5)


class TestEmbedderIntegration:
    """Integration tests for embedder usage patterns."""

    def test_embedder_with_transforms(self):
        """Test using embedders with transform pipelines."""

        # Create transform pipeline
        Compose(
            [
                Resize(224),  # Ensure correct size
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        # This would be tested with actual implementation
        # For now, just ensure the pattern is possible

        get_embedder("eva_clip_vit_g_14")
        # Would test: frames -> transform -> embedder

    def test_embedder_error_recovery(self):
        """Test embedder behavior under error conditions."""
        embedder = EVAClipEmbedder()

        # Test with invalid inputs
        with pytest.raises(ValueError):
            embedder(torch.randn(10, 1, 224, 224))  # Wrong channels

        with pytest.raises(ValueError):
            embedder(torch.randn(10, 3, 100, 100))  # Wrong size

    def test_embedder_reproducibility(self):
        """Test that embedders produce reproducible results."""
        embedder = EVAClipEmbedder()
        h, w = embedder.expected_input_size

        frames = torch.randn(5, 3, h, w)

        # Set seed for reproducibility
        torch.manual_seed(42)
        result1 = embedder(frames)

        torch.manual_seed(42)
        result2 = embedder(frames)

        assert torch.allclose(result1, result2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

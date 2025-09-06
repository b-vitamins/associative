"""Tests for video embedding extractors.

Focus on behavioral contracts and practical usage patterns.
Tests should verify correctness without constraining implementation.
"""

import pytest
import torch
from torch import nn

from associative.utils.video.embeddings import (
    CLIPEmbedder,
    EmbeddingExtractor,
    EVAClipEmbedder,
    get_embedder,
    list_embedders,
)


class TestEmbeddingExtractorContract:
    """Test the abstract interface contract for embedding extractors."""

    def test_cannot_instantiate_abstract_class(self):
        """Abstract class should not be directly instantiable."""
        with pytest.raises(TypeError, match="abstract"):
            EmbeddingExtractor()  # type: ignore[abstract]

    def test_concrete_class_must_implement_abstract_methods(self):
        """Concrete classes must implement all abstract methods."""

        class IncompleteEmbedder(EmbeddingExtractor):
            def forward(self, frames):
                return frames

            @property
            def embed_dim(self):
                return 768

            # Missing: expected_input_size

        with pytest.raises(TypeError, match="abstract"):
            IncompleteEmbedder()  # type: ignore[abstract]

    def test_freeze_unfreeze_behavior(self):
        """Test parameter freezing functionality."""
        embedder = EVAClipEmbedder()

        # Test freeze
        embedder.freeze()
        for param in embedder.parameters():
            assert not param.requires_grad

        # Test unfreeze
        embedder.unfreeze()
        for param in embedder.parameters():
            assert param.requires_grad

    def test_training_mode_respects_freeze_state(self):
        """Training mode should respect freeze state."""
        embedder = EVAClipEmbedder(freeze_vit=True, freeze_qformer=True)

        # When frozen, should stay in eval mode
        embedder.train(True)
        assert not embedder.training  # Should remain False

        # After unfreezing, should respect train mode
        embedder.unfreeze()
        embedder.train(True)
        assert embedder.training


class TestEVAClipEmbedder:
    """Test EVA-CLIP embedder implementation."""

    def test_default_configuration(self):
        """Test default initialization values."""
        embedder = EVAClipEmbedder()

        assert embedder.model_name == "eva_clip_vit_g_14"
        assert embedder.num_query_tokens == 32
        assert embedder.freeze_vit is True
        assert embedder.freeze_qformer is True
        assert embedder.pooling_strategy == "mean"
        assert embedder.embed_dim == 768
        assert embedder.expected_input_size == (224, 224)

    def test_custom_configuration(self):
        """Test custom initialization parameters."""
        embedder = EVAClipEmbedder(
            num_query_tokens=64,
            freeze_vit=False,
            pooling_strategy="cls",
        )

        assert embedder.num_query_tokens == 64
        assert embedder.freeze_vit is False
        assert embedder.pooling_strategy == "cls"

    def test_invalid_pooling_strategy(self):
        """Invalid pooling strategy should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid pooling_strategy"):
            EVAClipEmbedder(pooling_strategy="invalid")

    def test_invalid_model_name(self):
        """Invalid model name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            EVAClipEmbedder(model_name="nonexistent_model")

    @pytest.mark.parametrize(
        "input_shape,error_match",
        [
            ((100, 224, 224), "Expected 4D or 5D"),  # 3D
            ((100, 3, 224, 224, 5, 6), "Expected 4D or 5D"),  # 6D
            ((10, 1, 224, 224), "Expected 3 channels"),  # Wrong channels
            ((10, 4, 224, 224), "Expected 3 channels"),  # Wrong channels
            ((10, 3, 256, 256), "Expected input size"),  # Wrong size
        ],
    )
    def test_input_validation(self, input_shape, error_match):
        """Test input validation with various invalid shapes."""
        embedder = EVAClipEmbedder()
        with pytest.raises(ValueError, match=error_match):
            embedder(torch.randn(*input_shape))

    def test_forward_4d_input(self):
        """Test processing 4D input (N, C, H, W)."""
        embedder = EVAClipEmbedder()
        frames = torch.randn(10, 3, 224, 224)

        embeddings = embedder(frames)

        assert embeddings.shape == (10, 768)
        assert embeddings.dtype == torch.float32
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()

    def test_forward_5d_input(self):
        """Test processing 5D input (B, N, C, H, W)."""
        embedder = EVAClipEmbedder()
        batch_frames = torch.randn(4, 8, 3, 224, 224)

        embeddings = embedder(batch_frames)

        assert embeddings.shape == (4, 8, 768)
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()

    @pytest.mark.parametrize("pooling", ["mean", "cls", "max"])
    def test_pooling_strategies(self, pooling):
        """Test different pooling strategies produce valid outputs."""
        embedder = EVAClipEmbedder(pooling_strategy=pooling)
        embedder.eval()  # Put in eval mode for deterministic behavior
        frames = torch.randn(5, 3, 224, 224)

        embeddings = embedder(frames)

        assert embeddings.shape == (5, 768)
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()

        # Test determinism in eval mode
        embeddings2 = embedder(frames)
        assert torch.allclose(embeddings, embeddings2, atol=1e-5)

    def test_encode_batch_method(self):
        """Test batch encoding for memory efficiency."""
        embedder = EVAClipEmbedder()
        embedder.eval()  # Ensure deterministic behavior
        large_batch = torch.randn(20, 3, 224, 224)  # Reduced from 100 to 20

        # Process in smaller batches
        embeddings = embedder.encode_batch(large_batch, batch_size=8)

        assert embeddings.shape == (20, 768)

        # Should match processing all at once
        embeddings_full = embedder(large_batch)
        assert torch.allclose(embeddings, embeddings_full, atol=1e-5)

    def test_gradient_flow(self):
        """Test gradients flow through unfrozen embedder."""
        embedder = EVAClipEmbedder(freeze_vit=False, freeze_qformer=False)
        frames = torch.randn(2, 3, 224, 224, requires_grad=True)

        embeddings = embedder(frames)
        loss = embeddings.sum()
        loss.backward()

        assert frames.grad is not None
        # Check that gradients are non-zero (but may be small)
        assert frames.grad.abs().max() > 0

    def test_deterministic_output(self):
        """Test that embedder produces deterministic outputs in eval mode."""
        embedder = EVAClipEmbedder()
        embedder.eval()  # Put in eval mode for deterministic behavior
        frames = torch.randn(3, 3, 224, 224)

        # Multiple forward passes should give same result in eval mode
        emb1 = embedder(frames)
        emb2 = embedder(frames)

        assert torch.allclose(emb1, emb2, atol=1e-5)

    def test_device_handling(self):
        """Test embedder works on different devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        embedder = EVAClipEmbedder(device=torch.device("cuda"))
        frames = torch.randn(2, 3, 224, 224).cuda()

        embeddings = embedder(frames)
        assert embeddings.device.type == "cuda"


class TestCLIPEmbedder:
    """Test CLIP embedder implementation."""

    def test_initialization(self):
        """Test CLIP embedder initialization."""
        embedder = CLIPEmbedder()

        assert embedder.model_name == "ViT-B/32"
        assert embedder.embed_dim == 768
        assert embedder.expected_input_size == (224, 224)

    def test_unsupported_model_error(self):
        """Test error for unsupported CLIP models."""
        with pytest.raises(ValueError, match="Unsupported model"):
            CLIPEmbedder(model_name="ViT-L/14")

    def test_forward_4d_input(self):
        """Test CLIP processing of 4D input."""
        embedder = CLIPEmbedder()
        frames = torch.randn(5, 3, 224, 224)

        embeddings = embedder(frames)

        assert embeddings.shape == (5, 768)
        assert not torch.isnan(embeddings).any()

    def test_forward_5d_input(self):
        """Test CLIP processing of 5D batched input."""
        embedder = CLIPEmbedder()
        batch_frames = torch.randn(2, 6, 3, 224, 224)

        embeddings = embedder(batch_frames)

        assert embeddings.shape == (2, 6, 768)
        assert not torch.isnan(embeddings).any()

    def test_input_validation(self):
        """Test CLIP input validation."""
        embedder = CLIPEmbedder()

        # Wrong dimensions
        with pytest.raises(ValueError, match="Expected 4D or 5D"):
            embedder(torch.randn(224, 224))

        # Wrong channels
        with pytest.raises(ValueError, match="Expected 3 channels"):
            embedder(torch.randn(5, 1, 224, 224))

        # Wrong size
        with pytest.raises(ValueError, match="Expected input size"):
            embedder(torch.randn(5, 3, 256, 256))


class TestEmbedderRegistry:
    """Test embedder registry functionality."""

    def test_list_available_embedders(self):
        """Test listing registered embedders."""
        available = list_embedders()

        assert isinstance(available, list)
        assert "eva_clip_vit_g_14" in available
        assert "clip_vit_b_32" in available
        assert len(available) >= 2

    def test_get_embedder_by_name(self):
        """Test retrieving embedders from registry."""
        # EVA-CLIP
        eva_embedder = get_embedder("eva_clip_vit_g_14")
        assert isinstance(eva_embedder, EVAClipEmbedder)

        # CLIP
        clip_embedder = get_embedder("clip_vit_b_32")
        assert isinstance(clip_embedder, CLIPEmbedder)

    def test_get_embedder_with_kwargs(self):
        """Test passing configuration through registry."""
        embedder = get_embedder(
            "eva_clip_vit_g_14",
            num_query_tokens=128,
            pooling_strategy="max",
        )

        assert embedder.num_query_tokens == 128
        assert embedder.pooling_strategy == "max"

    def test_invalid_embedder_name(self):
        """Test error for non-existent embedder."""
        with pytest.raises(KeyError, match="not found"):
            get_embedder("nonexistent_embedder")


class TestEmbedderIntegration:
    """Integration tests for practical usage patterns."""

    def test_batch_processing_consistency(self):
        """Test that batch processing gives consistent results."""
        embedder = EVAClipEmbedder()
        embedder.eval()  # Ensure deterministic behavior

        # Individual frames
        frame1 = torch.randn(1, 3, 224, 224)
        frame2 = torch.randn(1, 3, 224, 224)

        emb1 = embedder(frame1)
        emb2 = embedder(frame2)

        # Batched processing
        batch = torch.cat([frame1, frame2], dim=0)
        batch_emb = embedder(batch)

        # Results should match
        assert torch.allclose(batch_emb[0], emb1[0], atol=1e-5)
        assert torch.allclose(batch_emb[1], emb2[0], atol=1e-5)

    def test_mixed_precision_support(self):
        """Test embedders work with different dtypes."""
        # Note: Model weights are initialized as float32 by transformers
        # We need to explicitly convert them
        embedder = EVAClipEmbedder()
        embedder = embedder.half()  # Convert to float16
        embedder.eval()

        frames_f16 = torch.randn(2, 3, 224, 224, dtype=torch.float16)

        embeddings = embedder(frames_f16)
        assert embeddings.dtype == torch.float16

    def test_embedder_as_feature_extractor(self):
        """Test using embedder as a frozen feature extractor."""
        embedder = EVAClipEmbedder(freeze_vit=True, freeze_qformer=True)

        # Create a simple classifier on top
        classifier = nn.Sequential(
            embedder,
            nn.Linear(768, 10),
        )

        # Check gradient requirements
        embedder_has_grad = False
        linear_has_grad = False

        for name, param in classifier.named_parameters():
            if "1." in name:  # Linear layer (second module in Sequential)
                linear_has_grad = linear_has_grad or param.requires_grad
            else:  # Embedder (first module)
                embedder_has_grad = embedder_has_grad or param.requires_grad

        assert linear_has_grad  # Linear layer should have gradients
        assert not embedder_has_grad  # Embedder should be frozen

    def test_memory_efficiency_with_large_batches(self):
        """Test memory-efficient processing of large video batches."""
        embedder = EVAClipEmbedder()

        # Simulate a moderately long video (reduced for test speed)
        num_frames = 50  # Reduced from 500
        frames = torch.randn(num_frames, 3, 224, 224)

        # Should not raise memory errors
        embeddings = embedder.encode_batch(frames, batch_size=10)

        assert embeddings.shape == (num_frames, 768)
        assert not torch.isnan(embeddings).any()

    @pytest.mark.parametrize(
        "num_frames,expected_shape",
        [
            (1, (1, 768)),  # Single frame
            (10, (10, 768)),  # Short clip
            (100, (100, 768)),  # Long clip
        ],
    )
    def test_variable_length_videos(self, num_frames, expected_shape):
        """Test handling videos of different lengths."""
        embedder = EVAClipEmbedder()
        frames = torch.randn(num_frames, 3, 224, 224)

        embeddings = embedder(frames)
        assert embeddings.shape == expected_shape

    def test_normalized_embeddings(self):
        """Test that embeddings have reasonable magnitudes."""
        embedder = EVAClipEmbedder()
        frames = torch.randn(10, 3, 224, 224)

        embeddings = embedder(frames)

        # Check embeddings are not degenerate
        norms = embeddings.norm(dim=1)
        assert (norms > 0).all()  # Non-zero
        assert (norms < 1000).all()  # Not exploding

        # Check diversity in embeddings
        std = embeddings.std(dim=0)
        assert (std > 0).all()  # Features have variance


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

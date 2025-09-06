"""Tests for audio embedding extractors.

Focus on behavioral contracts and public API, not implementation details.
Tests verify that embedders:
1. Produce correct output shapes
2. Handle various input formats
3. Maintain consistent behavior
4. Integrate properly with PyTorch
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from associative.utils.audio.embeddings import (
    MelSpectrogramEmbedding,
    VGGishEmbedding,
    Wav2Vec2Embedding,
    get_audio_embedder,
)


class TestVGGishEmbedding:
    """Test VGGish embedding behavior."""

    def test_output_shape_single_audio(self):
        """Single audio input produces correct embedding shape."""
        with patch("associative.utils.audio.embeddings.torchvggish") as mock_vggish:
            # Mock the model to return fixed-size embeddings
            mock_model = MagicMock()
            mock_model.return_value = torch.randn(5, 128)  # 5 frames
            mock_vggish.vggish.return_value = mock_model

            # Mock PCA to return the input unchanged
            mock_pca = MagicMock()
            mock_pca.side_effect = lambda x: x
            mock_vggish.PCA.return_value = mock_pca

            embedder = VGGishEmbedding()
            audio = torch.randn(16000 * 2)  # 2 seconds
            result = embedder(audio)

            # VGGish produces frame-level embeddings
            assert result.dim() == 2
            assert result.shape[1] == 128
            assert result.shape[0] > 0  # At least one frame

    def test_output_shape_batch_audio(self):
        """Batch audio input produces correct embedding shape."""
        with patch("associative.utils.audio.embeddings.torchvggish") as mock_vggish:
            mock_model = MagicMock()
            # Return different number of frames per batch item
            mock_model.side_effect = [
                torch.randn(5, 128),
                torch.randn(4, 128),
                torch.randn(6, 128),
            ]
            mock_vggish.vggish.return_value = mock_model

            # Mock PCA
            mock_pca = MagicMock()
            mock_pca.side_effect = lambda x: x
            mock_vggish.PCA.return_value = mock_pca

            embedder = VGGishEmbedding()
            audio = torch.randn(3, 16000 * 2)  # Batch of 3
            result = embedder(audio)

            # Should return padded batch
            assert result.dim() == 3
            assert result.shape[0] == 3  # batch size
            assert result.shape[2] == 128  # embedding dim
            assert result.shape[1] == 6  # max frames (padded)

    def test_embedding_dimension(self):
        """VGGish always produces 128-dimensional embeddings."""
        with patch("associative.utils.audio.embeddings.torchvggish"):
            embedder = VGGishEmbedding()
            assert embedder.dim == 128

    def test_sample_rate_fixed(self):
        """VGGish requires 16kHz sample rate."""
        with patch("associative.utils.audio.embeddings.torchvggish"):
            embedder = VGGishEmbedding()
            assert embedder.sample_rate == 16000

    def test_aggregation_methods(self):
        """Test different aggregation methods produce correct shapes."""
        with patch("associative.utils.audio.embeddings.torchvggish") as mock_vggish:
            mock_model = MagicMock()
            mock_model.return_value = torch.randn(10, 128)
            mock_vggish.vggish.return_value = mock_model

            # Mock PCA
            mock_pca = MagicMock()
            mock_pca.side_effect = lambda x: x
            mock_vggish.PCA.return_value = mock_pca

            # Test mean aggregation
            embedder = VGGishEmbedding(aggregate="mean")
            audio = torch.randn(16000)
            result = embedder(audio)
            assert result.shape == (128,)

            # Test no aggregation (default)
            embedder = VGGishEmbedding(aggregate=None)
            result = embedder(audio)
            assert result.shape == (10, 128)

    def test_invalid_input_shape(self):
        """Invalid input shapes raise appropriate errors."""
        with patch("associative.utils.audio.embeddings.torchvggish"):
            embedder = VGGishEmbedding()

            # 3D input should raise error
            audio = torch.randn(2, 2, 16000)
            with pytest.raises(ValueError, match="Expected 1D or 2D"):
                embedder(audio)

    @pytest.mark.parametrize("use_pca", [True, False])
    def test_pca_option(self, use_pca):
        """PCA option can be enabled/disabled."""
        with patch("associative.utils.audio.embeddings.torchvggish") as mock_vggish:
            mock_model = MagicMock()
            mock_model.return_value = torch.randn(5, 128)
            mock_vggish.vggish.return_value = mock_model

            if use_pca:
                mock_pca = MagicMock()
                mock_pca.return_value = torch.randn(5, 128)
                mock_vggish.PCA.return_value = mock_pca

            embedder = VGGishEmbedding(use_pca=use_pca)
            audio = torch.randn(16000)
            result = embedder(audio)

            assert result.dim() == 2
            assert result.shape[1] == 128


class TestWav2Vec2Embedding:
    """Test Wav2Vec2 embedding behavior."""

    def test_output_shape_base_model(self):
        """Base model produces 768-dimensional embeddings."""
        with (
            patch(
                "associative.utils.audio.embeddings.Wav2Vec2Model"
            ) as mock_model_class,
            patch(
                "associative.utils.audio.embeddings.Wav2Vec2Processor"
            ) as mock_proc_class,
        ):
            # Mock the model
            mock_model = MagicMock()
            mock_output = MagicMock()
            mock_output.last_hidden_state = torch.randn(1, 100, 768)
            mock_model.return_value = mock_output
            mock_model_class.from_pretrained.return_value = mock_model

            # Mock the processor
            mock_proc = MagicMock()
            mock_proc.return_value = {"input_values": torch.randn(1, 16000)}
            mock_proc_class.from_pretrained.return_value = mock_proc

            embedder = Wav2Vec2Embedding(model_size="base")
            audio = torch.randn(16000)
            result = embedder(audio)

            # With default mean aggregation
            assert result.shape == (768,)

    def test_output_shape_large_model(self):
        """Large model produces 1024-dimensional embeddings."""
        with (
            patch(
                "associative.utils.audio.embeddings.Wav2Vec2Model"
            ) as mock_model_class,
            patch(
                "associative.utils.audio.embeddings.Wav2Vec2Processor"
            ) as mock_proc_class,
        ):
            # Mock the model
            mock_model = MagicMock()
            mock_output = MagicMock()
            mock_output.last_hidden_state = torch.randn(1, 100, 1024)
            mock_model.return_value = mock_output
            mock_model_class.from_pretrained.return_value = mock_model

            # Mock the processor
            mock_proc = MagicMock()
            mock_proc.return_value = {"input_values": torch.randn(1, 16000)}
            mock_proc_class.from_pretrained.return_value = mock_proc

            embedder = Wav2Vec2Embedding(model_size="large")
            audio = torch.randn(16000)
            result = embedder(audio)

            assert result.shape == (1024,)

    def test_batch_processing(self):
        """Batch input produces correct shape."""
        with (
            patch(
                "associative.utils.audio.embeddings.Wav2Vec2Model"
            ) as mock_model_class,
            patch(
                "associative.utils.audio.embeddings.Wav2Vec2Processor"
            ) as mock_proc_class,
        ):
            # Mock the model to handle batch
            mock_model = MagicMock()
            mock_output = MagicMock()
            mock_output.last_hidden_state = torch.randn(4, 100, 768)
            mock_model.return_value = mock_output
            mock_model_class.from_pretrained.return_value = mock_model

            # Mock the processor
            mock_proc = MagicMock()
            mock_proc.return_value = {"input_values": torch.randn(4, 16000)}
            mock_proc_class.from_pretrained.return_value = mock_proc

            embedder = Wav2Vec2Embedding()
            audio = torch.randn(4, 16000)
            result = embedder(audio)

            assert result.shape == (4, 768)

    def test_aggregation_methods(self):
        """Different aggregation methods work correctly."""
        with (
            patch(
                "associative.utils.audio.embeddings.Wav2Vec2Model"
            ) as mock_model_class,
            patch(
                "associative.utils.audio.embeddings.Wav2Vec2Processor"
            ) as mock_proc_class,
        ):
            # Setup mocks
            mock_model = MagicMock()
            mock_output = MagicMock()
            seq_len = 50
            mock_output.last_hidden_state = torch.randn(1, seq_len, 768)
            mock_model.return_value = mock_output
            mock_model_class.from_pretrained.return_value = mock_model

            mock_proc = MagicMock()
            mock_proc.return_value = {"input_values": torch.randn(1, 16000)}
            mock_proc_class.from_pretrained.return_value = mock_proc

            audio = torch.randn(16000)

            # Test different aggregations
            for agg in ["mean", "max", "first", "last", None]:
                embedder = Wav2Vec2Embedding(aggregate=agg)
                result = embedder(audio)

                if agg is None:
                    # No aggregation returns sequence
                    assert result.shape == (seq_len, 768)
                else:
                    # Aggregation returns single vector
                    assert result.shape == (768,)

    def test_invalid_model_size(self):
        """Invalid model size raises error."""
        with pytest.raises(ValueError, match="model_size must be"):
            Wav2Vec2Embedding(model_size="invalid")  # type: ignore

    def test_invalid_aggregation(self):
        """Invalid aggregation method raises error."""
        with pytest.raises(ValueError, match="aggregate must be"):
            Wav2Vec2Embedding(aggregate="invalid")  # type: ignore

    def test_embedding_dimension_property(self):
        """Embedding dimension property returns correct value."""
        with (
            patch("associative.utils.audio.embeddings.Wav2Vec2Model"),
            patch("associative.utils.audio.embeddings.Wav2Vec2Processor"),
        ):
            embedder_base = Wav2Vec2Embedding(model_size="base")
            assert embedder_base.dim == 768

            embedder_large = Wav2Vec2Embedding(model_size="large")
            assert embedder_large.dim == 1024


class TestMelSpectrogramEmbedding:
    """Test Mel-spectrogram embedding behavior."""

    def test_output_shape_no_projection(self):
        """Without projection, output dimension equals n_mels."""
        embedder = MelSpectrogramEmbedding(n_mels=80)
        audio = torch.randn(16000)
        result = embedder(audio)

        # Default mean aggregation
        assert result.shape == (80,)

    def test_output_shape_with_projection(self):
        """With projection, output dimension equals projection_dim."""
        embedder = MelSpectrogramEmbedding(n_mels=80, projection_dim=256)
        audio = torch.randn(16000)
        result = embedder(audio)

        assert result.shape == (256,)

    def test_batch_processing(self):
        """Batch processing produces correct shape."""
        embedder = MelSpectrogramEmbedding(n_mels=128, projection_dim=256)
        audio = torch.randn(4, 16000)
        result = embedder(audio)

        assert result.shape == (4, 256)

    def test_aggregation_methods(self):
        """Different aggregation methods work correctly."""
        audio = torch.randn(16000)

        # Mean aggregation
        embedder = MelSpectrogramEmbedding(n_mels=80, aggregate="mean")
        result = embedder(audio)
        assert result.shape == (80,)

        # Max aggregation
        embedder = MelSpectrogramEmbedding(n_mels=80, aggregate="max")
        result = embedder(audio)
        assert result.shape == (80,)

        # No aggregation returns full spectrogram
        embedder = MelSpectrogramEmbedding(n_mels=80, aggregate=None)
        result = embedder(audio)
        assert result.dim() == 2
        assert result.shape[0] == 80  # n_mels
        assert result.shape[1] > 0  # time frames

    def test_custom_sample_rate(self):
        """Custom sample rates are supported."""
        embedder = MelSpectrogramEmbedding(sample_rate=22050)
        assert embedder.sample_rate == 22050

        audio = torch.randn(22050)  # 1 second at 22050 Hz
        result = embedder(audio)
        assert result.shape == (128,)  # Default n_mels

    def test_learnable_projection(self):
        """Projection layer is learnable."""
        embedder = MelSpectrogramEmbedding(n_mels=80, projection_dim=256)

        # Check projection has parameters
        assert hasattr(embedder, "projection")
        assert embedder.projection is not None
        assert sum(p.numel() for p in embedder.projection.parameters()) > 0

    def test_gradient_flow(self):
        """Gradients flow through the embedder."""
        embedder = MelSpectrogramEmbedding(n_mels=80, projection_dim=128)
        audio = torch.randn(16000, requires_grad=True)

        output = embedder(audio)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert audio.grad is not None
        for param in embedder.parameters():
            assert param.grad is not None

    def test_invalid_input_shape(self):
        """Invalid input shapes raise errors."""
        embedder = MelSpectrogramEmbedding()

        # 3D input should raise error
        audio = torch.randn(2, 2, 16000)
        with pytest.raises(ValueError, match="Expected 1D or 2D"):
            embedder(audio)

    def test_embedding_dimension_property(self):
        """Embedding dimension property returns correct value."""
        embedder1 = MelSpectrogramEmbedding(n_mels=80)
        assert embedder1.dim == 80

        embedder2 = MelSpectrogramEmbedding(n_mels=80, projection_dim=256)
        assert embedder2.dim == 256


class TestEmbedderRegistry:
    """Test embedder registry functionality."""

    def test_get_vggish_embedder(self):
        """Can retrieve VGGish embedder from registry."""
        with patch("associative.utils.audio.embeddings.torchvggish"):
            embedder = get_audio_embedder("vggish")
            assert isinstance(embedder, VGGishEmbedding)

    def test_get_wav2vec_embedder(self):
        """Can retrieve Wav2Vec2 embedder from registry."""
        with (
            patch("associative.utils.audio.embeddings.Wav2Vec2Model"),
            patch("associative.utils.audio.embeddings.Wav2Vec2Processor"),
        ):
            embedder = get_audio_embedder("wav2vec2", model_size="base")
            assert isinstance(embedder, Wav2Vec2Embedding)

    def test_get_mel_spectrogram_embedder(self):
        """Can retrieve Mel-spectrogram embedder from registry."""
        embedder = get_audio_embedder("mel_spectrogram", n_mels=128)
        assert isinstance(embedder, MelSpectrogramEmbedding)

    def test_invalid_embedder_name(self):
        """Invalid embedder name raises error."""
        with pytest.raises(KeyError, match="Unknown embedder"):
            get_audio_embedder("invalid_embedder")

    def test_embedder_with_kwargs(self):
        """Registry passes kwargs to embedder constructor."""
        embedder = get_audio_embedder(
            "mel_spectrogram", n_mels=80, projection_dim=256, aggregate="max"
        )
        assert embedder.dim == 256
        assert embedder.aggregate == "max"


class TestEmbedderIntegration:
    """Integration tests for embedder workflows."""

    def test_all_embedders_handle_variable_length_audio(self):
        """All embedders handle different audio lengths."""
        embedders = []

        # Create embedders (with mocked dependencies)
        with patch("associative.utils.audio.embeddings.torchvggish") as mock_vggish:
            mock_model = MagicMock()
            mock_model.return_value = torch.randn(5, 128)
            mock_vggish.vggish.return_value = mock_model
            mock_pca = MagicMock()
            mock_pca.side_effect = lambda x: x
            mock_vggish.PCA.return_value = mock_pca
            embedders.append(("vggish", VGGishEmbedding(aggregate="mean")))

        with (
            patch("associative.utils.audio.embeddings.Wav2Vec2Model") as mock_w2v_model,
            patch(
                "associative.utils.audio.embeddings.Wav2Vec2Processor"
            ) as mock_w2v_proc,
        ):
            mock_model = MagicMock()
            mock_output = MagicMock()
            mock_output.last_hidden_state = torch.randn(1, 50, 768)
            mock_model.return_value = mock_output
            mock_w2v_model.from_pretrained.return_value = mock_model
            mock_w2v_proc.from_pretrained.return_value = MagicMock()
            embedders.append(("wav2vec2", Wav2Vec2Embedding()))

        embedders.append(("mel", MelSpectrogramEmbedding()))

        # Test different audio lengths
        for _, embedder in embedders:
            short_audio = torch.randn(8000)  # 0.5 seconds
            long_audio = torch.randn(48000)  # 3 seconds

            short_result = embedder(short_audio)
            long_result = embedder(long_audio)

            # Both should produce valid embeddings
            assert short_result.shape[0] == embedder.dim
            assert long_result.shape[0] == embedder.dim

    def test_device_consistency(self):
        """Embedders maintain device consistency."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        embedder = MelSpectrogramEmbedding(n_mels=80, device=device)

        audio = torch.randn(16000).to(device)
        result = embedder(audio)

        assert result.device == device

    def test_dtype_consistency(self):
        """Embedders maintain dtype consistency."""
        embedder = MelSpectrogramEmbedding(n_mels=80, dtype=torch.float64)

        audio = torch.randn(16000, dtype=torch.float64)
        result = embedder(audio)

        assert result.dtype == torch.float64

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_batch_sizes(self, batch_size):
        """Different batch sizes work correctly."""
        embedder = MelSpectrogramEmbedding(n_mels=128, projection_dim=256)

        audio = torch.randn(batch_size, 16000)
        result = embedder(audio)

        assert result.shape == (batch_size, 256)

    def test_training_mode_behavior(self):
        """Training/eval mode works correctly."""
        embedder = MelSpectrogramEmbedding(n_mels=80, projection_dim=256)

        # Training mode
        embedder.train()
        assert embedder.training

        # Eval mode
        embedder.eval()
        assert not embedder.training

    def test_sequential_processing_consistency(self):
        """Processing audio sequentially gives same result as batch."""
        embedder = MelSpectrogramEmbedding(n_mels=80, projection_dim=256)

        # Create batch
        audio_batch = torch.randn(3, 16000)

        # Process as batch
        embedder.eval()  # Ensure deterministic behavior
        with torch.no_grad():
            batch_result = embedder(audio_batch)

        # Process sequentially
        sequential_results = []
        for i in range(3):
            with torch.no_grad():
                result = embedder(audio_batch[i])
                sequential_results.append(result)
        sequential_result = torch.stack(sequential_results)

        # Should be identical
        assert torch.allclose(batch_result, sequential_result, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

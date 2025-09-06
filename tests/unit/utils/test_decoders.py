"""Unit tests for decoder modules.

Tests focus on behavioral contracts and public API functionality.
Each decoder should:
1. Accept embeddings and produce correct output shapes
2. Handle various input configurations gracefully
3. Generate outputs in expected ranges
"""

import pytest
import torch

from associative.utils.decoders import (
    AudioDecoder,
    CrossModalDecoder,
    HierarchicalVideoDecoder,
    VideoDecoder,
    create_decoder,
)


class TestVideoDecoder:
    """Test video decoder behavior."""

    def test_basic_video_reconstruction(self):
        """Test basic video frame reconstruction."""
        decoder = VideoDecoder(
            input_dim=768,
            num_frames=16,
            height=224,
            width=224,
            channels=3,
        )

        # Test with frame embeddings
        embeddings = torch.randn(2, 16, 768)
        output = decoder(embeddings)

        assert output.shape == (2, 16, 3, 224, 224)
        assert -1.5 <= output.min() <= output.max() <= 1.5

    def test_global_embedding_expansion(self):
        """Test handling of global embeddings."""
        decoder = VideoDecoder(
            input_dim=256,
            num_frames=8,
            height=64,
            width=64,
        )

        # Global embedding (no frame dimension)
        embeddings = torch.randn(4, 256)
        output = decoder(embeddings)

        assert output.shape == (4, 8, 3, 64, 64)

    def test_different_hidden_dimensions(self):
        """Test decoder with custom hidden dimensions."""
        decoder = VideoDecoder(
            input_dim=128,
            num_frames=4,
            height=32,
            width=32,
            hidden_dims=[256, 128, 64],
        )

        embeddings = torch.randn(1, 4, 128)
        output = decoder(embeddings)

        assert output.shape == (1, 4, 3, 32, 32)

    def test_grayscale_output(self):
        """Test single channel (grayscale) output."""
        decoder = VideoDecoder(
            input_dim=128,
            num_frames=8,
            height=128,
            width=128,
            channels=1,
        )

        embeddings = torch.randn(2, 8, 128)
        output = decoder(embeddings)

        assert output.shape == (2, 8, 1, 128, 128)

    def test_parameter_count(self):
        """Test parameter counting."""
        decoder = VideoDecoder(
            input_dim=64,
            num_frames=2,
            height=16,
            width=16,
        )

        param_count = decoder.num_parameters
        assert param_count > 0

        # Verify manual count matches
        manual_count = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        assert param_count == manual_count

    def test_different_aspect_ratios(self):
        """Test non-square video frames."""
        decoder = VideoDecoder(
            input_dim=256,
            num_frames=4,
            height=240,  # 16:9 aspect ratio
            width=426,
        )

        embeddings = torch.randn(1, 4, 256)
        output = decoder(embeddings)

        assert output.shape == (1, 4, 3, 240, 426)

    @pytest.mark.parametrize("num_frames", [1, 4, 16, 32])
    def test_variable_frame_counts(self, num_frames):
        """Test decoder with different frame counts."""
        decoder = VideoDecoder(
            input_dim=128,
            num_frames=num_frames,
            height=32,
            width=32,
        )

        embeddings = torch.randn(2, num_frames, 128)
        output = decoder(embeddings)

        assert output.shape == (2, num_frames, 3, 32, 32)

    def test_gradient_flow(self):
        """Test gradients flow through decoder."""
        decoder = VideoDecoder(
            input_dim=64,
            num_frames=2,
            height=16,
            width=16,
        )

        embeddings = torch.randn(1, 2, 64, requires_grad=True)
        output = decoder(embeddings)
        loss = output.mean()
        loss.backward()

        assert embeddings.grad is not None
        assert not torch.isnan(embeddings.grad).any()


class TestAudioDecoder:
    """Test audio decoder behavior."""

    def test_basic_audio_reconstruction(self):
        """Test basic audio waveform reconstruction."""
        decoder = AudioDecoder(
            input_dim=128,
            sample_rate=16000,
            duration=2.0,
            channels=1,
        )

        embeddings = torch.randn(2, 128)
        output = decoder(embeddings)

        expected_samples = 32000  # 16000 * 2.0
        assert output.shape == (2, expected_samples)
        assert -1.5 <= output.min() <= output.max() <= 1.5

    def test_stereo_audio(self):
        """Test stereo audio generation."""
        decoder = AudioDecoder(
            input_dim=256,
            sample_rate=22050,
            duration=1.5,
            channels=2,
        )

        embeddings = torch.randn(2, 256)
        output = decoder(embeddings)

        expected_samples = int(22050 * 1.5)
        assert output.shape == (2, 2, expected_samples)

    def test_frame_embedding_averaging(self):
        """Test handling of frame-wise embeddings."""
        decoder = AudioDecoder(
            input_dim=64,
            sample_rate=8000,
            duration=1.0,
        )

        # Frame embeddings - should be averaged
        frame_embeddings = torch.randn(2, 10, 64)
        output = decoder(frame_embeddings)

        assert output.shape == (2, 8000)

    def test_different_sample_rates(self):
        """Test various sample rates."""
        for sample_rate in [8000, 16000, 22050, 44100, 48000]:
            decoder = AudioDecoder(
                input_dim=128,
                sample_rate=sample_rate,
                duration=0.5,
            )

            embeddings = torch.randn(1, 128)
            output = decoder(embeddings)

            expected_samples = int(sample_rate * 0.5)
            assert output.shape == (1, expected_samples)

    def test_hidden_dimension_configuration(self):
        """Test custom hidden dimensions."""
        decoder = AudioDecoder(
            input_dim=64,
            sample_rate=16000,
            duration=1.0,
            hidden_dims=[128, 256, 128],
        )

        embeddings = torch.randn(2, 64)
        output = decoder(embeddings)

        assert output.shape == (2, 16000)

    def test_very_short_audio(self):
        """Test generation of very short audio clips."""
        decoder = AudioDecoder(
            input_dim=128,
            sample_rate=16000,
            duration=0.01,  # 10ms
        )

        embeddings = torch.randn(4, 128)
        output = decoder(embeddings)

        expected_samples = 160  # 16000 * 0.01
        assert output.shape == (4, expected_samples)

    def test_parameter_count(self):
        """Test parameter counting."""
        decoder = AudioDecoder(
            input_dim=64,
            sample_rate=8000,
            duration=1.0,
        )

        param_count = decoder.num_parameters
        assert param_count > 0

        manual_count = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        assert param_count == manual_count

    def test_gradient_flow(self):
        """Test gradients flow properly."""
        decoder = AudioDecoder(
            input_dim=32,
            sample_rate=8000,
            duration=0.5,
        )

        embeddings = torch.randn(2, 32, requires_grad=True)
        output = decoder(embeddings)
        loss = output.mean()
        loss.backward()

        assert embeddings.grad is not None
        assert not torch.isnan(embeddings.grad).any()


class TestCrossModalDecoder:
    """Test cross-modal decoder behavior."""

    def test_video_to_audio_translation(self):
        """Test translating video embeddings to audio."""
        decoder = CrossModalDecoder(
            input_dim=768,
            output_shape=(16000,),  # 1 second audio
            source_modality="video",
            target_modality="audio",
        )

        video_embeddings = torch.randn(2, 768)
        audio_output = decoder(video_embeddings)

        assert audio_output.shape == (2, 16000)
        assert -1.5 <= audio_output.min() <= audio_output.max() <= 1.5

    def test_audio_to_video_translation(self):
        """Test translating audio embeddings to video."""
        decoder = CrossModalDecoder(
            input_dim=512,
            output_shape=(3, 64, 64),  # Single frame
            source_modality="audio",
            target_modality="video",
        )

        audio_embeddings = torch.randn(2, 512)
        video_output = decoder(audio_embeddings)

        assert video_output.shape == (2, 3, 64, 64)

    def test_text_modality_translation(self):
        """Test text as source or target modality."""
        # Text to video
        decoder1 = CrossModalDecoder(
            input_dim=256,
            output_shape=(3, 32, 32),
            source_modality="text",
            target_modality="video",
        )

        text_embeddings = torch.randn(2, 256)
        video_output = decoder1(text_embeddings)
        assert video_output.shape == (2, 3, 32, 32)

        # Audio to text (token logits)
        decoder2 = CrossModalDecoder(
            input_dim=128,
            output_shape=(100,),  # Vocabulary size
            source_modality="audio",
            target_modality="text",
        )

        audio_embeddings = torch.randn(2, 128)
        text_output = decoder2(audio_embeddings)
        assert text_output.shape == (2, 100)

    def test_same_modality_raises_error(self):
        """Test that same source and target modality raises error."""
        with pytest.raises(ValueError, match="must be different"):
            CrossModalDecoder(
                input_dim=256,
                output_shape=(100,),
                source_modality="video",
                target_modality="video",
            )

    def test_bridge_dimension_configuration(self):
        """Test configurable bridge dimension."""
        decoder = CrossModalDecoder(
            input_dim=512,
            output_shape=(8000,),
            source_modality="video",
            target_modality="audio",
            bridge_dim=256,
        )

        embeddings = torch.randn(2, 512)
        output = decoder(embeddings)

        assert output.shape == (2, 8000)
        # Check bridge dimension is used internally
        assert any(
            p.shape[0] == 256 or p.shape[1] == 256
            for p in decoder.parameters()
            if len(p.shape) >= 2
        )

    def test_parameter_count(self):
        """Test parameter counting."""
        decoder = CrossModalDecoder(
            input_dim=128,
            output_shape=(64,),
            source_modality="audio",
            target_modality="text",
        )

        param_count = decoder.num_parameters
        assert param_count > 0

        manual_count = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        assert param_count == manual_count

    @pytest.mark.parametrize(
        "source,target",
        [
            ("video", "audio"),
            ("audio", "video"),
            ("text", "video"),
            ("video", "text"),
            ("text", "audio"),
            ("audio", "text"),
        ],
    )
    def test_all_modality_pairs(self, source, target):
        """Test all valid modality combinations."""
        decoder = CrossModalDecoder(
            input_dim=128,
            output_shape=(100,) if target == "text" else (64, 64),
            source_modality=source,
            target_modality=target,
        )

        embeddings = torch.randn(2, 128)
        output = decoder(embeddings)

        # Check output shape is correct
        if target == "text":
            assert output.shape == (2, 100)
        else:
            assert len(output.shape) >= 3  # Has spatial dimensions

    def test_gradient_flow(self):
        """Test gradients flow through cross-modal translation."""
        decoder = CrossModalDecoder(
            input_dim=64,
            output_shape=(100,),
            source_modality="video",
            target_modality="audio",
        )

        embeddings = torch.randn(2, 64, requires_grad=True)
        output = decoder(embeddings)
        loss = output.mean()
        loss.backward()

        assert embeddings.grad is not None
        assert not torch.isnan(embeddings.grad).any()


class TestHierarchicalVideoDecoder:
    """Test hierarchical video decoder behavior."""

    def test_progressive_resolution_generation(self):
        """Test generation at multiple resolutions."""
        decoder = HierarchicalVideoDecoder(
            input_dim=256,
            num_frames=4,
            height=64,
            width=64,
            num_levels=3,
        )

        embeddings = torch.randn(2, 256)
        output = decoder(embeddings)

        # Final output should be at full resolution
        assert output.shape == (2, 4, 3, 64, 64)

    def test_single_level_decoder(self):
        """Test decoder with single level (no hierarchy)."""
        decoder = HierarchicalVideoDecoder(
            input_dim=128,
            num_frames=2,
            height=32,
            width=32,
            num_levels=1,
        )

        embeddings = torch.randn(2, 128)
        output = decoder(embeddings)

        assert output.shape == (2, 2, 3, 32, 32)

    def test_custom_scale_factors(self):
        """Test with custom upsampling factors."""
        decoder = HierarchicalVideoDecoder(
            input_dim=256,
            num_frames=4,
            height=128,
            width=128,
            num_levels=3,
            scale_factors=[4, 2, 2],  # 8x8 -> 32x32 -> 64x64 -> 128x128
        )

        embeddings = torch.randn(1, 256)
        output = decoder(embeddings)

        assert output.shape == (1, 4, 3, 128, 128)

    def test_return_all_levels(self):
        """Test returning outputs from all hierarchy levels."""
        decoder = HierarchicalVideoDecoder(
            input_dim=128,
            num_frames=2,
            height=64,
            width=64,
            num_levels=3,
        )

        embeddings = torch.randn(1, 128)
        all_outputs = decoder(embeddings, return_all_levels=True)

        assert isinstance(all_outputs, list)
        assert len(all_outputs) == 3

        # Check progressive resolution increase
        prev_size = 0
        for level_output in all_outputs:
            assert level_output.shape[0] == 1  # Batch size
            assert level_output.shape[1] == 2  # Num frames
            assert level_output.shape[2] == 3  # Channels
            # Resolution should increase
            current_size = level_output.shape[3] * level_output.shape[4]
            assert current_size > prev_size
            prev_size = current_size

    def test_different_fusion_methods(self):
        """Test different fusion strategies between levels."""
        from typing import Literal

        fusion_methods: list[Literal["add", "concat", "attention"]] = [
            "add",
            "concat",
            "attention",
        ]
        for fusion in fusion_methods:
            decoder = HierarchicalVideoDecoder(
                input_dim=128,
                num_frames=2,
                height=32,
                width=32,
                num_levels=2,
                fusion_method=fusion,
            )

            embeddings = torch.randn(2, 128)
            output = decoder(embeddings)

            assert output.shape == (2, 2, 3, 32, 32)

    def test_grayscale_hierarchical(self):
        """Test hierarchical generation of grayscale video."""
        decoder = HierarchicalVideoDecoder(
            input_dim=256,
            num_frames=8,
            height=128,
            width=128,
            channels=1,
            num_levels=4,
        )

        embeddings = torch.randn(2, 256)
        output = decoder(embeddings)

        assert output.shape == (2, 8, 1, 128, 128)

    def test_parameter_count(self):
        """Test parameter counting across all levels."""
        decoder = HierarchicalVideoDecoder(
            input_dim=64,
            num_frames=2,
            height=32,
            width=32,
            num_levels=2,
        )

        param_count = decoder.num_parameters
        assert param_count > 0

        manual_count = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        assert param_count == manual_count

    def test_gradient_flow_through_levels(self):
        """Test gradients flow through hierarchical structure."""
        decoder = HierarchicalVideoDecoder(
            input_dim=64,
            num_frames=2,
            height=32,
            width=32,
            num_levels=3,
        )

        embeddings = torch.randn(1, 64, requires_grad=True)
        output = decoder(embeddings)
        loss = output.mean()
        loss.backward()

        assert embeddings.grad is not None
        assert not torch.isnan(embeddings.grad).any()


class TestDecoderFactory:
    """Test decoder factory function."""

    def test_create_video_decoder(self):
        """Test creating video decoder via factory."""
        decoder = create_decoder(
            "video",
            input_dim=256,
            output_shape=(4, 3, 64, 64),  # (frames, channels, H, W)
        )

        assert isinstance(decoder, VideoDecoder)
        embeddings = torch.randn(2, 256)
        output = decoder(embeddings)
        assert output.shape == (2, 4, 3, 64, 64)

    def test_create_audio_decoder(self):
        """Test creating audio decoder via factory."""
        decoder = create_decoder(
            "audio",
            input_dim=128,
            output_shape=(16000,),  # 1 second at 16kHz
        )

        assert isinstance(decoder, AudioDecoder)
        embeddings = torch.randn(2, 128)
        output = decoder(embeddings)
        assert output.shape == (2, 16000)

    def test_create_cross_modal_decoder(self):
        """Test creating cross-modal decoder via factory."""
        decoder = create_decoder(
            "cross_modal",
            input_dim=256,
            output_shape=(8000,),
            source_modality="video",
            target_modality="audio",
        )

        assert isinstance(decoder, CrossModalDecoder)
        embeddings = torch.randn(2, 256)
        output = decoder(embeddings)
        assert output.shape == (2, 8000)

    def test_create_hierarchical_decoder(self):
        """Test creating hierarchical decoder via factory."""
        decoder = create_decoder(
            "hierarchical_video",
            input_dim=128,
            output_shape=(4, 3, 64, 64),
            num_levels=3,
        )

        assert isinstance(decoder, HierarchicalVideoDecoder)
        embeddings = torch.randn(2, 128)
        output = decoder(embeddings)
        assert output.shape == (2, 4, 3, 64, 64)

    def test_invalid_decoder_type(self):
        """Test that invalid decoder type raises error."""
        with pytest.raises(ValueError, match="Unknown decoder type"):
            create_decoder(
                "invalid_type",
                input_dim=128,
                output_shape=(100,),
            )

    def test_factory_with_device(self):
        """Test creating decoder with specific device."""
        device = torch.device("cpu")
        decoder = create_decoder(
            "video",
            input_dim=128,
            output_shape=(2, 3, 32, 32),
            device=device,
        )

        # Check parameters are on correct device
        for param in decoder.parameters():
            assert param.device == device

    def test_factory_with_dtype(self):
        """Test creating decoder with specific dtype."""
        decoder = create_decoder(
            "audio",
            input_dim=64,
            output_shape=(8000,),
            dtype=torch.float64,
        )

        # Check parameters have correct dtype
        for param in decoder.parameters():
            assert param.dtype == torch.float64


class TestDecoderIntegration:
    """Integration tests for decoder workflows."""

    def test_encoder_decoder_pipeline(self):
        """Test full encoding-decoding pipeline."""
        # Simulate encoder output
        encoder_dim = 512
        batch_size = 4

        # Create decoder
        decoder = VideoDecoder(
            input_dim=encoder_dim,
            num_frames=8,
            height=64,
            width=64,
        )

        # Simulate encoded features
        encoded = torch.randn(batch_size, encoder_dim)

        # Decode
        reconstructed = decoder(encoded)

        assert reconstructed.shape == (batch_size, 8, 3, 64, 64)
        assert torch.isfinite(reconstructed).all()

    def test_multi_modal_pipeline(self):
        """Test cross-modal encoding and decoding."""
        # Video encoder output
        video_features = torch.randn(2, 768)

        # Video to audio decoder
        v2a_decoder = CrossModalDecoder(
            input_dim=768,
            output_shape=(16000,),
            source_modality="video",
            target_modality="audio",
        )

        # Generate audio from video
        audio = v2a_decoder(video_features)
        assert audio.shape == (2, 16000)

        # Audio to video decoder (inverse)
        a2v_decoder = CrossModalDecoder(
            input_dim=768,
            output_shape=(3, 64, 64),
            source_modality="audio",
            target_modality="video",
        )

        # Simulate audio encoder output
        audio_features = torch.randn(2, 768)
        video = a2v_decoder(audio_features)
        assert video.shape == (2, 3, 64, 64)

    def test_hierarchical_with_loss_at_each_level(self):
        """Test hierarchical decoder with multi-scale supervision."""
        decoder = HierarchicalVideoDecoder(
            input_dim=256,
            num_frames=4,
            height=64,
            width=64,
            num_levels=3,
        )

        embeddings = torch.randn(2, 256, requires_grad=True)
        all_outputs = decoder(embeddings, return_all_levels=True)

        # Compute loss at each level (simulating multi-scale supervision)
        total_loss = torch.tensor(0.0, requires_grad=True)
        for level_output in all_outputs:
            level_loss = level_output.mean()  # Dummy loss
            total_loss = total_loss + level_loss

        total_loss.backward()

        assert embeddings.grad is not None
        assert not torch.isnan(embeddings.grad).any()

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_different_batch_sizes(self, batch_size):
        """Test decoders handle various batch sizes."""
        decoder = VideoDecoder(
            input_dim=128,
            num_frames=4,
            height=32,
            width=32,
        )

        embeddings = torch.randn(batch_size, 128)
        output = decoder(embeddings)

        assert output.shape == (batch_size, 4, 3, 32, 32)

    def test_mixed_precision_compatibility(self):
        """Test decoder works with different precisions."""
        decoder = AudioDecoder(
            input_dim=128,
            sample_rate=16000,
            duration=1.0,
            dtype=torch.float16,
        )

        embeddings = torch.randn(2, 128, dtype=torch.float16)
        output = decoder(embeddings)

        assert output.dtype == torch.float16
        assert output.shape == (2, 16000)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

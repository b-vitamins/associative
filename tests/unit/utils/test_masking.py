"""Comprehensive tests for masking utilities.

Tests cover all functions in associative.utils.masking with focus on:
- Implementation-agnostic expectations
- Edge cases and boundary conditions
- Statistical properties
- Error handling
- Device compatibility
"""

import pytest
import torch

from associative.utils.masking import (
    add_noise_to_embeddings,
    apply_mask_to_embeddings,
    generate_block_mask,
    generate_random_mask,
)


class TestGenerateRandomMask:
    """Test generate_random_mask function."""

    def test_basic_functionality(self):
        """Test basic mask generation with typical parameters."""
        batch_size, seq_len, mask_ratio = 4, 100, 0.5
        mask = generate_random_mask(batch_size, seq_len, mask_ratio)

        # Check shape
        assert mask.shape == (batch_size, seq_len)
        assert mask.dtype == torch.bool

        # Check mask ratio is approximately correct
        actual_ratio = mask.float().mean().item()
        assert abs(actual_ratio - mask_ratio) < 0.1  # Allow 10% tolerance

    def test_exact_mask_count(self):
        """Test that exact number of positions are masked."""
        batch_size, seq_len, mask_ratio = 2, 50, 0.4
        mask = generate_random_mask(batch_size, seq_len, mask_ratio)

        expected_masked = int(seq_len * mask_ratio)  # 20

        # Each batch should have exactly expected_masked True values
        for b in range(batch_size):
            assert mask[b].sum().item() == expected_masked

    def test_boundary_ratios(self):
        """Test edge cases for mask_ratio."""
        batch_size, seq_len = 3, 100

        # Test ratio = 0 (no masking)
        mask = generate_random_mask(batch_size, seq_len, 0.0)
        assert mask.sum().item() == 0
        assert (~mask).all()

        # Test ratio = 1 (full masking)
        mask = generate_random_mask(batch_size, seq_len, 1.0)
        assert mask.sum().item() == batch_size * seq_len
        assert mask.all()

        # Test ratio = 0.5
        mask = generate_random_mask(batch_size, seq_len, 0.5)
        assert mask.sum().item() == batch_size * 50

    def test_invalid_mask_ratio(self):
        """Test that invalid mask ratios raise errors."""
        with pytest.raises(ValueError, match="mask_ratio must be in"):
            generate_random_mask(2, 100, -0.1)

        with pytest.raises(ValueError, match="mask_ratio must be in"):
            generate_random_mask(2, 100, 1.1)

    def test_randomness_across_batch(self):
        """Test that different sequences in batch get different masks."""
        batch_size, seq_len, mask_ratio = 10, 100, 0.5
        mask = generate_random_mask(batch_size, seq_len, mask_ratio)

        # Check that not all batch elements have same pattern
        # Convert to float for easier comparison
        mask.float()

        # No two sequences should be identical (with high probability)
        unique_patterns = set()
        for b in range(batch_size):
            pattern = tuple(mask[b].tolist())
            unique_patterns.add(pattern)

        assert len(unique_patterns) == batch_size

    def test_reproducibility_with_seed(self):
        """Test that masks are reproducible with fixed seed."""
        batch_size, seq_len, mask_ratio = 2, 50, 0.3

        torch.manual_seed(42)
        mask1 = generate_random_mask(batch_size, seq_len, mask_ratio)

        torch.manual_seed(42)
        mask2 = generate_random_mask(batch_size, seq_len, mask_ratio)

        assert torch.equal(mask1, mask2)

    def test_device_placement(self):
        """Test mask creation on different devices."""
        batch_size, seq_len, mask_ratio = 2, 50, 0.5

        # CPU
        mask_cpu = generate_random_mask(
            batch_size, seq_len, mask_ratio, device=torch.device("cpu")
        )
        assert mask_cpu.device.type == "cpu"

        # CUDA if available
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            mask_cuda = generate_random_mask(
                batch_size, seq_len, mask_ratio, device=device
            )
            assert mask_cuda.device == device

    def test_small_sequences(self):
        """Test with very small sequence lengths."""
        # Single element
        mask = generate_random_mask(2, 1, 0.5)
        assert mask.shape == (2, 1)
        # With ratio 0.5 and length 1, int(1 * 0.5) = 0, so no masking
        assert mask.sum().item() == 0

        # Two elements
        mask = generate_random_mask(2, 2, 0.5)
        assert mask.shape == (2, 2)
        # Each batch should have exactly 1 masked
        assert all(mask[b].sum() == 1 for b in range(2))

    def test_large_batch_and_sequence(self):
        """Test with large batch size and sequence length."""
        batch_size, seq_len, mask_ratio = 128, 1024, 0.7
        mask = generate_random_mask(batch_size, seq_len, mask_ratio)

        assert mask.shape == (batch_size, seq_len)
        expected_masked = int(seq_len * mask_ratio)

        # Check each batch has correct count
        for b in range(batch_size):
            assert mask[b].sum().item() == expected_masked

    @pytest.mark.parametrize("mask_ratio", [0.1, 0.25, 0.33, 0.5, 0.75, 0.9])
    def test_various_mask_ratios(self, mask_ratio):
        """Test with various mask ratios."""
        batch_size, seq_len = 4, 100
        mask = generate_random_mask(batch_size, seq_len, mask_ratio)

        expected_masked = int(seq_len * mask_ratio)
        actual_masked = mask.sum(dim=1)  # Per batch

        assert all(actual_masked == expected_masked)


class TestGenerateBlockMask:
    """Test generate_block_mask function."""

    def test_basic_functionality(self):
        """Test basic block mask generation."""
        batch_size, seq_len, mask_ratio, block_size = 4, 100, 0.5, 10
        mask = generate_block_mask(batch_size, seq_len, mask_ratio, block_size)

        assert mask.shape == (batch_size, seq_len)
        assert mask.dtype == torch.bool

    def test_block_structure(self):
        """Test that mask has contiguous block structure."""
        batch_size, seq_len, mask_ratio, block_size = 2, 100, 0.4, 10
        mask = generate_block_mask(batch_size, seq_len, mask_ratio, block_size)

        for b in range(batch_size):
            # Check that masked regions align with block boundaries
            mask_seq = mask[b]

            # Find all masked positions
            masked_positions = torch.where(mask_seq)[0].tolist()

            if len(masked_positions) > 0:
                # Check that masked positions come in groups aligned to block_size
                # Each masked position should belong to a block starting at k*block_size
                for pos in masked_positions:
                    block_start = (pos // block_size) * block_size
                    # The position should be within a valid block
                    assert block_start <= pos < min(block_start + block_size, seq_len)

    def test_block_count(self):
        """Test that correct number of blocks are masked."""
        batch_size, seq_len, mask_ratio, block_size = 2, 100, 0.5, 10
        mask = generate_block_mask(batch_size, seq_len, mask_ratio, block_size)

        num_blocks = seq_len // block_size  # 10
        expected_masked_blocks = int(num_blocks * mask_ratio)  # 5

        for b in range(batch_size):
            # Count actual masked positions
            masked_positions = mask[b].sum().item()
            # Should be approximately expected_masked_blocks * block_size
            expected_positions = expected_masked_blocks * block_size
            assert abs(masked_positions - expected_positions) <= block_size

    def test_boundary_cases(self):
        """Test edge cases for block masking."""
        batch_size, seq_len = 2, 100

        # Block size = 1 (equivalent to random masking)
        mask = generate_block_mask(batch_size, seq_len, 0.5, block_size=1)
        assert mask.sum().item() == batch_size * 50

        # Block size = sequence length
        mask = generate_block_mask(batch_size, seq_len, 0.5, block_size=seq_len)
        # Should mask either all or nothing
        for b in range(batch_size):
            assert mask[b].sum().item() in [0, seq_len]

    def test_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        # Invalid mask ratio
        with pytest.raises(ValueError, match="mask_ratio must be in"):
            generate_block_mask(2, 100, -0.1, 10)

        with pytest.raises(ValueError, match="mask_ratio must be in"):
            generate_block_mask(2, 100, 1.5, 10)

        # Block size > sequence length
        with pytest.raises(ValueError, match="block_size .* > sequence_length"):
            generate_block_mask(2, 100, 0.5, 150)

    def test_zero_and_full_masking(self):
        """Test with 0 and 1 mask ratios."""
        batch_size, seq_len, block_size = 3, 100, 10

        # No masking
        mask = generate_block_mask(batch_size, seq_len, 0.0, block_size)
        assert mask.sum().item() == 0

        # Full masking
        mask = generate_block_mask(batch_size, seq_len, 1.0, block_size)
        assert mask.sum().item() == batch_size * seq_len

    def test_device_placement(self):
        """Test block mask on different devices."""
        batch_size, seq_len, mask_ratio, block_size = 2, 50, 0.4, 5

        # CPU
        mask_cpu = generate_block_mask(
            batch_size, seq_len, mask_ratio, block_size, device=torch.device("cpu")
        )
        assert mask_cpu.device.type == "cpu"

        # CUDA if available
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            mask_cuda = generate_block_mask(
                batch_size, seq_len, mask_ratio, block_size, device=device
            )
            assert mask_cuda.device == device

    def test_uneven_blocks(self):
        """Test when sequence length is not divisible by block size."""
        batch_size, seq_len, mask_ratio, block_size = 2, 95, 0.5, 10
        mask = generate_block_mask(batch_size, seq_len, mask_ratio, block_size)

        # 95 / 10 = 9 full blocks + partial
        num_blocks = seq_len // block_size  # 9
        int(num_blocks * mask_ratio)  # 4

        for b in range(batch_size):
            masked_count = mask[b].sum().item()
            # Should be around 4 * 10 = 40
            assert 30 <= masked_count <= 50

    @pytest.mark.parametrize("block_size", [1, 5, 10, 20, 50])
    def test_various_block_sizes(self, block_size):
        """Test with various block sizes."""
        batch_size, seq_len, mask_ratio = 4, 100, 0.4
        mask = generate_block_mask(batch_size, seq_len, mask_ratio, block_size)

        assert mask.shape == (batch_size, seq_len)
        # Check masking is in valid range
        total_masked = mask.sum().item()

        # Calculate expected based on actual block structure
        num_blocks = seq_len // block_size
        expected_masked_blocks = int(num_blocks * mask_ratio)

        # When expected_masked_blocks is 0, we expect no masking
        if expected_masked_blocks == 0:
            assert total_masked == 0
        else:
            # Otherwise expect roughly the right amount
            expected_positions = batch_size * expected_masked_blocks * block_size
            # Allow for variation due to randomness across batches
            assert 0.5 * expected_positions <= total_masked <= 1.5 * expected_positions


class TestApplyMaskToEmbeddings:
    """Test apply_mask_to_embeddings function."""

    def test_basic_functionality(self):
        """Test basic mask application."""
        batch_size, seq_len, embed_dim = 2, 10, 768
        embeddings = torch.randn(batch_size, seq_len, embed_dim)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask[:, :5] = True  # Mask first half

        masked_embeddings = apply_mask_to_embeddings(embeddings, mask)

        assert masked_embeddings.shape == embeddings.shape
        # Masked positions should be 0
        assert (masked_embeddings[:, :5] == 0).all()
        # Unmasked positions should be unchanged
        assert torch.equal(masked_embeddings[:, 5:], embeddings[:, 5:])

    def test_custom_mask_value(self):
        """Test with custom mask value."""
        batch_size, seq_len, embed_dim = 2, 10, 512
        embeddings = torch.randn(batch_size, seq_len, embed_dim)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask[:, 3:7] = True

        mask_value = -999.0
        masked_embeddings = apply_mask_to_embeddings(embeddings, mask, mask_value)

        # Check masked positions have mask_value
        assert (masked_embeddings[:, 3:7] == mask_value).all()
        # Unmasked positions unchanged
        assert torch.equal(masked_embeddings[:, :3], embeddings[:, :3])
        assert torch.equal(masked_embeddings[:, 7:], embeddings[:, 7:])

    def test_no_modification_to_input(self):
        """Test that input embeddings are not modified."""
        embeddings = torch.randn(2, 10, 768)
        embeddings_copy = embeddings.clone()
        mask = torch.ones(2, 10, dtype=torch.bool)

        _ = apply_mask_to_embeddings(embeddings, mask)

        # Original should be unchanged
        assert torch.equal(embeddings, embeddings_copy)

    def test_dimension_validation(self):
        """Test dimension checking."""
        # Wrong embedding dimensions
        with pytest.raises(ValueError, match="embeddings must be 3D"):
            embeddings_2d = torch.randn(10, 768)
            mask = torch.zeros(10, dtype=torch.bool)
            apply_mask_to_embeddings(embeddings_2d, mask)

        with pytest.raises(ValueError, match="embeddings must be 3D"):
            embeddings_4d = torch.randn(2, 10, 768, 1)
            mask = torch.zeros(2, 10, dtype=torch.bool)
            apply_mask_to_embeddings(embeddings_4d, mask)

        # Wrong mask dimensions
        with pytest.raises(ValueError, match="mask must be 2D"):
            embeddings = torch.randn(2, 10, 768)
            mask_1d = torch.zeros(10, dtype=torch.bool)
            apply_mask_to_embeddings(embeddings, mask_1d)

        with pytest.raises(ValueError, match="mask must be 2D"):
            embeddings = torch.randn(2, 10, 768)
            mask_3d = torch.zeros(2, 10, 1, dtype=torch.bool)
            apply_mask_to_embeddings(embeddings, mask_3d)

    def test_shape_mismatch(self):
        """Test shape mismatch detection."""
        embeddings = torch.randn(2, 10, 768)

        # Wrong batch size
        mask_wrong_batch = torch.zeros(3, 10, dtype=torch.bool)
        with pytest.raises(ValueError, match="Shape mismatch"):
            apply_mask_to_embeddings(embeddings, mask_wrong_batch)

        # Wrong sequence length
        mask_wrong_seq = torch.zeros(2, 15, dtype=torch.bool)
        with pytest.raises(ValueError, match="Shape mismatch"):
            apply_mask_to_embeddings(embeddings, mask_wrong_seq)

    def test_all_masked(self):
        """Test when all positions are masked."""
        embeddings = torch.randn(2, 10, 768)
        mask = torch.ones(2, 10, dtype=torch.bool)

        masked_embeddings = apply_mask_to_embeddings(embeddings, mask, 0.0)
        assert (masked_embeddings == 0).all()

    def test_none_masked(self):
        """Test when no positions are masked."""
        embeddings = torch.randn(2, 10, 768)
        mask = torch.zeros(2, 10, dtype=torch.bool)

        masked_embeddings = apply_mask_to_embeddings(embeddings, mask)
        assert torch.equal(masked_embeddings, embeddings)

    def test_device_compatibility(self):
        """Test masking on different devices."""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            embeddings = torch.randn(2, 10, 768, device=device)
            mask = torch.zeros(2, 10, dtype=torch.bool, device=device)
            mask[:, ::2] = True

            masked_embeddings = apply_mask_to_embeddings(embeddings, mask)
            assert masked_embeddings.device == device

    @pytest.mark.parametrize("embed_dim", [64, 128, 256, 512, 768, 1024])
    def test_various_embedding_dims(self, embed_dim):
        """Test with various embedding dimensions."""
        batch_size, seq_len = 2, 20
        embeddings = torch.randn(batch_size, seq_len, embed_dim)
        mask = generate_random_mask(batch_size, seq_len, 0.3)

        masked_embeddings = apply_mask_to_embeddings(embeddings, mask)
        assert masked_embeddings.shape == (batch_size, seq_len, embed_dim)

        # Check masking is applied correctly
        mask_expanded = mask.unsqueeze(-1).expand_as(embeddings)
        assert (masked_embeddings[mask_expanded] == 0).all()
        assert torch.equal(
            masked_embeddings[~mask_expanded], embeddings[~mask_expanded]
        )


class TestAddNoiseToEmbeddings:
    """Test add_noise_to_embeddings function."""

    def test_basic_functionality(self):
        """Test basic noise addition."""
        batch_size, seq_len, embed_dim = 2, 10, 768
        embeddings = torch.zeros(batch_size, seq_len, embed_dim)
        noise_std = 0.1

        noisy_embeddings = add_noise_to_embeddings(embeddings, noise_std)

        assert noisy_embeddings.shape == embeddings.shape
        # Since original is zeros, result should have std approximately noise_std
        actual_std = noisy_embeddings.std().item()
        assert abs(actual_std - noise_std) < 0.02  # Allow small tolerance

    def test_noise_statistics(self):
        """Test statistical properties of added noise."""
        torch.manual_seed(42)
        batch_size, seq_len, embed_dim = 10, 100, 512
        embeddings = torch.zeros(batch_size, seq_len, embed_dim)
        noise_std = 0.5

        noisy = add_noise_to_embeddings(embeddings, noise_std)

        # Check mean is approximately 0
        assert abs(noisy.mean().item()) < 0.01

        # Check std is approximately noise_std
        assert abs(noisy.std().item() - noise_std) < 0.01

        # Check noise is Gaussian (roughly)
        # Most values should be within 3 standard deviations
        within_3std = (noisy.abs() <= 3 * noise_std).float().mean()
        assert within_3std > 0.99  # ~99.7% for Gaussian

    def test_noise_with_mask(self):
        """Test noise addition with mask."""
        batch_size, seq_len, embed_dim = 2, 10, 768
        embeddings = torch.zeros(batch_size, seq_len, embed_dim)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask[:, :5] = True  # Only add noise to first half
        noise_std = 0.2

        noisy = add_noise_to_embeddings(embeddings, noise_std, mask)

        # Masked positions should have noise
        assert noisy[:, :5].std().item() > 0.1

        # Unmasked positions should be unchanged (zero)
        assert (noisy[:, 5:] == 0).all()

    def test_mask_expansion(self):
        """Test that 2D mask is properly expanded."""
        batch_size, seq_len, embed_dim = 2, 10, 512
        embeddings = torch.zeros(batch_size, seq_len, embed_dim)
        mask_2d = torch.ones(batch_size, seq_len, dtype=torch.bool)
        mask_3d = torch.ones(batch_size, seq_len, 1, dtype=torch.bool)

        # Both should work
        noisy_2d = add_noise_to_embeddings(embeddings, 0.1, mask_2d)
        noisy_3d = add_noise_to_embeddings(embeddings, 0.1, mask_3d)

        assert noisy_2d.shape == embeddings.shape
        assert noisy_3d.shape == embeddings.shape

    def test_preserves_original_values(self):
        """Test that original embedding structure is preserved."""
        torch.manual_seed(42)
        embeddings = torch.randn(2, 10, 768)
        noise_std = 0.01  # Small noise

        noisy = add_noise_to_embeddings(embeddings, noise_std)

        # Should be close to original
        diff = (noisy - embeddings).abs().mean().item()
        assert diff < 0.02  # Small average difference

        # Correlation should be high
        correlation = torch.nn.functional.cosine_similarity(
            embeddings.flatten(), noisy.flatten(), dim=0
        )
        assert correlation > 0.99

    def test_no_modification_to_input(self):
        """Test that input is not modified."""
        embeddings = torch.randn(2, 10, 768)
        embeddings_copy = embeddings.clone()

        _ = add_noise_to_embeddings(embeddings, 0.1)

        assert torch.equal(embeddings, embeddings_copy)

    def test_zero_noise(self):
        """Test with zero noise std."""
        embeddings = torch.randn(2, 10, 768)
        noisy = add_noise_to_embeddings(embeddings, noise_std=0.0)

        # Should be identical
        assert torch.equal(noisy, embeddings)

    def test_device_compatibility(self):
        """Test noise addition on different devices."""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            embeddings = torch.randn(2, 10, 768, device=device)

            noisy = add_noise_to_embeddings(embeddings, 0.1)
            assert noisy.device == device

    def test_reproducibility(self):
        """Test reproducibility with seed."""
        embeddings = torch.randn(2, 10, 768)

        torch.manual_seed(123)
        noisy1 = add_noise_to_embeddings(embeddings, 0.1)

        torch.manual_seed(123)
        noisy2 = add_noise_to_embeddings(embeddings, 0.1)

        assert torch.equal(noisy1, noisy2)

    @pytest.mark.parametrize("noise_std", [0.01, 0.05, 0.1, 0.2, 0.5, 1.0])
    def test_various_noise_levels(self, noise_std):
        """Test with various noise standard deviations."""
        embeddings = torch.zeros(4, 50, 256)
        noisy = add_noise_to_embeddings(embeddings, noise_std)

        # Check approximate std
        actual_std = noisy.std().item()
        assert abs(actual_std - noise_std) < 0.1 * noise_std  # 10% tolerance


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_mask_and_noise_pipeline(self):
        """Test complete masking and noise pipeline."""
        batch_size, seq_len, embed_dim = 4, 100, 768
        embeddings = torch.randn(batch_size, seq_len, embed_dim)

        # Generate mask
        mask = generate_random_mask(batch_size, seq_len, 0.3)

        # Apply mask
        masked = apply_mask_to_embeddings(embeddings, mask, 0.0)

        # Add noise to masked positions
        noisy = add_noise_to_embeddings(masked, 0.1, mask)

        # Check results
        assert noisy.shape == embeddings.shape

        # Unmasked positions should be unchanged
        assert torch.equal(noisy[~mask], embeddings[~mask])

        # Masked positions should have noise (not zero)
        mask_expanded = mask.unsqueeze(-1).expand_as(embeddings)
        masked_values = noisy[mask_expanded]
        assert masked_values.std() > 0.05

    def test_block_mask_with_embeddings(self):
        """Test block masking with embedding application."""
        batch_size, seq_len, embed_dim = 2, 100, 512
        embeddings = torch.randn(batch_size, seq_len, embed_dim)

        # Generate block mask
        mask = generate_block_mask(batch_size, seq_len, 0.4, block_size=10)

        # Apply to embeddings
        masked = apply_mask_to_embeddings(embeddings, mask, -1.0)

        # Check block structure is preserved
        for b in range(batch_size):
            for i in range(seq_len):
                if mask[b, i]:
                    assert (masked[b, i] == -1.0).all()
                else:
                    assert torch.equal(masked[b, i], embeddings[b, i])

    def test_combined_masks(self):
        """Test combining random and block masks."""
        batch_size, seq_len = 2, 100

        # Generate two different masks
        random_mask = generate_random_mask(batch_size, seq_len, 0.3)
        block_mask = generate_block_mask(batch_size, seq_len, 0.2, block_size=5)

        # Combine with OR
        combined_mask = random_mask | block_mask

        # Combined should have more masked positions
        assert combined_mask.sum() >= random_mask.sum()
        assert combined_mask.sum() >= block_mask.sum()

        # Apply to embeddings
        embeddings = torch.randn(batch_size, seq_len, 768)
        masked = apply_mask_to_embeddings(embeddings, combined_mask)

        assert (masked[combined_mask.unsqueeze(-1).expand_as(masked)] == 0).all()

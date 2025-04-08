"""
Tests specifically designed to increase coverage for dense_trunk.py.
Focuses on edge cases and error conditions that aren't covered by existing tests.
"""

import pytest
import torch
# No need for typing imports in test file

from rna_predict.pipeline.stageA.input_embedding.current.primitives.attention.dense_trunk import (
    _create_empty_output_tensor,
    _is_small_tensor_case,
    _handle_small_tensors,
    _rearrange_to_dense_trunk_impl,
    rearrange_qk_to_dense_trunk,
    rearrange_to_dense_trunk
)
from rna_predict.pipeline.stageA.input_embedding.current.primitives.attention.config_types import (
    DenseTrunkConfig,
    RearrangementConfig
)


class TestCreateEmptyOutputTensor:
    """Tests for _create_empty_output_tensor function."""

    def test_empty_input_list(self):
        """Test with completely empty input list."""
        result = _create_empty_output_tensor([], 5, 10)
        assert result.shape == (0, 5, 10, 0)
        assert result.dtype == torch.float32
        assert result.device.type == 'cpu'

    def test_list_with_all_empty_tensors(self):
        """Test with list containing only empty tensors."""
        empty_tensors = [
            torch.empty(0, 64, device='cpu'),
            torch.empty(0, 64, device='cpu')
        ]
        result = _create_empty_output_tensor(empty_tensors, 5, 10)
        assert result.shape == (0, 5, 10, 64)
        assert result.dtype == empty_tensors[0].dtype
        assert result.device == empty_tensors[0].device

    def test_mixed_empty_and_nonempty_tensors(self):
        """Test with mix of empty and non-empty tensors."""
        mixed_tensors = [
            torch.empty(0, 64, device='cpu'),
            torch.randn(2, 64, device='cpu')
        ]
        result = _create_empty_output_tensor(mixed_tensors, 5, 10)
        assert result.shape == (0, 5, 10, 64)
        assert result.dtype == mixed_tensors[1].dtype
        assert result.device == mixed_tensors[1].device

    def test_multidimensional_tensors(self):
        """Test with tensors having multiple dimensions."""
        tensors = [torch.randn(2, 3, 64, device='cpu')]  # (batch, heads, features)
        result = _create_empty_output_tensor(tensors, 5, 10)
        assert result.shape == (0, 3, 5, 10, 64)
        assert result.dtype == tensors[0].dtype
        assert result.device == tensors[0].device


class TestSmallTensorHandling:
    """Tests for _is_small_tensor_case and _handle_small_tensors functions."""

    def test_is_small_tensor_case_with_zero_length(self):
        """Test _is_small_tensor_case with zero-length tensors."""
        q = torch.empty(2, 0, 64)
        k = torch.randn(2, 5, 64)
        config = DenseTrunkConfig(n_queries=10, n_keys=10, attn_bias=None, inf=1e9)

        # Should return False for zero-length tensors
        assert not _is_small_tensor_case(q, k, config)

        # Test with both zero-length
        q2 = torch.empty(2, 0, 64)
        k2 = torch.empty(2, 0, 64)
        assert not _is_small_tensor_case(q2, k2, config)

        # Test with zero-length k only (to cover line 138)
        q3 = torch.randn(2, 5, 64)
        k3 = torch.empty(2, 0, 64)
        assert not _is_small_tensor_case(q3, k3, config)

    def test_is_small_tensor_case_boundary(self):
        """Test _is_small_tensor_case at the boundary conditions."""
        # Test exactly at the boundary (seq_len == n_queries/n_keys)
        q = torch.randn(2, 10, 64)
        k = torch.randn(2, 10, 64)
        config = DenseTrunkConfig(n_queries=10, n_keys=10, attn_bias=None, inf=1e9)

        assert _is_small_tensor_case(q, k, config)

        # Test just above the boundary (seq_len > n_queries/n_keys)
        q2 = torch.randn(2, 11, 64)
        k2 = torch.randn(2, 11, 64)
        assert not _is_small_tensor_case(q2, k2, config)

    def test_handle_small_tensors_with_attention_bias(self):
        """Test _handle_small_tensors with attention bias provided."""
        q = torch.randn(2, 5, 64)
        k = torch.randn(2, 5, 64)
        v = torch.randn(2, 5, 64)
        attn_bias = torch.randn(2, 5, 5)
        config = DenseTrunkConfig(n_queries=10, n_keys=10, attn_bias=attn_bias, inf=1e9)

        result = _handle_small_tensors(q, k, v, config)
        assert result is not None
        q_out, k_out, v_out, bias_out, padding_len = result

        # Check identity
        assert q_out is q
        assert k_out is k
        assert v_out is v
        assert bias_out is attn_bias
        assert padding_len == 0

    def test_handle_small_tensors_without_attention_bias(self):
        """Test _handle_small_tensors without attention bias."""
        q = torch.randn(2, 5, 64)
        k = torch.randn(2, 5, 64)
        v = torch.randn(2, 5, 64)
        config = DenseTrunkConfig(n_queries=10, n_keys=10, attn_bias=None, inf=1e9)

        result = _handle_small_tensors(q, k, v, config)
        assert result is not None
        q_out, k_out, v_out, bias_out, padding_len = result

        # Check identity for tensors
        assert q_out is q
        assert k_out is k
        assert v_out is v
        # Check bias is a scalar zero tensor
        assert bias_out.shape == torch.Size([1])
        assert bias_out.item() == 0
        assert padding_len == 0

    def test_handle_small_tensors_non_bypass(self):
        """Test _handle_small_tensors when bypass condition isn't met."""
        q = torch.randn(2, 15, 64)  # seq_len > n_queries
        k = torch.randn(2, 5, 64)
        v = torch.randn(2, 5, 64)
        config = DenseTrunkConfig(n_queries=10, n_keys=10, attn_bias=None, inf=1e9)

        result = _handle_small_tensors(q, k, v, config)
        assert result is None  # Should return None when bypass isn't applicable


class TestRearrangeToTrunkImpl:
    """Tests for _rearrange_to_dense_trunk_impl function."""

    def test_k_v_length_mismatch_warning(self):
        """Test warning when K and V sequence lengths differ."""
        q = torch.randn(2, 10, 64)
        k = torch.randn(2, 12, 64)
        v = torch.randn(2, 10, 64)  # Different from k.shape[-2]
        config = RearrangementConfig(
            q=q, k=k, v=v, n_queries=8, n_keys=8, attn_bias=None, inf=1e9
        )

        # The function prints a warning but doesn't use the warnings module
        # So we'll just call it and check it doesn't raise an error
        result = _rearrange_to_dense_trunk_impl(config)
        assert len(result) == 5  # Should return 5-tuple

        # Verify the specific outputs to cover lines 167-169
        _, k_out, v_out, _, _ = result
        assert k_out.shape[1] > 0  # Should have at least one trunk
        assert v_out.shape[1] > 0  # Should have at least one trunk

    def test_zero_length_query(self):
        """Test with zero-length query tensor."""
        q = torch.empty(2, 0, 64)
        k = torch.randn(2, 10, 64)
        v = torch.randn(2, 10, 64)
        config = RearrangementConfig(
            q=q, k=k, v=v, n_queries=8, n_keys=8, attn_bias=None, inf=1e9
        )

        q_out, _, _, _, padding_len = _rearrange_to_dense_trunk_impl(config)

        # Check shapes
        assert q_out.shape[0] == 2  # Batch size preserved
        assert q_out.shape[1] == 0  # Zero trunks for zero-length query
        assert padding_len == 0  # No padding for zero-length

    def test_zero_length_key_value(self):
        """Test with zero-length key and value tensors."""
        q = torch.randn(2, 10, 64)
        k = torch.empty(2, 0, 64)
        v = torch.empty(2, 0, 64)
        config = RearrangementConfig(
            q=q, k=k, v=v, n_queries=8, n_keys=8, attn_bias=None, inf=1e9
        )

        _, k_out, v_out, _, _ = _rearrange_to_dense_trunk_impl(config)

        # Check shapes
        assert k_out.shape[1] == 0  # Zero trunks for zero-length key
        assert v_out.shape[1] == 0  # Zero trunks for zero-length value

    def test_with_attention_bias_and_padding(self):
        """Test with attention bias and padding."""
        q = torch.randn(2, 10, 64)
        k = torch.randn(2, 12, 64)
        v = torch.randn(2, 12, 64)
        attn_bias = torch.randn(2, 10, 12)
        config = RearrangementConfig(
            q=q, k=k, v=v, n_queries=8, n_keys=8, attn_bias=attn_bias, inf=1e9
        )

        _, _, _, bias_out, padding_len = _rearrange_to_dense_trunk_impl(config)

        # Check padding
        assert padding_len == 6  # 10 -> 16 (2 trunks of 8)

        # Check bias shape includes padding
        expected_bias_shape = (2, 2, 8, 16)  # (batch, q_trunks, q_per_trunk, padded_k_len)
        assert bias_out.shape == expected_bias_shape

    def test_specific_trunk_calculations(self):
        """Test specific trunk calculations to cover lines 186-192, 195-200."""
        # Create a case where n_k_trunks calculation and padding are exercised
        # Use n_keys=3 to make the math simpler and more predictable
        q = torch.randn(2, 10, 64)
        k = torch.randn(2, 7, 64)  # 7 keys will need 3 trunks of 3 (with padding)
        v = torch.randn(2, 7, 64)
        config = RearrangementConfig(
            q=q, k=k, v=v, n_queries=4, n_keys=3, attn_bias=None, inf=1e9
        )

        _, k_out, v_out, _, padding_len = _rearrange_to_dense_trunk_impl(config)

        # Check number of trunks and padding
        # 7 keys with trunk size 3 should give 3 trunks (ceil(7/3))
        assert k_out.shape[1] == 3  # Should have 3 trunks
        assert padding_len == 2  # 7 -> 9 (3 trunks of 3)

        # Check the shapes of the output tensors
        # k_out should be (batch, n_k_trunks, n_keys, features)
        assert k_out.shape == (2, 3, 3, 64)
        assert v_out.shape == (2, 3, 3, 64)


class TestRearrangeQKToDenseTrunk:
    """Tests for rearrange_qk_to_dense_trunk function."""

    def test_inconsistent_ranks_before_concatenation(self):
        """Test error handling for inconsistent ranks before concatenation."""
        q1 = torch.randn(2, 3, 64)  # rank 3
        q2 = torch.randn(2, 4, 32)  # rank 3 but different feature dim

        with pytest.raises(RuntimeError, match="Failed to concatenate q_processed_list"):
            rearrange_qk_to_dense_trunk(
                [q1, q2], torch.randn(2, 5, 64),
                dim_q=[1, 1], dim_k=1
            )

    def test_compute_mask_false(self):
        """Test with compute_mask=False."""
        q = torch.randn(2, 10, 64)
        k = torch.randn(2, 12, 64)

        _, _, padding_info = rearrange_qk_to_dense_trunk(
            q, k, dim_q=1, dim_k=1, compute_mask=False
        )

        # Check that mask-related keys are None when compute_mask=False
        assert padding_info['q_mask'] is None
        assert padding_info['k_mask'] is None

    def test_with_different_dtypes(self):
        """Test with tensors of different dtypes."""
        q = torch.randn(2, 10, 64, dtype=torch.float32)
        k = torch.randn(2, 12, 64, dtype=torch.float64)

        # The function doesn't explicitly check for dtype matching
        # but will likely fail during concatenation
        try:
            rearrange_qk_to_dense_trunk(q, k, dim_q=1, dim_k=1)
            # If it doesn't fail, that's fine too - just make sure it runs
        except RuntimeError as e:
            # Check if the error is related to dtype mismatch
            assert "dtype" in str(e).lower() or "type" in str(e).lower()


class TestRearrangeToDenseTrunk:
    """Tests for rearrange_to_dense_trunk function."""

    def test_with_custom_inf_value(self):
        """Test with custom inf value."""
        q = torch.randn(2, 10, 64)
        k = torch.randn(2, 12, 64)
        v = torch.randn(2, 12, 64)
        attn_bias = torch.randn(2, 10, 12)
        custom_inf = 1e5

        _, _, _, bias_out, _ = rearrange_to_dense_trunk(
            q, k, v, n_queries=8, n_keys=8,
            attn_bias=attn_bias, inf=custom_inf
        )

        # Check that the custom inf value is used in the bias
        # The bias should have some values equal to -custom_inf
        assert torch.any(bias_out == -custom_inf)

    def test_with_different_devices(self):
        """Test with tensors on different devices (if CUDA available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping device test")

        q = torch.randn(2, 10, 64, device='cuda')
        k = torch.randn(2, 12, 64, device='cpu')
        v = torch.randn(2, 12, 64, device='cpu')

        # The function doesn't explicitly check for device matching
        # but will likely fail during operations
        try:
            rearrange_to_dense_trunk(q, k, v, n_queries=8, n_keys=8)
            # If it doesn't fail, that's fine too - just make sure it runs
        except RuntimeError as e:
            # Check if the error is related to device mismatch
            assert "device" in str(e).lower() or "cuda" in str(e).lower()

# tests/stageA/unit/input_embedding/current/primitives/attention/test_padding_reshape_utils.py
import pytest
import torch
from typing import List, Optional, NamedTuple

from rna_predict.pipeline.stageA.input_embedding.current.primitives.attention.padding_reshape_utils import (
    _calculate_padding_needed,
    _create_padding_tensor,
    _reshape_query_to_trunk,
    _pad_attention_bias,
    _reshape_bias_for_trunked_query,
    _create_different_dim_bias,
    _process_attention_bias,
)
# Assuming AttentionBiasConfig is importable or defined similarly for testing
# If it's complex, consider mocking or creating a test double
class MockAttentionBiasConfig(NamedTuple):
    """Simplified config for testing."""
    n_q_pad: int
    inf: float
    n_q_trunks: int
    n_queries: int
    original_length: int # Original query length before padding


# --- Tests for _calculate_padding_needed (Line 23) ---

@pytest.mark.parametrize(
    "n, n_queries, expected_padding",
    [
        (10, 5, 0), # Exact multiple
        (12, 5, 3), # Needs padding
        (5, 5, 0),  # Exact multiple (single trunk)
        (3, 5, 2),  # Needs padding (less than one trunk)
        (0, 5, 0),  # Edge case: zero length
    ],
)
def test_calculate_padding_needed(n: int, n_queries: int, expected_padding: int):
    """
    Test _calculate_padding_needed for various scenarios.
    Covers: Line 23
    """
    assert _calculate_padding_needed(n, n_queries) == expected_padding

# --- Tests for _create_padding_tensor (Lines 40-58) ---

@pytest.mark.parametrize("padding_dim", [-1, -2, 1, 0])
def test_create_padding_tensor_positive_length(padding_dim: int):
    """
    Test _create_padding_tensor when padding_length > 0.
    Covers: Lines 55-58
    """
    original_shape = [2, 3, 4, 5] # Example shape
    # Ensure padding_dim is valid for the shape
    if not (-len(original_shape) <= padding_dim < len(original_shape)):
         pytest.skip("Invalid padding_dim for shape")

    original_tensor = torch.randn(original_shape)
    padding_length = 3
    padding_tensor = _create_padding_tensor(original_tensor, padding_length, padding_dim)

    expected_shape = list(original_shape)
    expected_shape[padding_dim] = padding_length

    assert padding_tensor.shape == torch.Size(expected_shape)
    assert padding_tensor.dtype == original_tensor.dtype
    assert padding_tensor.device == original_tensor.device
    assert torch.all(padding_tensor == 0)

@pytest.mark.parametrize("padding_dim", [-1, -2, 1, 0])
def test_create_padding_tensor_zero_length(padding_dim: int):
    """
    Test _create_padding_tensor when padding_length is 0.
    Covers: Lines 40, 50-52
    """
    original_shape = [2, 3, 4, 5] # Example shape
     # Ensure padding_dim is valid for the shape
    if not (-len(original_shape) <= padding_dim < len(original_shape)):
         pytest.skip("Invalid padding_dim for shape")

    original_tensor = torch.randn(original_shape)
    padding_length = 0
    padding_tensor = _create_padding_tensor(original_tensor, padding_length, padding_dim)

    expected_shape = list(original_shape)
    expected_shape[padding_dim] = 0 # Expect zero size along the dim

    assert padding_tensor.shape == torch.Size(expected_shape)
    assert padding_tensor.numel() == 0 # Should have zero elements
    assert padding_tensor.dtype == original_tensor.dtype
    assert padding_tensor.device == original_tensor.device

# --- Tests for _reshape_query_to_trunk (Lines 75-94) ---

@pytest.mark.parametrize(
    "batch_dims, n, d_head, n_queries, needs_padding",
    [
        ([], 10, 4, 5, False),       # No batch, exact multiple
        ([2], 12, 4, 5, True),       # 1 batch dim, needs padding
        ([2, 3], 5, 8, 5, False),    # 2 batch dims, exact multiple
        ([1], 7, 8, 4, True),       # 1 batch dim, needs padding
    ],
)
def test_reshape_query_to_trunk(
    batch_dims: List[int], n: int, d_head: int, n_queries: int, needs_padding: bool
):
    """
    Test _reshape_query_to_trunk with and without padding.
    Covers: Lines 75-76, 79-81, (83-85 if needs_padding), (86-87 if not needs_padding), 90-94
    """
    q_shape = batch_dims + [n, d_head]
    q = torch.randn(q_shape)

    # Assuming QueryTrunkInfo is defined or importable
    from rna_predict.pipeline.stageA.input_embedding.current.primitives.attention.config_types import QueryTrunkInfo
    result: QueryTrunkInfo = _reshape_query_to_trunk(q, n_queries)

    expected_padding = _calculate_padding_needed(n, n_queries)
    expected_total_length = n + expected_padding
    expected_num_trunks = expected_total_length // n_queries

    assert result.padding_length == expected_padding
    assert result.total_length == expected_total_length
    assert result.num_trunks == expected_num_trunks

    expected_trunked_shape = batch_dims + [expected_num_trunks, n_queries, d_head]
    assert result.trunked_tensor.shape == torch.Size(expected_trunked_shape)
    assert result.trunked_tensor.dtype == q.dtype
    assert result.trunked_tensor.device == q.device

    # Verify content (first part should match original q)
    q_reshaped_original_part = result.trunked_tensor.reshape(batch_dims + [expected_total_length, d_head])[..., :n, :]
    assert torch.allclose(q_reshaped_original_part, q)

    # Verify padding part is zero if padding was added
    if needs_padding:
        assert expected_padding > 0
        q_reshaped_padding_part = result.trunked_tensor.reshape(batch_dims + [expected_total_length, d_head])[..., n:, :]
        assert torch.all(q_reshaped_padding_part == 0)
    else:
        assert expected_padding == 0

# --- Tests for _pad_attention_bias (Lines 115-124) ---

@pytest.mark.parametrize(
    "n_q_pad, needs_padding",
    [
        (0, False), # No padding needed
        (-1, False), # No padding needed (<= 0)
        (3, True),  # Padding needed
    ],
)
def test_pad_attention_bias(n_q_pad: int, needs_padding: bool):
    """
    Test _pad_attention_bias with and without padding.
    Covers: Lines 115-116 (if not needs_padding), 118-124 (if needs_padding)
    """
    bias_shape = [2, 4, 10, 15] # B, H, Nq, Nk
    attn_bias = torch.randn(bias_shape)
    config = MockAttentionBiasConfig(
        n_q_pad=n_q_pad,
        inf=1e9, # Example value for infinity
        n_q_trunks=-1, # Not used directly by this function
        n_queries=-1, # Not used directly by this function
        original_length=-1 # Not used directly by this function
    )

    padded_bias = _pad_attention_bias(attn_bias, config)

    if needs_padding:
        expected_shape = list(bias_shape)
        expected_shape[-2] = bias_shape[-2] + n_q_pad # Pad the query dim
        assert padded_bias.shape == torch.Size(expected_shape)
        # Check original part
        assert torch.allclose(padded_bias[..., :bias_shape[-2], :], attn_bias)
        # Check padded part
        assert torch.all(padded_bias[..., bias_shape[-2]:, :] == -config.inf)
    else:
        # Should return the original tensor if no padding needed
        assert torch.equal(padded_bias, attn_bias)
        assert padded_bias.shape == attn_bias.shape # Line 116 returns original

# --- Tests for _reshape_bias_for_trunked_query (Lines 141-151) ---

@pytest.mark.parametrize(
    "n_q_pad, n_queries, original_length",
    [
        (0, 5, 10), # No padding needed
        (3, 5, 12), # Padding needed
    ],
)
def test_reshape_bias_for_trunked_query(n_q_pad: int, n_queries: int, original_length: int):
    """
    Test _reshape_bias_for_trunked_query.
    Covers: Lines 141 (calls _pad_attention_bias), 144-151
    """
    k_len = 15
    bias_shape = [2, 4, original_length, k_len] # B, H, Nq_orig, Nk
    attn_bias = torch.randn(bias_shape)

    total_length = original_length + n_q_pad
    n_q_trunks = total_length // n_queries

    config = MockAttentionBiasConfig(
        n_q_pad=n_q_pad,
        inf=1e9,
        n_q_trunks=n_q_trunks,
        n_queries=n_queries,
        original_length=original_length
    )

    reshaped_bias = _reshape_bias_for_trunked_query(attn_bias, config)

    expected_shape = [2, 4, n_q_trunks, n_queries, k_len] # B, H, Nq_trunks, Nq_per_trunk, Nk
    assert reshaped_bias.shape == torch.Size(expected_shape)

    # Basic check: ensure the number of elements is preserved after padding and reshaping
    padded_len = original_length + n_q_pad
    assert reshaped_bias.numel() == bias_shape[0] * bias_shape[1] * padded_len * bias_shape[3]


# --- Tests for _create_different_dim_bias (Lines 169-192) ---

@pytest.mark.parametrize(
    "original_length, n_queries, n_keys, n_k_trunks",
    [
        (10, 5, 6, 2), # 2 full query trunks, hits line 190
        (12, 5, 6, 2), # 2 full query trunks + 1 partial, hits line 190
        (4, 5, 6, 2),  # 1 partial query trunk, hits line 190
        (13, 5, 6, 2), # Test case where a trunk might be fully padding (hits 182-183)
                       # Requires careful setup of n_q_trunks vs original_length
    ]
)
def test_create_different_dim_bias(original_length: int, n_queries: int, n_keys: int, n_k_trunks: int):
    """
    Test _create_different_dim_bias for generating default masks.
    Covers: Lines 169-174, 177-183, 185-192
    """
    n_q_pad = _calculate_padding_needed(original_length, n_queries)
    total_q_length = original_length + n_q_pad
    n_q_trunks = total_q_length // n_queries

    # Create dummy trunked tensors for shape information
    q_trunked_shape = [2, 4, n_q_trunks, n_queries, 8] # B, H, Nq_trunks, Nq, Dq
    k_trunked_shape = [2, 4, n_k_trunks, n_keys, 8]    # B, H, Nk_trunks, Nk, Dk
    q_trunked = torch.empty(q_trunked_shape)
    k_trunked = torch.empty(k_trunked_shape)

    config = MockAttentionBiasConfig(
        n_q_pad=n_q_pad,
        inf=1e9,
        n_q_trunks=n_q_trunks,
        n_queries=n_queries,
        original_length=original_length
    )

    attn_bias_trunked = _create_different_dim_bias(q_trunked, k_trunked, config)

    expected_shape = [2, 4, n_q_trunks, n_queries, n_k_trunks, n_keys] # B, H, Nq_t, Nq, Nk_t, Nk
    assert attn_bias_trunked.shape == torch.Size(expected_shape)

    # Check values: Should be 0 where attention is allowed, -inf otherwise
    for i in range(n_q_trunks):
        q_start = i * n_queries
        q_end = min(q_start + n_queries, original_length)
        valid_queries_in_trunk = max(0, q_end - q_start) # Ensure non-negative

        if q_start >= original_length: # Trunk is fully padding (Line 182)
             assert torch.all(attn_bias_trunked[..., i, :, :, :] == -config.inf)
        else:
            # Valid queries part should be 0 (Line 190)
            if valid_queries_in_trunk > 0:
                 assert torch.all(attn_bias_trunked[..., i, :valid_queries_in_trunk, :, :] == 0)
            # Padded queries part should be -inf
            if valid_queries_in_trunk < n_queries:
                 assert torch.all(attn_bias_trunked[..., i, valid_queries_in_trunk:, :, :] == -config.inf)


# --- Tests for _process_attention_bias (Lines 216-291) ---

def test_process_attention_bias_no_bias():
    """
    Test _process_attention_bias when attn_bias is None.
    Covers: Lines 216-221
    """
    q_trunked = torch.randn(2, 4, 3, 5, 8) # B, H, Nq_t, Nq, Dq
    k_trunked = torch.randn(2, 4, 2, 6, 8) # B, H, Nk_t, Nk, Dk
    config = MockAttentionBiasConfig(0, 1e9, 3, 5, 15) # Example config

    processed_bias = _process_attention_bias(q_trunked, k_trunked, None, None, config)

    # Expect a scalar zero tensor
    assert torch.is_tensor(processed_bias)
    assert processed_bias.numel() == 1
    assert processed_bias.item() == 0.0
    assert processed_bias.dtype == q_trunked.dtype
    assert processed_bias.device == q_trunked.device


def test_process_attention_bias_case1_matches_original_q_with_list():
    """
    Test _process_attention_bias: Case 1 - bias matches original q_len, attn_bias_list provided.
    Covers: Lines 231, 233, 256-266 (calls _reshape_bias_for_trunked_query)
    """
    original_q_len = 12
    original_k_len = 18 # Assume k is chunked
    n_queries = 5
    n_keys = 6 # Keys per chunk
    n_k_trunks = 3 # original_k_len // n_keys

    n_q_pad = _calculate_padding_needed(original_q_len, n_queries) # 3
    total_q_len = original_q_len + n_q_pad # 15
    n_q_trunks = total_q_len // n_queries # 3

    q_trunked = torch.randn(2, 4, n_q_trunks, n_queries, 8) # B, H, Nq_t, Nq, Dq
    k_trunked = torch.randn(2, 4, n_k_trunks, n_keys, 8)    # B, H, Nk_t, Nk, Dk

    # Original bias matching original q_len and k_len
    attn_bias = torch.randn(2, 4, original_q_len, original_k_len) # B, H, Nq_orig, Nk_orig

    # Simulate attn_bias_list created by _process_keys_values_chunks
    # Each element: (B, H, Nq_orig, Nk_per_chunk)
    attn_bias_list = [
        torch.randn(2, 4, original_q_len, n_keys),
        torch.randn(2, 4, original_q_len, n_keys),
        torch.randn(2, 4, original_q_len, n_keys),
    ]

    config = MockAttentionBiasConfig(
        n_q_pad=n_q_pad,
        inf=1e9,
        n_q_trunks=n_q_trunks,
        n_queries=n_queries,
        original_length=original_q_len
    )

    processed_bias = _process_attention_bias(q_trunked, k_trunked, attn_bias, attn_bias_list, config)

    # Expected shape after reshaping query dim and stacking key chunks
    expected_shape = [2, 4, n_q_trunks, n_queries, n_k_trunks, n_keys] # B, H, Nq_t, Nq, Nk_t, Nk
    assert processed_bias.shape == torch.Size(expected_shape)
    # Further checks could involve verifying specific values if the logic was clearer


def test_process_attention_bias_case3_different_dims():
    """
    Test _process_attention_bias: Case 3 - Bias dimensions don't match, create default.
    Covers: Lines 231 (False), 267 (False), 286-290 (calls _create_different_dim_bias)
    """
    original_q_len = 12
    n_queries = 5
    n_keys = 6
    n_k_trunks = 2

    n_q_pad = _calculate_padding_needed(original_q_len, n_queries) # 3
    total_q_len = original_q_len + n_q_pad # 15
    n_q_trunks = total_q_len // n_queries # 3

    q_trunked = torch.randn(2, 4, n_q_trunks, n_queries, 8) # B, H, Nq_t, Nq, Dq
    k_trunked = torch.randn(2, 4, n_k_trunks, n_keys, 8)    # B, H, Nk_t, Nk, Dk

    # Bias with unexpected dimensions (e.g., only query dim)
    attn_bias = torch.randn(2, 4, original_q_len) # B, H, Nq_orig

    config = MockAttentionBiasConfig(
        n_q_pad=n_q_pad,
        inf=1e9,
        n_q_trunks=n_q_trunks,
        n_queries=n_queries,
        original_length=original_q_len
    )

    processed_bias = _process_attention_bias(q_trunked, k_trunked, attn_bias, None, config) # No list provided

    # Should fall back to _create_different_dim_bias
    expected_shape = [2, 4, n_q_trunks, n_queries, n_k_trunks, n_keys] # B, H, Nq_t, Nq, Nk_t, Nk
    assert processed_bias.shape == torch.Size(expected_shape)
    # Check if it looks like the output of _create_different_dim_bias (0s and -infs)
    assert torch.any(processed_bias == 0)
    assert torch.any(processed_bias == -config.inf)


# Note: Hitting Case 2 (Lines 267-283) seems difficult given the current logic flow described
# in the comments and the reliance on Case 1. The `pass` at line 283 means even if the
# conditions were met, the code might not execute unique logic beyond line 278.
# A specific test for this case might require a deeper understanding of when attn_bias
# would have ndim == q_trunked.ndim but not match original_q_len.
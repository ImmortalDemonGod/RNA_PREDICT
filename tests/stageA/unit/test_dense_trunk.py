import pytest
import torch
import math
from typing import List, Union, Tuple

from rna_predict.pipeline.stageA.input_embedding.current.primitives.attention_utils import PaddingInfo # Correct import path
from rna_predict.pipeline.stageA.input_embedding.current.primitives.attention.dense_trunk import (
    rearrange_qk_to_dense_trunk,
    rearrange_to_dense_trunk,
)

# Define a simple PaddingInfo structure if not importable, for type hints and assertions
# from collections import namedtuple
# PaddingInfo = namedtuple("PaddingInfo", ["q_padding", "k_padding", "num_q_trunks", "num_k_trunks"])
# Or if it's just a tuple:
# PaddingInfo = Tuple[int, int, int, int]
# For now, assume it exists and is importable or globally defined for the tests.


# Helper function to calculate expected shapes and padding
def calculate_expected_trunk_params(seq_len: int, trunk_size: int) -> Tuple[int, int]:
    """Calculates expected number of trunks and padding length."""
    if trunk_size <= 0:
        raise ValueError("Trunk size must be positive")
    if seq_len < 0:
        raise ValueError("Sequence length cannot be negative")
    if seq_len == 0:
        return 0, 0 # Handle zero length input gracefully

    num_trunks = math.ceil(seq_len / trunk_size)
    padded_len = num_trunks * trunk_size
    padding_len = padded_len - seq_len
    return num_trunks, padding_len

# == Tests for rearrange_qk_to_dense_trunk ==

# Removed dummy PaddingInfo definition block as the real one is importable


@pytest.mark.parametrize(
    "batch_size, q_len, k_len, n_queries, n_keys, d_head, dim_q, dim_k, use_list_input",
    [
        # --- Happy Path: Partial Trunks ---
        (2, 50, 60, 16, 16, 32, -2, -2, False), # Standard case, padding needed
        (1, 5, 7, 4, 4, 8, -2, -2, False),    # Small lengths, padding needed
        (3, 100, 90, 32, 20, 16, -2, -2, False), # Different n_queries/n_keys
        (2, 50, 60, 16, 16, 32, -2, -2, True),  # List input, padding needed
        (2, 50, 60, 16, 16, 32, 1, 1, False),   # Different dims (B, S, D), padding needed
        (1, 0, 10, 4, 4, 8, -2, -2, False),   # Zero length Q
        (1, 10, 0, 4, 4, 8, -2, -2, False),   # Zero length K
        (1, 0, 0, 4, 4, 8, -2, -2, False),    # Zero length Q and K
        # --- Happy Path: Exact Division ---
        (2, 64, 64, 16, 16, 32, -2, -2, False), # Exact division
        (1, 12, 8, 4, 4, 8, -2, -2, False),    # Small lengths, exact division
        (3, 96, 100, 32, 20, 16, -2, -2, False), # Exact division q, partial k
        (2, 64, 64, 16, 16, 32, -2, -2, True),  # List input, exact division
    ]
)
def test_rearrange_qk_to_dense_trunk_happy_path(
    batch_size: int, q_len: int, k_len: int, n_queries: int, n_keys: int, d_head: int,
    dim_q: int, dim_k: int, use_list_input: bool
):
    """
    Tests rearrange_qk_to_dense_trunk for various valid inputs,
    including partial trunks (padding) and exact divisions.
    Checks output shapes, padding values, and padding_info.
    Handles zero-length inputs.
    """
    # Adjust shapes based on dim_q/dim_k
    # Handle potential negative dims correctly for insertion
    q_dim_idx = dim_q if dim_q >= 0 else dim_q + 3 # Assuming 3 dims B, S, D initially
    k_dim_idx = dim_k if dim_k >= 0 else dim_k + 3

    q_shape = [batch_size, d_head]
    # Ensure index is valid before inserting
    if not (0 <= q_dim_idx <= len(q_shape)):
         pytest.skip(f"Calculated q_dim_idx {q_dim_idx} is invalid for shape insertion.")
    q_shape.insert(q_dim_idx, q_len)

    k_shape = [batch_size, d_head]
    if not (0 <= k_dim_idx <= len(k_shape)):
         pytest.skip(f"Calculated k_dim_idx {k_dim_idx} is invalid for shape insertion.")
    k_shape.insert(k_dim_idx, k_len)

    # Define input variables with correct Union type hint
    q_input: Union[torch.Tensor, List[torch.Tensor]]
    k_input: Union[torch.Tensor, List[torch.Tensor]]

    if use_list_input:
        # Create lists of tensors (e.g., two tensors per list)
        # Handle zero length gracefully
        q_input = [torch.randn(*q_shape) if q_len > 0 else torch.empty(*q_shape),
                   torch.randn(*q_shape) if q_len > 0 else torch.empty(*q_shape)]
        k_input = [torch.randn(*k_shape) if k_len > 0 else torch.empty(*k_shape),
                   torch.randn(*k_shape) if k_len > 0 else torch.empty(*k_shape)]
        num_tensors_in_list = 2
    else:
        q_input = torch.randn(*q_shape) if q_len > 0 else torch.empty(*q_shape)
        k_input = torch.randn(*k_shape) if k_len > 0 else torch.empty(*k_shape)
        num_tensors_in_list = 1 # For consistent shape calculation below

    # Call function with dim_q, dim_k as positional args
    # TODO: The function's return type hint for padding_info (PaddingInfo TypedDict)
    # seems inconsistent with its actual return (a dict with lengths/counts).
    # The test currently checks against the actual returned dict structure.
    padding_info: dict # Use dict type hint based on actual return
    q_trunked, k_trunked, padding_info = rearrange_qk_to_dense_trunk(
        q_input, k_input, dim_q, dim_k, n_queries=n_queries, n_keys=n_keys
    )

    # Calculate expected shapes and padding
    num_q_trunks, q_padding = calculate_expected_trunk_params(q_len, n_queries)
    num_k_trunks, k_padding = calculate_expected_trunk_params(k_len, n_keys)

    # Expected shapes: (ListSize * Batch, NumTrunks, TrunkSize, D_Head)
    # This assumes the output format standardizes the dimensions.
    expected_q_shape = (batch_size * num_tensors_in_list, num_q_trunks, n_queries, d_head)
    expected_k_shape = (batch_size * num_tensors_in_list, num_k_trunks, n_keys, d_head)

    # Handle zero trunks case
    if num_q_trunks == 0:
        expected_q_shape = (batch_size * num_tensors_in_list, 0, n_queries, d_head)
    if num_k_trunks == 0:
        expected_k_shape = (batch_size * num_tensors_in_list, 0, n_keys, d_head)


    # --- Assertions ---
    assert isinstance(q_trunked, torch.Tensor)
    assert isinstance(k_trunked, torch.Tensor)
    # Removed isinstance check for TypedDict

    assert q_trunked.shape == torch.Size(expected_q_shape)
    assert k_trunked.shape == torch.Size(expected_k_shape)

    # Check padding info
    # Access dictionary keys directly (matches actual return, despite function type hint)
    assert padding_info["q_padding"] == q_padding
    assert padding_info["k_padding"] == k_padding
    assert padding_info["num_q_trunks"] == num_q_trunks
    assert padding_info["num_k_trunks"] == num_k_trunks

    # Check padding values (should be zero) only if there are trunks and padding
    # Slicing needs to target the sequence dimension within the trunk (dim=-2)
    if q_padding > 0 and num_q_trunks > 0:
        assert torch.all(q_trunked[:, -1, -q_padding:, :] == 0.0)
        # Check non-padded area of the last trunk is not all zero (sanity check)
        if n_queries - q_padding > 0:
             # Use torch.any for potentially sparse tensors
             assert torch.any(q_trunked[:, -1, :-q_padding, :] != 0.0)
    if k_padding > 0 and num_k_trunks > 0:
        assert torch.all(k_trunked[:, -1, -k_padding:, :] == 0.0)
        # Check non-padded area of the last trunk is not all zero (sanity check)
        if n_keys - k_padding > 0:
            assert torch.any(k_trunked[:, -1, :-k_padding, :] != 0.0)

    # Check exact division case (no padding in last trunk if padding == 0)
    if q_padding == 0 and q_len > 0 and num_q_trunks > 0:
         assert torch.any(q_trunked[:, -1, :, :] != 0.0) # Last trunk shouldn't be all zero
    if k_padding == 0 and k_len > 0 and num_k_trunks > 0:
         assert torch.any(k_trunked[:, -1, :, :] != 0.0) # Last trunk shouldn't be all zero


@pytest.mark.parametrize(
    "batch_size, q_len, k_len, n_queries, n_keys, d_head, dim_q, dim_k, expected_error",
    [
        (1, 10, 10, 8, 8, 16, 3, -2, IndexError), # Invalid dim_q (assuming B,S,D -> index 3 out of bounds)
        (1, 10, 10, 8, 8, 16, -2, 3, IndexError), # Invalid dim_k
        (1, 10, 10, 8, 8, 16, -4, -2, IndexError), # Invalid dim_q (assuming B,S,D -> index -4 out of bounds)
        (1, 10, 10, 0, 8, 16, -2, -2, ValueError), # Invalid n_queries (expect ValueError or ZeroDivisionError)
        (1, 10, 10, 8, 0, 16, -2, -2, ValueError), # Invalid n_keys (expect ValueError or ZeroDivisionError)
        (1, 10, 10, -1, 8, 16, -2, -2, ValueError), # Invalid n_queries
        (1, 10, 10, 8, -1, 16, -2, -2, ValueError), # Invalid n_keys
    ]
)
def test_rearrange_qk_to_dense_trunk_error_cases(
    batch_size: int, q_len: int, k_len: int, n_queries: int, n_keys: int, d_head: int,
    dim_q: int, dim_k: int, expected_error: type
):
    """Tests rearrange_qk_to_dense_trunk with invalid inputs expecting errors."""
    # Use standard B, S, D shape for error setup simplicity
    q_shape = [batch_size, q_len, d_head]
    k_shape = [batch_size, k_len, d_head]
    q_input = torch.randn(*q_shape) if q_len > 0 else torch.empty(*q_shape)
    k_input = torch.randn(*k_shape) if k_len > 0 else torch.empty(*k_shape)

    # Need to handle list input case for errors too, though less likely to cause index errors
    # For simplicity, testing errors primarily with single tensor inputs.

    with pytest.raises(expected_error):
        # Call function with dim_q, dim_k as positional args
        rearrange_qk_to_dense_trunk(
            q_input, k_input, dim_q, dim_k, n_queries=n_queries, n_keys=n_keys
        )

# == Tests for rearrange_to_dense_trunk ==

@pytest.mark.parametrize(
    "batch_size, q_len, k_len, v_len, n_queries, n_keys, d_head, provide_attn_bias",
    [
        # --- Happy Path: Partial Trunks ---
        (2, 50, 60, 60, 16, 16, 32, False), # Standard case, padding needed, no bias
        (1, 5, 7, 7, 4, 4, 8, True),      # Small lengths, padding needed, with bias
        (3, 100, 90, 90, 32, 20, 16, False), # Different n_queries/n_keys, padding
        (2, 55, 65, 65, 16, 16, 32, True),  # Padding needed, with bias
        (1, 10, 0, 0, 4, 4, 8, False),    # Zero length K/V
        (1, 0, 10, 10, 4, 4, 8, True),     # Zero length Q
        (1, 0, 0, 0, 4, 4, 8, False),     # Zero length Q/K/V
        # --- Happy Path: Exact Division ---
        (2, 64, 64, 64, 16, 16, 32, False), # Exact division, no bias
        (1, 12, 8, 8, 4, 4, 8, True),      # Small lengths, exact division, with bias
        (3, 96, 100, 100, 32, 20, 16, False), # Exact division q, partial k, no bias
        (2, 64, 80, 80, 16, 16, 32, True),  # Exact division, with bias
    ]
)
def test_rearrange_to_dense_trunk_happy_path(
    batch_size: int, q_len: int, k_len: int, v_len: int, n_queries: int, n_keys: int, d_head: int,
    provide_attn_bias: bool
):
    """
    Tests rearrange_to_dense_trunk for various valid inputs (excluding small tensor bypass).
    Checks output shapes, padding values, padding_length, and attn_bias handling.
    Handles zero-length inputs.
    """
    # Small tensor bypass condition
    if q_len <= n_queries and k_len <= n_keys and q_len > 0 and k_len > 0:
        pytest.skip("Skipping non-bypass case for small tensor input")
    if q_len == 0 or k_len == 0: # Handle zero length cases here, not bypass
         pass # Allow zero length tests

    # Basic assertion for test setup validity
    if k_len != v_len:
         # This function might allow k_len != v_len, but trunking logic usually assumes they match.
         # If the function *does* support mismatch, this test needs adjustment.
         # For now, assume they must match for standard trunking.
         pytest.skip("Skipping test where k_len != v_len as standard trunking assumes match")


    q_shape = (batch_size, q_len, d_head)
    k_shape = (batch_size, k_len, d_head)
    v_shape = (batch_size, v_len, d_head)
    attn_bias_shape = (batch_size, q_len, k_len)

    q = torch.randn(*q_shape) if q_len > 0 else torch.empty(*q_shape)
    k = torch.randn(*k_shape) if k_len > 0 else torch.empty(*k_shape)
    v = torch.randn(*v_shape) if v_len > 0 else torch.empty(*v_shape)
    attn_bias = torch.randn(*attn_bias_shape) if provide_attn_bias and q_len > 0 and k_len > 0 else None

    q_trunked, k_trunked, v_trunked, padding_length, attn_bias_trunked = rearrange_to_dense_trunk(
        q, k, v, n_queries, n_keys, attn_bias=attn_bias
    )

    # Calculate expected shapes and padding
    num_q_trunks, q_padding = calculate_expected_trunk_params(q_len, n_queries)
    num_k_trunks, k_padding = calculate_expected_trunk_params(k_len, n_keys)
    # V uses k padding logic
    num_v_trunks, v_padding = calculate_expected_trunk_params(v_len, n_keys) # Assuming v follows k trunking

    # If function allows k_len != v_len, v_padding might differ. Assuming v follows k.
    assert v_padding == k_padding, "Test logic assumes V padding follows K padding"
    assert num_v_trunks == num_k_trunks, "Test logic assumes V num_trunks follows K num_trunks"

    # Expected shapes: (Batch, NumTrunks, TrunkSize, D_Head)
    expected_q_shape = (batch_size, num_q_trunks, n_queries, d_head)
    expected_k_shape = (batch_size, num_k_trunks, n_keys, d_head)
    expected_v_shape = (batch_size, num_v_trunks, n_keys, d_head) # V follows K trunking
    # Attn Bias shape: (Batch, NumQTrunks, QTrunkSize, PaddedKLength)
    padded_k_len = num_k_trunks * n_keys
    expected_attn_bias_shape = (batch_size, num_q_trunks, n_queries, padded_k_len)

    # Handle zero trunks case
    if num_q_trunks == 0:
        expected_q_shape = (batch_size, 0, n_queries, d_head)
        expected_attn_bias_shape = (batch_size, 0, n_queries, padded_k_len)
    if num_k_trunks == 0:
        expected_k_shape = (batch_size, 0, n_keys, d_head)
        expected_v_shape = (batch_size, 0, n_keys, d_head)
        # If K has zero trunks, padded_k_len is 0
        expected_attn_bias_shape = (batch_size, num_q_trunks, n_queries, 0)


    # --- Assertions ---
    assert isinstance(q_trunked, torch.Tensor)
    assert isinstance(k_trunked, torch.Tensor)
    assert isinstance(v_trunked, torch.Tensor)
    assert isinstance(padding_length, int)
    assert isinstance(attn_bias_trunked, torch.Tensor)

    assert q_trunked.shape == torch.Size(expected_q_shape)
    assert k_trunked.shape == torch.Size(expected_k_shape)
    assert v_trunked.shape == torch.Size(expected_v_shape)
    assert attn_bias_trunked.shape == torch.Size(expected_attn_bias_shape)
    assert padding_length == q_padding # Function returns query padding length

    # Check padding values (should be zero for Q, K, V)
    if q_padding > 0 and num_q_trunks > 0:
        assert torch.all(q_trunked[:, -1, -q_padding:, :] == 0.0)
        if n_queries - q_padding > 0:
             assert torch.any(q_trunked[:, -1, :-q_padding, :] != 0.0)
    if k_padding > 0 and num_k_trunks > 0: # K and V share padding logic
        assert torch.all(k_trunked[:, -1, -k_padding:, :] == 0.0)
        assert torch.all(v_trunked[:, -1, -k_padding:, :] == 0.0)
        if n_keys - k_padding > 0:
            assert torch.any(k_trunked[:, -1, :-k_padding, :] != 0.0)
            assert torch.any(v_trunked[:, -1, :-k_padding, :] != 0.0)

    # Check attn_bias padding and masking (only if bias has non-zero dimensions)
    if attn_bias_trunked.numel() > 0:
        # Check padding in the K dimension (last dim)
        if k_padding > 0 and padded_k_len > 0:
            # Bias corresponding to padded K should also be masked (large negative)
            # Check the last k_padding columns across all Q trunks/positions
            k_padded_indices_start = padded_k_len - k_padding
            assert torch.all(attn_bias_trunked[..., k_padded_indices_start:] <= -1e9) # Check for large negative

        # Check masking in the Q dimension (dim=-2)
        if q_padding > 0 and num_q_trunks > 0:
            # Bias corresponding to padded Q should be masked (large negative)
            # Check the last q_padding rows of the last q_trunk
            assert torch.all(attn_bias_trunked[:, -1, -q_padding:, :] <= -1e9) # Check for large negative

        # Sanity check: if no padding, bias should not be all large negative (unless input was?)
        if q_padding == 0 and k_padding == 0 and q_len > 0 and k_len > 0:
             # Check that *not all* elements are masked. Some could be negative in original bias.
             assert torch.any(attn_bias_trunked > -1e9)


@pytest.mark.parametrize(
    "batch_size, q_len, k_len, v_len, n_queries, n_keys, d_head, provide_attn_bias",
    [
        (2, 10, 12, 12, 16, 16, 32, False), # q_len <= n_queries, k_len <= n_keys
        (1, 4, 4, 4, 8, 8, 16, True),     # Very small lengths, with bias
        (3, 15, 15, 15, 16, 20, 8, False),  # q_len <= n_queries, k_len <= n_keys
        (2, 8, 16, 16, 8, 16, 64, True),   # Exact match q_len=n_queries, k_len=n_keys
        (1, 0, 5, 5, 8, 8, 16, False),    # Zero length Q triggers bypass? (Check behavior) -> Should likely not bypass
        (1, 5, 0, 0, 8, 8, 16, True),     # Zero length K/V triggers bypass? -> Should likely not bypass
        (1, 0, 0, 0, 8, 8, 16, False),    # Zero length all -> Should likely not bypass
    ]
)
def test_rearrange_to_dense_trunk_small_tensor_bypass(
    batch_size: int, q_len: int, k_len: int, v_len: int, n_queries: int, n_keys: int, d_head: int,
    provide_attn_bias: bool
):
    """
    Tests the small tensor bypass condition in rearrange_to_dense_trunk.
    Expects no trunking/padding and padding_length = 0.
    Zero-length inputs should generally *not* trigger the bypass unless n_queries/n_keys are also zero/negative (error case).
    """
    # Bypass condition requires *positive* lengths <= trunk sizes
    is_bypass_expected = q_len > 0 and k_len > 0 and q_len <= n_queries and k_len <= n_keys

    if not is_bypass_expected:
         # Rerun the standard test logic if bypass isn't expected
         # This avoids duplicating the non-bypass tests here
         # Or simply skip if the parameters don't meet bypass criteria
         pytest.skip("Skipping bypass test for parameters not meeting bypass criteria (positive lengths <= trunk sizes)")


    # Assertions for test setup validity
    assert v_len == k_len, "Setup Error: V length must match K length for this test"
    assert is_bypass_expected, "Setup Error: Test logic assumes bypass condition is met"

    q_shape = (batch_size, q_len, d_head)
    k_shape = (batch_size, k_len, d_head)
    v_shape = (batch_size, v_len, d_head)
    attn_bias_shape = (batch_size, q_len, k_len)

    q = torch.randn(*q_shape)
    k = torch.randn(*k_shape)
    v = torch.randn(*v_shape)
    attn_bias_in = torch.randn(*attn_bias_shape) if provide_attn_bias else None

    q_out, k_out, v_out, padding_length, attn_bias_out = rearrange_to_dense_trunk(
        q, k, v, n_queries, n_keys, attn_bias=attn_bias_in
    )

    # --- Assertions for Bypass ---
    # Expect shape (Batch, 1, SeqLen, D_Head) - assuming it adds a 'trunk' dim of 1
    expected_q_shape = (batch_size, 1, q_len, d_head)
    expected_k_shape = (batch_size, 1, k_len, d_head)
    expected_v_shape = (batch_size, 1, v_len, d_head)

    assert q_out.shape == expected_q_shape
    assert k_out.shape == expected_k_shape
    assert v_out.shape == expected_v_shape

    # Check padding length is zero
    assert padding_length == 0

    # Check attn_bias handling
    # Expect shape (Batch, 1, QLen, KLen)
    expected_attn_bias_shape = (batch_size, 1, q_len, k_len)

    # Ensure attn_bias_out is a tensor before checking shape/content
    assert isinstance(attn_bias_out, torch.Tensor), f"Expected attn_bias_out to be Tensor, got {type(attn_bias_out)}"
    assert attn_bias_out.shape == expected_attn_bias_shape

    if provide_attn_bias:
        # Ensure the content matches the input bias (potentially reshaped)
        # Use squeeze(1) to remove the added trunk dimension for comparison
        assert attn_bias_in is not None, "attn_bias_in should be a Tensor when provide_attn_bias is True"
        assert torch.allclose(attn_bias_out.squeeze(1), attn_bias_in)
    else:
        # Expect a zero tensor with the correct shape
        assert torch.all(attn_bias_out == 0.0) # Correct usage of torch.all

    # Check that original tensors were not modified (if function avoids in-place)
    # Compare content after removing the added dimension
    assert torch.allclose(q_out.squeeze(1), q)
    assert torch.allclose(k_out.squeeze(1), k)
    assert torch.allclose(v_out.squeeze(1), v)


@pytest.mark.parametrize(
    "batch_size, q_len, k_len, v_len, n_queries, n_keys, d_head, expected_error",
    [
        (1, 10, 10, 10, 0, 8, 16, ValueError), # Invalid n_queries (expect ValueError or ZeroDivisionError)
        (1, 10, 10, 10, 8, 0, 16, ValueError), # Invalid n_keys (expect ValueError or ZeroDivisionError)
        (1, 10, 10, 10, -1, 8, 16, ValueError), # Invalid n_queries
        (1, 10, 10, 10, 8, -1, 16, ValueError), # Invalid n_keys
        # Test case for mismatched K/V lengths, if the function should raise an error
        # (1, 20, 20, 18, 8, 8, 16, ValueError), # Mismatched K/V length (if checked by func)
    ]
)
def test_rearrange_to_dense_trunk_error_cases(
    batch_size: int, q_len: int, k_len: int, v_len: int, n_queries: int, n_keys: int, d_head: int,
    expected_error: type
):
    """Tests rearrange_to_dense_trunk with invalid inputs expecting errors."""
    # Skip if it's a bypass case, as errors might differ or not occur for invalid n_queries/n_keys
    # Bypass requires positive lengths, so zero/negative n_queries/n_keys are tested here.
    # if q_len > 0 and k_len > 0 and q_len <= n_queries and k_len <= n_keys:
    #      pytest.skip("Skipping error case that might trigger bypass with valid lengths")
    # Let invalid n_queries/n_keys proceed regardless of bypass potential

    q_shape = (batch_size, q_len, d_head)
    k_shape = (batch_size, k_len, d_head)
    v_shape = (batch_size, v_len, d_head) # Use potentially mismatched v_len

    q = torch.randn(*q_shape) if q_len > 0 else torch.empty(*q_shape)
    k = torch.randn(*k_shape) if k_len > 0 else torch.empty(*k_shape)
    v = torch.randn(*v_shape) if v_len > 0 else torch.empty(*v_shape)

    with pytest.raises(expected_error):
        rearrange_to_dense_trunk(q, k, v, n_queries, n_keys, attn_bias=None)
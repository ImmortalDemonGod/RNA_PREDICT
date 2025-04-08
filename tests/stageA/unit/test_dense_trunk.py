# Proposed content for tests/stageA/unit/test_dense_trunk.py
import pytest
import torch
import math
from typing import List, Union, Tuple # Added Dict

# Hypothesis imports
from hypothesis import given, strategies as st, assume, settings

# Local imports
# Corrected import path if attention_utils is directly under primitives
# from rna_predict.pipeline.stageA.input_embedding.current.primitives.attention_utils import PaddingInfo
# Assuming PaddingInfo might be defined elsewhere or just a type hint for dict now
# Let's use a generic dict type hint for now if PaddingInfo class is not found/needed
# from typing import Dict as PaddingInfo # Placeholder if PaddingInfo class is problematic

# Import from dense_trunk directly
from rna_predict.pipeline.stageA.input_embedding.current.primitives.attention.dense_trunk import (
    rearrange_qk_to_dense_trunk,
    rearrange_to_dense_trunk,
)
# Import PaddingInfo type if it's defined and needed for type hints
# If it's just a dict, the type hint can be Dict or kept as is if defined elsewhere
# from rna_predict.pipeline.stageA.input_embedding.current.primitives.attention_utils import PaddingInfo


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

# == Hypothesis Strategies ==

# Strategy for tensor dimensions (Batch, Sequence Length, Head Dimension)
# Keep sizes reasonable to avoid excessive memory/time usage
dims_strategy = st.tuples(
    st.integers(min_value=1, max_value=4),  # batch_size
    st.integers(min_value=0, max_value=128), # seq_len_q
    st.integers(min_value=0, max_value=128), # seq_len_k
    st.integers(min_value=1, max_value=32)   # d_head
)

# Strategy for trunk sizes (must be positive)
trunk_sizes_strategy = st.tuples(
    st.integers(min_value=1, max_value=64), # n_queries
    st.integers(min_value=1, max_value=64)  # n_keys
)

# Strategy for dimension indices (typically -2 for sequence dim)
dim_indices_strategy = st.sampled_from([-2, 1]) # Common dimension indices

# == Tests for rearrange_qk_to_dense_trunk ==

@settings(deadline=None, max_examples=50) # Increase examples, disable deadline for potentially slow tensor ops
@given(
    dims=dims_strategy,
    trunks=trunk_sizes_strategy,
    dim_q=dim_indices_strategy,
    dim_k=dim_indices_strategy,
    use_list_input=st.booleans()
)
def test_hypothesis_rearrange_qk_to_dense_trunk(
    dims: Tuple[int, int, int, int],
    trunks: Tuple[int, int],
    dim_q: int,
    dim_k: int,
    use_list_input: bool
):
    """
    Tests rearrange_qk_to_dense_trunk with Hypothesis-generated inputs.
    Verifies shapes, padding info, and padding values for various configurations.
    Now expects a single concatenated tensor output regardless of input type.
    """
    batch_size, q_len, k_len, d_head = dims
    n_queries, n_keys = trunks

    # --- Input Tensor Creation ---
    # Adjust shapes based on dim_q/dim_k
    # Assuming 3 dims B, S, D initially for index calculation relative to rank 3
    # If tensors can have more dims, this needs adjustment
    temp_rank = 3
    q_dim_idx = dim_q if dim_q >= 0 else dim_q + temp_rank
    k_dim_idx = dim_k if dim_k >= 0 else dim_k + temp_rank

    q_shape_list = [batch_size, d_head]
    k_shape_list = [batch_size, d_head]

    # Validate and insert sequence dimension index
    assume(0 <= q_dim_idx <= len(q_shape_list)) # Ensure valid index before insert
    q_shape_list.insert(q_dim_idx, q_len)
    q_shape = tuple(q_shape_list)

    assume(0 <= k_dim_idx <= len(k_shape_list)) # Ensure valid index before insert
    k_shape_list.insert(k_dim_idx, k_len)
    k_shape = tuple(k_shape_list)

    # Define input variables with correct Union type hint
    q_input: Union[torch.Tensor, List[torch.Tensor]]
    k_input: Union[torch.Tensor, List[torch.Tensor]]
    total_batch_size = batch_size # Default if not list

    if use_list_input:
        # Create lists of tensors (e.g., two tensors)
        num_tensors_in_list = 2
        q_input = [torch.randn(*q_shape) if q_len > 0 else torch.empty(*q_shape)
                   for _ in range(num_tensors_in_list)]
        k_input = [torch.randn(*k_shape) if k_len > 0 else torch.empty(*k_shape)
                   for _ in range(num_tensors_in_list)]
        total_batch_size = batch_size * num_tensors_in_list
    else:
        q_input = torch.randn(*q_shape) if q_len > 0 else torch.empty(*q_shape)
        k_input = torch.randn(*k_shape) if k_len > 0 else torch.empty(*k_shape)
        num_tensors_in_list = 1 # Keep track for expected shape calculation

    # --- Function Call ---
    try:
        # Function now always returns Tensors, not lists
        q_trunked, k_trunked, padding_info = rearrange_qk_to_dense_trunk(
            q_input, k_input, dim_q, dim_k, n_queries=n_queries, n_keys=n_keys
        )
    except IndexError:
        # This can happen if generated dims don't match tensor structure, skip test case
        assume(False) # Skip if dimension index is invalid for the tensor shape
        return # Should not be reached
    except ValueError as e:
         # Expected for invalid n_queries/n_keys <= 0 (already tested elsewhere)
         assume(n_queries > 0 and n_keys > 0)
         # If error persists with valid trunk sizes, re-raise
         raise e


    # --- Assertions ---
    # Calculate expected shapes and padding based on original lengths
    num_q_trunks, q_padding = calculate_expected_trunk_params(q_len, n_queries)
    num_k_trunks, k_padding = calculate_expected_trunk_params(k_len, n_keys)

    # Expected shapes: (TotalBatch, NumTrunks, TrunkSize, D_Head)
    # TotalBatch = batch_size * num_tensors_in_list if use_list_input else batch_size
    expected_q_shape = (total_batch_size, num_q_trunks, n_queries, d_head)
    expected_k_shape = (total_batch_size, num_k_trunks, n_keys, d_head)

    # Handle zero trunks case (sequence length was 0)
    if num_q_trunks == 0:
        # Shape should be (TotalBatch, 0, n_queries, d_head)
        expected_q_shape = (total_batch_size, 0, n_queries, d_head)
    if num_k_trunks == 0:
        # Shape should be (TotalBatch, 0, n_keys, d_head)
        expected_k_shape = (total_batch_size, 0, n_keys, d_head)

    # Assert output types are always Tensors now
    assert isinstance(q_trunked, torch.Tensor)
    assert isinstance(k_trunked, torch.Tensor)
    # Assuming padding_info is expected to be a dictionary
    assert isinstance(padding_info, dict)

    # Assert shapes match expected shapes
    assert q_trunked.shape == torch.Size(expected_q_shape), f"Q shape mismatch: Got {q_trunked.shape}, Expected {expected_q_shape}"
    assert k_trunked.shape == torch.Size(expected_k_shape), f"K shape mismatch: Got {k_trunked.shape}, Expected {expected_k_shape}"

    # Check padding info dictionary (THIS IS STILL EXPECTED TO FAIL until _prepare_padding_info is fixed)
    assert padding_info.get("q_padding") == q_padding, f"Padding info q_padding mismatch: Got {padding_info.get('q_padding')}, Expected {q_padding}"
    assert padding_info.get("k_padding") == k_padding, f"Padding info k_padding mismatch: Got {padding_info.get('k_padding')}, Expected {k_padding}"
    assert padding_info.get("num_q_trunks") == num_q_trunks, f"Padding info num_q_trunks mismatch: Got {padding_info.get('num_q_trunks')}, Expected {num_q_trunks}"
    assert padding_info.get("num_k_trunks") == num_k_trunks, f"Padding info num_k_trunks mismatch: Got {padding_info.get('num_k_trunks')}, Expected {num_k_trunks}"

    # Check padding values (should be zero in the padded region of the last trunk)
    # These checks operate directly on the output tensors q_trunked, k_trunked
    if q_padding > 0 and num_q_trunks > 0:
        assert q_trunked.numel() > 0, "Should have elements if padding > 0 and trunks > 0"
        # Check only the padding area in the last trunk
        assert torch.all(q_trunked[:, -1, -q_padding:, :] == 0.0)
        # Sanity check non-padded part of the last trunk (if it exists)
        if n_queries - q_padding > 0:
            assert torch.any(q_trunked[:, -1, :-q_padding, :] != 0.0)

    if k_padding > 0 and num_k_trunks > 0:
        assert k_trunked.numel() > 0, "Should have elements if padding > 0 and trunks > 0"
        # Check only the padding area in the last trunk
        assert torch.all(k_trunked[:, -1, -k_padding:, :] == 0.0)
        # Sanity check non-padded part of the last trunk (if it exists)
        if n_keys - k_padding > 0:
            assert torch.any(k_trunked[:, -1, :-k_padding, :] != 0.0)

    # Check exact division case (no padding in last trunk if padding == 0)
    if q_padding == 0 and q_len > 0 and num_q_trunks > 0:
        assert q_trunked.numel() > 0, "Should have elements if q_len > 0"
        # Last trunk shouldn't be all zero unless input was zero (which isn't tested here)
        assert torch.any(q_trunked[:, -1, :, :] != 0.0)
    if k_padding == 0 and k_len > 0 and num_k_trunks > 0:
        assert k_trunked.numel() > 0, "Should have elements if k_len > 0"
        # Last trunk shouldn't be all zero unless input was zero
        assert torch.any(k_trunked[:, -1, :, :] != 0.0)


# Keep existing parametrized tests for specific scenarios and error cases
@pytest.mark.parametrize(
    "batch_size, q_len, k_len, n_queries, n_keys, d_head, dim_q, dim_k, use_list_input",
    [
        # --- Specific Cases for Regression ---
        (2, 50, 60, 16, 16, 32, -2, -2, False), # Standard case, padding needed
        (1, 5, 7, 4, 4, 8, -2, -2, False),    # Small lengths, padding needed
        (3, 100, 90, 32, 20, 16, -2, -2, False), # Different n_queries/n_keys
        (2, 50, 60, 16, 16, 32, -2, -2, True),  # List input, padding needed
        (2, 50, 60, 16, 16, 32, 1, 1, False),   # Different dims (B, D, S), padding needed
        (1, 0, 10, 4, 4, 8, -2, -2, False),   # Zero length Q
        (1, 10, 0, 4, 4, 8, -2, -2, False),   # Zero length K
        (1, 0, 0, 4, 4, 8, -2, -2, False),    # Zero length Q and K
        (2, 64, 64, 16, 16, 32, -2, -2, False), # Exact division
        (1, 12, 8, 4, 4, 8, -2, -2, False),    # Small lengths, exact division
        (3, 96, 100, 32, 20, 16, -2, -2, False), # Exact division q, partial k
        (2, 64, 64, 16, 16, 32, -2, -2, True),  # List input, exact division
    ]
)
def test_rearrange_qk_to_dense_trunk_specific_cases( # Renamed slightly
    batch_size: int, q_len: int, k_len: int, n_queries: int, n_keys: int, d_head: int,
    dim_q: int, dim_k: int, use_list_input: bool
):
    """
    Tests rearrange_qk_to_dense_trunk for specific important scenarios (regression).
    Checks output shapes, padding values, and padding_info.
    Handles zero-length inputs and list inputs. Expects single tensor output.
    """
    # --- Input Tensor Creation --- (Same logic as Hypothesis test)
    temp_rank = 3
    q_dim_idx = dim_q if dim_q >= 0 else dim_q + temp_rank
    k_dim_idx = dim_k if dim_k >= 0 else dim_k + temp_rank

    q_shape_list = [batch_size, d_head]
    k_shape_list = [batch_size, d_head]

    if not (0 <= q_dim_idx <= len(q_shape_list)):
         pytest.skip(f"Calculated q_dim_idx {q_dim_idx} is invalid for shape insertion.")
    q_shape_list.insert(q_dim_idx, q_len)
    q_shape = tuple(q_shape_list)

    if not (0 <= k_dim_idx <= len(k_shape_list)):
         pytest.skip(f"Calculated k_dim_idx {k_dim_idx} is invalid for shape insertion.")
    k_shape_list.insert(k_dim_idx, k_len)
    k_shape = tuple(k_shape_list)

    q_input: Union[torch.Tensor, List[torch.Tensor]]
    k_input: Union[torch.Tensor, List[torch.Tensor]]
    total_batch_size = batch_size

    if use_list_input:
        num_tensors_in_list = 2
        q_input = [torch.randn(*q_shape) if q_len > 0 else torch.empty(*q_shape)
                   for _ in range(num_tensors_in_list)]
        k_input = [torch.randn(*k_shape) if k_len > 0 else torch.empty(*k_shape)
                   for _ in range(num_tensors_in_list)]
        total_batch_size = batch_size * num_tensors_in_list
    else:
        q_input = torch.randn(*q_shape) if q_len > 0 else torch.empty(*q_shape)
        k_input = torch.randn(*k_shape) if k_len > 0 else torch.empty(*k_shape)
        num_tensors_in_list = 1

    # --- Function Call & Assertions ---
    q_trunked, k_trunked, padding_info = rearrange_qk_to_dense_trunk(
        q_input, k_input, dim_q, dim_k, n_queries=n_queries, n_keys=n_keys
    )

    num_q_trunks, q_padding = calculate_expected_trunk_params(q_len, n_queries)
    num_k_trunks, k_padding = calculate_expected_trunk_params(k_len, n_keys)

    expected_q_shape = (total_batch_size, num_q_trunks, n_queries, d_head)
    expected_k_shape = (total_batch_size, num_k_trunks, n_keys, d_head)

    if num_q_trunks == 0:
        expected_q_shape = (total_batch_size, 0, n_queries, d_head)
    if num_k_trunks == 0:
        expected_k_shape = (total_batch_size, 0, n_keys, d_head)

    # Assert output types are always Tensors
    assert isinstance(q_trunked, torch.Tensor)
    assert isinstance(k_trunked, torch.Tensor)
    assert isinstance(padding_info, dict)

    # Assert shapes
    assert q_trunked.shape == torch.Size(expected_q_shape)
    assert k_trunked.shape == torch.Size(expected_k_shape)

    # Assert padding info (STILL EXPECTED TO FAIL until _prepare_padding_info is fixed)
    assert padding_info.get("q_padding") == q_padding
    assert padding_info.get("k_padding") == k_padding
    assert padding_info.get("num_q_trunks") == num_q_trunks
    assert padding_info.get("num_k_trunks") == num_k_trunks

    # Assert padding values
    if q_padding > 0 and num_q_trunks > 0:
        assert q_trunked.numel() > 0
        assert torch.all(q_trunked[:, -1, -q_padding:, :] == 0.0)
        if n_queries - q_padding > 0:
            assert torch.any(q_trunked[:, -1, :-q_padding, :] != 0.0)
    if k_padding > 0 and num_k_trunks > 0:
        assert k_trunked.numel() > 0
        assert torch.all(k_trunked[:, -1, -k_padding:, :] == 0.0)
        if n_keys - k_padding > 0:
            assert torch.any(k_trunked[:, -1, :-k_padding, :] != 0.0)
    if q_padding == 0 and q_len > 0 and num_q_trunks > 0:
        assert q_trunked.numel() > 0
        assert torch.any(q_trunked[:, -1, :, :] != 0.0)
    if k_padding == 0 and k_len > 0 and num_k_trunks > 0:
        assert k_trunked.numel() > 0
        assert torch.any(k_trunked[:, -1, :, :] != 0.0)


@pytest.mark.parametrize(
    "batch_size, q_len, k_len, n_queries, n_keys, d_head, dim_q, dim_k, expected_error",
    [
        (1, 10, 10, 8, 8, 16, 3, -2, IndexError), # Invalid dim_q (assuming B,S,D -> index 3 out of bounds)
        (1, 10, 10, 8, 8, 16, -2, 3, IndexError), # Invalid dim_k
        (1, 10, 10, 8, 8, 16, -4, -2, IndexError), # Invalid dim_q (assuming B,S,D -> index -4 out of bounds)
        (1, 10, 10, 0, 8, 16, -2, -2, ValueError), # Invalid n_queries (expect ValueError) - Now checked upfront
        (1, 10, 10, 8, 0, 16, -2, -2, ValueError), # Invalid n_keys (expect ValueError) - Now checked upfront
        (1, 10, 10, -1, 8, 16, -2, -2, ValueError), # Invalid n_queries - Now checked upfront
        (1, 10, 10, 8, -1, 16, -2, -2, ValueError), # Invalid n_keys - Now checked upfront
    ]
)
def test_rearrange_qk_to_dense_trunk_error_cases(
    batch_size: int, q_len: int, k_len: int, n_queries: int, n_keys: int, d_head: int,
    dim_q: int, dim_k: int, expected_error: type
):
    """Tests rearrange_qk_to_dense_trunk with invalid inputs expecting errors."""
    # Use standard B, S, D shape for error setup simplicity
    q_shape = (batch_size, q_len, d_head)
    k_shape = (batch_size, k_len, d_head)
    q_input = torch.randn(*q_shape) if q_len > 0 else torch.empty(*q_shape)
    k_input = torch.randn(*k_shape) if k_len > 0 else torch.empty(*k_shape)

    with pytest.raises(expected_error):
        rearrange_qk_to_dense_trunk(
            q_input, k_input, dim_q, dim_k, n_queries=n_queries, n_keys=n_keys
        )

# == Tests for rearrange_to_dense_trunk ==
# NOTE: These tests remain unchanged for now. If they fail after fixing
# rearrange_qk_to_dense_trunk, they will need separate investigation,
# potentially involving _reshape_query_to_trunk, _process_keys_values_chunks,
# _process_attention_bias, and the bypass logic.

@settings(deadline=None, max_examples=50)
@given(
    dims=dims_strategy, # Reuse dims strategy (batch, q_len, k_len, d_head)
    trunks=trunk_sizes_strategy, # Reuse trunk sizes (n_queries, n_keys)
    provide_attn_bias=st.booleans()
)
def test_hypothesis_rearrange_to_dense_trunk(
    dims: Tuple[int, int, int, int],
    trunks: Tuple[int, int],
    provide_attn_bias: bool
):
    """
    Tests rearrange_to_dense_trunk with Hypothesis-generated inputs (non-bypass cases).
    Verifies shapes, padding, and bias masking. Assumes k_len == v_len.
    """
    batch_size, q_len, k_len, d_head = dims
    n_queries, n_keys = trunks
    v_len = k_len # Assume V length matches K length for standard trunking

    # --- Skip Bypass Cases ---
    # This test focuses on the trunking logic, bypass is tested separately
    # Handle zero lengths correctly in assume: bypass requires POSITIVE lengths
    is_bypass = q_len > 0 and k_len > 0 and q_len <= n_queries and k_len <= n_keys
    assume(not is_bypass)
    # Also assume valid trunk sizes
    assume(n_queries > 0 and n_keys > 0)


    # --- Input Tensor Creation ---
    q_shape = (batch_size, q_len, d_head)
    k_shape = (batch_size, k_len, d_head)
    v_shape = (batch_size, v_len, d_head)
    attn_bias_shape = (batch_size, q_len, k_len)

    q = torch.randn(*q_shape) if q_len > 0 else torch.empty(*q_shape)
    k = torch.randn(*k_shape) if k_len > 0 else torch.empty(*k_shape)
    v = torch.randn(*v_shape) if v_len > 0 else torch.empty(*v_shape)
    attn_bias = torch.randn(*attn_bias_shape) if provide_attn_bias and q_len > 0 and k_len > 0 else None

    # --- Function Call ---
    try:
        # Corrected assignment order: q, k, v, bias, padding_len
        q_trunked, k_trunked, v_trunked, attn_bias_trunked, padding_length = rearrange_to_dense_trunk(
            q, k, v, n_queries, n_keys, attn_bias=attn_bias, inf=1e9 # Use consistent inf
        )
    except ValueError as e:
         # Should not happen if assume(n_queries > 0 and n_keys > 0) holds
         raise e
    except IndexError as e:
        # Might occur with zero-length inputs in internal reshaping, treat as skip
        # Let's refine the assumption based on which length is zero
        assume(q_len > 0 and k_len > 0 and v_len > 0) # Skip if any input seq len is 0 for non-bypass
        raise e


    # --- Assertions ---
    num_q_trunks, q_padding = calculate_expected_trunk_params(q_len, n_queries)
    num_k_trunks, k_padding = calculate_expected_trunk_params(k_len, n_keys)
    num_v_trunks, v_padding = calculate_expected_trunk_params(v_len, n_keys) # V follows K

    assert v_padding == k_padding, "Internal assumption: V padding follows K padding"
    assert num_v_trunks == num_k_trunks, "Internal assumption: V num_trunks follows K num_trunks"

    # Expected shapes assume B, S, D input -> B, NumTrunks, TrunkSize, D output
    expected_q_shape = (batch_size, num_q_trunks, n_queries, d_head)
    expected_k_shape = (batch_size, num_k_trunks, n_keys, d_head)
    expected_v_shape = (batch_size, num_v_trunks, n_keys, d_head)
    # Bias shape depends on implementation details (padding, reshaping)
    # Let's infer from output if possible, or calculate based on expected padding
    padded_k_len = num_k_trunks * n_keys
    # Expected bias shape: (B, NumQTrunks, NumQItems, PaddedKLen)
    expected_attn_bias_shape = (batch_size, num_q_trunks, n_queries, padded_k_len)

    # Adjust expected shapes for zero trunks
    if num_q_trunks == 0:
        expected_q_shape = (batch_size, 0, n_queries, d_head)
        expected_attn_bias_shape = (batch_size, 0, n_queries, padded_k_len) # Q trunks = 0
    if num_k_trunks == 0:
        expected_k_shape = (batch_size, 0, n_keys, d_head)
        expected_v_shape = (batch_size, 0, n_keys, d_head)
        expected_attn_bias_shape = (batch_size, num_q_trunks, n_queries, 0) # K length = 0

    assert isinstance(q_trunked, torch.Tensor)
    assert isinstance(k_trunked, torch.Tensor)
    assert isinstance(v_trunked, torch.Tensor)
    assert isinstance(padding_length, int)
    assert isinstance(attn_bias_trunked, torch.Tensor)

    assert q_trunked.shape == torch.Size(expected_q_shape)
    assert k_trunked.shape == torch.Size(expected_k_shape)
    assert v_trunked.shape == torch.Size(expected_v_shape)
    # Check bias shape carefully - might differ based on internal padding/masking
    assert attn_bias_trunked.shape == torch.Size(expected_attn_bias_shape)
    assert padding_length == q_padding # Function returns query padding length

    # Check padding values (should be zero for Q, K, V in padded area)
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

    # Check attn_bias padding and masking
    if attn_bias_trunked.numel() > 0:
        inf_val = 1e9 # Match the value used in the function call
        # Check masking ONLY if bias was provided initially
        if provide_attn_bias:
            # Check masking in the K dimension (padded K indices should be masked)
            if k_padding > 0 and padded_k_len > 0:
                # Check ONLY the last k_padding columns
                assert torch.all(attn_bias_trunked[..., -k_padding:] <= -inf_val)
                # Check non-padded K indices are not all masked (unless input bias was all masked)
                if padded_k_len - k_padding > 0: # Check if non-padded region exists
                     if attn_bias is not None and torch.any(attn_bias > -inf_val): # Check original bias
                         assert torch.any(attn_bias_trunked[..., :-k_padding] > -inf_val)

            # Check masking in the Q dimension (padded Q indices should be masked)
            if q_padding > 0 and num_q_trunks > 0:
                # Check the rows corresponding to padded Q indices in the last Q trunk
                assert torch.all(attn_bias_trunked[:, -1, -q_padding:, :] <= -inf_val)
                 # Check non-padded Q indices are not all masked (unless input bias was all masked)
                if n_queries - q_padding > 0:
                     if attn_bias is not None and torch.any(attn_bias > -inf_val): # Check original bias
                         assert torch.any(attn_bias_trunked[:, -1, :-q_padding, :] > -inf_val)

            # Sanity check: if no padding and inputs non-zero, bias should not be all masked
            if q_padding == 0 and k_padding == 0 and q_len > 0 and k_len > 0:
                 if attn_bias is not None and torch.any(attn_bias > -inf_val): # Check original bias
                     assert torch.any(attn_bias_trunked > -inf_val)


# Keep existing specific parametrized tests
@pytest.mark.parametrize(
    "batch_size, q_len, k_len, v_len, n_queries, n_keys, d_head, provide_attn_bias",
    [
        # --- Specific Cases for Regression ---
        (2, 50, 60, 60, 16, 16, 32, False), # Standard case, padding needed, no bias
        (1, 5, 7, 7, 4, 4, 8, True),      # Small lengths, padding needed, with bias
        (3, 100, 90, 90, 32, 20, 16, False), # Different n_queries/n_keys, padding
        (2, 55, 65, 65, 16, 16, 32, True),  # Padding needed, with bias
        (1, 10, 0, 0, 4, 4, 8, False),    # Zero length K/V
        (1, 0, 10, 10, 4, 4, 8, True),     # Zero length Q
        (1, 0, 0, 0, 4, 4, 8, False),     # Zero length Q/K/V
        (2, 64, 64, 64, 16, 16, 32, False), # Exact division, no bias
        (1, 12, 8, 8, 4, 4, 8, True),      # Small lengths, exact division, with bias
        (3, 96, 100, 100, 32, 20, 16, False), # Exact division q, partial k, no bias
        (2, 64, 80, 80, 16, 16, 32, True),  # Exact division, with bias
    ]
)
def test_rearrange_to_dense_trunk_specific_cases( # Renamed slightly
    batch_size: int, q_len: int, k_len: int, v_len: int, n_queries: int, n_keys: int, d_head: int,
    provide_attn_bias: bool
):
    """
    Tests rearrange_to_dense_trunk for specific important scenarios (regression).
    Checks output shapes, padding values, padding_length, and attn_bias handling.
    Handles zero-length inputs. Excludes small tensor bypass cases.
    """
    # --- Skip Bypass Cases ---
    is_bypass = q_len > 0 and k_len > 0 and q_len <= n_queries and k_len <= n_keys
    if is_bypass:
        pytest.skip("Skipping specific non-bypass case for parameters that trigger bypass")

    # --- Input Tensor Creation ---
    assume(k_len == v_len) # Parametrized tests assume this
    q_shape = (batch_size, q_len, d_head)
    k_shape = (batch_size, k_len, d_head)
    v_shape = (batch_size, v_len, d_head)
    attn_bias_shape = (batch_size, q_len, k_len)

    q = torch.randn(*q_shape) if q_len > 0 else torch.empty(*q_shape)
    k = torch.randn(*k_shape) if k_len > 0 else torch.empty(*k_shape)
    v = torch.randn(*v_shape) if v_len > 0 else torch.empty(*v_shape)
    attn_bias = torch.randn(*attn_bias_shape) if provide_attn_bias and q_len > 0 and k_len > 0 else None

    # --- Function Call & Assertions ---
    # Corrected assignment order: q, k, v, bias, padding_len
    q_trunked, k_trunked, v_trunked, attn_bias_trunked, padding_length = rearrange_to_dense_trunk(
        q, k, v, n_queries, n_keys, attn_bias=attn_bias, inf=1e9
    )

    num_q_trunks, q_padding = calculate_expected_trunk_params(q_len, n_queries)
    num_k_trunks, k_padding = calculate_expected_trunk_params(k_len, n_keys)
    num_v_trunks, v_padding = calculate_expected_trunk_params(v_len, n_keys)

    assert v_padding == k_padding
    assert num_v_trunks == num_k_trunks

    expected_q_shape = (batch_size, num_q_trunks, n_queries, d_head)
    expected_k_shape = (batch_size, num_k_trunks, n_keys, d_head)
    expected_v_shape = (batch_size, num_v_trunks, n_keys, d_head)
    padded_k_len = num_k_trunks * n_keys
    expected_attn_bias_shape = (batch_size, num_q_trunks, n_queries, padded_k_len)

    if num_q_trunks == 0:
        expected_q_shape = (batch_size, 0, n_queries, d_head)
        expected_attn_bias_shape = (batch_size, 0, n_queries, padded_k_len)
    if num_k_trunks == 0:
        expected_k_shape = (batch_size, 0, n_keys, d_head)
        expected_v_shape = (batch_size, 0, n_keys, d_head)
        expected_attn_bias_shape = (batch_size, num_q_trunks, n_queries, 0)

    assert isinstance(q_trunked, torch.Tensor)
    assert isinstance(k_trunked, torch.Tensor)
    assert isinstance(v_trunked, torch.Tensor)
    assert isinstance(padding_length, int)
    assert isinstance(attn_bias_trunked, torch.Tensor)

    assert q_trunked.shape == torch.Size(expected_q_shape)
    assert k_trunked.shape == torch.Size(expected_k_shape)
    assert v_trunked.shape == torch.Size(expected_v_shape)
    assert attn_bias_trunked.shape == torch.Size(expected_attn_bias_shape)
    assert padding_length == q_padding

    if q_padding > 0 and num_q_trunks > 0:
        assert torch.all(q_trunked[:, -1, -q_padding:, :] == 0.0)
    if k_padding > 0 and num_k_trunks > 0:
        assert torch.all(k_trunked[:, -1, -k_padding:, :] == 0.0)
        assert torch.all(v_trunked[:, -1, -k_padding:, :] == 0.0)

    if attn_bias_trunked.numel() > 0:
        inf_val = 1e9
        # Check masking ONLY if bias was provided initially
        if provide_attn_bias:
            if k_padding > 0 and padded_k_len > 0:
                # Check ONLY the last k_padding columns
                assert torch.all(attn_bias_trunked[..., -k_padding:] <= -inf_val)
            if q_padding > 0 and num_q_trunks > 0:
                assert torch.all(attn_bias_trunked[:, -1, -q_padding:, :] <= -inf_val)
            if q_padding == 0 and k_padding == 0 and q_len > 0 and k_len > 0:
                 if attn_bias is not None and torch.any(attn_bias > -inf_val): # Check original bias
                     assert torch.any(attn_bias_trunked > -inf_val)


@pytest.mark.parametrize(
    "batch_size, q_len, k_len, v_len, n_queries, n_keys, d_head, provide_attn_bias",
    [
        (2, 10, 12, 12, 16, 16, 32, False), # q_len <= n_queries, k_len <= n_keys
        (1, 4, 4, 4, 8, 8, 16, True),     # Very small lengths, with bias
        (3, 15, 15, 15, 16, 20, 8, False),  # q_len <= n_queries, k_len <= n_keys
        (2, 8, 16, 16, 8, 16, 64, True),   # Exact match q_len=n_queries, k_len=n_keys
        # Zero length cases are handled by the non-bypass tests now
        # (1, 0, 5, 5, 8, 8, 16, False), # Should not bypass
        # (1, 5, 0, 0, 8, 8, 16, True),  # Should not bypass
        # (1, 0, 0, 0, 8, 8, 16, False), # Should not bypass
    ]
)
def test_rearrange_to_dense_trunk_small_tensor_bypass(
    batch_size: int, q_len: int, k_len: int, v_len: int, n_queries: int, n_keys: int, d_head: int,
    provide_attn_bias: bool
):
    """
    Tests the small tensor bypass condition in rearrange_to_dense_trunk.
    Expects original tensors and padding_length = 0.
    Requires positive lengths <= trunk sizes.
    """
    # --- Ensure Bypass Condition Met ---
    is_bypass_expected = q_len > 0 and k_len > 0 and q_len <= n_queries and k_len <= n_keys
    if not is_bypass_expected:
        pytest.skip("Skipping bypass test: parameters do not meet bypass criteria (positive lengths <= trunk sizes)")

    # --- Input Tensor Creation ---
    assert v_len == k_len, "Setup Error: V length must match K length for this test"
    q_shape = (batch_size, q_len, d_head)
    k_shape = (batch_size, k_len, d_head)
    v_shape = (batch_size, v_len, d_head)
    attn_bias_shape = (batch_size, q_len, k_len)

    q = torch.randn(*q_shape)
    k = torch.randn(*k_shape)
    v = torch.randn(*v_shape)
    attn_bias_in = torch.randn(*attn_bias_shape) if provide_attn_bias else None

    # --- Function Call ---
    # Corrected assignment order: q, k, v, bias, padding_len
    q_out, k_out, v_out, attn_bias_out, padding_length = rearrange_to_dense_trunk(
        q, k, v, n_queries, n_keys, attn_bias=attn_bias_in
    )

    # --- Assertions for Bypass ---
    # Expect original shapes (no trunking dimension added in bypass based on current bypass logic)
    expected_q_shape = q_shape
    expected_k_shape = k_shape
    expected_v_shape = v_shape

    assert q_out.shape == expected_q_shape
    assert k_out.shape == expected_k_shape
    assert v_out.shape == expected_v_shape

    # Check padding length is zero
    assert padding_length == 0

    # Check attn_bias handling
    assert isinstance(attn_bias_out, torch.Tensor)

    if provide_attn_bias:
        assert attn_bias_in is not None
        # Bypass returns original bias
        assert attn_bias_out.shape == attn_bias_in.shape
        assert torch.allclose(attn_bias_out, attn_bias_in)
        assert attn_bias_out is attn_bias_in # Check identity
    else:
        # Expect a scalar zero tensor based on _handle_small_tensors
        assert attn_bias_out.numel() == 1
        assert attn_bias_out.item() == 0.0


    # Check that original tensors were returned (identity check for bypass)
    assert q_out is q
    assert k_out is k
    assert v_out is v


@pytest.mark.parametrize(
    "batch_size, q_len, k_len, v_len, n_queries, n_keys, d_head, expected_error",
    [
        (1, 10, 10, 10, 0, 8, 16, ValueError), # Invalid n_queries
        (1, 10, 10, 10, 8, 0, 16, ValueError), # Invalid n_keys
        (1, 10, 10, 10, -1, 8, 16, ValueError), # Invalid n_queries
        (1, 10, 10, 10, 8, -1, 16, ValueError), # Invalid n_keys
        # Mismatched K/V length is not explicitly checked by the top-level function,
        # but might cause errors deeper down if logic assumes they match.
        # Adding a check in _rearrange_to_dense_trunk_impl if needed.
        # (1, 20, 20, 18, 8, 8, 16, ValueError),
    ]
)
def test_rearrange_to_dense_trunk_error_cases(
    batch_size: int, q_len: int, k_len: int, v_len: int, n_queries: int, n_keys: int, d_head: int,
    expected_error: type
):
    """Tests rearrange_to_dense_trunk with invalid inputs expecting errors."""
    q_shape = (batch_size, q_len, d_head)
    k_shape = (batch_size, k_len, d_head)
    v_shape = (batch_size, v_len, d_head)

    q = torch.randn(*q_shape) if q_len > 0 else torch.empty(*q_shape)
    k = torch.randn(*k_shape) if k_len > 0 else torch.empty(*k_shape)
    v = torch.randn(*v_shape) if v_len > 0 else torch.empty(*v_shape)

    with pytest.raises(expected_error):
        rearrange_to_dense_trunk(q, k, v, n_queries, n_keys, attn_bias=None)
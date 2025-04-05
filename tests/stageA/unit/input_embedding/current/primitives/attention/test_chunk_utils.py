# tests/stageA/unit/input_embedding/current/primitives/attention/test_chunk_utils.py
import pytest
import torch
from typing import Optional, List, Tuple

from rna_predict.pipeline.stageA.input_embedding.current.primitives.attention.chunk_utils import (
    _process_keys_values_chunks,
    KeysValuesChunkParams,
    _get_chunk_info,  # Import helper if needed for setup/verification
    _process_chunk,   # Import helper if needed for setup/verification
    _process_bias_chunk # Import helper if needed for setup/verification
)

# Constants for testing
BATCH_SIZE = 2
N_HEADS = 4
SEQ_LEN_K = 15  # Sequence length for keys/values
SEQ_LEN_Q = 10  # Sequence length for queries (relevant for bias shape)
D_HEAD = 8
CHUNK_SIZE = 5 # n_keys
INF = 1e9

@pytest.fixture
def base_tensors() -> Tuple[torch.Tensor, torch.Tensor]:
    """Provides base key and value tensors."""
    k = torch.randn(BATCH_SIZE, N_HEADS, SEQ_LEN_K, D_HEAD)
    v = torch.randn(BATCH_SIZE, N_HEADS, SEQ_LEN_K, D_HEAD)
    return k, v

@pytest.fixture
def attention_bias() -> torch.Tensor:
    """Provides a sample attention bias tensor."""
    # Bias shape: (batch_size, n_heads, seq_len_q, seq_len_k)
    return torch.randn(BATCH_SIZE, N_HEADS, SEQ_LEN_Q, SEQ_LEN_K)

@pytest.mark.parametrize(
    "seq_len_k_override, n_keys_override, provide_bias",
    [
        (15, 5, False),  # Case 1: Exact multiple chunks, no bias
        (15, 5, True),   # Case 2: Exact multiple chunks, with bias
        (17, 5, False),  # Case 3: Padding needed for last chunk, no bias
        (17, 5, True),   # Case 4: Padding needed for last chunk, with bias
        (5, 5, False),   # Case 5: Single chunk, no bias
        (5, 5, True),    # Case 6: Single chunk, with bias
        (3, 5, False),   # Case 7: Single partial chunk, no bias
        (3, 5, True),    # Case 8: Single partial chunk, with bias
    ],
    ids=[
        "exact_chunks_no_bias",
        "exact_chunks_with_bias",
        "padding_no_bias",
        "padding_with_bias",
        "single_chunk_no_bias",
        "single_chunk_with_bias",
        "single_partial_chunk_no_bias",
        "single_partial_chunk_with_bias",
    ]
)
def test_process_keys_values_chunks(
    base_tensors: Tuple[torch.Tensor, torch.Tensor],
    attention_bias: torch.Tensor,
    seq_len_k_override: int,
    n_keys_override: int,
    provide_bias: bool
):
    """
    Tests _process_keys_values_chunks for various scenarios.

    Covers lines 146-182 by testing:
    - Loop execution (range(params.n_k_trunks)) - lines 151-177
    - _get_chunk_info call - line 153
    - _process_chunk calls for k and v - lines 156, 162
    - Conditional _process_bias_chunk call - lines 168-176
    - Stacking results - lines 179-180
    - Returning correct tuple structure - lines 182-185
    """
    k_base, v_base = base_tensors
    k = k_base[..., :seq_len_k_override, :]
    v = v_base[..., :seq_len_k_override, :]
    bias = attention_bias[..., :seq_len_k_override] if provide_bias else None

    n_k_trunks = (seq_len_k_override + n_keys_override - 1) // n_keys_override

    params = KeysValuesChunkParams(
        k=k,
        v=v,
        attn_bias=bias,
        n_keys=n_keys_override,
        n_k_trunks=n_k_trunks,
        inf=INF,
    )

    k_trunked, v_trunked, attn_bias_list = _process_keys_values_chunks(params)

    # --- Assertions ---
    # 1. Check output shapes
    expected_k_shape = (BATCH_SIZE, N_HEADS, n_k_trunks, n_keys_override, D_HEAD)
    expected_v_shape = (BATCH_SIZE, N_HEADS, n_k_trunks, n_keys_override, D_HEAD)
    assert k_trunked.shape == expected_k_shape, f"Expected K shape {expected_k_shape}, got {k_trunked.shape}"
    assert v_trunked.shape == expected_v_shape, f"Expected V shape {expected_v_shape}, got {v_trunked.shape}"

    # 2. Check bias output structure and shapes
    if provide_bias:
        assert attn_bias_list is not None, "Expected bias list, got None"
        assert len(attn_bias_list) == n_k_trunks, f"Expected {n_k_trunks} bias chunks, got {len(attn_bias_list)}"
        # Bias shape: (batch_size, n_heads, seq_len_q, n_keys) - padding on last dim
        expected_bias_chunk_shape = (BATCH_SIZE, N_HEADS, SEQ_LEN_Q, n_keys_override)
        for i, bias_chunk in enumerate(attn_bias_list):
            assert bias_chunk.shape == expected_bias_chunk_shape, \
                f"Expected bias chunk {i} shape {expected_bias_chunk_shape}, got {bias_chunk.shape}"
    else:
        assert attn_bias_list is None, "Expected None for bias list, got a list"

    # 3. Check content consistency (spot check first chunk and padding)
    # Verify first chunk matches original data
    first_chunk_len = min(n_keys_override, seq_len_k_override)
    torch.testing.assert_close(k_trunked[..., 0, :first_chunk_len, :], k[..., :first_chunk_len, :])
    torch.testing.assert_close(v_trunked[..., 0, :first_chunk_len, :], v[..., :first_chunk_len, :])

    if provide_bias:
         torch.testing.assert_close(attn_bias_list[0][..., :first_chunk_len], bias[..., :first_chunk_len])


    # 4. Check padding values if padding occurred
    if seq_len_k_override % n_keys_override != 0:
        last_chunk_idx = n_k_trunks - 1
        original_len_last_chunk = seq_len_k_override % n_keys_override
        padding_len = n_keys_override - original_len_last_chunk

        # Check K/V padding (should be zeros)
        assert torch.all(k_trunked[..., last_chunk_idx, original_len_last_chunk:, :] == 0.0)
        assert torch.all(v_trunked[..., last_chunk_idx, original_len_last_chunk:, :] == 0.0)

        # Check bias padding (should be -inf)
        if provide_bias:
            assert torch.all(attn_bias_list[last_chunk_idx][..., original_len_last_chunk:] == -INF)

    # 5. Check device and dtype consistency (optional but good practice)
    assert k_trunked.dtype == k.dtype
    assert v_trunked.dtype == v.dtype
    assert k_trunked.device == k.device
    assert v_trunked.device == v.device
    if provide_bias:
        for bias_chunk in attn_bias_list:
            assert bias_chunk.dtype == bias.dtype
            assert bias_chunk.device == bias.device
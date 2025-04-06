# tests/stageA/unit/input_embedding/current/primitives/test_attention_utils.py
import pytest
import torch
from typing import Tuple, Optional
from dataclasses import replace
from unittest.mock import patch # Add import

# Module to test
from rna_predict.pipeline.stageA.input_embedding.current.primitives import attention_utils
from rna_predict.pipeline.stageA.input_embedding.current.primitives.attention_utils import (
    LocalAttentionInputs,
    BiasCreationInputs,
    AttentionChunkConfig,
    ChunkProcessingInputs,
)
from rna_predict.pipeline.stageA.input_embedding.current.primitives.attention_base import (
    AttentionInputs,
    _attention,
)


# --- Fixtures ---

@pytest.fixture
def device():
    """Fixture to provide the device (CPU for consistency in tests)."""
    return torch.device("cpu")


@pytest.fixture
def default_local_attention_inputs(device) -> LocalAttentionInputs:
    """Provides default LocalAttentionInputs."""
    batch_size = 2
    n_heads = 4
    n_queries = 16
    n_keys = 16
    d_head = 8
    return LocalAttentionInputs(
        q=torch.randn(batch_size, n_heads, n_queries, d_head, device=device),
        k=torch.randn(batch_size, n_heads, n_keys, d_head, device=device),
        v=torch.randn(batch_size, n_heads, n_keys, d_head, device=device),
        n_queries=n_queries,
        n_keys=n_keys,
        inf=1e9,
        # device=device, # Removed: Not a field of LocalAttentionInputs
    )


# --- Tests for optimized_concat_split (Lines 87-95) ---

@pytest.mark.parametrize(
    "bias_shape, n_queries_split, expected_output_shape",
    [
        ((1, 8, 32, 64), 16, (1, 16, 16, 64)), # Split 32 queries into 2 chunks of 16, concat along dim -3 (size 8)
        ((2, 4, 64, 128), 32, (2, 8, 32, 128)), # Split 64 queries into 2 chunks of 32, concat along dim -3 (size 4)
        ((1, 1, 10, 20), 10, (1, 1, 10, 20)),  # n_queries_split equals original query dim
        ((1, 1, 10, 20), 5, (1, 2, 5, 20)),   # Split 10 queries into 2 chunks of 5
        ((1, 1, 10, 20), 1, (1, 10, 1, 20)),  # Split 10 queries into 10 chunks of 1
        ((1, 1, 10, 20), 12, (1, 1, 10, 20)), # n_queries_split > original query dim (no split)
    ],
)
def test_optimized_concat_split(
    bias_shape: Tuple[int, ...],
    n_queries_split: int,
    expected_output_shape: Tuple[int, ...],
    device,
):
    """
    Tests optimized_concat_split correctly reshapes the bias tensor.
    Covers lines 87-95.
    """
    attn_bias = torch.randn(bias_shape, device=device)
    result = attention_utils.optimized_concat_split(attn_bias, n_queries_split)
    assert result.shape == expected_output_shape
    # Check if total elements remain the same
    assert result.numel() == attn_bias.numel()


# --- Tests for _calculate_bias_shape (Lines 108-109) ---

@pytest.mark.parametrize(
    "n, n_queries, n_keys, expected_shape",
    [
        (32, 16, 64, (1, 2, 16, 64)),  # n divisible by n_queries
        (30, 16, 64, (1, 2, 16, 64)),  # n not divisible by n_queries
        (16, 16, 64, (1, 1, 16, 64)),  # n equals n_queries
        (10, 16, 64, (1, 1, 16, 64)),  # n less than n_queries
        (100, 10, 20, (1, 10, 10, 20)),
    ],
)
def test_calculate_bias_shape(
    n: int, n_queries: int, n_keys: int, expected_shape: Tuple[int, int, int, int]
):
    """
    Tests the calculation of the bias shape based on input dimensions.
    Covers lines 108-109.
    """
    inputs = BiasCreationInputs(n=n, n_queries=n_queries, n_keys=n_keys)
    result_shape = attention_utils._calculate_bias_shape(inputs)
    assert result_shape == expected_shape


# --- Tests for create_local_attn_bias (Lines 123-147) ---

@pytest.mark.parametrize(
    "n, n_queries, n_keys",
    [
        (32, 16, 64),
        (30, 16, 64),
        (16, 16, 64),
        (10, 16, 64),
        (5, 4, 10),
    ],
)
def test_create_local_attn_bias_shape_and_device(
    n: int, n_queries: int, n_keys: int, device
):
    """
    Tests the shape and device placement of the created local attention bias.
    Covers lines 123, 126-127.
    """
    inf_val = 1e7
    inputs = BiasCreationInputs(
        n=n, n_queries=n_queries, n_keys=n_keys, inf=inf_val, device=device
    )
    expected_shape = attention_utils._calculate_bias_shape(inputs)
    attn_bias = attention_utils.create_local_attn_bias(inputs)

    assert attn_bias.shape == expected_shape
    assert attn_bias.device == device


def test_create_local_attn_bias_values(device):
    """
    Tests the values within the created local attention bias (0 in window, -inf outside).
    Covers lines 130-145.
    """
    n = 7
    n_queries = 4
    n_keys = 10
    inf_val = 1e9
    inputs = BiasCreationInputs(
        n=n, n_queries=n_queries, n_keys=n_keys, inf=inf_val, device=device
    )
    attn_bias = attention_utils.create_local_attn_bias(inputs)

    # Expected shape: (1, 2, 4, 10)
    assert attn_bias.shape == (1, 2, 4, 10)

    # Chunk 0 (queries 0-3)
    # Window should cover keys max(0, 0-4) to min(7, 0 + 2*4) => 0 to 7
    # Bias shape [0, 0, :4, :7] should be 0
    assert torch.all(attn_bias[0, 0, :4, :7] == 0)
    # Bias shape [0, 0, :4, 7:] should be -inf
    assert torch.all(attn_bias[0, 0, :4, 7:] == -inf_val)

    # Chunk 1 (queries 4-6, shape is [..., :3, :])
    # Window should cover keys max(0, 4-4) to min(7, 4 + 2*4) => 0 to 7
    # Bias shape [0, 1, :3, :7] should be 0
    assert torch.all(attn_bias[0, 1, :3, :7] == 0)
    # Bias shape [0, 1, :3, 7:] should be -inf
    assert torch.all(attn_bias[0, 1, :3, 7:] == -inf_val)
    # Bias shape [0, 1, 3, :] should be -inf (padding query)
    assert torch.all(attn_bias[0, 1, 3, :] == -inf_val)


# --- Tests for _determine_chunking (Lines 166-180) ---

@pytest.mark.parametrize(
    "q_shape, n_queries, n_keys, chunk_size_override, expected_config",
    [
        # Override chunk size
        ((2, 4, 32, 8), 16, 64, 8, AttentionChunkConfig(chunk_size=8, n_chunks_q=4, n_chunks_k=4)),
        # Heuristic: small batch size (<= 32), query_len > 128 -> chunk_size 128
        ((2, 4, 150, 8), 16, 64, None, AttentionChunkConfig(chunk_size=128, n_chunks_q=2, n_chunks_k=2)),
        # Heuristic: small batch size (<= 32), query_len < 128 -> chunk_size query_len
        ((2, 4, 50, 8), 16, 64, None, AttentionChunkConfig(chunk_size=50, n_chunks_q=1, n_chunks_k=1)),
         # Heuristic: batch size == 32 -> chunk_size 100 (due to <= 32 condition)
        ((32, 4, 100, 8), 16, 64, None, AttentionChunkConfig(chunk_size=100, n_chunks_q=1, n_chunks_k=1)),
       # Heuristic: large batch size (> 32), query_len > 64 -> chunk_size 64
       ((33, 4, 100, 8), 16, 64, None, AttentionChunkConfig(chunk_size=64, n_chunks_q=2, n_chunks_k=2)), # REVERTED EXPECTATION
       # Heuristic: large batch size (> 32), query_len < 64 -> chunk_size query_len
       ((33, 4, 40, 8), 16, 64, None, AttentionChunkConfig(chunk_size=40, n_chunks_q=1, n_chunks_k=1)),
        # Edge case: query_len = 1
        ((2, 4, 1, 8), 16, 64, None, AttentionChunkConfig(chunk_size=1, n_chunks_q=1, n_chunks_k=1)),
    ],
)
def test_determine_chunking(
    q_shape: Tuple[int, ...],
    n_queries: int,
    n_keys: int,
    chunk_size_override: Optional[int],
    expected_config: AttentionChunkConfig,
    device,
):
    """
    Tests the logic for determining attention chunking configuration.
    Covers lines 166-180.
    """
    q = torch.randn(q_shape, device=device)
    result_config = attention_utils._determine_chunking(
        q, n_queries, n_keys, chunk_size_override
    )
    assert result_config == expected_config


# --- Tests for _process_attention_chunk (Lines 196-221) ---

@pytest.mark.parametrize(
    "bias_shape, expected_bias_shape_in_call",
    [
        (None, None), # No bias
        ((2, 4, 16, 32), (2, 4, 16, 32)), # Standard 4D bias, no change needed
        ((2, 4, 1, 16, 32), (2, 4, 8, 32)), # 5D bias (trunked), squeeze dim -3, THEN SLICE to match n_queries_chunk=8 # Corrected expectation
        ((2, 4, 8, 32), (2, 4, 8, 32)), # Bias query dim < chunk query dim (needs slicing)
        ((2, 4, 1, 8, 32), (2, 4, 8, 32)), # 5D bias query dim < chunk query dim (squeeze + slice)
    ]
)
def test_process_attention_chunk_bias_handling(
    bias_shape: Optional[Tuple[int, ...]],
    expected_bias_shape_in_call: Optional[Tuple[int, ...]],
    # mocker, # Use mocker fixture - Removed, using patch directly
    device
):
    """
    Tests how _process_attention_chunk handles and reshapes the bias slice.
    Covers lines 197-208, 214.
    """
    batch_size = 2
    n_heads = 4
    n_queries_chunk = 8 # Chunk size smaller than bias query dim in some cases
    n_keys = 32
    d_head = 16

    q_chunk = torch.randn(batch_size, n_heads, n_queries_chunk, d_head, device=device)
    k = torch.randn(batch_size, n_heads, n_keys, d_head, device=device)
    v = torch.randn(batch_size, n_heads, n_keys, d_head, device=device)
    bias_slice = torch.randn(bias_shape, device=device) if bias_shape else None

    # Mock the _attention function using unittest.mock.patch
    with patch("rna_predict.pipeline.stageA.input_embedding.current.primitives.attention_utils._attention", return_value=torch.randn_like(q_chunk)) as mock_attention:

        chunk_inputs = ChunkProcessingInputs(
            q_chunk=q_chunk,
            k=k,
            v=v,
            bias_slice=bias_slice,
            use_efficient_implementation=False,
            attn_weight_dropout_p=0.0,
            inplace_safe=False,
        )

        attention_utils._process_attention_chunk(chunk_inputs)

        # Assert _attention was called
        mock_attention.assert_called_once()
        call_args = mock_attention.call_args[0][0] # Get the AttentionInputs object

        # Check the shape of the bias passed to _attention
        passed_bias = call_args.attn_bias
        if expected_bias_shape_in_call is None:
            assert passed_bias is None
        else:
            assert passed_bias is not None
            assert passed_bias.shape == expected_bias_shape_in_call


def test_process_attention_chunk_calls_attention(device): # Removed mocker
    """
    Tests that _process_attention_chunk correctly calls the underlying _attention function.
    Covers lines 210-221.
    """
    batch_size, n_heads, n_queries_chunk, n_keys, d_head = 2, 4, 8, 16, 32
    q_chunk = torch.randn(batch_size, n_heads, n_queries_chunk, d_head, device=device)
    k = torch.randn(batch_size, n_heads, n_keys, d_head, device=device)
    v = torch.randn(batch_size, n_heads, n_keys, d_head, device=device)
    bias = torch.randn(batch_size, n_heads, n_queries_chunk, n_keys, device=device)
    dropout = 0.1
    inplace = True
    efficient = True

    expected_output = torch.randn_like(q_chunk)
    # Mock the _attention function using unittest.mock.patch
    with patch("rna_predict.pipeline.stageA.input_embedding.current.primitives.attention_utils._attention", return_value=expected_output) as mock_attention:

        chunk_inputs = ChunkProcessingInputs(
            q_chunk=q_chunk,
            k=k,
            v=v,
            bias_slice=bias,
            use_efficient_implementation=efficient,
            attn_weight_dropout_p=dropout,
            inplace_safe=inplace,
        )

        result = attention_utils._process_attention_chunk(chunk_inputs)

        assert torch.equal(result, expected_output)
        mock_attention.assert_called_once()
        call_args: AttentionInputs = mock_attention.call_args[0][0]

        assert torch.equal(call_args.q, q_chunk)
        assert torch.equal(call_args.k, k)
        assert torch.equal(call_args.v, v)
        assert torch.equal(call_args.attn_bias, bias)
        assert call_args.use_efficient_implementation == efficient
        assert call_args.attn_weight_dropout_p == dropout
        assert call_args.inplace_safe == inplace


# --- Tests for _get_attention_bias (Lines 237-251) ---

def test_get_attention_bias_priority(default_local_attention_inputs, device): # Removed mocker
    """
    Tests the priority order for selecting the attention bias.
    Covers lines 237-242.
    """
    inputs = default_local_attention_inputs
    q_shape = inputs.q.shape
    inputs.trunked_attn_bias = torch.randn(1, device=device) # Dummy tensors
    inputs.attn_bias = torch.randn(1, device=device)

    # Use patch context manager
    with patch("rna_predict.pipeline.stageA.input_embedding.current.primitives.attention_utils.create_local_attn_bias") as mock_create_bias:
        # Case 1: trunked_attn_bias is present
        result1 = attention_utils._get_attention_bias(inputs, q_shape)
        assert result1 is inputs.trunked_attn_bias
        mock_create_bias.assert_not_called()

        # Case 2: trunked_attn_bias is None, attn_bias is present
        inputs.trunked_attn_bias = None
        result2 = attention_utils._get_attention_bias(inputs, q_shape)
        assert result2 is inputs.attn_bias
        mock_create_bias.assert_not_called()


def test_get_attention_bias_creation_path(default_local_attention_inputs, device): # Removed mocker
    """
    Tests the path where local attention bias is created.
    Covers lines 244-251.
    """
    inputs = default_local_attention_inputs
    q_shape = inputs.q.shape
    inputs.trunked_attn_bias = None
    inputs.attn_bias = None
    expected_bias = torch.randn(1, device=device) # Dummy

    # Use patch context manager
    with patch("rna_predict.pipeline.stageA.input_embedding.current.primitives.attention_utils.create_local_attn_bias", return_value=expected_bias) as mock_create_bias:
        result = attention_utils._get_attention_bias(inputs, q_shape)

        assert result is expected_bias
        mock_create_bias.assert_called_once()
        call_args: BiasCreationInputs = mock_create_bias.call_args[0][0]
        assert call_args.n == q_shape[-2]
        assert call_args.n_queries == inputs.n_queries
        assert call_args.n_keys == inputs.n_keys
        assert call_args.inf == inputs.inf
        assert call_args.device == inputs.q.device


def test_get_attention_bias_all_none(default_local_attention_inputs): # Removed mocker
    """
    Tests that None is returned if no bias is provided or created (though create should always run).
    This test mainly ensures create_local_attn_bias is called when others are None.
    """
    inputs = default_local_attention_inputs
    q_shape = inputs.q.shape
    inputs.trunked_attn_bias = None
    inputs.attn_bias = None

    # Mock create_local_attn_bias to return None (unrealistic but tests the flow)
    # Use patch context manager
    with patch("rna_predict.pipeline.stageA.input_embedding.current.primitives.attention_utils.create_local_attn_bias", return_value=None) as mock_create_bias:
        result = attention_utils._get_attention_bias(inputs, q_shape)
        assert result is None
        mock_create_bias.assert_called_once()


# --- Tests for _get_bias_slice (Lines 360-367) ---

@pytest.mark.parametrize(
    "bias_ndim, expected_slice_expr",
    [
        (None, lambda b, i, s, e: None), # No bias -> None slice
        (4, lambda b, i, s, e: b[..., s:e, :]), # 4D bias -> slice query dim
        (5, lambda b, i, s, e: b[..., i : i + 1, :, :]), # 5D bias -> slice chunk dim
    ]
)
def test_get_bias_slice(bias_ndim, expected_slice_expr, device):
    """
    Tests selecting the correct bias slice based on bias dimensionality.
    Covers lines 360-367.
    """
    batch, heads, chunks, queries, keys = 1, 4, 5, 8, 16
    chunk_idx = 2
    start_idx = chunk_idx * queries
    end_idx = start_idx + queries

    local_bias = None
    expected_slice = None

    if bias_ndim == 4:
        local_bias = torch.randn(batch, heads, chunks * queries, keys, device=device)
        expected_slice = expected_slice_expr(local_bias, chunk_idx, start_idx, end_idx)
    elif bias_ndim == 5:
        local_bias = torch.randn(batch, heads, chunks, queries, keys, device=device)
        expected_slice = expected_slice_expr(local_bias, chunk_idx, start_idx, end_idx)

    result_slice = attention_utils._get_bias_slice(local_bias, chunk_idx, start_idx, end_idx)

    if expected_slice is None:
        assert result_slice is None
    else:
        # Use allclose for floating point comparisons
        assert torch.allclose(result_slice, expected_slice)


# Note: Testing _process_small_tensors, _process_chunks, _local_attention,
# and local_attention requires more complex setup, potentially involving mocking
# internal calls or carefully crafting inputs to trigger specific paths.
# These will be added in subsequent steps to keep this manageable.

# --- Placeholder for future tests ---

# TODO: Add tests for _process_small_tensors (Lines 270-342)
# TODO: Add tests for _process_chunks (Lines 387-420)
# TODO: Add tests for _local_attention (Lines 434-469)
# TODO: Add tests for local_attention (public API) (Lines 502-521)

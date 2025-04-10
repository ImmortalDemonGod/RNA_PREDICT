# tests/stageA/unit/input_embedding/current/primitives/test_attention_processing.py
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from rna_predict.pipeline.stageA.input_embedding.current.primitives.attention_core import (
    AttentionInputs,
    ProcessQueryInputs,
    ProcessDifferentQueryInputs,
)
from rna_predict.pipeline.stageA.input_embedding.current.primitives.attention_processing import (
    process_same_query_keyvalue,
    _process_with_batch_matmul,
    _reshape_attention_bias,
    process_different_query_keyvalue,
    process_small_tensors,
    SmallTensorParams,
    TensorInputs,
    SmallTensorInputs,
    _ensure_batch_dimensions,
    _compute_attention_scores,
    _apply_attention_bias,
    _apply_attention_mask,
)


@pytest.fixture
def device():
    """Fixture to provide the device (CPU for consistency in tests)."""
    return torch.device("cpu")


# --- Tests for process_same_query_keyvalue ---

@pytest.mark.parametrize(
    "use_efficient_implementation, sdpa_raises",
    [
        (True, False),  # Use efficient implementation, no error
        (True, True),   # Use efficient implementation, but it raises error
        (False, False), # Don't use efficient implementation
    ],
)
def test_process_same_query_keyvalue(use_efficient_implementation, sdpa_raises, device):
    """
    Tests process_same_querykeyvalue under various configurations.
    
    This test simulates scenarios by toggling the use of an efficient attention implementation and by forcing a simulated error in the scaled dot product attention function. It verifies that when the efficient implementation is enabled without errors, the function bypasses the fallback batch matrix multiplication, and when an error is simulated, it appropriately falls back to the alternative path. The test asserts that the output tensor has the expected shape and that the correct internal function is invoked with the proper parameters.
    
    Args:
        use_efficient_implementation: Flag to choose the efficient attention path.
        sdpa_raises: Flag to simulate a RuntimeError from the scaled dot product attention.
        device: The device (e.g., CPU) on which the tensors are allocated.
    """
    batch_size = 2
    n_heads = 4
    n_queries = 8
    d_head = 16

    # Create test tensors
    q = torch.randn(batch_size, n_queries, n_heads, d_head, device=device)
    k = torch.randn(batch_size, n_queries, n_heads, d_head, device=device)
    v = torch.randn(batch_size, n_queries, n_heads, d_head, device=device)
    attn_bias = torch.randn(batch_size, n_heads, n_queries, n_queries, device=device)

    inputs = ProcessQueryInputs(
        q=q,
        k=k,
        v=v,
        q_x=torch.randn(batch_size, n_queries, n_heads * d_head, device=device),
        attn_bias=attn_bias,
        inplace_safe=True,
    )

    # Mock F.scaled_dot_product_attention

    def mock_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
        """
        Mock implementation of scaled dot-product attention.
        
        This function simulates scaled dot-product attention by returning a random tensor
        with the same shape as the query tensor. It accepts query, key, and value tensors,
        as well as optional parameters for attention masking, dropout probability, and causal
        attention. These additional parameters are retained for interface compatibility and
        do not affect the output. If the global flag sdpa_raises is set, a RuntimeError is raised
        to simulate an SDPA error.
        
        Parameters:
            query: Tensor containing the query data.
            key: Tensor containing the key data.
            value: Tensor containing the value data.
            attn_mask: Optional tensor representing an attention mask.
            dropout_p: Dropout probability (ignored in this mock).
            is_causal: Boolean flag indicating causal attention (ignored in this mock).
        
        Returns:
            A tensor with the same shape as `query` containing random values.
        
        Raises:
            RuntimeError: If the global flag sdpa_raises is True.
        """
        if sdpa_raises:
            raise RuntimeError("Simulated SDPA error")
        # Return a tensor with the expected shape
        return torch.randn_like(query)

    # Mock _process_with_batch_matmul
    with patch(
        "rna_predict.pipeline.stageA.input_embedding.current.primitives.attention_processing._process_with_batch_matmul"
    ) as mock_batch_matmul, patch.object(
        F, "scaled_dot_product_attention", side_effect=mock_sdpa
    ):
        # Set up mock return value
        expected_output = torch.randn_like(q)
        mock_batch_matmul.return_value = expected_output

        # Call the function
        result = process_same_query_keyvalue(
            inputs,
            num_heads=n_heads,
            attn_weight_dropout_p=0.1,
            use_efficient_implementation=use_efficient_implementation
        )

        # Check if the result has the expected shape
        assert result.shape == q.shape

        # Check if the appropriate function was called
        if use_efficient_implementation and not sdpa_raises:
            # SDPA should have been called
            assert mock_batch_matmul.call_count == 0
        else:
            # Batch matmul should have been called
            assert mock_batch_matmul.call_count == 1
            # Check arguments
            call_args = mock_batch_matmul.call_args[0][0]  # Get the BatchMatmulParams object
            assert call_args.tensors.q.shape == (batch_size, n_heads, n_queries, d_head)  # q transposed
            assert call_args.tensors.k.shape == (batch_size, n_heads, n_queries, d_head)  # k transposed
            assert call_args.tensors.v.shape == (batch_size, n_heads, n_queries, d_head)  # v transposed
            assert call_args.attn_bias is attn_bias  # attn_bias
            assert call_args.config.num_heads == n_heads  # num_heads
            assert call_args.config.attn_weight_dropout_p == 0.1  # attn_weight_dropout_p
            assert call_args.config.use_efficient_implementation == use_efficient_implementation  # use_efficient_implementation
            assert call_args.config.inplace_safe is True  # inplace_safe


# --- Tests for _process_with_batch_matmul ---

def test_process_with_batch_matmul(device):
    """
    Test _process_with_batch_matmul function.
    """
    batch_size = 2
    n_heads = 4
    n_queries = 8
    n_keys = 10
    d_head = 16

    # Create test tensors
    q = torch.randn(batch_size, n_heads, n_queries, d_head, device=device)
    k = torch.randn(batch_size, n_heads, n_keys, d_head, device=device)
    v = torch.randn(batch_size, n_heads, n_keys, d_head, device=device)
    attn_bias = torch.randn(batch_size, n_heads, n_queries, n_keys, device=device)

    # Mock _reshape_attention_bias and attention
    with patch(
        "rna_predict.pipeline.stageA.input_embedding.current.primitives.attention_processing._reshape_attention_bias"
    ) as mock_reshape_bias, patch(
        "rna_predict.pipeline.stageA.input_embedding.current.primitives.attention_processing.attention"
    ) as mock_attention:
        # Set up mock return values
        reshaped_bias = torch.randn(batch_size * n_heads, n_queries, n_keys, device=device)
        mock_reshape_bias.return_value = reshaped_bias

        attn_output = torch.randn(batch_size * n_heads, n_queries, d_head, device=device)
        mock_attention.return_value = attn_output

        # Create parameter object
        from rna_predict.pipeline.stageA.input_embedding.current.primitives.attention_processing import (
            TensorInputs, AttentionConfig, BatchMatmulParams
        )
        tensor_inputs = TensorInputs(q=q, k=k, v=v)
        config = AttentionConfig(
            num_heads=n_heads,
            attn_weight_dropout_p=0.1,
            use_efficient_implementation=True,
            inplace_safe=True
        )
        params = BatchMatmulParams(
            tensors=tensor_inputs,
            attn_bias=attn_bias,
            config=config
        )

        # Call the function
        result = _process_with_batch_matmul(params)

        # Check if the result has the expected shape
        assert result.shape == (batch_size, n_queries, n_heads, d_head)

        # Check if the appropriate functions were called
        mock_reshape_bias.assert_called_once_with(attn_bias, n_heads)
        mock_attention.assert_called_once()

        # Check attention inputs
        attention_inputs = mock_attention.call_args[0][0]
        assert isinstance(attention_inputs, AttentionInputs)
        assert attention_inputs.q.shape == (batch_size * n_heads, n_queries, d_head)
        assert attention_inputs.k.shape == (batch_size * n_heads, n_keys, d_head)
        assert attention_inputs.v.shape == (batch_size * n_heads, n_keys, d_head)
        assert attention_inputs.attn_bias is reshaped_bias
        assert attention_inputs.use_efficient_implementation is True
        assert attention_inputs.attn_weight_dropout_p == 0.1
        assert attention_inputs.inplace_safe is True


# --- Tests for _reshape_attention_bias ---

@pytest.mark.parametrize(
    "attn_bias_shape, num_heads, expected_shape, should_raise",
    [
        (None, 4, None, False),  # No bias
        ((2, 4, 8, 10), 4, (8, 8, 10), False),  # Standard 4D bias, correct heads
        ((2, 1, 8, 10), 4, None, True),  # Bias with 1 head, but raises warning due to implementation
        ((2, 4, 8, 10), 8, None, True),  # Wrong number of heads, should raise warning
    ],
)
def test_reshape_attention_bias(attn_bias_shape, num_heads, expected_shape, should_raise, device):
    """
    Test _reshape_attention_bias function with different bias shapes.
    """
    # Create test tensor if shape is provided
    attn_bias = None
    if attn_bias_shape is not None:
        attn_bias = torch.randn(attn_bias_shape, device=device)

    # Capture warnings
    if should_raise:
        with pytest.warns(UserWarning) as warning_record:
            result = _reshape_attention_bias(attn_bias, num_heads)

        if expected_shape is None:
            if should_raise:
                # Should have raised a warning and returned None
                assert result is None
                assert len(warning_record) > 0
                assert "Could not reshape attn_bias" in str(warning_record[0].message)
            else:
                # Should have returned None without warning
                assert result is None
        else:
            # Should have returned a tensor with the expected shape
            assert result is not None
            assert result.shape == expected_shape
    else:
        # No warning expected
        result = _reshape_attention_bias(attn_bias, num_heads)
        if expected_shape is None:
            assert result is None
        else:
            assert result is not None
            assert result.shape == expected_shape


# --- Tests for process_different_query_keyvalue ---

@pytest.mark.parametrize(
    "local_attention_method, use_local_attention",
    [
        ("global_attention_with_bias", True),
        ("standard_attention", False),
    ],
)
def test_process_different_query_keyvalue(local_attention_method, use_local_attention, device):
    """
    Test process_different_query_keyvalue with different local attention methods.
    """
    batch_size = 2
    n_heads = 4
    n_queries = 8
    n_keys = 10
    d_head = 16

    # Create test tensors
    q = torch.randn(batch_size, n_queries, n_heads, d_head, device=device)
    k = torch.randn(batch_size, n_keys, n_heads, d_head, device=device)
    v = torch.randn(batch_size, n_keys, n_heads, d_head, device=device)
    attn_bias = torch.randn(batch_size, n_heads, n_queries, n_keys, device=device)

    inputs = ProcessDifferentQueryInputs(
        q=q,
        k=k,
        v=v,
        q_x=torch.randn(batch_size, n_queries, n_heads * d_head, device=device),
        attn_bias=attn_bias,
        trunked_attn_bias=None,
        n_queries=n_queries,
        n_keys=n_keys,
        inf=1e10,
        inplace_safe=True,
        chunk_size=None,
    )

    # Mock _local_attention and attention
    with patch(
        "rna_predict.pipeline.stageA.input_embedding.current.primitives.attention_utils._local_attention"
    ) as mock_local_attention, patch(
        "rna_predict.pipeline.stageA.input_embedding.current.primitives.attention_processing.attention"
    ) as mock_attention:
        # Set up mock return values
        expected_output = torch.randn_like(q)
        mock_local_attention.return_value = expected_output
        mock_attention.return_value = expected_output

        # Call the function
        result = process_different_query_keyvalue(
            inputs,
            use_efficient_implementation=True,
            attn_weight_dropout_p=0.1,
            local_attention_method=local_attention_method
        )

        # Check if the result has the expected shape
        assert result.shape == q.shape

        # Check if the appropriate function was called
        if use_local_attention:
            assert mock_local_attention.call_count == 1
            assert mock_attention.call_count == 0

            # Check local_attention inputs
            local_attn_inputs = mock_local_attention.call_args[0][0]
            assert local_attn_inputs.q is q
            assert local_attn_inputs.k is k
            assert local_attn_inputs.v is v
            assert local_attn_inputs.n_queries == n_queries
            assert local_attn_inputs.n_keys == n_keys
            assert local_attn_inputs.attn_bias is attn_bias
            assert local_attn_inputs.trunked_attn_bias is None
            assert local_attn_inputs.inf == 1e10
            assert local_attn_inputs.use_efficient_implementation is True
            assert local_attn_inputs.attn_weight_dropout_p == 0.1
            assert local_attn_inputs.inplace_safe is True
            assert local_attn_inputs.chunk_size is None
        else:
            assert mock_local_attention.call_count == 0
            assert mock_attention.call_count == 1

            # Check attention inputs
            attention_inputs = mock_attention.call_args[0][0]
            assert attention_inputs.q is q
            assert attention_inputs.k is k
            assert attention_inputs.v is v
            assert attention_inputs.attn_bias is attn_bias
            assert attention_inputs.use_efficient_implementation is True
            assert attention_inputs.attn_weight_dropout_p == 0.1
            assert attention_inputs.inplace_safe is True


# --- Tests for SmallTensorParams and helper functions ---

def test_ensure_batch_dimensions(device):
    """
    Test _ensure_batch_dimensions function.
    """
    # Create test tensors with different batch dimensions
    q = torch.randn(2, 3, 8, 16, device=device)  # [B1, B2, N_q, d]
    k = torch.randn(1, 1, 10, 16, device=device)  # [1, 1, N_k, d]
    v = torch.randn(1, 1, 10, 16, device=device)  # [1, 1, N_k, d]
    bias = None
    mask = None

    # Create TensorInputs object first
    from rna_predict.pipeline.stageA.input_embedding.current.primitives.attention_processing import TensorInputs
    tensor_inputs = TensorInputs(q=q, k=k, v=v)
    params = SmallTensorParams(tensors=tensor_inputs, bias=bias, mask=mask)

    # Call the function
    _ensure_batch_dimensions(params)

    # Check if k and v have been expanded to match q's batch dimensions
    adjusted_tensors = _ensure_batch_dimensions(params)
    assert adjusted_tensors.k.shape[:-2] == q.shape[:-2]
    assert adjusted_tensors.v.shape[:-2] == q.shape[:-2]
    assert adjusted_tensors.k.shape[-2:] == (10, 16)  # N_k and d should remain unchanged
    assert adjusted_tensors.v.shape[-2:] == (10, 16)  # N_k and d should remain unchanged


def test_compute_attention_scores(device):
    """
    Test _compute_attention_scores function.
    """
    # Create test tensors
    q = torch.randn(2, 8, 16, device=device)  # [B, N_q, d]
    k = torch.randn(2, 10, 16, device=device)  # [B, N_k, d]
    v = torch.randn(2, 10, 16, device=device)  # [B, N_k, d]
    bias = None
    mask = None

    # Create TensorInputs object first
    from rna_predict.pipeline.stageA.input_embedding.current.primitives.attention_processing import TensorInputs
    tensor_inputs = TensorInputs(q=q, k=k, v=v)
    params = SmallTensorParams(tensors=tensor_inputs, bias=bias, mask=mask)

    # Call the function
    scores = _compute_attention_scores(params.tensors)

    # Check if scores have the expected shape
    assert scores.shape == (2, 8, 10)  # [B, N_q, N_k]

    # Verify the computation manually
    expected_scores = torch.matmul(q, k.transpose(-2, -1)) / (16 ** 0.5)
    assert torch.allclose(scores, expected_scores)


def test_apply_attention_bias(device):
    """
    Test _apply_attention_bias function.
    """
    # Create test tensors
    scores = torch.randn(2, 8, 10, device=device)  # [B, N_q, N_k]
    bias = torch.randn(2, 8, 10, device=device)  # [B, N_q, N_k]

    # Mock adjust_attention_bias
    with patch(
        "rna_predict.utils.shape_utils.adjust_attention_bias",
        return_value=bias
    ) as mock_adjust_bias:
        # Call the function
        result = _apply_attention_bias(scores, bias)

        # Check if the result has the expected shape
        assert result.shape == scores.shape

        # Check if adjust_attention_bias was called
        mock_adjust_bias.assert_called_once()

        # Verify the computation manually
        expected_result = scores + bias
        assert torch.allclose(result, expected_result)


def test_apply_attention_mask(device):
    """
    Test _apply_attention_mask function.
    """
    # Create test tensors
    scores = torch.randn(2, 8, 10, device=device)  # [B, N_q, N_k]
    mask = torch.ones(2, 8, 10, dtype=torch.bool, device=device)  # [B, N_q, N_k]

    # Set some elements to False
    mask[0, 0, 0] = False
    mask[1, 1, 1] = False

    # Call the function
    result = _apply_attention_mask(scores, mask)

    # Check if the result has the expected shape
    assert result.shape == scores.shape

    # Check if the masked elements are -inf
    assert result[0, 0, 0].item() == float("-inf")
    assert result[1, 1, 1].item() == float("-inf")

    # Check if the non-masked elements are unchanged
    # Create a mask for non-masked elements
    mask.clone()
    # Compare only non-masked elements
    for i in range(2):
        for j in range(8):
            for k in range(10):
                if mask[i, j, k]:
                    assert scores[i, j, k] == result[i, j, k]


# --- Tests for process_small_tensors ---

@pytest.mark.parametrize(
    "has_bias, has_mask",
    [
        (False, False),  # No bias, no mask
        (True, False),   # Has bias, no mask
        (False, True),   # No bias, has mask
        (True, True),    # Has bias, has mask
    ],
)
def test_process_small_tensors(has_bias, has_mask, device):
    """
    Test process_small_tensors function with different configurations.
    """
    # Create test tensors
    q = torch.randn(2, 8, 16, device=device)  # [B, N_q, d]
    k = torch.randn(2, 10, 16, device=device)  # [B, N_k, d]
    v = torch.randn(2, 10, 16, device=device)  # [B, N_k, d]
    bias = torch.randn(2, 8, 10, device=device) if has_bias else None  # [B, N_q, N_k]
    mask = torch.ones(2, 8, 10, dtype=torch.bool, device=device) if has_mask else None  # [B, N_q, N_k]

    # Set up mocks for helper functions
    with patch(
        "rna_predict.pipeline.stageA.input_embedding.current.primitives.attention_processing._ensure_batch_dimensions",
        return_value=TensorInputs(q=q, k=k, v=v)
    ) as mock_ensure_batch, patch(
        "rna_predict.pipeline.stageA.input_embedding.current.primitives.attention_processing._compute_attention_scores",
        return_value=torch.randn(2, 8, 10, device=device)
    ) as mock_compute_scores, patch(
        "rna_predict.pipeline.stageA.input_embedding.current.primitives.attention_processing._apply_attention_bias",
        return_value=torch.randn(2, 8, 10, device=device)
    ) as mock_apply_bias, patch(
        "rna_predict.pipeline.stageA.input_embedding.current.primitives.attention_processing._apply_attention_mask",
        return_value=torch.randn(2, 8, 10, device=device)
    ) as mock_apply_mask:
        # Create input object
        inputs = SmallTensorInputs(q=q, k=k, v=v, bias=bias, mask=mask)

        # Call the function
        result = process_small_tensors(inputs)

        # Check if the result has the expected shape
        assert result.shape == (2, 8, 16)  # [B, N_q, d]

        # Check if the appropriate functions were called
        mock_ensure_batch.assert_called_once()

        # Verify the calls to the mocked functions
        mock_compute_scores.assert_called_once()
        if has_bias:
            mock_apply_bias.assert_called_once()
        else:
            mock_apply_bias.assert_not_called()

        if has_mask:
            mock_apply_mask.assert_called_once()
        else:
            mock_apply_mask.assert_not_called()

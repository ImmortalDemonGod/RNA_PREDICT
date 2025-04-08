import torch
import pytest
from rna_predict.pipeline.stageA.input_embedding.current.primitives.attention.dense_trunk import (
    _create_empty_output_tensor,
    rearrange_qk_to_dense_trunk,
    rearrange_to_dense_trunk
)


def test_create_empty_output_tensor_edge_cases():
    """Test edge cases for _create_empty_output_tensor function."""
    # Test with empty list
    empty_tensor = _create_empty_output_tensor([], 2, 32)
    assert empty_tensor.shape == (0, 2, 32, 0)
    assert empty_tensor.dtype == torch.float32
    assert empty_tensor.device.type == 'cpu'

    # Test with list of empty tensors
    empty_tensors = [torch.empty(0, 64), torch.empty(0, 64)]
    empty_tensor = _create_empty_output_tensor(empty_tensors, 2, 32)
    assert empty_tensor.shape == (0, 2, 32, 64)

    # Test with mixed empty and non-empty tensors
    mixed_tensors = [torch.empty(0, 64), torch.randn(2, 64)]
    empty_tensor = _create_empty_output_tensor(mixed_tensors, 2, 32)
    assert empty_tensor.shape == (0, 2, 32, 64)

    # Test with multi-dimensional tensor
    multi_dim_tensor = torch.randn(2, 3, 64)  # (batch, heads, features)
    empty_tensor = _create_empty_output_tensor([multi_dim_tensor], 2, 32)
    assert empty_tensor.shape == (0, 3, 2, 32, 64)

def test_rearrange_qk_to_dense_trunk_validation():
    """Test input validation in rearrange_qk_to_dense_trunk."""
    # Test invalid n_queries
    with pytest.raises(ValueError, match="n_queries must be positive, got 0"):
        rearrange_qk_to_dense_trunk(
            torch.randn(2, 64), torch.randn(2, 64),
            dim_q=1, dim_k=1, n_queries=0
        )

    # Test invalid n_keys
    with pytest.raises(ValueError, match="n_keys must be positive, got 0"):
        rearrange_qk_to_dense_trunk(
            torch.randn(2, 64), torch.randn(2, 64),
            dim_q=1, dim_k=1, n_keys=0
        )

def test_rearrange_qk_to_dense_trunk_concatenation_errors():
    """Test error handling during tensor concatenation."""
    # Create tensors with inconsistent ranks
    q1 = torch.randn(2, 3, 64)  # rank 3
    q2 = torch.randn(2, 64)     # rank 2
    k = torch.randn(2, 64)

    with pytest.raises(RuntimeError, match="Inconsistent ranks found in q_processed_list before concatenation"):
        rearrange_qk_to_dense_trunk(
            [q1, q2], k,
            dim_q=[1, 1], dim_k=1
        )

    # Test concatenation failure due to shape mismatch
    q1 = torch.randn(2, 3, 64)
    q2 = torch.randn(2, 4, 64)  # Different middle dimension
    k = torch.randn(2, 64)

    with pytest.raises(RuntimeError, match="Failed to concatenate q_processed_list"):
        rearrange_qk_to_dense_trunk(
            [q1, q2], k,
            dim_q=[1, 1], dim_k=1
        )

def test_rearrange_to_dense_trunk_validation():
    """Test input validation in rearrange_to_dense_trunk."""
    q = torch.randn(2, 64)
    k = torch.randn(2, 64)
    v = torch.randn(2, 64)

    # Test invalid n_queries
    with pytest.raises(ValueError, match="n_queries must be positive"):
        rearrange_to_dense_trunk(q, k, v, n_queries=0, n_keys=32)

    # Test invalid n_keys
    with pytest.raises(ValueError, match="n_keys must be positive"):
        rearrange_to_dense_trunk(q, k, v, n_queries=32, n_keys=0)

def test_rearrange_qk_to_dense_trunk_empty_inputs():
    """Test handling of empty inputs in rearrange_qk_to_dense_trunk."""
    # Test with empty tensor lists
    q_result, k_result, padding_info = rearrange_qk_to_dense_trunk(
        [], [],
        dim_q=[], dim_k=[],
        n_queries=32, n_keys=64
    )
    assert q_result.shape[0] == 0
    assert k_result.shape[0] == 0
    assert isinstance(padding_info, dict)

    # Test with empty tensors
    empty_q = torch.empty(0, 64)
    empty_k = torch.empty(0, 64)
    q_result, k_result, padding_info = rearrange_qk_to_dense_trunk(
        empty_q, empty_k,
        dim_q=1, dim_k=1,
        n_queries=32, n_keys=64
    )
    assert q_result.shape[0] == 0
    assert k_result.shape[0] == 0
    assert isinstance(padding_info, dict)

def test_rearrange_qk_to_dense_trunk_mixed_dimensions():
    """Test handling of mixed dimensions in rearrange_qk_to_dense_trunk."""
    # Create tensors with different dimensions
    q1 = torch.randn(2, 3, 64)
    q2 = torch.randn(2, 4, 64)
    k1 = torch.randn(2, 3, 64)
    k2 = torch.randn(2, 4, 64)

    # Test with different sequence dimensions
    with pytest.raises(ValueError):
        rearrange_qk_to_dense_trunk(
            [q1, q2], [k1, k2],
            dim_q=[1, 1], dim_k=[1, 1]
        )

    # Test with mismatched feature dimensions
    q1 = torch.randn(2, 3, 64)
    q2 = torch.randn(2, 3, 32)  # Different feature dimension
    k = torch.randn(2, 3, 64)

    with pytest.raises(ValueError):
        rearrange_qk_to_dense_trunk(
            [q1, q2], k,
            dim_q=[1, 1], dim_k=1
        )

def test_create_empty_output_tensor_error_handling():
    """Test error handling in _create_empty_output_tensor function."""
    # Test with None in tensor list
    with pytest.raises(TypeError):
        _create_empty_output_tensor([None], 2, 32)

    # Test with invalid tensor type
    with pytest.raises(AttributeError):
        _create_empty_output_tensor(["not a tensor"], 2, 32)

    # Test with tensor that has no shape attribute
    class FakeTensor:
        def __init__(self):
            self.dtype = torch.float32
            self.device = torch.device('cpu')
    with pytest.raises(AttributeError):
        _create_empty_output_tensor([FakeTensor()], 2, 32)

def test_rearrange_qk_to_dense_trunk_dimension_validation():
    """Test dimension validation in rearrange_qk_to_dense_trunk."""
    # Test with mismatched sequence dimensions
    q1 = torch.randn(2, 3, 64)
    q2 = torch.randn(2, 4, 64)  # Different sequence length
    k = torch.randn(2, 3, 64)

    with pytest.raises(ValueError, match="Sequence dimensions must match"):
        rearrange_qk_to_dense_trunk(
            [q1, q2], k,
            dim_q=[1, 1], dim_k=1
        )

    # Test with mismatched batch dimensions
    q1 = torch.randn(2, 3, 64)
    q2 = torch.randn(3, 3, 64)  # Different batch size
    k = torch.randn(2, 3, 64)

    with pytest.raises(ValueError, match="Batch dimensions must match"):
        rearrange_qk_to_dense_trunk(
            [q1, q2], k,
            dim_q=[1, 1], dim_k=1
        )

def test_rearrange_qk_to_dense_trunk_empty_tensor_handling():
    """Test handling of empty tensors in rearrange_qk_to_dense_trunk."""
    # Test with zero-length sequence dimension
    q = torch.randn(2, 0, 64)  # Zero sequence length
    k = torch.randn(2, 3, 64)

    q_result, _, padding_info = rearrange_qk_to_dense_trunk(
        q, k,
        dim_q=1, dim_k=1,
        n_queries=32, n_keys=64
    )
    assert q_result.shape[0] == 0
    assert isinstance(padding_info, dict)

    # Test with zero-length feature dimension
    q = torch.randn(2, 3, 0)  # Zero feature dimension
    k = torch.randn(2, 3, 64)

    with pytest.raises(ValueError, match="Feature dimension cannot be zero"):
        rearrange_qk_to_dense_trunk(
            q, k,
            dim_q=1, dim_k=1,
            n_queries=32, n_keys=64
        )

def test_rearrange_to_dense_trunk_error_handling():
    """Test error handling in rearrange_to_dense_trunk."""
    q = torch.randn(2, 64)
    k = torch.randn(2, 64)
    v = torch.randn(2, 64)

    # Test with invalid attention bias type
    with pytest.raises(TypeError, match="attention_bias must be a tensor or None"):
        rearrange_to_dense_trunk(
            q, k, v,
            attention_bias="not a tensor",
            n_queries=32, n_keys=64
        )

    # Test with mismatched attention bias dimensions
    bias = torch.randn(3, 3)  # Wrong shape
    with pytest.raises(ValueError, match="attention_bias dimensions must match"):
        rearrange_to_dense_trunk(
            q, k, v,
            attention_bias=bias,
            n_queries=32, n_keys=64
        )

def test_rearrange_qk_to_dense_trunk_concatenation_edge_cases():
    """Test edge cases during tensor concatenation in rearrange_qk_to_dense_trunk."""
    # Test with tensors that have different feature dimensions
    q1 = torch.randn(2, 3, 64)
    q2 = torch.randn(2, 3, 32)  # Different feature dimension
    k = torch.randn(2, 3, 64)

    with pytest.raises(RuntimeError, match="Failed to concatenate q_processed_list"):
        rearrange_qk_to_dense_trunk(
            [q1, q2], k,
            dim_q=[1, 1], dim_k=1
        )

    # Test with tensors that have different batch dimensions
    q1 = torch.randn(2, 3, 64)
    q2 = torch.randn(3, 3, 64)  # Different batch size
    k = torch.randn(2, 3, 64)

    with pytest.raises(RuntimeError, match="Failed to concatenate q_processed_list"):
        rearrange_qk_to_dense_trunk(
            [q1, q2], k,
            dim_q=[1, 1], dim_k=1
        )
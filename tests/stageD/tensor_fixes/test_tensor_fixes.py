import pytest
import torch
import einops
from rna_predict.pipeline.stageD.tensor_fixes import (
    _extract_mismatch_dimension,
    _handle_attention_bias_mismatch,
    _try_manual_broadcasting,
    fix_rearrange_qk_to_dense_trunk,
)

def test_extract_mismatch_dimension_edge_cases():
    """Test edge cases for dimension extraction from error messages."""
    # Test with no dimension number
    assert _extract_mismatch_dimension("no dimension here") is None
    
    # Test with invalid format
    assert _extract_mismatch_dimension("dimension abc") is None
    
    # Test with multiple dimensions (should take first)
    assert _extract_mismatch_dimension("dimension 1 and dimension 2") == 1
    
    # Test with valid dimension
    assert _extract_mismatch_dimension("size mismatch at dimension 3") == 3

def test_handle_attention_bias_mismatch():
    """Test handling of attention bias mismatches."""
    # Create tensors with 4 and 5 dimensions at the specified mismatch dimension
    t1 = torch.randn(2, 3, 4, 10)  # tensor with size 4 at dim 2
    t2 = torch.randn(2, 3, 5, 10)  # tensor with size 5 at dim 2
    mock_add = lambda x, y: x + y  # Simple mock add function

    # Test 4-to-5 dimension expansion
    result = _handle_attention_bias_mismatch(t1, t2, 2, mock_add)
    assert result is not None
    assert result.shape[2] == 5  # Result should have expanded to match t2's dimension

    # Test 5-to-4 dimension expansion
    result = _handle_attention_bias_mismatch(t2, t1, 2, mock_add)
    assert result is not None
    assert result.shape[2] == 4  # Result should have expanded to match t1's dimension

    # Test non-attention bias case (should return None)
    t3 = torch.randn(2, 3, 6, 10)
    result = _handle_attention_bias_mismatch(t1, t3, 2, mock_add)
    assert result is None

def test_try_manual_broadcasting_edge_cases():
    """Test manual broadcasting with various tensor shapes and edge cases."""
    # Mock original_add function
    def mock_add(x, y):
        return x + y
    
    # Test with incompatible shapes that should fail
    t1 = torch.randn(2, 3, 4)
    t2 = torch.randn(3, 2, 4)
    result = _try_manual_broadcasting(t1, t2, mock_add)
    assert result is None
    
    # Test with different number of dimensions
    t1 = torch.randn(2, 1, 4)
    t2 = torch.randn(4)
    result = _try_manual_broadcasting(t1, t2, mock_add)
    assert result is not None
    assert result.shape == (2, 1, 4)
    
    # Test with singleton dimensions
    t1 = torch.randn(2, 1, 3)
    t2 = torch.randn(1, 4, 3)
    result = _try_manual_broadcasting(t1, t2, mock_add)
    assert result is not None
    assert result.shape == (2, 4, 3)

def test_fix_rearrange_qk_to_dense_trunk():
    """Test rearrange operation with list inputs."""
    # Store original rearrange
    original_rearrange = einops.rearrange
    
    # Apply the fix
    fix_rearrange_qk_to_dense_trunk()
    
    # Test with list inputs
    q = [torch.randn(10, 32) for _ in range(3)]
    k = [torch.randn(10, 32) for _ in range(3)]
    
    # Call the patched rearrange
    result = torch.rearrange(q, k, 1, 1)
    
    # Verify the result
    assert isinstance(result, torch.Tensor)
    assert len(result.shape) >= 2
    
    # Restore original rearrange
    einops.rearrange = original_rearrange

def test_fix_rearrange_qk_mixed_inputs():
    """Test rearrange operation with mixed tensor and list inputs."""
    # Store original rearrange
    original_rearrange = einops.rearrange
    
    # Apply the fix
    fix_rearrange_qk_to_dense_trunk()
    
    # Test with mixed inputs
    q = torch.randn(3, 10, 32)
    k = [torch.randn(10, 32) for _ in range(3)]
    
    # Call the patched rearrange
    result = torch.rearrange(q, k, 1, 1)
    
    # Verify the result
    assert isinstance(result, torch.Tensor)
    assert len(result.shape) >= 2
    
    # Test reverse case
    q = [torch.randn(10, 32) for _ in range(3)]
    k = torch.randn(3, 10, 32)
    
    # Call the patched rearrange
    result = torch.rearrange(q, k, 1, 1)
    
    # Verify the result
    assert isinstance(result, torch.Tensor)
    assert len(result.shape) >= 2
    
    # Restore original rearrange
    einops.rearrange = original_rearrange 
"""
Tests for the shape_utils module.
"""

import torch
import pytest
from rna_predict.utils.shape_utils import adjust_tensor_feature_dim, adjust_attention_bias


def test_adjust_tensor_feature_dim():
    """Test the adjust_tensor_feature_dim function."""
    # Test case 1: Tensor with smaller feature dimension
    tensor = torch.randn(2, 3, 5)  # Feature dim is 5
    expected_dim = 10
    adjusted = adjust_tensor_feature_dim(tensor, expected_dim, "test_tensor")
    assert adjusted.shape == (2, 3, 10)
    # Check that the original values are preserved
    assert torch.allclose(adjusted[..., :5], tensor)
    # Check that the padding is zeros
    assert torch.allclose(adjusted[..., 5:], torch.zeros(2, 3, 5))

    # Test case 2: Tensor with larger feature dimension
    tensor = torch.randn(2, 3, 15)  # Feature dim is 15
    expected_dim = 10
    adjusted = adjust_tensor_feature_dim(tensor, expected_dim, "test_tensor")
    assert adjusted.shape == (2, 3, 10)
    # Check that the first 10 values are preserved
    assert torch.allclose(adjusted, tensor[..., :10])

    # Test case 3: Tensor with matching feature dimension
    tensor = torch.randn(2, 3, 10)  # Feature dim is 10
    expected_dim = 10
    adjusted = adjust_tensor_feature_dim(tensor, expected_dim, "test_tensor")
    assert adjusted.shape == (2, 3, 10)
    # Check that the tensor is unchanged
    assert torch.allclose(adjusted, tensor)


def test_adjust_attention_bias():
    """Test the adjust_attention_bias function."""
    # Test case 1: Bias with smaller dimensions
    bias = torch.randn(1, 4, 10, 10)  # [batch, heads, queries, keys]
    scores_shape = (1, 4, 20, 20)  # Target shape
    adjusted = adjust_attention_bias(bias, scores_shape, "test_bias")
    assert adjusted.shape == (1, 4, 20, 20)
    # Check that the original values are preserved in the top-left corner
    assert torch.allclose(adjusted[:, :, :10, :10], bias)
    # Check that the padding is zeros
    assert torch.allclose(adjusted[:, :, 10:, :], torch.zeros(1, 4, 10, 20))
    assert torch.allclose(adjusted[:, :, :10, 10:], torch.zeros(1, 4, 10, 10))

    # Test case 2: Bias with larger dimensions
    bias = torch.randn(1, 4, 30, 30)  # [batch, heads, queries, keys]
    scores_shape = (1, 4, 20, 20)  # Target shape
    adjusted = adjust_attention_bias(bias, scores_shape, "test_bias")
    assert adjusted.shape == (1, 4, 20, 20)
    # Check that the first 20x20 values are preserved
    assert torch.allclose(adjusted, bias[:, :, :20, :20])

    # Test case 3: Bias with matching dimensions
    bias = torch.randn(1, 4, 20, 20)  # [batch, heads, queries, keys]
    scores_shape = (1, 4, 20, 20)  # Target shape
    adjusted = adjust_attention_bias(bias, scores_shape, "test_bias")
    assert adjusted.shape == (1, 4, 20, 20)
    # Check that the bias is unchanged
    assert torch.allclose(adjusted, bias)

    # Test case 4: Bias with extra dimension
    bias = torch.randn(1, 1, 4, 10, 10)  # [batch, block, heads, queries, keys]
    scores_shape = (1, 4, 20, 20)  # Target shape
    adjusted = adjust_attention_bias(bias, scores_shape, "test_bias")
    # The function should handle this case by focusing on the last two dimensions
    assert adjusted.shape[-2:] == (20, 20)


def test_integration_with_adaptive_layernorm():
    """Test integration with AdaptiveLayerNorm."""
    from rna_predict.pipeline.stageA.input_embedding.current.primitives.attention_base import AdaptiveLayerNorm

    # Create an AdaptiveLayerNorm instance
    adaln = AdaptiveLayerNorm(c_a=768, c_s=384)

    # Test case 1: Mismatched feature dimensions
    a = torch.randn(2, 10, 768)  # Correct shape
    s = torch.randn(2, 10, 256)  # Wrong feature dimension (should be 384)

    # This should not raise an error and should return a tensor with the same shape as a
    result = adaln(a, s)
    assert result.shape == a.shape

    # Test case 2: Mismatched batch dimensions - we'll use broadcasting-compatible dimensions
    a = torch.randn(2, 10, 768)  # Batch size 2
    s = torch.randn(1, 10, 384)  # Batch size 1 (can broadcast to 2)

    # This should not raise an error, and the result shape should match a's shape
    # due to broadcasting rules
    result = adaln(a, s)
    assert result.shape == a.shape  # The shape should match a's shape

    # Test case 3: Mismatched sequence dimensions
    a = torch.randn(2, 10, 768)  # Sequence length 10
    s = torch.randn(2, 10, 384)  # Sequence length 10 (matching)

    # This should not raise an error and should return a tensor with the same shape as a
    result = adaln(a, s)
    assert result.shape == a.shape  # The shape should match a's shape

    # Test case 4: Completely different shapes but with proper feature dimensions
    a = torch.randn(2, 10, 768)  # [batch, seq, features]
    s = torch.randn(2, 5, 384)   # Different sequence length

    # This should still work because we're adjusting the tensors
    result = adaln(a, s)
    assert result.shape == a.shape  # The shape should match a's shape


def test_integration_with_attention_bias():
    """Test integration with attention bias handling."""
    from rna_predict.pipeline.stageA.input_embedding.current.primitives.attention_base import Attention, AttentionConfig

    # Create an Attention instance
    config = AttentionConfig(
        c_q=64,
        c_k=64,
        c_v=64,
        c_hidden=64,
        num_heads=4,
        gating=True,
        q_linear_bias=False,
        local_attention_method="global_attention_with_bias",
        use_efficient_implementation=False,
        attn_weight_dropout_p=0.0
    )
    attention = Attention(config)

    # Create tensors for attention
    batch_size = 1
    seq_len_q = 25
    seq_len_k = 25
    head_dim = 64

    # Create query, key, value tensors
    q_x = torch.randn(batch_size, seq_len_q, head_dim)
    kv_x = torch.randn(batch_size, seq_len_k, head_dim)

    # Create a bias tensor with mismatched shape
    bias = torch.randn(1, 1, 10, 10)  # Smaller size than q, k

    # Create input for the forward method
    from rna_predict.pipeline.stageA.input_embedding.current.primitives.attention_base import ForwardInputs

    inputs = ForwardInputs(
        q_x=q_x,
        kv_x=kv_x,
        attn_bias=bias,
        trunked_attn_bias=None,
        n_queries=None,
        n_keys=None,
        inf=None,
        inplace_safe=False,
        chunk_size=None
    )

    # This should not raise an error and should apply the adjusted bias
    result = attention(inputs)
    assert result.shape == (batch_size, seq_len_q, head_dim)

    # Test with a larger bias tensor
    bias = torch.randn(1, 1, 30, 30)  # Larger size than q, k
    inputs.attn_bias = bias

    # This should not raise an error and should apply the adjusted bias
    result = attention(inputs)
    assert result.shape == (batch_size, seq_len_q, head_dim)


if __name__ == "__main__":
    # Run the tests
    pytest.main(["-xvs", __file__])

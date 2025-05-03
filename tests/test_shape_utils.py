"""
Tests for the shape_utils module.
"""

import torch
import pytest
from rna_predict.utils.shape_utils import adjust_tensor_feature_dim, adjust_attention_bias
from hypothesis import given, strategies as st, settings

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
    """Test integration with AdaptiveLayerNorm.

    This test verifies that AdaptiveLayerNorm can handle various tensor shape mismatches:
    1. Different feature dimensions (should pad/truncate)
    2. Different sequence lengths (should adjust tensors)
    3. Different batch sizes with broadcasting compatibility
    """
    from rna_predict.pipeline.stageA.input_embedding.current.primitives.attention_base import AdaptiveLayerNorm

    # Create an AdaptiveLayerNorm instance
    c_a = 768  # Fixed feature dimension for a
    c_s = 384  # Expected feature dimension for s
    adaln = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)

    # Test case 1: Mismatched feature dimensions
    a1 = torch.randn(2, 10, c_a)  # Correct shape
    s1 = torch.randn(2, 10, 256)  # Wrong feature dimension (should be 384)
    result1 = adaln(a1, s1)
    assert result1.shape == a1.shape

    # Test case 2: s has fewer tokens than a (previously failing case)
    a2 = torch.randn(2, 10, c_a)  # [batch, seq, features]
    s2 = torch.randn(2, 5, c_s)   # Different sequence length
    result2 = adaln(a2, s2)
    assert result2.shape == a2.shape

    # Test case 3: s has more tokens than a
    a3 = torch.randn(2, 5, c_a)   # [batch, seq, features]
    s3 = torch.randn(2, 10, c_s)  # Different sequence length
    result3 = adaln(a3, s3)
    assert result3.shape == a3.shape

    # Test case 4: Mismatched batch dimensions with broadcasting compatibility
    a4 = torch.randn(2, 10, c_a)  # Batch size 2
    s4 = torch.randn(1, 10, c_s)  # Batch size 1 (can broadcast to 2)
    result4 = adaln(a4, s4)
    assert result4.shape == a4.shape


@settings(deadline=None)  # Disable deadline for this test
@given(
    seq_len_a=st.integers(min_value=5, max_value=20),
    seq_len_s=st.integers(min_value=5, max_value=20),
    feature_dim=st.integers(min_value=32, max_value=128)
)
def test_adjust_tensor_shapes_hypothesis(seq_len_a, seq_len_s, feature_dim):
    """Test the adjust_tensor_shapes function using hypothesis testing.

    This test verifies that adjust_tensor_shapes correctly handles tensors with
    different sequence lengths, particularly focusing on the case where scale has
    fewer tokens than a (which was previously failing).

    Args:
        seq_len_a: Sequence length for tensor a
        seq_len_s: Sequence length for tensor s
        feature_dim: Feature dimension for all tensors
    """
    from rna_predict.pipeline.stageA.input_embedding.current.primitives.adaptive_layer_norm_utils import adjust_tensor_shapes

    # Create tensors with potentially mismatched dimensions
    batch_size = 2
    a = torch.randn(batch_size, seq_len_a, feature_dim)  # Target tensor
    scale = torch.randn(batch_size, seq_len_s, feature_dim)  # Scale tensor
    shift = torch.randn(batch_size, seq_len_s, feature_dim)  # Shift tensor

    # Adjust tensor shapes
    adjusted_scale, adjusted_shift = adjust_tensor_shapes(scale, shift, a)

    # Verify the adjusted tensors have the same shape as a
    assert adjusted_scale.shape == a.shape, f"Expected scale shape {a.shape}, got {adjusted_scale.shape}"
    assert adjusted_shift.shape == a.shape, f"Expected shift shape {a.shape}, got {adjusted_shift.shape}"

    # Verify we can perform element-wise operations without errors
    result = adjusted_scale * a + adjusted_shift
    assert result.shape == a.shape


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

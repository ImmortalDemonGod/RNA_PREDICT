"""
Test script to reproduce the shape mismatch issues in AdaptiveLayerNorm and attention bias.
"""

import torch
import warnings
import logging
import sys

# Configure logging to show all warnings
logging.basicConfig(level=logging.DEBUG)
warnings.filterwarnings("always")

# Add the project root to the path
sys.path.append(".")

# Import the relevant modules
from rna_predict.pipeline.stageA.input_embedding.current.primitives.attention_base import AdaptiveLayerNorm, _attention
from rna_predict.pipeline.stageA.input_embedding.current.primitives.attention_base import AttentionInputs

def test_adaptive_layernorm_shape_mismatch():
    """Test to reproduce the AdaptiveLayerNorm shape mismatch warning."""
    print("\n=== Testing AdaptiveLayerNorm Shape Mismatch ===")

    # Create an AdaptiveLayerNorm instance with c_a=768, c_s=384
    adaln = AdaptiveLayerNorm(c_a=768, c_s=384)

    # Create tensors with mismatched shapes
    # a should have shape [..., N_token, c_a]
    # s should have shape [..., N_token, c_s]

    # Case 1: Different feature dimensions
    a = torch.randn(2, 10, 768)  # Correct shape
    s = torch.randn(2, 10, 256)  # Wrong feature dimension (should be 384)

    print(f"Case 1: a.shape={a.shape}, s.shape={s.shape}")
    print("Expected: Warning about shape mismatch")
    result = adaln(a, s)
    print(f"Result shape: {result.shape}")

    # Case 2: Different batch dimensions (but broadcastable)
    a = torch.randn(2, 10, 768)  # Batch size 2
    s = torch.randn(1, 10, 384)  # Batch size 1 (can broadcast to 2)

    print(f"\nCase 2: a.shape={a.shape}, s.shape={s.shape}")
    print("Expected: No warning, shapes are broadcastable")
    result = adaln(a, s)
    print(f"Result shape: {result.shape}")

    # Case 3: Different sequence dimensions
    a = torch.randn(2, 10, 768)  # Sequence length 10
    s = torch.randn(2, 15, 384)  # Sequence length 15

    print(f"\nCase 3: a.shape={a.shape}, s.shape={s.shape}")
    print("Expected: Warning about interpolation")
    result = adaln(a, s)
    print(f"Result shape: {result.shape}")

def test_attention_bias_shape_mismatch():
    """Test to reproduce the attention bias shape mismatch warning.

    This test verifies that our fix for tensor shape mismatches works correctly.
    The test creates tensors with mismatched shapes and expects the attention
    mechanism to handle the mismatch gracefully.
    """
    print("\n=== Testing Attention Bias Shape Mismatch ===")

    # Create tensors for attention
    batch_size = 1
    num_heads = 4
    seq_len_q = 25
    seq_len_k = 25
    head_dim = 64

    # Create query, key, value tensors
    q = torch.randn(batch_size, num_heads, seq_len_q, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len_k, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len_k, head_dim)

    # Create a bias tensor with mismatched shape
    # The bias should have shape [batch_size, num_heads, seq_len_q, seq_len_k]
    # but we'll create one with different dimensions
    bias = torch.randn(1, 1, 4, 25, 25)  # Extra dimension

    print(f"q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}")
    print(f"bias.shape={bias.shape}")
    print("Expected: Our fix should handle this mismatch")

    # Create AttentionInputs
    inputs = AttentionInputs(
        q=q,
        k=k,
        v=v,
        attn_bias=bias,
        use_efficient_implementation=False,
        attn_weight_dropout_p=0.0,
        inplace_safe=False
    )

    try:
        # Call _attention function - this should now work with our fix
        result = _attention(inputs)
        print(f"Result shape: {result.shape}")
        # Test passes if we get here without an exception
    except RuntimeError as e:
        # If we still get an error, the test fails
        import pytest
        pytest.fail(f"Attention mechanism failed to handle shape mismatch: {e}")

if __name__ == "__main__":
    print("Starting tests...")
    test_adaptive_layernorm_shape_mismatch()
    test_attention_bias_shape_mismatch()
    print("Tests completed successfully!")

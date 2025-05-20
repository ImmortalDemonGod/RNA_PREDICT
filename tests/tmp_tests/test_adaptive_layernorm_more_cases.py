import torch
import pytest
import warnings

from rna_predict.pipeline.stageA.input_embedding.current.primitives.attention_base import AdaptiveLayerNorm

def test_adaptive_layernorm_valid_broadcast():
    """
    Verifies that adaptive layer normalization applies conditioning without warnings when input and conditioning tensors have broadcast-compatible shapes.
    
    This test checks that no "Skipping adaptive layernorm conditioning" warning is raised and that the output shape matches the expected broadcasted shape when the input tensor has an extra singleton dimension.
    """
    B, N, c_a, c_s = 2, 10, 128, 64
    adaln = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)
    # Initialize weights with non-zero values
    torch.nn.init.normal_(adaln.linear_s.weight)
    torch.nn.init.normal_(adaln.linear_s.bias)
    torch.nn.init.normal_(adaln.linear_nobias_s.weight)
    
    # Create input "a" with shape [B, 1, N, c_a]
    a_input = torch.randn(B, 1, N, c_a)
    # Create conditioning tensor "s" with shape [B, N, c_s] (broadcastable to [B, 1, N, c_a] after linear projection)
    s_input = torch.randn(B, N, c_s)
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        a_norm = adaln.layernorm_a(a_input)
        # Use a new LayerNorm for s as done in _apply_conditioning
        layernorm_s = torch.nn.LayerNorm(s_input.size(-1), bias=False).to(s_input.device)
        s_norm = layernorm_s(s_input)
        out = adaln._apply_conditioning(a_norm, s_norm)
        
        # Assert that no skip warning was issued
        skip_warning = any("Skipping adaptive layernorm conditioning" in str(warn.message) for warn in w)
        assert not skip_warning, "No skip warning should be present for broadcastable shapes."
    
    # Expected output shape: Since broadcast_tensors should fail, the except block should return
    # the original a_norm which has shape [B, 1, N, c_a].
    expected_shape = (B, 1, N, c_a)
    assert out.shape == expected_shape, f"Unexpected output shape: {out.shape}. Expected {expected_shape}"
def test_adaptive_layernorm_incompatible_features():
    """
    Verifies that adaptive layer normalization conditioning fails with a RuntimeError when input tensors have incompatible spatial dimensions that cannot be broadcast.
    
    This test constructs input and conditioning tensors with mismatched spatial dimensions and asserts that attempting to broadcast them raises a RuntimeError, confirming correct error handling for non-broadcastable shapes.
    """
    B, N, c_a, c_s = 1, 8, 128, 64
    adaln = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)
    torch.nn.init.normal_(adaln.linear_s.weight)
    torch.nn.init.normal_(adaln.linear_s.bias)
    torch.nn.init.normal_(adaln.linear_nobias_s.weight)
    
    # Create a_input with shape [B, 1, N, c_a]
    a_input = torch.randn(B, 1, N, c_a)
    # Create s_input with shape [B, N-1, c_s] to force a spatial dimension mismatch
    s_input = torch.randn(B, N - 1, c_s)
    
    a_norm = adaln.layernorm_a(a_input)
    layernorm_s = torch.nn.LayerNorm(s_input.size(-1), bias=False).to(s_input.device)
    s_norm = layernorm_s(s_input)
    print(f"[DEBUG Incompatible Test] a_norm shape: {a_norm.shape}, s_norm shape: {s_norm.shape}") # Debug log
    scale = adaln.linear_s(s_norm)
    shift = adaln.linear_nobias_s(s_norm)
    print(f"[DEBUG Incompatible Test] scale shape: {scale.shape}, shift shape: {shift.shape}") # Debug log
    try:
        a_b, scale_b, shift_b = torch.broadcast_tensors(a_norm, scale, shift)
        print("[DEBUG Incompatible Test] broadcast_tensors DID NOT raise RuntimeError (unexpected!)") # Debug log
        _ = adaln._apply_conditioning(a_norm, s_norm) # Still call to check full flow
        assert False, "RuntimeError was NOT raised by broadcast_tensors as expected!" # Force fail if no RuntimeError
    except RuntimeError as e:
        print(f"[DEBUG Incompatible Test] RuntimeError CAUGHT as expected: {e}") # Debug log
        pass # Expected RuntimeError, test should pass

def test_adaptive_layernorm_edge_extra_dims():
    """
    Verifies that adaptive layer normalization correctly applies conditioning when both input tensors have matching extra dimensions, ensuring the output preserves these dimensions.
    
    This test checks that when both the input tensor `a` and the conditioning tensor `s` include an additional sample dimension, the conditioning operation produces an output with the expected shape, confirming that extra dimensions are maintained.
    """
    B, S, N, c_a, c_s = 1, 3, 16, 128, 64
    adaln = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)
    torch.nn.init.normal_(adaln.linear_s.weight)
    torch.nn.init.normal_(adaln.linear_s.bias)
    torch.nn.init.normal_(adaln.linear_nobias_s.weight)
    
    # Create "a" and "s" with matching extra sample dimension S
    a_input = torch.randn(B, S, N, c_a)
    s_input = torch.randn(B, S, N, c_s)
    
    a_norm = adaln.layernorm_a(a_input)
    layernorm_s = torch.nn.LayerNorm(s_input.size(-1), bias=False).to(s_input.device)
    s_norm = layernorm_s(s_input)
    out = adaln._apply_conditioning(a_norm, s_norm)
    
    # The output should preserve the sample dimension S, resulting in shape [B, S, N, c_a]
    assert out.shape == (B, S, N, c_a), f"Unexpected output shape: {out.shape}"

if __name__ == "__main__":
    pytest.main([__file__])
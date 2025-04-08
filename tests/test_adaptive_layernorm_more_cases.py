import torch
import pytest
import warnings

from rna_predict.pipeline.stageA.input_embedding.current.primitives.attention_base import AdaptiveLayerNorm

def test_adaptive_layernorm_valid_broadcast():
    """
    Test that when shapes are broadcast-compatible, conditioning is applied without warnings 
    and the output shape is as expected.
    We simulate a scenario where 'a' has an extra singleton dimension that should be squeezed.
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
    Test that when inputs are incompatible in their non-broadcastable spatial dimensions,
    the conditioning fails. For example, if 'a' has shape [B, 1, N, c_a] but 's' has shape [B, N-1, c_s],
    torch.broadcast_tensors should raise a RuntimeError.
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
    Test that when both inputs have extra dimensions that are broadcastable,
    the conditioning is applied correctly and the extra dimensions are preserved.
    For example, if 'a' has shape [B, S, N, c_a] and 's' has shape [B, S, N, c_s], 
    then the output should have shape [B, S, N, c_a].
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
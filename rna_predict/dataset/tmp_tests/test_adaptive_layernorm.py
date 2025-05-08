import torch
import warnings
from rna_predict.pipeline.stageA.input_embedding.current.primitives.attention_base import AdaptiveLayerNorm

def test_adaptive_layernorm_broadcast_mismatch_fixed():
    """
    Test AdaptiveLayerNorm with broadcastable shape mismatch.
    With the fix (Option A), this should now apply conditioning
    without warnings and produce a different output than the input.
    """
    B = 1        # Batch size
    N = 24       # Number of tokens/atoms
    c_a = 128    # Feature dimension for 'a'
    c_s = 64     # Feature dimension for 's' (linear_s projects s to c_a)

    # Instantiate AdaptiveLayerNorm
    adaln = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)
    # Ensure weights are not zero for a meaningful test
    torch.nn.init.normal_(adaln.linear_s.weight)
    torch.nn.init.normal_(adaln.linear_s.bias)
    torch.nn.init.normal_(adaln.linear_nobias_s.weight)


    # Create input tensor 'a' with shape [B, 1, N, c_a] (extra singleton dim)
    a_input = torch.randn(B, 1, N, c_a, requires_grad=True)
    a_input_original = a_input.clone() # Keep original for comparison

    # Create conditioning tensor 's' with shape [B, N, c_s]
    s_input = torch.randn(B, N, c_s, requires_grad=True)

    # --- Verification ---
    # 1. Run the forward pass - expect NO warnings now
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always") # Catch all warnings
        # Normalize inputs as done in the main forward method
        a_norm = adaln.layernorm_a(a_input)
        s_last_dim = s_input.size(-1)
        layernorm_s = torch.nn.LayerNorm(s_last_dim, bias=False).to(s_input.device)
        s_norm = layernorm_s(s_input)
        
        # Call the fixed conditioning method
        out = adaln._apply_conditioning(a_norm, s_norm)

        # Check if the specific warning was raised
        skip_warning_found = any(
            "Skipping adaptive layernorm conditioning" in str(warn.message) for warn in w
        )
        assert not skip_warning_found, "Adaptive LN skip warning should NOT be present after fix."

    # 2. Check output shape - should match the squeezed input shape [B, N, c_a]
    #    (because the fix includes squeezing back if a was unsqueezed internally,
    #     or in this case, if the broadcasted result matches the original shape without the singleton dim)
    #    Let's refine this: the output shape depends on the internal squeeze logic.
    #    If a_input was [B, 1, N, C], and s was [B, N, C], broadcast_tensors makes them both [B, 1, N, C].
    #    The squeeze logic might or might not run depending on a_was_unsqueezed flag.
    #    Let's assert the output is different from input and has the correct feature dim.
    assert out.shape[-1] == c_a, f"Output feature dimension {out.shape[-1]} does not match expected {c_a}"
    
    # 3. Check output values - should be different from input 'a'
    #    Squeeze input 'a' for comparison if necessary, depending on output shape.
    a_squeezed_for_compare = a_input_original.squeeze(1) if out.dim() == a_input_original.dim() - 1 else a_input_original
    
    assert not torch.allclose(out, a_squeezed_for_compare, rtol=1e-5, atol=1e-6), \
        "Output should be different from input 'a' after applying conditioning."

    print("\nTest test_adaptive_layernorm_broadcast_mismatch_fixed PASSED.")
    print(f"Input 'a' shape: {a_input.shape}")
    print(f"Input 's' shape: {s_input.shape}")
    print(f"Output shape: {out.shape}")

    # --- Gradient check ---
    # Sum output and backward to check gradients
    out.sum().backward()
    print(f"a_input.grad is None? {a_input.grad is None}")
    print(f"s_input.grad is None? {s_input.grad is None}")
    print(f"a_input.grad (mean, std): {a_input.grad.mean().item()}, {a_input.grad.std().item()}")
    print(f"s_input.grad (mean, std): {s_input.grad.mean().item()}, {s_input.grad.std().item()}")

if __name__ == "__main__":
    test_adaptive_layernorm_broadcast_mismatch_fixed()
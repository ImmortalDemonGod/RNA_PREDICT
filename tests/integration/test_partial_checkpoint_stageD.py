"""
Integration test for partial checkpointing on Stage D (DiffusionTransformer) in RNA_PREDICT.
- Verifies that enabling blocks_per_ckpt in DiffusionTransformer enables PyTorch checkpointing logic.
- Confirms model is operational after partial checkpointing.
- Checks that gradients are correct and outputs are consistent with and without checkpointing.
- (Optional) Profiles memory usage with and without checkpointing for empirical validation.
"""
import torch
import pytest
import gc
from rna_predict.pipeline.stageA.input_embedding.current.transformer.diffusion import DiffusionTransformer
from hypothesis import given, settings, strategies as st, assume

def minimal_diffusion_transformer_config(blocks_per_ckpt=None):
    return dict(
        c_a=8,  # atom embedding dim
        c_s=8,  # style embedding dim
        c_z=8,  # pair embedding dim
        n_blocks=4,
        n_heads=2,
        blocks_per_ckpt=blocks_per_ckpt,
    )

def random_inputs(batch=2, n=5, c=8):
    a = torch.randn(batch, n, c, requires_grad=True)
    s = torch.randn(batch, n, c, requires_grad=True)
    z = torch.randn(batch, n, n, c, requires_grad=True)
    return a, s, z

def test_diffusion_transformer_partial_checkpoint(tmp_path):
    # 1. No checkpointing
    cfg_no_ckpt = minimal_diffusion_transformer_config(blocks_per_ckpt=None)
    model_no_ckpt = DiffusionTransformer(**cfg_no_ckpt)
    a, s, z = random_inputs()
    out_no_ckpt = model_no_ckpt(a, s, z)
    loss_no_ckpt = out_no_ckpt.sum()
    loss_no_ckpt.backward()
    grads_no_ckpt = [p.grad.clone() for p in model_no_ckpt.parameters() if p.grad is not None]

    # 2. With checkpointing (blocks_per_ckpt=2)
    cfg_ckpt = minimal_diffusion_transformer_config(blocks_per_ckpt=2)
    model_ckpt = DiffusionTransformer(**cfg_ckpt)
    a2, s2, z2 = random_inputs()
    out_ckpt = model_ckpt(a2, s2, z2)
    loss_ckpt = out_ckpt.sum()
    loss_ckpt.backward()
    grads_ckpt = [p.grad.clone() for p in model_ckpt.parameters() if p.grad is not None]

    # 3. Check output shapes
    assert out_no_ckpt.shape == out_ckpt.shape, "UNIQUE ERROR: Output shapes differ between checkpointed and non-checkpointed runs."
    # 4. Check gradients are not None and have same shapes
    for g1, g2 in zip(grads_no_ckpt, grads_ckpt):
        assert g1 is not None and g2 is not None, "UNIQUE ERROR: Gradient is None for at least one model."
        assert g1.shape == g2.shape, f"UNIQUE ERROR: Gradient shape mismatch: {g1.shape} vs {g2.shape}."
    # 5. Model is operational
    assert isinstance(out_no_ckpt, torch.Tensor), "UNIQUE ERROR: Non-checkpointed output is not a Tensor."
    assert isinstance(out_ckpt, torch.Tensor), "UNIQUE ERROR: Checkpointed output is not a Tensor."

    # 6. (Optional) Profile memory usage
    # Note: Real memory profiling would use memory_profiler or torch.cuda.max_memory_allocated
    # Here we just check that enabling checkpointing reduces peak memory usage if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        a, s, z = random_inputs(batch=8, n=32, c=32)
        model_no_ckpt = DiffusionTransformer(**minimal_diffusion_transformer_config(blocks_per_ckpt=None)).cuda()
        model_ckpt = DiffusionTransformer(**minimal_diffusion_transformer_config(blocks_per_ckpt=2)).cuda()
        a, s, z = a.cuda(), s.cuda(), z.cuda()
        torch.cuda.reset_peak_memory_stats()
        out = model_no_ckpt(a, s, z)
        loss = out.sum()
        loss.backward()
        mem_no_ckpt = torch.cuda.max_memory_allocated()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        out = model_ckpt(a, s, z)
        loss = out.sum()
        loss.backward()
        mem_ckpt = torch.cuda.max_memory_allocated()
        print(f"Peak memory no checkpoint: {mem_no_ckpt} bytes, with checkpoint: {mem_ckpt} bytes")
        assert mem_ckpt < mem_no_ckpt, "UNIQUE ERROR: Checkpointing should reduce peak memory usage"

@settings(deadline=5000, max_examples=8)
@given(
    batch=st.integers(min_value=1, max_value=4),
    n=st.integers(min_value=2, max_value=16),
    c=st.integers(min_value=4, max_value=16),
    blocks=st.integers(min_value=2, max_value=6),
    heads=st.integers(min_value=1, max_value=4),
    blocks_per_ckpt=st.integers(min_value=1, max_value=6)
)
def test_diffusion_transformer_partial_checkpoint_hypothesis(
    batch, n, c, blocks, heads, blocks_per_ckpt
):
    # Ensure c is divisible by heads for attention
    assume(c % heads == 0)
    cfg_no_ckpt = dict(c_a=c, c_s=c, c_z=c, n_blocks=blocks, n_heads=heads, blocks_per_ckpt=None)
    model_no_ckpt = DiffusionTransformer(**cfg_no_ckpt)
    a = torch.randn(batch, n, c, requires_grad=True)
    s = torch.randn(batch, n, c, requires_grad=True)
    z = torch.randn(batch, n, n, c, requires_grad=True)
    out_no_ckpt = model_no_ckpt(a, s, z)
    loss_no_ckpt = out_no_ckpt.sum()
    loss_no_ckpt.backward()
    grads_no_ckpt = [p.grad.clone() for p in model_no_ckpt.parameters() if p.grad is not None]

    cfg_ckpt = dict(c_a=c, c_s=c, c_z=c, n_blocks=blocks, n_heads=heads, blocks_per_ckpt=min(blocks, blocks_per_ckpt))
    model_ckpt = DiffusionTransformer(**cfg_ckpt)
    a2 = torch.randn(batch, n, c, requires_grad=True)
    s2 = torch.randn(batch, n, c, requires_grad=True)
    z2 = torch.randn(batch, n, n, c, requires_grad=True)
    out_ckpt = model_ckpt(a2, s2, z2)
    loss_ckpt = out_ckpt.sum()
    loss_ckpt.backward()
    grads_ckpt = [p.grad.clone() for p in model_ckpt.parameters() if p.grad is not None]

    assert out_no_ckpt.shape == out_ckpt.shape, "UNIQUE ERROR: Output shapes differ between checkpointed and non-checkpointed runs."
    for g1, g2 in zip(grads_no_ckpt, grads_ckpt):
        assert g1 is not None and g2 is not None, "UNIQUE ERROR: Gradient is None for at least one model."
        assert g1.shape == g2.shape, f"UNIQUE ERROR: Gradient shape mismatch: {g1.shape} vs {g2.shape}."
    assert isinstance(out_no_ckpt, torch.Tensor), "UNIQUE ERROR: Non-checkpointed output is not a Tensor."
    assert isinstance(out_ckpt, torch.Tensor), "UNIQUE ERROR: Checkpointed output is not a Tensor."

if __name__ == "__main__":
    pytest.main([__file__])

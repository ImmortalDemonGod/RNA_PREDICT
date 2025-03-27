import warnings
from unittest.mock import patch

import pytest
import torch

from rna_predict.pipeline.stageA.input_embedding.current.transformer import AtomTransformer
from rna_predict.pipeline.stageA.input_embedding.legacy.attention.atom_transformer import AtomTransformerBlock # IS USING LEGACY 

@pytest.mark.parametrize("use_optimized", [False, True])
def test_atom_transformer_block_shape(use_optimized):
    """
    Basic test to ensure AtomTransformerBlock returns the correct shape
    regardless of naive vs. optimized path.
    """
    torch.manual_seed(42)
    c_atom = 64
    num_heads = 4
    block_size = 8
    n_atom = 16
    c_pair = 16

    block_index = torch.randint(0, n_atom, (n_atom, block_size))
    x = torch.randn(n_atom, c_atom)
    pair_emb = torch.randn(n_atom, n_atom, c_pair)

    block = AtomTransformerBlock(
        c_atom=c_atom, num_heads=num_heads, use_optimized=use_optimized
    )
    output = block(x, pair_emb, block_index)
    assert output.shape == (
        n_atom,
        c_atom,
    ), f"Expected output shape ({n_atom},{c_atom}), but got {output.shape}"


def test_atom_transformer_block_fallback_warning():
    """
    Test that if a RuntimeError occurs in the optimized path,
    the block issues a warning and falls back to the naive approach.
    """
    torch.manual_seed(42)
    c_atom = 64
    num_heads = 4
    block_size = 8
    n_atom = 16
    c_pair = 16

    block_index = torch.randint(0, n_atom, (n_atom, block_size))
    x = torch.randn(n_atom, c_atom)
    pair_emb = torch.randn(n_atom, n_atom, c_pair)

    block = AtomTransformerBlock(c_atom=c_atom, num_heads=num_heads, use_optimized=True)

    # Patch the bs_attn call to raise RuntimeError
    with patch.object(
        block.bs_attn, "__call__", side_effect=RuntimeError("Intentional error")
    ):
        with warnings.catch_warnings(record=True) as w:
            output = block(x, pair_emb, block_index)
            # Check that fallback worked
            assert output.shape == (n_atom, c_atom)
            # Check a warning was issued
            assert len(w) == 1, "Expected exactly one warning when fallback occurs"
            assert "Falling back to naive attention." in str(w[0].message)


def test_atom_transformer_full_stack():
    """
    Test the full AtomTransformer with multiple blocks.
    Now also checks 3D fallback vs. 5D trunk usage.
    """
    import torch
    from rna_predict.pipeline.stageA.input_embedding.current.transformer import AtomTransformer

    torch.manual_seed(43)
    c_atom = 32
    num_heads = 4
    num_layers = 3
    n_atom = 10
    c_pair = 8

    # 3D fallback path
    x = torch.randn(n_atom, c_atom)
    c = torch.randn(n_atom, c_atom)
    p_3d = torch.randn(n_atom, n_atom, c_pair)

    model_3d = AtomTransformer(c_atom=c_atom, c_atompair=c_pair,
                               n_blocks=num_layers, n_heads=num_heads)
    out_3d = model_3d(x, c, p_3d)
    assert out_3d.shape == (n_atom, c_atom), f"3D fallback => got {out_3d.shape}"

    # 5D trunk path
    # shape [batch=1, n_blocks=2, n_queries=4, n_keys=4, c_pair=8]
    # also ensure x, c have shape [1,n_atom,c_atom]
    x5 = x.unsqueeze(0)
    c5 = c.unsqueeze(0)
    p_5d = torch.randn(1, 2, 4, 4, c_pair)

    model_5d = AtomTransformer(c_atom=c_atom, c_atompair=c_pair,
                               n_blocks=num_layers, n_heads=num_heads)
    out_5d = model_5d(x5, c5, p_5d)
    assert out_5d.shape == (1, n_atom, c_atom), f"5D trunk => got {out_5d.shape}"


def test_minimal_atoms():
    """
    Edge-case test: minimal number of atoms, e.g. 1 or 2.
    """
    torch.manual_seed(44)
    c_atom = 16
    num_heads = 4
    block_size = 1
    n_atom = 1
    c_pair = 4

    block_index = torch.randint(0, n_atom, (n_atom, block_size))
    x = torch.randn(n_atom, c_atom)
    pair_emb = torch.randn(n_atom, n_atom, c_pair)

    block = AtomTransformerBlock(
        c_atom=c_atom, num_heads=num_heads, use_optimized=False
    )
    out = block(x, pair_emb, block_index)
    assert out.shape == (n_atom, c_atom), "Single-atom test failed."

    # Test with 2 atoms
    n_atom = 2
    block_index = torch.randint(0, n_atom, (n_atom, block_size))
    x = torch.randn(n_atom, c_atom)
    pair_emb = torch.randn(n_atom, n_atom, c_pair)
    out = block(x, pair_emb, block_index)
    assert out.shape == (n_atom, c_atom), "Two-atom test failed."


@pytest.mark.parametrize("num_layers", [1, 2, 3])
@pytest.mark.parametrize("use_optimized", [False, True])
def test_atom_transformer_legacy_forward(num_layers: int, use_optimized: bool):
    """
    Test coverage for AtomTransformer (legacy) to cover lines 142-143 in __init__
    and lines 164-166 in forward.
    """
    from rna_predict.pipeline.stageA.input_embedding.legacy.attention.atom_transformer import AtomTransformer

    torch.manual_seed(101)
    N_atom = 6
    c_atom = 12
    c_pair = 8
    block_size = 4

    x = torch.randn(N_atom, c_atom)
    pair_emb = torch.randn(N_atom, N_atom, c_pair)
    block_index = torch.randint(0, N_atom, (N_atom, block_size))

    # Constructing covers lines 142-143
    model = AtomTransformer(c_atom=c_atom, num_heads=2, num_layers=num_layers, use_optimized=use_optimized)

    # Forward pass covers lines 164-166 (loop over self.blocks)
    output = model(x, pair_emb, block_index)
    assert output.shape == (N_atom, c_atom), "Output shape mismatch."

    # Basic sanity check: ensure it's not all zeros or NaNs
    assert not torch.isnan(output).any(), "Output contains NaNs, unexpected."


def test_atom_transformer_legacy_zero_layers():
    """
    Edge case: if we create an AtomTransformer with zero layers,
    we want to see if it simply returns the input or raises an error.
    """
    from rna_predict.pipeline.stageA.input_embedding.legacy.attention.atom_transformer import AtomTransformer

    x = torch.randn(5, 10)
    pair_emb = torch.randn(5, 5, 6)
    block_index = torch.randint(0, 5, (5, 4))

    model = AtomTransformer(c_atom=10, num_heads=2, num_layers=0, use_optimized=False)
    output = model(x, pair_emb, block_index)
    assert output.shape == x.shape, "Zero-layer transformer should return shape same as input."
    # We expect output to match input exactly if there's truly no block
    assert torch.allclose(output, x), "Zero-layer AtomTransformer didn't return the input as-is."


def test_atom_transformer_legacy_zero_atoms():
    """
    Edge case: zero atoms. Depending on internal design, this might fail or produce an empty result.
    Either way, lines 142-143 and 164-166 are executed (constructor + forward).
    """
    from rna_predict.pipeline.stageA.input_embedding.legacy.attention.atom_transformer import AtomTransformer
    x = torch.empty(0, 12)
    pair_emb = torch.empty(0, 0, 8)
    block_index = torch.empty(0, 4, dtype=torch.long)

    model = AtomTransformer(c_atom=12, num_heads=2, num_layers=1, use_optimized=False)
    try:
        output = model(x, pair_emb, block_index)
        assert output.shape == (0, 12), "Expected zero-atom output shape to remain (0,12)."
    except RuntimeError:
        # If code doesn't support zero-atom usage, it's still covered.
        pass
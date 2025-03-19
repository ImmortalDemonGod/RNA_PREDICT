import warnings
from unittest.mock import patch

import pytest
import torch

from rna_predict.models.attention.atom_transformer import (
    AtomTransformer,
    AtomTransformerBlock,
)


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
    """
    torch.manual_seed(43)
    c_atom = 32
    num_heads = 4
    num_layers = 3
    block_size = 4
    n_atom = 10
    c_pair = 8

    block_index = torch.randint(0, n_atom, (n_atom, block_size))
    x = torch.randn(n_atom, c_atom)
    pair_emb = torch.randn(n_atom, n_atom, c_pair)

    # Try naive stack
    naive_transformer = AtomTransformer(
        c_atom=c_atom, num_heads=num_heads, num_layers=num_layers, use_optimized=False
    )
    naive_out = naive_transformer(x, pair_emb, block_index)
    assert naive_out.shape == (
        n_atom,
        c_atom,
    ), f"Expected shape ({n_atom},{c_atom}); got {naive_out.shape}"

    # Try optimized stack (not forcing any error)
    opt_transformer = AtomTransformer(
        c_atom=c_atom, num_heads=num_heads, num_layers=num_layers, use_optimized=True
    )
    opt_out = opt_transformer(x, pair_emb, block_index)
    assert opt_out.shape == (n_atom, c_atom)


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

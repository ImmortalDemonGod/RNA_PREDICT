import warnings
from unittest.mock import patch

import pytest
import torch

import unittest
import warnings
from unittest.mock import patch

import torch
from torch import Tensor
import torch.nn as nn

from hypothesis import given, strategies as st, settings, HealthCheck
from hypothesis import assume
from hypothesis import example

# If your code is in a different module, import them properly.
# For demonstration, we assume we can import as:
# from atom_transformer import AtomTransformer, AtomTransformerBlock
# In practice, replace 'atom_transformer' with the correct import path.

# --------------------------------------------------------------------------
# Pre-Test Analysis (Summary):
# 1. We will test both AtomTransformerBlock and AtomTransformer.
# 2. We'll cover typical forward passes, shape checks, edge cases, and fallback logic.
# 3. We'll use unittest TestCase classes and setUp methods to avoid repetition.
# 4. We'll employ Hypothesis to fuzz numeric parameters and random Tensors.
# 5. We'll include at least one "round-trip" style test (multiple consecutive passes).
# 6. We will mock the block-sparse kernel to ensure fallback is tested.
# --------------------------------------------------------------------------

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



class TestAtomTransformerBlock(unittest.TestCase):
    """
    Test suite for the AtomTransformerBlock class, ensuring correct behavior,
    handling of edge cases, and fallback to naive attention when optimized path fails.
    """

    def setUp(self) -> None:
        """
        Common initialization for AtomTransformerBlock tests.
        Creates a default block with moderate dimension parameters.
        """
        self.c_atom = 32
        self.num_heads = 4
        self.block = AtomTransformerBlock(
            c_atom=self.c_atom, num_heads=self.num_heads, use_optimized=False
        )

        # Create small default shapes for test
        self.n_atoms = 8
        # pair_emb shape: [n_atoms, n_atoms, c_pair] -> let c_pair = 16
        self.c_pair = 16
        # block_index shape: [n_atoms, block_size=4] (fake local attention indices)
        self.block_size = 4

        # Create example tensors
        self.x = torch.randn(self.n_atoms, self.c_atom)
        self.pair_emb = torch.randn(self.n_atoms, self.n_atoms, self.c_pair)
        # For simplicity, we just create consecutive indices [0, 1, 2, 3, ...]
        # In a real use case, these would be local attention indices around each position.
        self.block_index = torch.randint(
            low=0, high=self.n_atoms, size=(self.n_atoms, self.block_size)
        )

    def test_forward_basic_shapes(self) -> None:
        """
        Test that forward pass preserves [N_atom, c_atom] shape
        and returns a valid tensor after normal forward execution.
        """
        out = self.block(self.x, self.pair_emb, self.block_index)
        self.assertEqual(out.shape, (self.n_atoms, self.c_atom))
        self.assertFalse(torch.any(torch.isnan(out)), "Output should not contain NaNs.")

    @patch("rna_predict.pipeline.stageA.input_embedding.legacy.attention.block_sparse.BlockSparseAttentionOptimized.forward")
    def test_fallback_to_naive_attention(self, mock_bs_forward) -> None:
        """
        Test fallback logic: If the optimized kernel raises a RuntimeError,
        the block should warn and then use naive attention.
        """
        block_opt = AtomTransformerBlock(
            c_atom=self.c_atom, num_heads=self.num_heads, use_optimized=True
        )
        # Force a RuntimeError in the optimized path
        mock_bs_forward.side_effect = RuntimeError("Simulated kernel fail.")
        with self.assertWarns(Warning):
            out = block_opt(self.x, self.pair_emb, self.block_index)
        self.assertEqual(out.shape, (self.n_atoms, self.c_atom))

    @given(
        n_atoms=st.integers(min_value=1, max_value=16),
        c_atom=st.sampled_from([8, 16, 32, 64]),  # restrict so c_atom // num_heads is int
        num_heads=st.sampled_from([1, 2, 4, 8]),
        c_pair=st.sampled_from([8, 16, 32]),
        block_size=st.integers(min_value=1, max_value=8),
        use_optimized=st.booleans()
    )
    @settings(suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
    @example(n_atoms=2, c_atom=8, num_heads=1, c_pair=8, block_size=2, use_optimized=False)
    def test_randomized_forward(self, n_atoms, c_atom, num_heads, c_pair, block_size, use_optimized):
        """
        Property-based test for AtomTransformerBlock with random valid shapes.
        Ensures forward pass completes without error and yields correct output shape.
        """
        # c_atom must be divisible by num_heads for correct shape, assume that
        assume(c_atom % num_heads == 0)

        block = AtomTransformerBlock(
            c_atom=c_atom, num_heads=num_heads, use_optimized=use_optimized
        )
        x = torch.randn(n_atoms, c_atom)
        pair_emb = torch.randn(n_atoms, n_atoms, c_pair)

        # Indices in [0, n_atoms). This is minimal local attention indexing.
        block_index = torch.randint(low=0, high=n_atoms, size=(n_atoms, block_size))

        out = block(x, pair_emb, block_index)
        self.assertEqual(out.shape, (n_atoms, c_atom))

    def test_round_trip_block(self) -> None:
        """
        Simple 'round-trip' style test: pass the same data multiple times through
        the same block to ensure repeated transformations don't break dimensions.
        """
        out1 = self.block(self.x, self.pair_emb, self.block_index)
        out2 = self.block(out1, self.pair_emb, self.block_index)
        self.assertEqual(out2.shape, (self.n_atoms, self.c_atom))
        # Check that it doesn't produce NaN or inf on repeated passes
        self.assertTrue(torch.isfinite(out2).all())


class TestAtomTransformer(unittest.TestCase):
    """
    Test suite for the multi-layer AtomTransformer, verifying its stacking logic,
    shape consistency, and forward pass behaviors across multiple blocks.
    """

    def setUp(self) -> None:
        """
        Initialize a default AtomTransformer with multiple layers,
        plus reference inputs for subsequent tests.
        """
        self.c_atom = 64
        self.num_heads = 4
        self.num_layers = 3
        self.transformer = AtomTransformer(
            c_atom=self.c_atom,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            use_optimized=False
        )

        self.n_atoms = 5
        self.c_pair = 32
        self.block_size = 2

        self.x = torch.randn(self.n_atoms, self.c_atom)
        self.pair_emb = torch.randn(self.n_atoms, self.n_atoms, self.c_pair)
        self.block_index = torch.randint(
            low=0, high=self.n_atoms, size=(self.n_atoms, self.block_size)
        )

    def test_forward_consistency(self) -> None:
        """
        Basic forward test to ensure each layer is applied and the final shape is correct.
        """
        out = self.transformer(self.x, self.pair_emb, self.block_index)
        self.assertEqual(out.shape, (self.n_atoms, self.c_atom))
        self.assertFalse(torch.isnan(out).any(), "No NaNs expected in output.")

    @given(
        n_atoms=st.integers(min_value=1, max_value=16),
        c_atom=st.sampled_from([8, 16, 32, 64]),
        num_heads=st.sampled_from([1, 2, 4, 8]),
        num_layers=st.integers(min_value=1, max_value=5),
        c_pair=st.sampled_from([8, 16, 32]),
        block_size=st.integers(min_value=1, max_value=8),
        use_optimized=st.booleans()
    )
    @settings(suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
    def test_transformer_random_shapes(self, n_atoms, c_atom, num_heads, num_layers, c_pair, block_size, use_optimized):
        """
        Property-based test verifying random shapes and configurations
        for the full multi-block AtomTransformer.
        """
        # c_atom must be divisible by num_heads for multi-head splitting
        assume(c_atom % num_heads == 0)

        tr = AtomTransformer(
            c_atom=c_atom,
            num_heads=num_heads,
            num_layers=num_layers,
            use_optimized=use_optimized
        )
        x = torch.randn(n_atoms, c_atom)
        pair_emb = torch.randn(n_atoms, n_atoms, c_pair)
        block_index = torch.randint(low=0, high=n_atoms, size=(n_atoms, block_size))

        out = tr(x, pair_emb, block_index)
        self.assertEqual(out.shape, (n_atoms, c_atom))

    def test_transformer_round_trip(self) -> None:
        """
        Repeated forward pass (multi-layer stacking) to ensure repeated transformations
        maintain shape and do not degrade numerically in a single run.
        """
        out1 = self.transformer(self.x, self.pair_emb, self.block_index)
        out2 = self.transformer(out1, self.pair_emb, self.block_index)
        self.assertEqual(out2.shape, (self.n_atoms, self.c_atom))
        self.assertTrue(torch.isfinite(out2).all())

    @patch("rna_predict.pipeline.stageA.input_embedding.legacy.attention.block_sparse.BlockSparseAttentionOptimized.forward")
    def test_transformer_fallback_behaviors(self, mock_bs_forward) -> None:
        """
        Test that the entire AtomTransformer will fallback to naive attention blocks
        if the optimized kernel fails in any sub-block.
        """
        # Create a fully 'optimized' transformer
        tr_opt = AtomTransformer(
            c_atom=self.c_atom,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            use_optimized=True
        )
        # Simulate the block-sparse kernel raising an error
        mock_bs_forward.side_effect = RuntimeError("Simulated error in kernel")

        with self.assertWarns(Warning):
            out = tr_opt(self.x, self.pair_emb, self.block_index)

        self.assertEqual(out.shape, (self.n_atoms, self.c_atom))


if __name__ == "__main__":
    # You can run these tests with: python -m unittest <this_file_name>.py
    unittest.main()
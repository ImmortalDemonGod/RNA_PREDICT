import math
import unittest
import pytest
import torch
import torch.nn.functional as F
import sys
from typing import Tuple

from rna_predict.models.attention.block_sparse import (
    _HAS_BSA,
    BlockSparseAttentionOptimized,
    LocalBlockMaskConfig,
    LocalSparseInput,
    LocalBlockSparseAttentionNaive,
    build_local_blockmask,
)


###############################################################################
# Existing unittest-based tests
###############################################################################
class TestBlockMask(unittest.TestCase):
    def test_build_local_blockmask_basic(self):
        """
        Tests basic creation of a local blockmask with default config.
        Verifies shape of the mask for a known N_atom.
        """
        N_atom = 16
        config = LocalBlockMaskConfig()  # defaults
        mask = build_local_blockmask(N_atom, config)
        # If _HAS_BSA is False, mask might be None
        if mask is None:
            self.assertFalse(
                _HAS_BSA, "Mask is None but library claims to be available."
            )
        else:
            # shape should be [1, config.nheads, nrow, ncol]
            nrow = math.ceil(N_atom / config.block_size)
            ncol = math.ceil(N_atom / config.block_size)
            expected_shape = (1, config.nheads, nrow, ncol)
            self.assertEqual(mask.shape, expected_shape, "Blockmask shape mismatch.")

    def test_build_local_blockmask_causal(self):
        """
        Tests the causal option in the blockmask config.
        """
        N_atom = 32
        cfg = LocalBlockMaskConfig(block_size=8, local_window=8, nheads=2, causal=True)
        mask = build_local_blockmask(N_atom, cfg)
        if mask is not None:
            # Check that any row i does not allow columns > i if config.causal
            # in block terms, this is more coarse than a direct row/col check
            pass


class TestBlockSparseOptimized(unittest.TestCase):
    def test_optimized_forward(self):
        """
        If _HAS_BSA is True, we test forward pass with BlockSparseAttentionOptimized.
        Otherwise, we skip.
        """
        if not _HAS_BSA:
            self.skipTest(
                "block_sparse_attn not installed, skipping optimized attention test."
            )

        N_atom, n_heads, c_per_head = 8, 2, 8
        q = torch.randn(N_atom, n_heads, c_per_head)
        k = torch.randn(N_atom, n_heads, c_per_head)
        v = torch.randn(N_atom, n_heads, c_per_head)
        pair_bias = torch.zeros(N_atom, N_atom, n_heads)  # for simplicity

        attn = BlockSparseAttentionOptimized(
            nheads=n_heads, block_size=4, local_window=4, causal=False
        )
        out = attn(q, k, v, pair_bias)
        self.assertEqual(
            out.shape,
            (N_atom, n_heads, c_per_head),
            "Output shape from optimized local attention is incorrect.",
        )


###############################################################################
# New pytest-based tests
###############################################################################
@pytest.fixture
def random_tensors() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fixture that generates random q, k, v, and pair_bias tensors for testing
    local block-sparse attention.
    """
    N_atom, n_heads, c_per_head = 6, 2, 4
    q = torch.randn(N_atom, n_heads, c_per_head, requires_grad=True)
    k = torch.randn(N_atom, n_heads, c_per_head, requires_grad=True)
    v = torch.randn(N_atom, n_heads, c_per_head, requires_grad=True)
    pair_bias = torch.randn(N_atom, N_atom, n_heads, requires_grad=True)
    return q, k, v, pair_bias


@pytest.fixture
def block_index() -> torch.Tensor:
    """
    Fixture that generates a block_index for local attention.
    Each row in block_index is a set of neighbor indices for the corresponding atom.
    """
    idx = torch.tensor([
        [0, 1, 2],
        [1, 2, 0],
        [2, 3, 4],
        [3, 4, 5],
        [4, 3, 2],
        [5, 4, 1],
    ], dtype=torch.long)
    return idx


def test_local_block_sparse_attention_naive_forward_backward(
    random_tensors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    block_index: torch.Tensor
) -> None:
    """
    Test the forward/backward pass of LocalBlockSparseAttentionNaive.
    """
    q, k, v, pair_bias = random_tensors
    inputs = LocalSparseInput(q=q, k=k, v=v, pair_bias=pair_bias, block_index=block_index)

    out = LocalBlockSparseAttentionNaive.apply(inputs)
    assert out.shape == q.shape, "Output shape must match [N_atom, n_heads, c_per_head]."

    grad = torch.ones_like(out)
    out.backward(grad, retain_graph=True)

    assert q.grad is not None and torch.any(q.grad != 0)
    assert k.grad is not None and torch.any(k.grad != 0)
    assert v.grad is not None and torch.any(v.grad != 0)
    assert pair_bias.grad is not None and torch.any(pair_bias.grad != 0)


def test_local_block_sparse_attention_naive_zero_neighbors() -> None:
    """
    Test naive local block-sparse attention in an edge case where block_size=0
    or block_index is empty for each atom. Ensures it produces all-zero output.
    """
    N_atom, n_heads, c_per_head = 3, 2, 4
    q = torch.randn(N_atom, n_heads, c_per_head, requires_grad=True)
    k = torch.randn(N_atom, n_heads, c_per_head, requires_grad=True)
    v = torch.randn(N_atom, n_heads, c_per_head, requires_grad=True)
    pair_bias = torch.randn(N_atom, N_atom, n_heads, requires_grad=True)

    zero_index = torch.empty((N_atom, 0), dtype=torch.long)
    inputs = LocalSparseInput(q=q, k=k, v=v, pair_bias=pair_bias, block_index=zero_index)

    out = LocalBlockSparseAttentionNaive.apply(inputs)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-7)

    grad = torch.ones_like(out)
    out.backward(grad)

    for tensor in (q, k, v, pair_bias):
        assert torch.allclose(tensor.grad, torch.zeros_like(tensor.grad), atol=1e-7)


@pytest.mark.parametrize("causal_setting", [False, True])
def test_build_local_blockmask(causal_setting: bool) -> None:
    """
    Test build_local_blockmask for both causal and non-causal settings.
    """
    N_atom = 32
    config = LocalBlockMaskConfig(
        block_size=8,
        local_window=8,
        nheads=2,
        causal=causal_setting
    )

    mask = build_local_blockmask(N_atom, config)
    if not _HAS_BSA:
        assert mask is None, "Mask should be None when _HAS_BSA=False."
        pytest.skip("Skipping extended checks because _HAS_BSA=False.")
    else:
        assert mask is not None, "Mask should not be None with _HAS_BSA=True."
        nrow = math.ceil(N_atom / config.block_size)
        ncol = math.ceil(N_atom / config.block_size)
        assert mask.shape == (1, config.nheads, nrow, ncol)

        if causal_setting:
            # Ensure columns > row_idx are false
            for row_idx in range(nrow):
                if row_idx + 1 < ncol:
                    region = mask[0, :, row_idx, row_idx + 1:]
                    assert not torch.any(region)
        else:
            assert torch.any(mask), "Non-causal mask must have True values."


def test_block_sparse_attention_optimized_no_bsa(
    random_tensors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
) -> None:
    """
    If _HAS_BSA is False, verify that we raise a RuntimeError.
    """
    if _HAS_BSA:
        pytest.skip("Skipping because block_sparse_attn is installed. This test is for fallback.")

    q, k, v, pair_bias = random_tensors
    attn_module = BlockSparseAttentionOptimized(
        nheads=2,
        block_size=4,
        local_window=4,
        causal=False
    )

    with pytest.raises(RuntimeError, match="block_sparse_attn not installed"):
        _ = attn_module(q, k, v, pair_bias)


@pytest.mark.skipif(not _HAS_BSA, reason="Requires block_sparse_attn to be installed.")
def test_block_sparse_attention_optimized_yes_bsa(
    random_tensors: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
) -> None:
    """
    If _HAS_BSA is True, ensure the optimized path runs without error.
    """
    q, k, v, pair_bias = random_tensors

    attn_module = BlockSparseAttentionOptimized(
        nheads=2,
        block_size=4,
        local_window=4,
        causal=False
    )
    out = attn_module(q, k, v, pair_bias)
    assert out.shape == q.shape

    loss = out.sum()
    loss.backward()
    assert q.grad is not None and torch.any(q.grad != 0)


###############################################################################
# Unified main block to run both unittest and pytest
###############################################################################
if __name__ == "__main__":
    # First, run unittest tests
    unittest.main(exit=False)

    # Then run pytest tests
    sys.exit(pytest.main([__file__]))
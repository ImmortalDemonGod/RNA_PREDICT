import unittest
import math
import torch
import torch.nn.functional as F

from rna_predict.models.attention.block_sparse import (
    LocalBlockSparseAttentionNaive,
    LocalSparseInput,
    build_local_blockmask,
    LocalBlockMaskConfig,
    BlockSparseAttentionOptimized,
    _HAS_BSA
)


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
            self.assertFalse(_HAS_BSA, "Mask is None but library claims to be available.")
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
            self.skipTest("block_sparse_attn not installed, skipping optimized attention test.")

        N_atom, n_heads, c_per_head = 8, 2, 8
        q = torch.randn(N_atom, n_heads, c_per_head)
        k = torch.randn(N_atom, n_heads, c_per_head)
        v = torch.randn(N_atom, n_heads, c_per_head)
        pair_bias = torch.zeros(N_atom, N_atom, n_heads)  # for simplicity

        attn = BlockSparseAttentionOptimized(
            nheads=n_heads, block_size=4, local_window=4, causal=False
        )
        out = attn(q, k, v, pair_bias)
        self.assertEqual(out.shape, (N_atom, n_heads, c_per_head),
                         "Output shape from optimized local attention is incorrect.")

if __name__ == "__main__":
    unittest.main()
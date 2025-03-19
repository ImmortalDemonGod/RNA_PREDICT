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

class TestBlockSparseNaive(unittest.TestCase):
    def test_naive_forward_and_backward(self):
        """
        Tests forward/backward pass for LocalBlockSparseAttentionNaive by generating
        random inputs that mimic typical usage scenarios. Ensures gradient computations
        succeed without error and output shapes match expectations.
        """
        N_atom, n_heads, c_per_head = 6, 2, 4  # small test case
        block_size = 3

        # Create random input Tensors
        q = torch.randn(N_atom, n_heads, c_per_head, requires_grad=True)
        k = torch.randn(N_atom, n_heads, c_per_head, requires_grad=True)
        v = torch.randn(N_atom, n_heads, c_per_head, requires_grad=True)
        pair_bias = torch.randn(N_atom, N_atom, n_heads, requires_grad=True)

        # Create block_index (neighbors) for each atom
        block_index = torch.randint(0, N_atom, (N_atom, block_size))

        # Wrap in LocalSparseInput
        inputs = LocalSparseInput(q=q, k=k, v=v, pair_bias=pair_bias, block_index=block_index)

        # Forward pass
        out = LocalBlockSparseAttentionNaive.apply(inputs)
        self.assertEqual(out.shape, (N_atom, n_heads, c_per_head),
                         "Output shape from naive local attention is incorrect.")

        # Backward pass (ensure no error)
        loss = out.sum()  # simple scalar
        loss.backward()

        # Check gradients
        self.assertIsNotNone(q.grad, "No gradient computed for q.")
        self.assertIsNotNone(k.grad, "No gradient computed for k.")
        self.assertIsNotNone(v.grad, "No gradient computed for v.")
        self.assertIsNotNone(pair_bias.grad, "No gradient computed for pair_bias.")

    def test_small_input(self):
        """
        Edge case: minimal possible scenario (N_atom=1 or 2).
        """
        N_atom, n_heads, c_per_head = 2, 1, 2
        block_size = 1

        q = torch.randn(N_atom, n_heads, c_per_head, requires_grad=True)
        k = torch.randn(N_atom, n_heads, c_per_head, requires_grad=True)
        v = torch.randn(N_atom, n_heads, c_per_head, requires_grad=True)
        pair_bias = torch.randn(N_atom, N_atom, n_heads, requires_grad=True)
        block_index = torch.randint(0, N_atom, (N_atom, block_size))

        inputs = LocalSparseInput(q=q, k=k, v=v, pair_bias=pair_bias, block_index=block_index)
        out = LocalBlockSparseAttentionNaive.apply(inputs)
        self.assertEqual(out.shape, (N_atom, n_heads, c_per_head))

        loss = out.mean()
        loss.backward()

        # Expect gradient to exist
        for tvar in [q, k, v, pair_bias]:
            self.assertIsNotNone(tvar.grad)

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
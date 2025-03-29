"""
Comprehensive, Single-File Test Suite for Pairformer-Related Modules
===================================================================

This file unifies and expands upon the strengths of multiple earlier test suites
(V1 through V5) to achieve robust, high-coverage testing of the following classes:

  • PairformerBlock
  • PairformerStack
  • MSAPairWeightedAveraging
  • MSAStack
  • MSABlock
  • MSAModule
  • TemplateEmbedder

It uses Python’s built-in unittest framework for organization and provides:
  1. Thorough docstrings for each test class and test method
  2. setUp fixtures where beneficial
  3. Extensive shape and flags testing
  4. Effective Hypothesis usage, leveraging composite strategies
  5. Mocking external calls when necessary
  6. Checks for edge cases and normal paths

How to Run
----------
Save this file (e.g. as test_pairformer.py) and run:

  python -m unittest test_pairformer.py

Or integrate it into your continuous integration / test suite as needed.

Dependencies
------------
  • Python 3.7+ recommended
  • PyTorch >= 1.7 (for tensor creation)
  • Hypothesis >= 6.0 (for property-based testing)
  • (Optional) CUDA environment if testing GPU caching code paths

Coverage Goals
--------------
We aim to cover:
  • Constructor initialization for each class with valid parameter ranges
  • Forward passes with random shapes under normal conditions
  • Edge behaviors, like c_s=0, n_blocks=0, or large z dimension triggering cache calls
  • Key optional flags: inplace_safe, use_memory_efficient_kernel, etc.
  • MSA sampling behavior if “msa” feature dict is present vs absent
  • TemplateEmbedder returning 0 if n_blocks=0 or if template keys are missing

Note on Mocking
---------------
We mock calls to torch.cuda.empty_cache() in certain tests to confirm the code path
is triggered for large shapes. We also optionally mock the MSA sampling logic if
you want to isolate external dependencies. Adjust these mocks as suits your environment.

Enjoy robust coverage with minimal duplication!
"""

import unittest
from unittest.mock import patch

import numpy as np
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as np_strategies

# If your code is located elsewhere, adjust the imports accordingly.
# Example path-based import for pairformer classes:
# from rna_predict.pipeline.stageB.pairwise.pairformer import (
#     PairformerBlock, PairformerStack, MSAPairWeightedAveraging,
#     MSAStack, MSABlock, MSAModule, TemplateEmbedder
# )
from rna_predict.pipeline.stageB.pairwise.pairformer import (
    MSABlock,
    MSAModule,
    MSAPairWeightedAveraging,
    MSAStack,
    PairformerBlock,
    PairformerStack,
    TemplateEmbedder,
)

# ------------------------------------------------------------------------
# HELPER STRATEGIES & FUNCTIONS
# ------------------------------------------------------------------------

# To avoid repeated warnings about large or slow generation:
settings.register_profile(
    "extended",
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
    deadline=None,
)
settings.load_profile("extended")


def float_arrays(shape, min_value=-1e3, max_value=1e3):
    """
    A Hypothesis strategy for creating float32 NumPy arrays of a given shape
    within [min_value, max_value].
    """
    return np_strategies.arrays(
        dtype=np.float32,
        shape=shape,
        elements=st.floats(
            min_value=min_value,
            max_value=max_value,
            allow_nan=False,
            allow_infinity=False,
        ),
    )


def bool_arrays(shape):
    """A Hypothesis strategy for boolean arrays of a given shape."""
    return np_strategies.arrays(dtype=np.bool_, shape=shape)


@st.composite
def s_z_mask_draw(
    draw, c_s_range=(0, 64), c_z_range=(4, 64), n_token_range=(1, 8), batch_range=(1, 2)
):
    """
    Composite strategy to produce random s, z, pair_mask suitable for
    testing PairformerBlock or PairformerStack. We allow c_s=0 or >0.
    """
    batch = draw(st.integers(*batch_range))
    n_token = draw(st.integers(*n_token_range))
    c_s = draw(st.integers(*c_s_range))
    c_z = draw(st.integers(*c_z_range))

    # s shape: (batch, n_token, c_s) if c_s > 0 else None
    s_array = None
    if c_s > 0:
        s_array = draw(float_arrays((batch, n_token, c_s)))

    # z shape: (batch, n_token, n_token, c_z)
    z_array = draw(float_arrays((batch, n_token, n_token, c_z)))

    # mask shape: (batch, n_token, n_token)
    mask_array = draw(bool_arrays((batch, n_token, n_token)))

    s_tensor = torch.from_numpy(s_array) if s_array is not None else None
    z_tensor = torch.from_numpy(z_array)
    mask_tensor = torch.from_numpy(mask_array)

    return s_tensor, z_tensor, mask_tensor, c_s, c_z


# ------------------------------------------------------------------------
# TEST CLASSES
# ------------------------------------------------------------------------


class TestPairformerBlock(unittest.TestCase):
    """
    Tests for the PairformerBlock class, covering:
      • constructor parameterization
      • forward pass shape consistency
      • handling of c_s=0 vs c_s>0
      • optional flags such as inplace_safe
    """

    def test_instantiate_basic(self):
        """
        Simple instantiation check with default parameters.
        """
        block = PairformerBlock()
        self.assertIsInstance(block, PairformerBlock)

    @given(
        n_heads=st.integers(min_value=1, max_value=8),
        c_z=st.integers(min_value=4, max_value=64),
        c_s=st.integers(min_value=0, max_value=64),
        c_hidden_mul=st.integers(min_value=4, max_value=64),
        c_hidden_pair_att=st.integers(min_value=4, max_value=64),
        no_heads_pair=st.integers(min_value=1, max_value=4),
        dropout=st.floats(min_value=0.0, max_value=0.5),
    )
    def test_init_random_params(
        self, n_heads, c_z, c_s, c_hidden_mul, c_hidden_pair_att, no_heads_pair, dropout
    ):
        """
        Hypothesis test verifying no errors are raised when instantiating
        a PairformerBlock with various numeric parameters.
        """
        block = PairformerBlock(
            n_heads=n_heads,
            c_z=c_z,
            c_s=c_s,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_pair=no_heads_pair,
            dropout=dropout,
        )
        self.assertIsInstance(block, PairformerBlock)

    @given(data=s_z_mask_draw())
    def test_forward_shapes(self, data):
        """
        For random s, z, pair_mask shapes and c_s=0 or >0, check that
        the block forward pass returns consistent shapes or None for s_out.
        """
        s_in, z_in, mask, c_s, c_z = data
        block = PairformerBlock(n_heads=2, c_z=c_z, c_s=c_s, dropout=0.1)
        s_out, z_out = block.forward(s_in, z_in, mask)

        # z_out should match z_in's shape
        self.assertEqual(z_out.shape, z_in.shape)
        if c_s > 0:
            self.assertIsNotNone(s_out)
            self.assertEqual(s_out.shape, s_in.shape)
        else:
            self.assertIsNone(s_out)

    @given(
        data=s_z_mask_draw(c_s_range=(4, 16), c_z_range=(4, 16), n_token_range=(2, 4)),
        inplace_flag=st.booleans(),
    )
    def test_forward_inplace_safe_toggle(self, data, inplace_flag):
        """
        Test toggling inplace_safe, ensuring the block forward pass
        does not break shapes or produce errors.
        """
        s_in, z_in, mask, c_s, c_z = data
        block = PairformerBlock(n_heads=2, c_z=c_z, c_s=c_s, dropout=0.1)
        s_out, z_out = block.forward(s_in, z_in, mask, inplace_safe=inplace_flag)
        self.assertEqual(z_out.shape, z_in.shape)
        if c_s > 0:
            self.assertIsNotNone(s_out)
            self.assertEqual(s_out.shape, s_in.shape)
        else:
            self.assertIsNone(s_out)


class TestPairformerStack(unittest.TestCase):
    """
    Tests for PairformerStack, verifying:
      • constructor with multiple blocks
      • forward pass across n_blocks
      • large z dimension triggers torch.cuda.empty_cache in eval mode
    """

    def test_instantiate_basic(self):
        """Check a simple instantiation with default params."""
        stack = PairformerStack()
        self.assertIsInstance(stack, PairformerStack)

    @given(
        n_blocks=st.integers(min_value=1, max_value=3),
        n_heads=st.integers(min_value=1, max_value=4),
        c_z=st.integers(min_value=4, max_value=32),
        c_s=st.integers(min_value=0, max_value=32),
        dropout=st.floats(min_value=0.0, max_value=0.5),
        blocks_per_ckpt=st.one_of(st.none(), st.integers(min_value=1, max_value=2)),
    )
    def test_init_random(self, n_blocks, n_heads, c_z, c_s, dropout, blocks_per_ckpt):
        """
        Hypothesis-based constructor test for PairformerStack.
        """
        stk = PairformerStack(
            n_blocks=n_blocks,
            n_heads=n_heads,
            c_z=c_z,
            c_s=c_s,
            dropout=dropout,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        self.assertIsInstance(stk, PairformerStack)
        self.assertEqual(len(stk.blocks), n_blocks)

    @given(data=s_z_mask_draw(n_token_range=(2, 4), batch_range=(1, 1)))
    def test_forward_normal(self, data):
        """
        Normal forward pass coverage for moderate shapes.
        """
        s_in, z_in, mask, c_s, c_z = data
        stack = PairformerStack(n_blocks=2, n_heads=2, c_z=c_z, c_s=c_s, dropout=0.1)
        s_out, z_out = stack.forward(s_in, z_in, mask)
        self.assertEqual(z_out.shape, z_in.shape)
        if c_s > 0 and s_in is not None:
            self.assertEqual(s_out.shape, s_in.shape)
        else:
            self.assertIsNone(s_out)

    def test_forward_large_z_eval_triggers_cache(self):
        """
        If z.shape[-2] > 2000 and not training, we expect a call to
        torch.cuda.empty_cache() inside forward. We'll patch it to confirm.
        """
        stack = PairformerStack(n_blocks=1, c_z=8, c_s=8)
        stack.eval()  # training=False
        s_in = torch.zeros((1, 2100, 8), dtype=torch.float32)
        z_in = torch.zeros((1, 2100, 2100, 8), dtype=torch.float32)
        mask = torch.ones((1, 2100, 2100), dtype=torch.float32)

        with patch("torch.cuda.empty_cache") as mock_cache:
            s_out, z_out = stack.forward(s_in, z_in, mask)
            self.assertTrue(mock_cache.called)
            self.assertEqual(z_out.shape, z_in.shape)
            self.assertEqual(s_out.shape, s_in.shape)


class TestMSAPairWeightedAveraging(unittest.TestCase):
    """
    Tests for MSAPairWeightedAveraging:
      • constructor parameter checks
      • forward pass shape validation
    """

    def test_instantiate_basic(self):
        mwa = MSAPairWeightedAveraging()
        self.assertIsInstance(mwa, MSAPairWeightedAveraging)

    @given(
        c_m=st.integers(min_value=4, max_value=64),
        c=st.integers(min_value=4, max_value=32),
        c_z=st.integers(min_value=4, max_value=64),
        n_heads=st.integers(min_value=1, max_value=8),
    )
    def test_init_random(self, c_m, c, c_z, n_heads):
        mod = MSAPairWeightedAveraging(c_m=c_m, c=c, c_z=c_z, n_heads=n_heads)
        self.assertIsInstance(mod, MSAPairWeightedAveraging)

    @given(
        # m shape: [n_msa, n_token, c_m]
        # z shape: [n_token, n_token, c_z]
        n_msa=st.integers(1, 4),
        n_token=st.integers(2, 6),
        c_m=st.integers(4, 16),
        c_z=st.integers(4, 16),
    )
    def test_forward_shapes(self, n_msa, n_token, c_m, c_z):
        """
        Create random Tensors for m, z and pass to forward, verifying shape.
        """
        mod = MSAPairWeightedAveraging(c_m=c_m, c=8, c_z=c_z, n_heads=2)
        m = torch.randn((n_msa, n_token, c_m), dtype=torch.float32)
        z = torch.randn((n_token, n_token, c_z), dtype=torch.float32)
        out = mod.forward(m, z)
        self.assertEqual(out.shape, m.shape)


class TestMSAStack(unittest.TestCase):
    """
    Tests for MSAStack: verifying constructor & forward pass shape correctness.
    """

    def test_instantiate_basic(self):
        ms = MSAStack()
        self.assertIsInstance(ms, MSAStack)

    @given(
        n_msa=st.integers(1, 3),
        n_token=st.integers(2, 5),
        c_m=st.integers(4, 16),
        c_z=st.integers(4, 16),
    )
    def test_forward_shapes(self, n_msa, n_token, c_m, c_z):
        """
        MSAStack forward pass:
          m => [n_msa, n_token, c_m],
          z => [n_token, n_token, c_z].
        Output should match m shape.
        """
        stack = MSAStack(c_m=c_m, c=8, dropout=0.1)
        # Adjust c_z in the nested MSAPairWeightedAveraging if needed:
        stack.msa_pair_weighted_averaging.c_z = c_z
        stack.msa_pair_weighted_averaging.layernorm_z = torch.nn.LayerNorm(c_z)
        stack.msa_pair_weighted_averaging.linear_no_bias_z = torch.nn.Linear(
            c_z, stack.msa_pair_weighted_averaging.n_heads, bias=False
        )

        m = torch.randn((n_msa, n_token, c_m), dtype=torch.float32)
        z = torch.randn((n_token, n_token, c_z), dtype=torch.float32)

        out = stack.forward(m, z)
        self.assertEqual(out.shape, (n_msa, n_token, c_m))


class TestMSABlock(unittest.TestCase):
    """
    Tests for MSABlock which integrates OuterProductMean (m->z) and
    MSA/Pairs transformations, verifying:
      • last_block => returns (None, z)
      • otherwise => returns (m, z)
    """

    def test_instantiate_basic(self):
        mb = MSABlock()
        self.assertIsInstance(mb, MSABlock)

    @given(
        n_msa=st.integers(1, 3),
        n_token=st.integers(2, 4),
        c_m=st.integers(4, 16),
        c_z=st.integers(4, 16),
        last_block=st.booleans(),
    )
    def test_forward_behaviors(self, n_msa, n_token, c_m, c_z, last_block):
        block = MSABlock(c_m=c_m, c_z=c_z, c_hidden=8, is_last_block=last_block)
        m = torch.randn((n_msa, n_token, c_m), dtype=torch.float32)
        z = torch.randn((n_token, n_token, c_z), dtype=torch.float32)
        pair_mask = torch.ones((n_token, n_token), dtype=torch.bool)

        m_out, z_out = block.forward(m, z, pair_mask)
        self.assertEqual(z_out.shape, (n_token, n_token, c_z))
        if last_block:
            self.assertIsNone(m_out)
        else:
            self.assertIsNotNone(m_out)
            self.assertEqual(m_out.shape, (n_msa, n_token, c_m))


class TestMSAModule(unittest.TestCase):
    """
    Tests for MSAModule, including:
      • n_blocks=0 => returns z unchanged
      • missing 'msa' => returns z unchanged
      • presence of 'msa' => shape updated, or at least tested for coverage
    """

    def test_instantiate_basic(self):
        mm = MSAModule()
        self.assertIsInstance(mm, MSAModule)

    def test_forward_nblocks_zero(self):
        """If n_blocks=0, forward always returns the original z."""
        module = MSAModule(n_blocks=0, c_m=8, c_z=16, c_s_inputs=8)
        z_in = torch.randn((1, 3, 3, 16), dtype=torch.float32)
        s_inputs = torch.randn((1, 3, 8), dtype=torch.float32)
        mask = torch.ones((1, 3, 3), dtype=torch.bool)
        out_z = module.forward({"msa": torch.zeros((2, 3))}, z_in, s_inputs, mask)
        self.assertTrue(torch.equal(out_z, z_in))

    def test_forward_no_msa_key(self):
        """If no 'msa' in feature dict, returns z unchanged."""
        module = MSAModule(n_blocks=1, c_m=8, c_z=16, c_s_inputs=8)
        z_in = torch.randn((1, 3, 3, 16), dtype=torch.float32)
        s_inputs = torch.randn((1, 3, 8), dtype=torch.float32)
        mask = torch.ones((1, 3, 3), dtype=torch.bool)
        out_z = module.forward({}, z_in, s_inputs, mask)
        self.assertTrue(torch.equal(out_z, z_in))

    @patch(
        "rna_predict.pipeline.stageB.pairwise.pairformer.sample_msa_feature_dict_random_without_replacement"
    )
    def test_forward_with_msa(self, mock_sample):
        """If 'msa' key is present, we try sampling and proceed in blocks > 0."""
        # Mock the returned MSA feature dict
        mock_sample.return_value = {
            "msa": torch.randint(0, 32, (2, 5)),  # shape [N_msa, N_token]
            "has_deletion": torch.zeros((2, 5), dtype=torch.bool),
            "deletion_value": torch.zeros((2, 5), dtype=torch.float32),
        }
        module = MSAModule(
            n_blocks=1, c_m=8, c_z=16, c_s_inputs=8, msa_configs={"enable": True}
        )
        z_in = torch.randn((1, 5, 5, 16), dtype=torch.float32)
        s_inputs = torch.randn((1, 5, 8), dtype=torch.float32)
        mask = torch.ones((1, 5, 5), dtype=torch.bool)
        input_dict = {"msa": torch.zeros((3, 5), dtype=torch.int64)}

        out_z = module.forward(input_dict, z_in, s_inputs, mask)
        self.assertEqual(out_z.shape, z_in.shape)
        self.assertTrue(mock_sample.called)


class TestTemplateEmbedder(unittest.TestCase):
    """
    Tests for TemplateEmbedder, verifying:
      • n_blocks=0 => returns 0
      • missing 'template_restype' => returns 0
      • presence of 'template_restype' but code is not fully implemented => also returns 0
    """

    def test_instantiate_basic(self):
        te = TemplateEmbedder()
        self.assertIsInstance(te, TemplateEmbedder)

    def test_forward_no_template_restype(self):
        embedder = TemplateEmbedder(n_blocks=2, c=8, c_z=16)
        z_in = torch.randn((1, 4, 4, 16), dtype=torch.float32)
        pair_mask = torch.ones((1, 4, 4), dtype=torch.bool)
        out = embedder.forward({}, z_in, pair_mask=pair_mask)
        self.assertEqual(out, 0)

    def test_forward_nblocks_zero(self):
        embedder = TemplateEmbedder(n_blocks=0, c=8, c_z=16)
        z_in = torch.randn((1, 4, 4, 16), dtype=torch.float32)
        out = embedder.forward({"template_restype": torch.zeros((1, 4))}, z_in)
        self.assertEqual(out, 0)

    def test_forward_template_present(self):
        """
        Even if 'template_restype' is present, the current logic returns 0
        unless there’s a deeper implementation. Checking coverage only.
        """
        embedder = TemplateEmbedder(n_blocks=2, c=8, c_z=16)
        input_dict = {"template_restype": torch.zeros((1, 4))}
        z_in = torch.randn((1, 4, 4, 16), dtype=torch.float32)
        out = embedder.forward(input_dict, z_in)
        self.assertEqual(out, 0)


if __name__ == "__main__":
    unittest.main()

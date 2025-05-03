"""
Comprehensive Verification Protocol for Stage B Pairformer Module
=================================================================

This test file provides a thorough verification of the PairformerWrapper component,
which integrates Protenix's PairformerStack into the RNA prediction pipeline for
global pairwise encoding.

The verification protocol includes:
1. Instantiation verification with various parameters
2. Weight management validation
3. Functional testing with appropriate test tensors
4. Shape consistency checks
5. Gradient flow verification
6. Variable sequence length testing

Each test is documented with clear assertions and expected outcomes.
"""

import gc
import unittest

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from rna_predict.pipeline.stageB.pairwise.pairformer_wrapper import PairformerWrapper


class TestPairformerWrapperVerification(unittest.TestCase):
    """
    Comprehensive verification tests for the PairformerWrapper class.
    """

    def setUp(self):
        """
        Set up common test parameters and configurations.
        Using reduced dimensions to minimize memory usage while still testing functionality.
        """
        # Reduced model parameters for testing
        self.default_n_blocks = 2  # Reduced from 48
        self.default_c_z = 32  # Reduced from 128
        self.default_c_s = 64  # Reduced from 384
        self.default_use_checkpoint = True  # Enable checkpointing by default

        # Reduced test tensor dimensions
        self.batch_size = 1
        self.seq_length = 10  # Reduced from 20
        self.node_features = 64  # Reduced from 384
        self.edge_features = 32  # Reduced from 128

        # Create test tensors
        self.s = torch.randn(self.batch_size, self.seq_length, self.node_features)
        self.z = torch.randn(
            self.batch_size, self.seq_length, self.seq_length, self.edge_features
        )
        self.pair_mask = torch.ones(
            self.batch_size, self.seq_length, self.seq_length, dtype=torch.float32
        )

        # Initialize cache for model instances
        self._wrapper_cache = {}

    def tearDown(self):
        """
        Clean up after each test to free memory.
        """
        # Clear the model cache
        self._wrapper_cache.clear()

        # Clear any stored tensors
        self.s = None
        self.z = None
        self.pair_mask = None

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_instantiation_default_parameters(self):
        """
        Verify that PairformerWrapper can be instantiated with default parameters.
        """
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({
            "stageB_pairformer": {
                "n_blocks": self.default_n_blocks,
                "n_heads": 8,
                "c_z": self.default_c_z,
                "c_s": self.default_c_s,
                "dropout": 0.1,
                "use_checkpoint": self.default_use_checkpoint,
                "use_memory_efficient_kernel": False,
                "use_deepspeed_evo_attention": False,
                "use_lma": False,
                "inplace_safe": False,
                "chunk_size": None,
                "device": "cpu"
            }
        })
        required_keys = ["n_blocks", "n_heads", "c_z", "c_s", "dropout", "use_checkpoint", "use_memory_efficient_kernel", "use_deepspeed_evo_attention", "use_lma", "inplace_safe", "chunk_size", "device"]
        for k in required_keys:
            assert k in cfg["stageB_pairformer"], f"Config missing required key: {k}"
        wrapper = PairformerWrapper(cfg["stageB_pairformer"])
        self.assertEqual(wrapper.n_blocks, self.default_n_blocks)
        self.assertEqual(wrapper.c_z, self.default_c_z)
        self.assertEqual(wrapper.c_s, self.default_c_s)
        self.assertEqual(wrapper.dropout, 0.1)
        del wrapper
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @given(
        n_blocks=st.integers(min_value=1, max_value=4),  # Further reduced from 12
        c_z=st.sampled_from([16, 32]),
        c_s=st.sampled_from([16, 32]),
        n_heads=st.integers(min_value=2, max_value=4),
        dropout=st.floats(min_value=0.0, max_value=0.2),
        use_checkpoint=st.booleans(),
        use_memory_efficient=st.booleans(),
    )
    @settings(
        deadline=None,
        max_examples=3,
    )
    def test_instantiation_custom_parameters(self, n_blocks, c_z, c_s, n_heads, dropout, use_checkpoint, use_memory_efficient):
        """
        Property-based test: PairformerWrapper can be instantiated with various custom parameters.

        This test verifies that the wrapper correctly handles a wide range of configuration values
        and properly initializes the underlying PairformerStack with the adjusted parameters.

        Args:
            n_blocks: Number of transformer blocks
            c_z: Dimension of pair embeddings
            c_s: Dimension of single embeddings
            n_heads: Number of attention heads
            dropout: Dropout rate
            use_checkpoint: Whether to use checkpointing
            use_memory_efficient: Whether to use memory efficient attention
        """
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({
            "stageB_pairformer": {
                "n_blocks": n_blocks,
                "n_heads": n_heads,
                "c_z": c_z,
                "c_s": c_s,
                "dropout": dropout,
                "use_checkpoint": use_checkpoint,
                "use_memory_efficient_kernel": use_memory_efficient,
                "use_deepspeed_evo_attention": False,
                "use_lma": False,
                "inplace_safe": False,
                "chunk_size": None,
                "device": "cpu"
            }
        })
        required_keys = ["n_blocks", "n_heads", "c_z", "c_s", "dropout", "use_checkpoint", "use_memory_efficient_kernel", "use_deepspeed_evo_attention", "use_lma", "inplace_safe", "chunk_size", "device"]
        for k in required_keys:
            assert k in cfg["stageB_pairformer"], f"Config missing required key: {k}"
        wrapper = PairformerWrapper(cfg["stageB_pairformer"])
        self.assertEqual(wrapper.n_blocks, n_blocks)
        self.assertEqual(wrapper.c_z, c_z)
        self.assertEqual(wrapper.c_s, c_s)
        self.assertEqual(wrapper.dropout, dropout)
        del wrapper
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_parameter_count(self):
        """
        Verify that the parameter count matches the expected architecture size.
        Using reduced model size for testing.
        """
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({
            "stageB_pairformer": {
                "n_blocks": self.default_n_blocks,
                "n_heads": 8,
                "c_z": self.default_c_z,
                "c_s": self.default_c_s,
                "dropout": 0.1,
                "use_checkpoint": self.default_use_checkpoint,
                "use_memory_efficient_kernel": False,
                "use_deepspeed_evo_attention": False,
                "use_lma": False,
                "inplace_safe": False,
                "chunk_size": None,
                "device": "cpu"
            }
        })
        required_keys = ["n_blocks", "n_heads", "c_z", "c_s", "dropout", "use_checkpoint", "use_memory_efficient_kernel", "use_deepspeed_evo_attention", "use_lma", "inplace_safe", "chunk_size", "device"]
        for k in required_keys:
            assert k in cfg["stageB_pairformer"], f"Config missing required key: {k}"
        wrapper = PairformerWrapper(cfg["stageB_pairformer"])
        stack_param_count = sum(p.numel() for p in wrapper.stack.parameters())
        param_count = sum(p.numel() for p in wrapper.parameters())
        self.assertEqual(param_count, stack_param_count)
        del wrapper
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_forward_shape_consistency(self):
        """
        Verify that the forward pass returns tensors with the expected shapes.
        Using reduced tensor sizes for testing.
        """
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({
            "stageB_pairformer": {
                "n_blocks": self.default_n_blocks,
                "n_heads": 8,
                "c_z": self.default_c_z,
                "c_s": self.default_c_s,
                "dropout": 0.1,
                "use_checkpoint": self.default_use_checkpoint,
                "use_memory_efficient_kernel": False,
                "use_deepspeed_evo_attention": False,
                "use_lma": False,
                "inplace_safe": False,
                "chunk_size": None,
                "device": "cpu"
            }
        })
        required_keys = ["n_blocks", "n_heads", "c_z", "c_s", "dropout", "use_checkpoint", "use_memory_efficient_kernel", "use_deepspeed_evo_attention", "use_lma", "inplace_safe", "chunk_size", "device"]
        for k in required_keys:
            assert k in cfg["stageB_pairformer"], f"Config missing required key: {k}"
        wrapper = PairformerWrapper(cfg["stageB_pairformer"])
        for idx, block in enumerate(wrapper.stack.blocks):
            msg = f"Block {idx} c_z mismatch: got {getattr(block, 'c_z', None)}, expected {self.default_c_z}"
            self.assertEqual(getattr(block, 'c_z', None), self.default_c_z, msg)
            ln_shape = getattr(block.tri_mul_out.layer_norm_in, 'normalized_shape', None)
            if ln_shape is not None:
                self.assertTrue(
                    ln_shape == (self.default_c_z,) or ln_shape == [self.default_c_z],
                    f"Block {idx} tri_mul_out.layer_norm_in.normalized_shape={ln_shape}, expected ({self.default_c_z},)"
                )
        s_updated, z_updated = wrapper(self.s, self.z, self.pair_mask)
        self.assertEqual(s_updated.shape, self.s.shape)
        self.assertEqual(z_updated.shape, self.z.shape)
        del wrapper, s_updated, z_updated
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @settings(deadline=None, max_examples=5)
    @given(
        batch_size=st.integers(min_value=1, max_value=2),
        seq_length=st.integers(min_value=4, max_value=12),
        c_z=st.sampled_from([16, 32, 64]),
        c_s=st.sampled_from([16, 32, 64]),
    )
    def test_forward_shape_consistency_hypothesis(self, batch_size, seq_length, c_z, c_s):
        """
        Property-based test: forward pass shape consistency for random valid dimensions.
        """
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({
            "stageB_pairformer": {
                "n_blocks": 2,
                "n_heads": 4,
                "c_z": c_z,
                "c_s": c_s,
                "dropout": 0.1,
                "use_checkpoint": True,
                "use_memory_efficient_kernel": False,
                "use_deepspeed_evo_attention": False,
                "use_lma": False,
                "inplace_safe": False,
                "chunk_size": None,
                "device": "cpu"
            }
        })
        wrapper = PairformerWrapper(cfg["stageB_pairformer"])
        s = torch.randn(batch_size, seq_length, c_s)
        z = torch.randn(batch_size, seq_length, seq_length, c_z)
        pair_mask = torch.ones(batch_size, seq_length, seq_length, dtype=torch.float32)
        for idx, block in enumerate(wrapper.stack.blocks):
            msg = f"Block {idx} c_z mismatch: got {getattr(block, 'c_z', None)}, expected {c_z}"
            self.assertEqual(getattr(block, 'c_z', None), c_z, msg)
            ln_shape = getattr(block.tri_mul_out.layer_norm_in, 'normalized_shape', None)
            if ln_shape is not None:
                self.assertTrue(
                    ln_shape == (c_z,) or ln_shape == [c_z],
                    f"Block {idx} tri_mul_out.layer_norm_in.normalized_shape={ln_shape}, expected ({c_z},)"
                )
        s_updated, z_updated = wrapper(s, z, pair_mask)
        self.assertEqual(s_updated.shape, s.shape)
        self.assertEqual(z_updated.shape, z.shape)
        del wrapper, s, z, s_updated, z_updated
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_forward_no_nan_inf(self):
        """
        Verify that the forward pass does not produce NaN or Inf values.
        Using minimal model size for faster execution.
        """
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({
            "stageB_pairformer": {
                "n_blocks": 1,  # Minimum size
                "n_heads": 4,  # Minimum size
                "c_z": 16,  # Minimum size
                "c_s": 32,  # Minimum size
                "dropout": 0.1,
                "use_checkpoint": True,
                "use_memory_efficient_kernel": False,
                "use_deepspeed_evo_attention": False,
                "use_lma": False,
                "inplace_safe": False,
                "chunk_size": None,
                "device": "cpu"
            }
        })
        required_keys = ["n_blocks", "n_heads", "c_z", "c_s", "dropout", "use_checkpoint", "use_memory_efficient_kernel", "use_deepspeed_evo_attention", "use_lma", "inplace_safe", "chunk_size", "device"]
        for k in required_keys:
            assert k in cfg["stageB_pairformer"], f"Config missing required key: {k}"
        wrapper = PairformerWrapper(cfg["stageB_pairformer"])
        for idx, block in enumerate(wrapper.stack.blocks):
            msg = f"Block {idx} c_z mismatch: got {getattr(block, 'c_z', None)}, expected 16"
            self.assertEqual(getattr(block, 'c_z', None), 16, msg)
            ln_shape = getattr(block.tri_mul_out.layer_norm_in, 'normalized_shape', None)
            if ln_shape is not None:
                self.assertTrue(
                    ln_shape == (16,) or ln_shape == [16],
                    f"Block {idx} tri_mul_out.layer_norm_in.normalized_shape={ln_shape}, expected (16,)"
                )
        s_test = torch.randn(1, 5, 32)  # Reduced size
        z_test = torch.randn(1, 5, 5, 16)  # Reduced size
        pair_mask = torch.ones(1, 5, 5)  # Reduced size
        s_updated, z_updated = wrapper(s_test, z_test, pair_mask)
        self.assertFalse(torch.isnan(s_updated).any())
        self.assertFalse(torch.isinf(s_updated).any())
        self.assertFalse(torch.isnan(z_updated).any())
        self.assertFalse(torch.isinf(z_updated).any())
        del wrapper, s_test, z_test, pair_mask, s_updated, z_updated
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_gradient_flow(self):
        """
        Verify that gradients flow through the module during backpropagation.
        Using minimal model size and tensor dimensions.
        """
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({
            "stageB_pairformer": {
                "n_blocks": 1,  # Minimum size
                "n_heads": 4,  # Minimum size
                "c_z": 16,  # Minimum size
                "c_s": 32,  # Minimum size
                "dropout": 0.1,
                "use_checkpoint": True,
                "use_memory_efficient_kernel": False,
                "use_deepspeed_evo_attention": False,
                "use_lma": False,
                "inplace_safe": False,
                "chunk_size": None,
                "device": "cpu"
            }
        })
        required_keys = ["n_blocks", "n_heads", "c_z", "c_s", "dropout", "use_checkpoint", "use_memory_efficient_kernel", "use_deepspeed_evo_attention", "use_lma", "inplace_safe", "chunk_size", "device"]
        for k in required_keys:
            assert k in cfg["stageB_pairformer"], f"Config missing required key: {k}"
        wrapper = PairformerWrapper(cfg["stageB_pairformer"])
        for idx, block in enumerate(wrapper.stack.blocks):
            msg = f"Block {idx} c_z mismatch: got {getattr(block, 'c_z', None)}, expected 16"
            self.assertEqual(getattr(block, 'c_z', None), 16, msg)
            ln_shape = getattr(block.tri_mul_out.layer_norm_in, 'normalized_shape', None)
            if ln_shape is not None:
                self.assertTrue(
                    ln_shape == (16,) or ln_shape == [16],
                    f"Block {idx} tri_mul_out.layer_norm_in.normalized_shape={ln_shape}, expected (16,)"
                )
        s = torch.randn(1, 5, 32, requires_grad=True)  # Reduced size
        z = torch.randn(1, 5, 5, 16, requires_grad=True)  # Reduced size
        pair_mask = torch.ones(1, 5, 5)  # Reduced size
        s_updated, z_updated = wrapper(s, z, pair_mask)
        loss = s_updated.mean() + z_updated.mean()
        loss.backward()
        self.assertIsNotNone(s.grad)
        self.assertIsNotNone(z.grad)
        self.assertFalse(torch.isnan(s.grad).any())
        self.assertFalse(torch.isnan(z.grad).any())
        del wrapper, s, z, pair_mask, s_updated, z_updated
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_variable_sequence_length(self):
        """
        Test the model with different sequence lengths.
        Using smaller sequence lengths and minimal model size.
        """
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({
            "stageB_pairformer": {
                "n_blocks": self.default_n_blocks,
                "n_heads": 8,
                "c_z": self.default_c_z,
                "c_s": self.default_c_s,
                "dropout": 0.1,
                "use_checkpoint": self.default_use_checkpoint,
                "use_memory_efficient_kernel": False,
                "use_deepspeed_evo_attention": False,
                "use_lma": False,
                "inplace_safe": False,
                "chunk_size": None,
                "device": "cpu"
            }
        })
        required_keys = ["n_blocks", "n_heads", "c_z", "c_s", "dropout", "use_checkpoint", "use_memory_efficient_kernel", "use_deepspeed_evo_attention", "use_lma", "inplace_safe", "chunk_size", "device"]
        for k in required_keys:
            assert k in cfg["stageB_pairformer"], f"Config missing required key: {k}"
        wrapper = PairformerWrapper(cfg["stageB_pairformer"])
        for seq_length in [4, 8, 12]:
            s = torch.randn(1, seq_length, self.default_c_s)
            z = torch.randn(1, seq_length, seq_length, self.default_c_z)
            pair_mask = torch.ones(1, seq_length, seq_length)
            s_updated, z_updated = wrapper(s, z, pair_mask)
            self.assertEqual(s_updated.shape, s.shape)
            self.assertEqual(z_updated.shape, z.shape)
        del wrapper
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()

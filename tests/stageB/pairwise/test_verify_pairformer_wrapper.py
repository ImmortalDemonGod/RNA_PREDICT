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
import torch.nn as nn
from hypothesis import given, settings
from hypothesis import strategies as st

from rna_predict.pipeline.stageB.pairwise.pairformer import PairformerStack
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
        # Create a configuration object
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

        wrapper = PairformerWrapper(cfg)

        # Check instance type
        self.assertIsInstance(wrapper, PairformerWrapper)
        self.assertIsInstance(wrapper, nn.Module)

        # Check default parameter values
        self.assertEqual(wrapper.n_blocks, self.default_n_blocks)
        self.assertEqual(wrapper.c_z, self.default_c_z)
        self.assertEqual(wrapper.c_s, self.default_c_s)
        self.assertEqual(wrapper.use_checkpoint, self.default_use_checkpoint)

        # Check that PairformerStack is properly initialized
        self.assertIsInstance(wrapper.stack, PairformerStack)
        self.assertEqual(wrapper.stack.n_blocks, self.default_n_blocks)
        self.assertEqual(wrapper.stack.c_z, self.default_c_z)

    @given(
        n_blocks=st.integers(min_value=1, max_value=4),  # Further reduced from 12
        c_z=st.integers(min_value=16, max_value=64),  # Test more values
        c_s=st.integers(min_value=32, max_value=128),  # Test more values
        n_heads=st.integers(min_value=2, max_value=8),  # Test different head counts
        dropout=st.floats(min_value=0.0, max_value=0.5),  # Test different dropout rates
        use_checkpoint=st.booleans(),  # Test both with and without checkpointing
        use_memory_efficient=st.booleans(),  # Test memory efficiency options
    )
    @settings(
        deadline=None,
        max_examples=10,  # Reduced from 20
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
        # Create a configuration object with custom parameters
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

        wrapper = PairformerWrapper(cfg)

        # Check parameter values
        self.assertEqual(wrapper.n_blocks, n_blocks)
        self.assertEqual(wrapper.c_z, c_z)
        # c_z_adjusted should be a multiple of 16 and at least 16
        expected_c_z_adjusted = max(16, ((c_z + 15) // 16) * 16)
        self.assertEqual(wrapper.c_z_adjusted, expected_c_z_adjusted)
        self.assertEqual(wrapper.c_s, c_s)
        self.assertEqual(wrapper.use_checkpoint, use_checkpoint)
        self.assertEqual(wrapper.use_memory_efficient_kernel, use_memory_efficient)

        # Check that PairformerStack is properly initialized
        self.assertEqual(wrapper.stack.n_blocks, n_blocks)
        self.assertEqual(wrapper.stack.c_z, wrapper.c_z_adjusted)
        self.assertEqual(wrapper.stack.c_s, c_s)
        self.assertEqual(wrapper.stack.dropout, dropout)

        # Check that blocks_per_ckpt is set correctly based on use_checkpoint
        expected_blocks_per_ckpt = 1 if use_checkpoint else None
        self.assertEqual(wrapper.stack.blocks_per_ckpt, expected_blocks_per_ckpt)

        # Clean up
        del wrapper
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_parameter_count(self):
        """
        Verify that the parameter count matches the expected architecture size.
        Using reduced model size for testing.
        """
        # Create a configuration object
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

        wrapper = PairformerWrapper(cfg)

        # Get parameter count
        param_count = sum(p.numel() for p in wrapper.parameters())

        # The parameter count should be non-zero
        self.assertGreater(param_count, 0)

        # The parameter count should match the PairformerStack parameter count
        # Create a PairformerStackConfig for the stack
        from rna_predict.conf.config_schema import PairformerStackConfig

        stack_cfg = PairformerStackConfig(
            n_blocks=self.default_n_blocks,
            n_heads=8,
            c_z=wrapper.c_z_adjusted,  # Use the adjusted c_z value
            c_s=self.default_c_s,
            dropout=0.1,
            blocks_per_ckpt=1 if self.default_use_checkpoint else None
        )

        stack = PairformerStack(stack_cfg)
        stack_param_count = sum(p.numel() for p in stack.parameters())

        self.assertEqual(param_count, stack_param_count)

        # Clean up
        del wrapper
        del stack
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_forward_shape_consistency(self):
        """
        Verify that the forward pass returns tensors with the expected shapes.
        Using reduced tensor sizes for testing.
        """
        # Create a configuration object
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

        wrapper = PairformerWrapper(cfg)

        # Run forward pass
        s_updated, z_updated = wrapper(self.s, self.z, self.pair_mask)

        # Check output shapes
        self.assertEqual(s_updated.shape, self.s.shape)
        self.assertEqual(z_updated.shape, self.z.shape)

        # Clean up
        del wrapper
        del s_updated
        del z_updated
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_forward_no_nan_inf(self):
        """
        Verify that the forward pass does not produce NaN or Inf values.
        Using minimal model size for faster execution.
        """
        # Create a configuration object
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

        wrapper = PairformerWrapper(cfg)

        # Create minimal test tensors
        s_test = torch.randn(1, 5, 32)  # Reduced size
        z_test = torch.randn(1, 5, 5, 16)  # Reduced size
        pair_mask = torch.ones(1, 5, 5)  # Reduced size

        # Run forward pass
        s_updated, z_updated = wrapper(s_test, z_test, pair_mask)

        # Check for NaN or Inf values
        self.assertFalse(torch.isnan(s_updated).any())
        self.assertFalse(torch.isinf(s_updated).any())
        self.assertFalse(torch.isnan(z_updated).any())
        self.assertFalse(torch.isinf(z_updated).any())

        # Clean up
        del wrapper
        del s_test
        del z_test
        del pair_mask
        del s_updated
        del z_updated
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_gradient_flow(self):
        """
        Verify that gradients flow through the module during backpropagation.
        Using minimal model size and tensor dimensions.
        """
        # Create a configuration object
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

        wrapper = PairformerWrapper(cfg)

        # Create minimal test tensors
        s = torch.randn(1, 5, 32, requires_grad=True)  # Reduced size
        z = torch.randn(1, 5, 5, 16, requires_grad=True)  # Reduced size
        pair_mask = torch.ones(1, 5, 5)  # Reduced size

        # Run forward pass
        s_updated, z_updated = wrapper(s, z, pair_mask)

        # Compute loss and backpropagate
        loss = s_updated.mean() + z_updated.mean()
        loss.backward()

        # Check that gradients exist and are not None
        self.assertIsNotNone(s.grad)
        self.assertIsNotNone(z.grad)
        self.assertFalse(torch.isnan(s.grad).any())
        self.assertFalse(torch.isnan(z.grad).any())

        # Clean up
        del wrapper
        del s
        del z
        del pair_mask
        del s_updated
        del z_updated
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_variable_sequence_length(self):
        """
        Test the model with different sequence lengths.
        Using smaller sequence lengths and minimal model size.
        """
        # Create a configuration object
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

        wrapper = PairformerWrapper(cfg)

        # Test with a small range of sequence lengths
        for seq_len in [5, 8, 10]:  # Reduced range
            # Create tensors for this sequence length
            s = torch.randn(1, seq_len, 32)
            z = torch.randn(1, seq_len, seq_len, 16)
            pair_mask = torch.ones(1, seq_len, seq_len)

            # Run forward pass
            s_updated, z_updated = wrapper(s, z, pair_mask)

            # Check shapes
            self.assertEqual(s_updated.shape, s.shape)
            self.assertEqual(z_updated.shape, z.shape)

            # Clean up intermediate tensors
            del s
            del z
            del pair_mask
            del s_updated
            del z_updated
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Final cleanup
        del wrapper
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()

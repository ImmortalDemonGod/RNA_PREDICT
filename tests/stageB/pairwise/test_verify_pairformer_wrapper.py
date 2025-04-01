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

import unittest
import torch
import torch.nn as nn
from hypothesis import given, settings, strategies as st

from rna_predict.pipeline.stageB.pairwise.pairformer_wrapper import PairformerWrapper
from rna_predict.pipeline.stageB.pairwise.pairformer import PairformerStack


class TestPairformerWrapperVerification(unittest.TestCase):
    """
    Comprehensive verification tests for the PairformerWrapper class.
    """

    def setUp(self):
        """
        Set up common test parameters and configurations.
        """
        self.default_n_blocks = 48
        self.default_c_z = 128
        self.default_c_s = 384
        self.default_use_checkpoint = False
        
        # Test tensor dimensions
        self.batch_size = 1
        self.seq_length = 20
        self.node_features = 384  # c_s
        self.edge_features = 128  # c_z
        
        # Create test tensors
        self.s = torch.randn(self.batch_size, self.seq_length, self.node_features)
        self.z = torch.randn(self.batch_size, self.seq_length, self.seq_length, self.edge_features)
        # Use float tensor for pair_mask instead of boolean to avoid subtraction issues
        self.pair_mask = torch.ones(self.batch_size, self.seq_length, self.seq_length, dtype=torch.float32)

    def test_instantiation_default_parameters(self):
        """
        Verify that PairformerWrapper can be instantiated with default parameters.
        """
        wrapper = PairformerWrapper()
        
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
        n_blocks=st.integers(min_value=1, max_value=48),
        c_z=st.integers(min_value=16, max_value=256).filter(lambda x: x % 16 == 0),
        c_s=st.integers(min_value=16, max_value=512).filter(lambda x: x % 16 == 0),
        use_checkpoint=st.booleans()
    )
    @settings(deadline=None)  # Disable deadline for potentially slow tests
    def test_instantiation_custom_parameters(self, n_blocks, c_z, c_s, use_checkpoint):
        """
        Verify that PairformerWrapper can be instantiated with custom parameters.
        Note: both c_s and c_z must be divisible by 16 to satisfy AttentionPairBias constraint
        where n_heads=16 by default and requires c_a % n_heads == 0.
        """
        wrapper = PairformerWrapper(
            n_blocks=n_blocks,
            c_z=c_z,
            c_s=c_s,
            use_checkpoint=use_checkpoint
        )
        
        # Check parameter values
        self.assertEqual(wrapper.n_blocks, n_blocks)
        self.assertEqual(wrapper.c_z, c_z)
        self.assertEqual(wrapper.c_z_adjusted, c_z)  # Should be equal since we filter for divisible by 16
        self.assertEqual(wrapper.c_s, c_s)
        self.assertEqual(wrapper.use_checkpoint, use_checkpoint)
        
        # Check that PairformerStack is properly initialized with the same parameters
        self.assertEqual(wrapper.stack.n_blocks, n_blocks)
        self.assertEqual(wrapper.stack.c_z, wrapper.c_z_adjusted)  # Using c_z_adjusted now

    def test_parameter_count(self):
        """
        Verify that the parameter count matches the expected architecture size.
        """
        wrapper = PairformerWrapper()
        
        # Get parameter count
        param_count = sum(p.numel() for p in wrapper.parameters())
        
        # The parameter count should be non-zero
        self.assertGreater(param_count, 0)
        
        # The parameter count should match the PairformerStack parameter count
        stack = PairformerStack(
            n_blocks=self.default_n_blocks,
            c_z=self.default_c_z,
            c_s=self.default_c_s
        )
        stack_param_count = sum(p.numel() for p in stack.parameters())
        
        self.assertEqual(param_count, stack_param_count)

    def test_forward_shape_consistency(self):
        """
        Verify that the forward pass returns tensors with the expected shapes.
        """
        wrapper = PairformerWrapper()
        
        # Run forward pass
        s_updated, z_updated = wrapper(self.s, self.z, self.pair_mask)
        
        # Check output shapes
        self.assertEqual(s_updated.shape, self.s.shape)
        self.assertEqual(z_updated.shape, self.z.shape)

    def test_forward_no_nan_inf(self):
        """
        Verify that the forward pass does not produce NaN or Inf values.
        """
        wrapper = PairformerWrapper()
        
        # Run forward pass
        s_updated, z_updated = wrapper(self.s, self.z, self.pair_mask)
        
        # Check for NaN or Inf values
        self.assertFalse(torch.isnan(s_updated).any())
        self.assertFalse(torch.isinf(s_updated).any())
        self.assertFalse(torch.isnan(z_updated).any())
        self.assertFalse(torch.isinf(z_updated).any())

    def test_gradient_flow(self):
        """
        Verify that gradients flow through the module during backpropagation.
        """
        wrapper = PairformerWrapper()
        
        # Set requires_grad=True for input tensors
        s = self.s.clone().detach().requires_grad_(True)
        z = self.z.clone().detach().requires_grad_(True)
        
        # Run forward pass
        s_updated, z_updated = wrapper(s, z, self.pair_mask)
        
        # Compute a loss and backpropagate
        loss = s_updated.mean() + z_updated.mean()
        loss.backward()
        
        # Check that gradients are computed
        for param in wrapper.parameters():
            self.assertIsNotNone(param.grad)
            # At least some parameters should have non-zero gradients
        
        # Check that input tensors have gradients
        self.assertIsNotNone(s.grad)
        self.assertIsNotNone(z.grad)

    def test_variable_sequence_length(self):
        """
        Verify that the module can handle variable sequence lengths.
        """
        wrapper = PairformerWrapper()
        
        # Test with different sequence lengths
        seq_lengths = [10, 15, 25]
        
        for seq_len in seq_lengths:
            # Create test tensors with different sequence length
            s = torch.randn(self.batch_size, seq_len, self.node_features)
            z = torch.randn(self.batch_size, seq_len, seq_len, self.edge_features)
            # Use float tensor for pair_mask instead of boolean to avoid subtraction issues
            pair_mask = torch.ones(self.batch_size, seq_len, seq_len, dtype=torch.float32)
            
            # Run forward pass
            s_updated, z_updated = wrapper(s, z, pair_mask)
            
            # Check output shapes
            self.assertEqual(s_updated.shape, s.shape)
            self.assertEqual(z_updated.shape, z.shape)
            
            # Check for NaN or Inf values
            self.assertFalse(torch.isnan(s_updated).any())
            self.assertFalse(torch.isinf(s_updated).any())
            self.assertFalse(torch.isnan(z_updated).any())
            self.assertFalse(torch.isinf(z_updated).any())

    def test_wrapper_delegates_to_stack(self):
        """
        Verify that the wrapper correctly delegates to the PairformerStack.
        """
        wrapper = PairformerWrapper()
        
        # Mock the PairformerStack forward method
        original_forward = wrapper.stack.forward
        
        call_count = [0]
        def mock_forward(s, z, pair_mask, **kwargs):
            call_count[0] += 1
            return original_forward(s, z, pair_mask, **kwargs)
        
        wrapper.stack.forward = mock_forward
        
        # Run forward pass
        s_updated, z_updated = wrapper(self.s, self.z, self.pair_mask)
        
        # Check that the stack's forward method was called
        self.assertEqual(call_count[0], 1)
        
        # Restore original forward method
        wrapper.stack.forward = original_forward


if __name__ == "__main__":
    unittest.main()
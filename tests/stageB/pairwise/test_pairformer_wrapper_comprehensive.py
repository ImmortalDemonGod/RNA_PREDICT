"""
Comprehensive Tests for PairformerWrapper
=========================================

This test file provides comprehensive tests for the PairformerWrapper class,
focusing on achieving high test coverage by testing all code paths and edge cases.

The tests cover:
1. Initialization with various parameters
2. Forward pass with different tensor shapes and configurations
3. Dimension adjustment logic
4. Edge cases and error handling
5. The adjust_z_dimensions method
"""

import unittest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from rna_predict.pipeline.stageB.pairwise.pairformer import PairformerStack
from rna_predict.pipeline.stageB.pairwise.pairformer_wrapper import PairformerWrapper


class TestPairformerWrapperComprehensive(unittest.TestCase):
    """
    Comprehensive tests for the PairformerWrapper class.
    """

    def setUp(self):
        """
        Initialize common test parameters and create test input tensors.
        
        This method configures the instance variables used by multiple tests, including batch size, sequence length, and channel dimensions. It creates random tensors for testing: an input tensor (self.s), a feature map tensor (self.z), a pair mask (self.pair_mask), and a tensor with a non-standard channel dimension (self.z_non_multiple) to simulate edge cases for dimension adjustment.
        """
        # Test parameters
        self.batch_size = 1
        self.seq_length = 8
        self.c_s = 64
        self.c_z = 32
        self.c_z_non_multiple = 30  # Not a multiple of 16

        # Create test tensors
        self.s = torch.randn(self.batch_size, self.seq_length, self.c_s)
        self.z = torch.randn(self.batch_size, self.seq_length, self.seq_length, self.c_z)
        self.pair_mask = torch.ones(self.batch_size, self.seq_length, self.seq_length)

        # Create non-multiple tensors
        self.z_non_multiple = torch.randn(
            self.batch_size, self.seq_length, self.seq_length, self.c_z_non_multiple
        )

    def test_init_with_c_z_adjustment(self):
        """
        Verifies that PairformerWrapper correctly adjusts c_z to the next multiple of 16.
        
        This test confirms that initializing the wrapper with a c_z value of 30 results
        in an adjusted c_z of 32, and that this corrected value is properly initialized in
        the internal stack.
        """
        # c_z = 30 should be adjusted to 32 (next multiple of 16)
        wrapper = PairformerWrapper(n_blocks=2, c_z=30, c_s=64)

        # Check that c_z_adjusted is properly calculated
        self.assertEqual(wrapper.c_z, 30)
        self.assertEqual(wrapper.c_z_adjusted, 32)

        # Check that the stack is initialized with the adjusted c_z
        self.assertEqual(wrapper.stack.c_z, 32)

    def test_forward_with_c_z_adjustment_padding(self):
        """Test forward pass with c_z that needs padding."""
        # Create wrapper with c_z that needs adjustment
        wrapper = PairformerWrapper(n_blocks=2, c_z=30, c_s=64)

        # Run forward pass with z tensor that has c_z=30
        s_updated, z_updated = wrapper(self.s, self.z_non_multiple, self.pair_mask)

        # Check output shapes
        self.assertEqual(s_updated.shape, self.s.shape)
        self.assertEqual(z_updated.shape, self.z_non_multiple.shape)

        # Verify that z_updated has the original c_z dimension (30)
        self.assertEqual(z_updated.shape[-1], 30)

    def test_forward_with_c_z_adjustment_truncation(self):
        """
        Tests the forward pass with forced c_z truncation.
        
        This test simulates an edge case by manually reducing the adjusted c_z value, which forces the 
        wrapper to truncate the input z tensor's channel dimension. The internal PairformerStack forward 
        method is replaced with a mock that returns tensors with 16 channels, and the test verifies that 
        the z tensor is truncated appropriately and that the output shapes match the expectations.
        """
        # This is an edge case that shouldn't happen with the current implementation,
        # but we test it for completeness

        # Instead of trying to run the full forward pass, which would require
        # modifying the internal PairformerStack, we'll mock the stack's forward method
        wrapper = PairformerWrapper(n_blocks=2, c_z=30, c_s=64)

        # Manually set c_z_adjusted to a smaller value to force truncation path
        wrapper.c_z_adjusted = 16

        # Create a mock for the stack's forward method
        mock_output_s = torch.randn_like(self.s)
        mock_output_z = torch.randn(self.batch_size, self.seq_length, self.seq_length, 16)  # Adjusted size

        # Replace the stack's forward method with a mock
        original_stack_forward = wrapper.stack.forward
        wrapper.stack.forward = MagicMock(return_value=(mock_output_s, mock_output_z))

        try:
            # Run forward pass
            s_updated, z_updated = wrapper(self.s, self.z_non_multiple, self.pair_mask)

            # Check that stack.forward was called with truncated z
            args, _ = wrapper.stack.forward.call_args
            _, z_arg, _ = args
            self.assertEqual(z_arg.shape[-1], 16)  # Should be truncated to 16

            # In this mocked case, z_updated will have shape [batch, seq, seq, 16]
            # because we're mocking the stack to return a tensor with 16 channels
            # and the wrapper doesn't expand it back to 30 in this case
            self.assertEqual(s_updated.shape, self.s.shape)
            self.assertEqual(z_updated.shape[-1], 16)
        finally:
            # Restore the original forward method
            wrapper.stack.forward = original_stack_forward

    def test_adjust_z_dimensions_padding(self):
        """Test the adjust_z_dimensions method with padding."""
        wrapper = PairformerWrapper(n_blocks=2, c_z=30, c_s=64)

        # Call adjust_z_dimensions directly
        z_adjusted = wrapper.adjust_z_dimensions(self.z_non_multiple)

        # Check that the output has the adjusted dimension
        self.assertEqual(z_adjusted.shape[-1], 32)

        # Check that the first 30 dimensions match the original tensor
        self.assertTrue(torch.allclose(
            z_adjusted[..., :30],
            self.z_non_multiple
        ))

        # Check that the padding dimensions are zeros
        self.assertTrue(torch.all(z_adjusted[..., 30:] == 0))

    def test_adjust_z_dimensions_truncation(self):
        """Test the adjust_z_dimensions method with truncation."""
        wrapper = PairformerWrapper(n_blocks=2, c_z=32, c_s=64)

        # Manually set c_z_adjusted to a smaller value to force truncation path
        wrapper.c_z_adjusted = 16

        # Call adjust_z_dimensions directly
        z_adjusted = wrapper.adjust_z_dimensions(self.z)

        # Check that the output has the adjusted dimension
        self.assertEqual(z_adjusted.shape[-1], 16)

        # Check that the output matches the truncated original tensor
        self.assertTrue(torch.allclose(
            z_adjusted,
            self.z[..., :16]
        ))

    def test_forward_with_mock_stack(self):
        """Test forward pass with a mocked PairformerStack to isolate the wrapper logic."""
        # Create a wrapper
        wrapper = PairformerWrapper(n_blocks=2, c_z=30, c_s=64)

        # Create a mock for the stack's forward method
        mock_output_s = torch.randn_like(self.s)
        mock_output_z = torch.randn_like(self.z_non_multiple[..., :32])  # Adjusted size

        # Replace the stack's forward method with a mock
        original_stack_forward = wrapper.stack.forward
        wrapper.stack.forward = MagicMock(return_value=(mock_output_s, mock_output_z))

        try:
            # Run forward pass
            s_updated, z_updated = wrapper(self.s, self.z_non_multiple, self.pair_mask)

            # Check that stack.forward was called with adjusted z
            args, _ = wrapper.stack.forward.call_args
            _, z_arg, _ = args
            self.assertEqual(z_arg.shape[-1], 32)  # Should be adjusted to 32

            # Check output shapes
            self.assertEqual(s_updated.shape, self.s.shape)
            self.assertEqual(z_updated.shape, self.z_non_multiple.shape)

            # Check that z_updated is truncated back to original size
            self.assertEqual(z_updated.shape[-1], 30)
        finally:
            # Restore the original forward method
            wrapper.stack.forward = original_stack_forward

    def test_device_consistency(self):
        """Test that the wrapper handles device placement correctly."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping device test")

        # Create tensors on GPU
        s_gpu = self.s.cuda()
        z_gpu = self.z_non_multiple.cuda()
        pair_mask_gpu = self.pair_mask.cuda()

        # Create wrapper
        wrapper = PairformerWrapper(n_blocks=2, c_z=30, c_s=64).cuda()

        # Run forward pass
        s_updated, z_updated = wrapper(s_gpu, z_gpu, pair_mask_gpu)

        # Check that outputs are on the same device
        self.assertEqual(s_updated.device, s_gpu.device)
        self.assertEqual(z_updated.device, z_gpu.device)

        # Check output shapes
        self.assertEqual(s_updated.shape, s_gpu.shape)
        self.assertEqual(z_updated.shape, z_gpu.shape)

    def test_dtype_consistency_in_adjust_dimensions(self):
        """
        Verifies that adjust_z_dimensions preserves the input tensor's data type.
        
        This test converts a sample tensor to float64 and passes it to the adjust_z_dimensions method of a PairformerWrapper.
        It asserts that the output tensor maintains the float64 type and that its last dimension is adjusted to 32.
        """
        # Create tensors with float64 dtype
        z_float64 = self.z_non_multiple.to(torch.float64)

        # Create wrapper
        wrapper = PairformerWrapper(n_blocks=2, c_z=30, c_s=64)

        # Call adjust_z_dimensions directly
        z_adjusted = wrapper.adjust_z_dimensions(z_float64)

        # Check that output has the same dtype
        self.assertEqual(z_adjusted.dtype, torch.float64)

        # Check output shape
        self.assertEqual(z_adjusted.shape[-1], 32)

    def test_adjust_z_dimensions_with_device(self):
        """Test adjust_z_dimensions with tensors on different devices."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping device test")

        # Create wrapper
        wrapper = PairformerWrapper(n_blocks=2, c_z=30, c_s=64)

        # Create tensor on GPU
        z_gpu = self.z_non_multiple.cuda()

        # Call adjust_z_dimensions
        z_adjusted = wrapper.adjust_z_dimensions(z_gpu)

        # Check that output is on the same device
        self.assertEqual(z_adjusted.device, z_gpu.device)

        # Check output shape
        self.assertEqual(z_adjusted.shape[-1], 32)

    def test_adjust_z_dimensions_with_dtype(self):
        """Test adjust_z_dimensions with tensors of different dtypes."""
        # Create wrapper
        wrapper = PairformerWrapper(n_blocks=2, c_z=30, c_s=64)

        # Create tensor with float64 dtype
        z_float64 = self.z_non_multiple.to(torch.float64)

        # Call adjust_z_dimensions
        z_adjusted = wrapper.adjust_z_dimensions(z_float64)

        # Check that output has the same dtype
        self.assertEqual(z_adjusted.dtype, torch.float64)

        # Check output shape
        self.assertEqual(z_adjusted.shape[-1], 32)


if __name__ == "__main__":
    unittest.main()

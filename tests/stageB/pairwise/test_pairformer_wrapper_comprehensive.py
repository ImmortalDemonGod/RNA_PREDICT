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
from omegaconf import OmegaConf, DictConfig # Import OmegaConf and DictConfig
from unittest.mock import MagicMock

from rna_predict.pipeline.stageB.pairwise.pairformer_wrapper import PairformerWrapper
# --- Helper function to create test configs ---
def create_test_pairformer_config(**overrides) -> DictConfig:
    """Creates a base DictConfig for pairformer tests, allowing overrides."""
    # Use stageB_pairformer as the key to match the expected configuration structure
    base_config = {
        "stageB_pairformer": {
            "n_blocks": 48,
            "n_heads": 16,
            "c_z": 128,
            "c_s": 384,
            "dropout": 0.25,
            "use_memory_efficient_kernel": False,
            "use_deepspeed_evo_attention": False,
            "use_lma": False,
            "inplace_safe": False,
            "chunk_size": None,
            "c_hidden_mul": 128,
            "c_hidden_pair_att": 32,
            "no_heads_pair": 4,
            "init_z_from_adjacency": False,
            "use_checkpoint": False, # Explicitly add use_checkpoint based on refactored code
             "lora": {              # Include LoRA defaults
                "enabled": False,
                "r": 8,
                "alpha": 16,
                "dropout": 0.1,
                "target_modules": []
            }
        }
        # Can add other top-level keys if ever needed by tests
    }
    cfg = OmegaConf.create(base_config)
    override_nested = {"stageB_pairformer": overrides}
    override_cfg = OmegaConf.create(override_nested)
    cfg = OmegaConf.merge(cfg, override_cfg)
    # Ensure the merge result is DictConfig, handle potential ListConfig case if necessary
    if not isinstance(cfg, DictConfig):
         # This case should ideally not happen with current structure, but added defensively
         raise TypeError(f"Merged config is not DictConfig: {type(cfg)}")
    return cfg



class TestPairformerWrapperComprehensive(unittest.TestCase):
    """
    Comprehensive tests for the PairformerWrapper class.
    """

    def setUp(self):
        """Set up common test parameters."""
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
        """Test initialization with c_z that needs adjustment."""
        # c_z = 30 should be adjusted to 32 (next multiple of 16)
        test_cfg = create_test_pairformer_config(n_blocks=2, c_z=30, c_s=64)
        wrapper = PairformerWrapper(cfg=test_cfg)

        # Check that c_z_adjusted is properly calculated
        self.assertEqual(wrapper.c_z, 30)
        self.assertEqual(wrapper.c_z_adjusted, 32)

        # Check that the stack is initialized with the adjusted c_z
        self.assertEqual(wrapper.stack.c_z, 32)

    def test_forward_with_c_z_adjustment_padding(self):
        """Test forward pass with c_z that needs padding."""
        # Create wrapper with c_z that needs adjustment
        test_cfg = create_test_pairformer_config(n_blocks=2, c_z=30, c_s=64)
        wrapper = PairformerWrapper(cfg=test_cfg)

        # Run forward pass with z tensor that has c_z=30
        s_updated, z_updated = wrapper(self.s, self.z_non_multiple, self.pair_mask)

        # Check output shapes
        self.assertEqual(s_updated.shape, self.s.shape)
        self.assertEqual(z_updated.shape, self.z_non_multiple.shape)

        # Verify that z_updated has the original c_z dimension (30)
        self.assertEqual(z_updated.shape[-1], 30)

    def test_forward_with_c_z_adjustment_truncation(self):
        """Test forward pass with c_z that needs truncation (edge case)."""
        # This is an edge case that shouldn't happen with the current implementation,
        # but we test it for completeness

        # Instead of trying to run the full forward pass, which would require
        # modifying the internal PairformerStack, we'll mock the stack's forward method
        test_cfg = create_test_pairformer_config(n_blocks=2, c_z=30, c_s=64)
        wrapper = PairformerWrapper(cfg=test_cfg)

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
        test_cfg = create_test_pairformer_config(n_blocks=2, c_z=30, c_s=64)
        wrapper = PairformerWrapper(cfg=test_cfg)

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
        test_cfg = create_test_pairformer_config(n_blocks=2, c_z=32, c_s=64)
        wrapper = PairformerWrapper(cfg=test_cfg)

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
        test_cfg = create_test_pairformer_config(n_blocks=2, c_z=30, c_s=64)
        wrapper = PairformerWrapper(cfg=test_cfg)

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
        # Use CPU device if CUDA is not available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create tensors on the selected device
        s_device = self.s.to(device)
        z_device = self.z_non_multiple.to(device)
        pair_mask_device = self.pair_mask.to(device)

        # Create wrapper
        test_cfg = create_test_pairformer_config(n_blocks=2, c_z=30, c_s=64)
        # Instantiate first, then move to the selected device
        wrapper = PairformerWrapper(cfg=test_cfg).to(device)

        # Run forward pass
        s_updated, z_updated = wrapper(s_device, z_device, pair_mask_device)

        # Check that outputs are on the same device
        self.assertEqual(s_updated.device, s_device.device)
        self.assertEqual(z_updated.device, z_device.device)

        # Check output shapes
        self.assertEqual(s_updated.shape, s_device.shape)
        self.assertEqual(z_updated.shape, z_device.shape)

    def test_dtype_consistency_in_adjust_dimensions(self):
        """Test that adjust_z_dimensions handles different dtypes correctly."""
        # Create tensors with float64 dtype
        z_float64 = self.z_non_multiple.to(torch.float64)

        # Create wrapper
        test_cfg = create_test_pairformer_config(n_blocks=2, c_z=30, c_s=64)
        wrapper = PairformerWrapper(cfg=test_cfg)

        # Call adjust_z_dimensions directly
        z_adjusted = wrapper.adjust_z_dimensions(z_float64)

        # Check that output has the same dtype
        self.assertEqual(z_adjusted.dtype, torch.float64)

        # Check output shape
        self.assertEqual(z_adjusted.shape[-1], 32)

    def test_adjust_z_dimensions_with_device(self):
        """Test adjust_z_dimensions with tensors on different devices."""
        # Use CPU device if CUDA is not available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create wrapper
        test_cfg = create_test_pairformer_config(n_blocks=2, c_z=30, c_s=64)
        wrapper = PairformerWrapper(cfg=test_cfg)

        # Create tensor on the selected device
        z_device = self.z_non_multiple.to(device)

        # Call adjust_z_dimensions
        z_adjusted = wrapper.adjust_z_dimensions(z_device)

        # Check that output is on the same device
        self.assertEqual(z_adjusted.device, z_device.device)

        # Check output shape
        self.assertEqual(z_adjusted.shape[-1], 32)

    def test_adjust_z_dimensions_with_dtype(self):
        """Test adjust_z_dimensions with tensors of different dtypes."""
        # Create wrapper
        test_cfg = create_test_pairformer_config(n_blocks=2, c_z=30, c_s=64)
        wrapper = PairformerWrapper(cfg=test_cfg)

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

import unittest
import torch
from rna_predict.pipeline.stageD.diffusion.components.diffusion_conditioning import DiffusionConditioning
from rna_predict.pipeline.stageD.diffusion.components.diffusion_utils import ShapeMismatchError

class TestDiffusionConditioning(unittest.TestCase):
    def setUp(self):
        self.c_s = 64  # Single feature dimension
        self.c_s_inputs = 449  # Expected input feature dimension
        self.c_z = 32  # Pair feature dimension
        self.c_hidden = 128  # Hidden dimension
        self.n_heads = 4  # Number of attention heads
        self.n_blocks = 2  # Number of transformer blocks
        self.dropout = 0.1
        self.blocks_per_ckpt = None
        
        self.module = DiffusionConditioning(
            c_s=self.c_s,
            c_s_inputs=self.c_s_inputs,
            c_z=self.c_z,
            c_hidden=self.c_hidden,
            n_heads=self.n_heads,
            n_blocks=self.n_blocks,
            dropout=self.dropout,
            blocks_per_ckpt=self.blocks_per_ckpt
        )

    def test_shape_validation_single_features(self):
        """Test that shape validation catches mismatched feature dimensions"""
        batch_size = 2
        seq_len = 24
        
        # Correct shapes
        s_trunk = torch.randn(batch_size, seq_len, self.c_s)
        s_inputs = torch.randn(batch_size, seq_len, self.c_s_inputs)
        
        # Should not raise error
        self.module._process_single_features(s_trunk, s_inputs, inplace_safe=True)
        
        # Wrong feature dimension for s_inputs
        wrong_s_inputs = torch.randn(batch_size, seq_len, 384)  # Wrong dimension
        with self.assertRaises(ShapeMismatchError) as context:
            self.module._process_single_features(s_trunk, wrong_s_inputs, inplace_safe=True)
        self.assertIn("Expected last dimension 449 for s_inputs", str(context.exception))

    def test_bias_shape_handling(self):
        """Test that bias shape mismatches are handled correctly"""
        batch_size = 2
        seq_len = 24
        
        # Create tensors with shapes that would trigger bias shape mismatch warnings
        s_trunk = torch.randn(batch_size, seq_len, self.c_s)
        s_inputs = torch.randn(batch_size, seq_len, self.c_s_inputs)
        z_pair = torch.randn(batch_size, seq_len, seq_len, self.c_z)
        
        # Test with different bias shapes
        with self.assertLogs(level='WARNING') as log:
            self.module.forward(s_trunk, s_inputs, z_pair, inplace_safe=True)
            self.assertTrue(any("Selected bias shape mismatch" in msg for msg in log.output))

    def test_feature_dimension_consistency(self):
        """Test that feature dimensions are consistent across the pipeline"""
        batch_size = 2
        seq_len = 24
        
        # Test with mismatched feature dimensions
        s_trunk = torch.randn(batch_size, seq_len, self.c_s)
        s_inputs = torch.randn(batch_size, seq_len, 384)  # Wrong dimension
        z_pair = torch.randn(batch_size, seq_len, seq_len, self.c_z)
        
        with self.assertRaises(ShapeMismatchError):
            self.module.forward(s_trunk, s_inputs, z_pair, inplace_safe=True)

    def test_batch_size_handling(self):
        """Test handling of different batch sizes"""
        seq_len = 24
        
        # Test with batch size 1
        s_trunk = torch.randn(1, seq_len, self.c_s)
        s_inputs = torch.randn(1, seq_len, self.c_s_inputs)
        z_pair = torch.randn(1, seq_len, seq_len, self.c_z)
        
        # Should not raise error
        self.module.forward(s_trunk, s_inputs, z_pair, inplace_safe=True)
        
        # Test with larger batch size
        batch_size = 4
        s_trunk = torch.randn(batch_size, seq_len, self.c_s)
        s_inputs = torch.randn(batch_size, seq_len, self.c_s_inputs)
        z_pair = torch.randn(batch_size, seq_len, seq_len, self.c_z)
        
        # Should not raise error
        self.module.forward(s_trunk, s_inputs, z_pair, inplace_safe=True)

if __name__ == '__main__':
    unittest.main() 
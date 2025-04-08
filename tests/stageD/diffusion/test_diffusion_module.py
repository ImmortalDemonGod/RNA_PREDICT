import unittest
import torch
from rna_predict.pipeline.stageD.diffusion.components.diffusion_module import DiffusionModule
from rna_predict.pipeline.stageD.diffusion.components.diffusion_utils import ShapeMismatchError

class TestDiffusionModule(unittest.TestCase):
    def setUp(self):
        self.c_s = 64  # Single feature dimension
        self.c_s_inputs = 449  # Expected input feature dimension
        self.c_z = 32  # Pair feature dimension
        self.c_token = 384  # Token feature dimension
        self.c_atom = 128  # Atom feature dimension
        self.c_atompair = 16  # Atom pair feature dimension
        self.c_noise_embedding = 256  # Noise embedding dimension
        self.blocks_per_ckpt = None
        
        self.module = DiffusionModule(
            c_s=self.c_s,
            c_s_inputs=self.c_s_inputs,
            c_z=self.c_z,
            c_token=self.c_token,
            c_atom=self.c_atom,
            c_atompair=self.c_atompair,
            c_noise_embedding=self.c_noise_embedding,
            blocks_per_ckpt=self.blocks_per_ckpt
        )

    def test_tensor_broadcasting(self):
        """Test that tensor broadcasting is handled correctly"""
        batch_size = 2
        seq_len = 24
        
        # Create input tensors
        x_noisy = torch.randn(batch_size, seq_len, seq_len, 3)
        t_hat = torch.randn(batch_size, 1)  # Time step tensor
        
        # Test broadcasting
        with self.assertLogs(level='WARNING') as log:
            self.module.forward(x_noisy, t_hat)
            self.assertTrue(any("Broadcasting t_hat" in msg for msg in log.output))

    def test_shape_validation(self):
        """Test that shape validation catches mismatched dimensions"""
        batch_size = 2
        seq_len = 24
        
        # Correct shapes
        x_noisy = torch.randn(batch_size, seq_len, seq_len, 3)
        t_hat = torch.randn(batch_size, 1)
        
        # Should not raise error
        self.module.forward(x_noisy, t_hat)
        
        # Wrong shape for x_noisy
        wrong_x_noisy = torch.randn(batch_size, seq_len, 3)  # Missing dimension
        with self.assertRaises(ShapeMismatchError):
            self.module.forward(wrong_x_noisy, t_hat)

    def test_bias_shape_handling(self):
        """Test that bias shape mismatches are handled correctly"""
        batch_size = 2
        seq_len = 24
        
        # Create tensors with shapes that would trigger bias shape mismatch warnings
        x_noisy = torch.randn(batch_size, seq_len, seq_len, 3)
        t_hat = torch.randn(batch_size, 1)
        
        # Test with different bias shapes
        with self.assertLogs(level='INFO') as log:  # Changed from WARNING to INFO to capture print output
            self.module.forward(x_noisy, t_hat)
            self.assertTrue(any("Selected bias shape mismatch" in msg for msg in log.output))

    def test_n_sample_handling(self):
        """Test handling of different N_sample values"""
        batch_size = 2
        seq_len = 24
        
        # Test with N_sample=1
        x_noisy = torch.randn(batch_size, 1, seq_len, 3)
        t_hat = torch.randn(batch_size, 1)
        
        # Should not raise error
        self.module.forward(x_noisy, t_hat)
        
        # Test with N_sample>1
        x_noisy = torch.randn(batch_size, 4, seq_len, 3)
        t_hat = torch.randn(batch_size, 1)
        
        # Should not raise error
        self.module.forward(x_noisy, t_hat)

    def test_feature_dimension_consistency(self):
        """Test that feature dimensions are consistent throughout the module"""
        batch_size = 2
        seq_len = 24
        
        # Test with correct feature dimensions
        x_noisy = torch.randn(batch_size, seq_len, seq_len, 3)
        t_hat = torch.randn(batch_size, 1)
        
        # Should not raise error
        self.module.forward(x_noisy, t_hat)
        
        # Test with mismatched feature dimensions in internal processing
        with self.assertRaises(ShapeMismatchError):
            # Create a tensor with wrong feature dimension
            wrong_feature = torch.randn(batch_size, seq_len, 384)  # Wrong dimension
            self.module._process_features(wrong_feature)

if __name__ == '__main__':
    unittest.main() 
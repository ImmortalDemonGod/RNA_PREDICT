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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Ensure tensors are on same device
        self.module.to(device) # Ensure module is on the same device

        # Create tensors with shapes that would trigger bias shape mismatch warnings
        # Corrected shape for x_noisy
        x_noisy = torch.randn(batch_size, seq_len, 3, device=device)
        t_hat = torch.randn(batch_size, 1, device=device) # Ensure t_hat is also on device

        # Create input feature dictionary with required fields and on device
        input_feature_dict = {
            "ref_pos": torch.randn(batch_size, 1, seq_len, 3, device=device),
            "ref_charge": torch.zeros(batch_size, seq_len, 1, device=device),
            "ref_mask": torch.ones(batch_size, seq_len, 1, device=device),
            "ref_element": torch.randn(batch_size, seq_len, 128, device=device),
            "ref_atom_name_chars": torch.randn(batch_size, seq_len, 4 * 64, device=device),
            "ref_space_uid": torch.randn(batch_size, 1, seq_len, 3, device=device),
            "atom_to_token_idx": torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1),
            "restype": torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        }

        # Create additional required tensors on device
        s_trunk = torch.randn(batch_size, seq_len, self.c_s, device=device)
        s_inputs = torch.randn(batch_size, seq_len, self.c_s_inputs, device=device)
        z_trunk = torch.randn(batch_size, seq_len, seq_len, self.c_z, device=device)

        # Test with different bias shapes
        # Note: This test expects warnings about bias shape mismatch.
        # The underlying cause of those warnings might be complex and related to
        # internal layer configurations vs. input shapes, but the test aims
        # to ensure the forward pass *completes* despite these warnings.
        # We also change assertLogs level back to WARNING as originally intended.
        with self.assertLogs(level='WARNING') as log:
            try:
                # Pass all required arguments
                self.module.forward(
                    x_noisy=x_noisy,
                    t_hat_noise_level=t_hat,
                    input_feature_dict=input_feature_dict,
                    s_inputs=s_inputs,
                    s_trunk=s_trunk,
                    z_trunk=z_trunk
                )
                # Check if *any* bias mismatch warning occurred, as expected by the test name
                self.assertTrue(any("bias shape mismatch" in msg for msg in log.output),
                                "Expected at least one bias shape mismatch warning, but none found.")
            except Exception as e:
                self.fail(f"forward failed unexpectedly: {e}")

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
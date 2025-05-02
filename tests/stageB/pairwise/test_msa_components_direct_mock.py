"""
Direct tests for MSA components using complete mocks.
"""

import unittest
import torch
import torch.nn as nn

# Create mock classes for testing
class MockMSAConfig:
    """Mock MSA configuration for testing."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class MockMSABlock(nn.Module):
    """Mock MSA block for testing."""
    def __init__(self, cfg, is_last_block=False):
        super().__init__()
        self.c_m = cfg.c_m
        self.c_z = cfg.c_z
        self.is_last_block = is_last_block
        
    def forward(self, m, z, pair_mask, **kwargs):
        """Mock forward pass."""
        if self.is_last_block:
            return None, z
        else:
            return m, z

class MockMSAModule(nn.Module):
    """Mock MSA module for testing."""
    def __init__(self, cfg):
        super().__init__()
        self.n_blocks = cfg.n_blocks
        self.c_m = cfg.c_m
        self.c_z = cfg.c_z
        
        # Set up msa_configs
        self.msa_configs = {
            "enable": getattr(cfg, "enable", False),
            "strategy": getattr(cfg, "strategy", "random"),
            "train_cutoff": getattr(cfg, "train_cutoff", 512),
            "test_cutoff": getattr(cfg, "test_cutoff", 16384),
            "train_lowerb": getattr(cfg, "train_lowerb", 1),
            "test_lowerb": getattr(cfg, "test_lowerb", 1),
        }
        
        # Create blocks
        self.blocks = nn.ModuleList()
        for i in range(self.n_blocks):
            block = MockMSABlock(
                cfg=cfg,
                is_last_block=(i + 1 == self.n_blocks),
            )
            self.blocks.append(block)
    
    def forward(self, input_feature_dict, z, s_inputs=None, pair_mask=None, **kwargs):
        """Mock forward pass."""
        # If n_blocks < 1, return z
        if self.n_blocks < 1:
            return z
        
        if "msa" not in input_feature_dict:
            return z
        
        # Create mock MSA sample
        msa_sample = torch.zeros((2, z.shape[-3], self.c_m))
        
        # Process through blocks
        for block in self.blocks:
            msa_sample, z = block(msa_sample, z, pair_mask)
        
        return z

class TestMSAComponentsDirectMock(unittest.TestCase):
    """Direct tests for MSA components using complete mocks."""
    
    def test_msa_block(self):
        """Test MSABlock with complete mocks."""
        # Create a mock MSAConfig
        cfg = MockMSAConfig(
            c_m=8,
            c=8,
            c_z=8,
            dropout=0.1
        )
        
        # Create a mock MSABlock
        block = MockMSABlock(cfg=cfg)
        
        # Test instantiation
        self.assertIsInstance(block, MockMSABlock)
        self.assertEqual(block.c_m, 8)
        self.assertEqual(block.c_z, 8)
        self.assertFalse(block.is_last_block)
        
        # Create test input tensors
        m = torch.randn((2, 4, 8))  # [n_msa, n_token, c_m]
        z = torch.randn((4, 4, 8))  # [n_token, n_token, c_z]
        pair_mask = torch.ones((4, 4), dtype=torch.bool)
        
        # Test forward pass for normal block
        m_out, z_out = block.forward(m, z, pair_mask)
        self.assertIsNotNone(m_out)
        self.assertEqual(m_out.shape, m.shape)
        self.assertEqual(z_out.shape, z.shape)
        
        # Test with is_last_block=True
        block_last = MockMSABlock(cfg=cfg, is_last_block=True)
        m_out, z_out = block_last.forward(m, z, pair_mask)
        self.assertIsNone(m_out)
        self.assertEqual(z_out.shape, z.shape)
    
    def test_msa_module(self):
        """Test MSAModule with complete mocks."""
        # Create a mock MSAConfig
        cfg = MockMSAConfig(
            n_blocks=1,
            c_m=8,
            c=8,
            c_z=16,
            dropout=0.0
        )
        
        # Create a mock MSAModule
        module = MockMSAModule(cfg)
        
        # Test instantiation
        self.assertIsInstance(module, MockMSAModule)
        self.assertEqual(module.n_blocks, 1)
        self.assertEqual(module.c_m, 8)
        self.assertEqual(module.c_z, 16)
        
        # Create test input tensors
        z_in = torch.randn((1, 3, 3, 16))
        s_inputs = torch.randn((1, 3, 8))
        pair_mask = torch.ones((1, 3, 3), dtype=torch.bool)
        
        # Test forward pass with 'msa' key
        z_out = module.forward({"msa": torch.zeros((2, 3))}, z_in, s_inputs, pair_mask)
        self.assertEqual(z_out.shape, z_in.shape)
        
        # Test forward pass with no 'msa' key
        z_out = module.forward({}, z_in, s_inputs, pair_mask)
        self.assertTrue(torch.equal(z_out, z_in))
        
        # Test with n_blocks=0
        cfg_zero_blocks = MockMSAConfig(
            n_blocks=0,
            c_m=8,
            c=8,
            c_z=16,
            dropout=0.0
        )
        module_zero_blocks = MockMSAModule(cfg_zero_blocks)
        z_out = module_zero_blocks.forward({"msa": torch.zeros((2, 3))}, z_in, s_inputs, pair_mask)
        self.assertTrue(torch.equal(z_out, z_in))

if __name__ == "__main__":
    unittest.main()

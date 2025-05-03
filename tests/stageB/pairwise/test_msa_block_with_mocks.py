"""
Test MSABlock with mocked dependencies.
"""

import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from rna_predict.pipeline.stageB.pairwise.pairformer import MSABlock, MSAConfig

# Create mock classes for dependencies
class MockOuterProductMean(nn.Module):
    def __init__(self, c_m, c_z, c_hidden):
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden

    def forward(self, m, **kwargs):
        n_token = m.shape[-2]
        return torch.zeros((n_token, n_token, self.c_z))

class MockMSAStack(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_m = cfg.c_m

    def forward(self, m, z):
        return m

class MockPairformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_z = cfg.c_z

    def forward(self, s, z, pair_mask, **kwargs):
        return None, z

class TestMSABlockWithMocks(unittest.TestCase):
    """Test MSABlock with mocked dependencies."""

    @patch('rna_predict.pipeline.stageB.pairwise.pairformer.OuterProductMean', MockOuterProductMean)
    @patch('rna_predict.pipeline.stageB.pairwise.pairformer.MSAStack', MockMSAStack)
    @patch('rna_predict.pipeline.stageB.pairwise.pairformer.PairformerBlock', MockPairformerBlock)
    def test_msa_block_instantiate(self):
        """Test MSABlock instantiation with mocked dependencies."""
        # Create a complete MSAConfig with all required parameters
        cfg = MSAConfig(
            c_m=8,
            c=8,
            c_z=8,
            dropout=0.1,
            pair_dropout=0.25,
            # Additional parameters needed for PairformerBlock inside MSABlock
            n_heads=2,
            # Other required parameters
            n_blocks=1,
            enable=False,
            strategy="random",
            train_cutoff=512,
            test_cutoff=16384,
            train_lowerb=1,
            test_lowerb=1,
            c_s_inputs=8,
            blocks_per_ckpt=1,
            input_feature_dims={"msa": 32, "has_deletion": 1, "deletion_value": 1}
        )

        # Create MSABlock
        mb = MSABlock(cfg=cfg)
        self.assertIsInstance(mb, MSABlock)

        # Test forward pass
        m = torch.randn((2, 4, 8))  # [n_msa, n_token, c_m]
        z = torch.randn((4, 4, 8))  # [n_token, n_token, c_z]
        pair_mask = torch.ones((4, 4), dtype=torch.bool)

        # Test forward pass for normal block
        m_out, z_out = mb.forward(m, z, pair_mask)
        self.assertIsNotNone(m_out)
        self.assertEqual(m_out.shape, m.shape)
        self.assertEqual(z_out.shape, z.shape)

        # Create MSABlock with is_last_block=True
        mb_last = MSABlock(cfg=cfg, is_last_block=True)

        # Test forward pass for last block
        m_out, z_out = mb_last.forward(m, z, pair_mask)
        self.assertIsNone(m_out)
        self.assertEqual(z_out.shape, z.shape)

if __name__ == "__main__":
    unittest.main()

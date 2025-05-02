"""
Tests for MSA components using mocks.
"""

import unittest
from unittest.mock import patch

import torch

from rna_predict.pipeline.stageB.pairwise.pairformer import MSABlock, MSAModule
from rna_predict.conf.config_schema import MSAConfig

# Import mock components for testing
from tests.stageB.pairwise.mock_msa_components import (
    MockMSABlock,
    MockMSAStack,
    MockPairformerBlock,
    MockOuterProductMean
)

class TestMSAComponentsMock(unittest.TestCase):
    """Tests for MSA components using mocks."""

    @patch('rna_predict.pipeline.stageB.pairwise.pairformer.OuterProductMean', MockOuterProductMean)
    @patch('rna_predict.pipeline.stageB.pairwise.pairformer.MSAStack', MockMSAStack)
    @patch('rna_predict.pipeline.stageB.pairwise.pairformer.PairformerBlock', MockPairformerBlock)
    def test_msa_block_instantiate(self):
        """Test MSABlock instantiation with mocks."""
        # Create a minimal MSAConfig with only required parameters
        cfg = MSAConfig(
            c_m=8,
            c=8,
            c_z=8,
            dropout=0.1
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
        # The MockOuterProductMean adds a batch dimension, so we need to check the last 3 dimensions
        self.assertEqual(z_out.shape[-3:], z.shape)

        # Create MSABlock with is_last_block=True
        mb_last = MSABlock(cfg=cfg, is_last_block=True)

        # Test forward pass for last block
        m_out, z_out = mb_last.forward(m, z, pair_mask)
        self.assertIsNone(m_out)
        # The MockOuterProductMean adds a batch dimension, so we need to check the last 3 dimensions
        self.assertEqual(z_out.shape[-3:], z.shape)

    @patch('rna_predict.pipeline.stageB.pairwise.pairformer.MSABlock', MockMSABlock)
    @patch('rna_predict.pipeline.stageB.pairwise.pairformer.OuterProductMean', MockOuterProductMean)
    @patch('rna_predict.pipeline.stageB.pairwise.pairformer.PairformerBlock', MockPairformerBlock)
    def test_msa_module_instantiate(self):
        """Test MSAModule instantiation with mocks."""
        # Create a minimal MSAConfig with only required parameters
        cfg = MSAConfig(
            n_blocks=1,
            c_m=8,
            c=8,
            c_z=16,
            dropout=0.0
        )

        # Create MSAModule
        mm = MSAModule(cfg)
        self.assertIsInstance(mm, MSAModule)

        # Test forward pass with n_blocks=0
        cfg_zero_blocks = MSAConfig(
            n_blocks=0,
            c_m=8,
            c=8,
            c_z=16,
            dropout=0.0
        )
        mm_zero_blocks = MSAModule(cfg_zero_blocks)

        z_in = torch.randn((1, 3, 3, 16))
        s_inputs = torch.randn((1, 3, 8))
        pair_mask = torch.ones((1, 3, 3), dtype=torch.bool)

        # Create a complete input feature dictionary
        input_dict = {
            "msa": torch.zeros((2, 3), dtype=torch.long),
            "has_deletion": torch.zeros((2, 3), dtype=torch.bool),
            "deletion_value": torch.zeros((2, 3), dtype=torch.float32)
        }

        # Test forward pass with n_blocks=0
        z_out = mm_zero_blocks.forward(input_dict, z_in, s_inputs, pair_mask)
        self.assertTrue(torch.equal(z_out, z_in))

        # Test forward pass with no 'msa' key
        z_out = mm.forward({}, z_in, s_inputs, pair_mask)
        self.assertTrue(torch.equal(z_out, z_in))

        # Test forward pass with 'msa' key
        z_out = mm.forward(input_dict, z_in, s_inputs, pair_mask)
        self.assertEqual(z_out.shape, z_in.shape)

if __name__ == "__main__":
    unittest.main()

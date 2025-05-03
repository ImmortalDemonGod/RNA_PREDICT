"""
Minimal test for MSABlock to identify the issue.
"""

import unittest

import torch
import torch.nn as nn

from rna_predict.conf.config_schema import MSAConfig

# Create a mock OuterProductMean class
class MockOuterProductMean(nn.Module):
    def __init__(self, c_m, c_z, c_hidden):
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden

    def forward(self, m, **kwargs):
        n_token = m.shape[-2]
        return torch.zeros((n_token, n_token, self.c_z))

# Create a mock MSAStack class
class MockMSAStack(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_m = cfg.c_m

    def forward(self, m, z):
        return m

# Create a mock PairformerBlock class
class MockPairformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_z = cfg.c_z

    def forward(self, s, z, pair_mask, **kwargs):
        return None, z

# Create a minimal MSABlock class for testing
class MinimalMSABlock(nn.Module):
    def __init__(self, cfg, is_last_block=False):
        super().__init__()
        self.c_m = cfg.c_m
        self.c_z = cfg.c_z
        self.is_last_block = is_last_block

        # Create components
        self.outer_product_mean_msa = MockOuterProductMean(
            c_m=self.c_m, c_z=self.c_z, c_hidden=cfg.c
        )

        if not self.is_last_block:
            self.msa_stack = MockMSAStack(cfg)

        self.pair_stack = MockPairformerBlock(cfg)

    def forward(self, m, z, pair_mask, **kwargs):
        # Communication
        z = z + self.outer_product_mean_msa(m)

        if not self.is_last_block:
            # MSA stack
            m = self.msa_stack(m, z)

        # Pair stack
        _, z = self.pair_stack(None, z, pair_mask)

        if not self.is_last_block:
            return m, z
        else:
            return None, z

class TestMSABlockMinimal(unittest.TestCase):
    """Minimal test for MSABlock."""

    def test_minimal_msa_block(self):
        """Test a minimal MSABlock implementation."""
        # Create a minimal MSAConfig
        cfg = MSAConfig(
            c_m=8,
            c=8,
            c_z=8,
            dropout=0.1,
            pair_dropout=0.25,
            n_heads=2,
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

        # Create a minimal MSABlock
        block = MinimalMSABlock(cfg)

        # Create test input tensors
        m = torch.randn((2, 4, 8))  # [n_msa, n_token, c_m]
        z = torch.randn((4, 4, 8))  # [n_token, n_token, c_z]
        pair_mask = torch.ones((4, 4), dtype=torch.bool)

        # Test forward pass
        m_out, z_out = block.forward(m, z, pair_mask)

        # Check output shapes
        self.assertIsNotNone(m_out)
        self.assertEqual(m_out.shape, m.shape)
        self.assertEqual(z_out.shape, z.shape)

        # Test with is_last_block=True
        block_last = MinimalMSABlock(cfg, is_last_block=True)
        m_out, z_out = block_last.forward(m, z, pair_mask)

        # Check output shapes
        self.assertIsNone(m_out)
        self.assertEqual(z_out.shape, z.shape)

if __name__ == "__main__":
    unittest.main()

"""
Mock implementations of MSA components for testing.
"""

import torch
import torch.nn as nn

class MockMSABlock(nn.Module):
    """Mock implementation of MSABlock for testing."""

    def __init__(self, cfg, is_last_block=False):
        super(MockMSABlock, self).__init__()
        self.c_m = cfg.c_m
        self.c_z = cfg.c_z
        self.is_last_block = is_last_block

    def forward(self, m, z, pair_mask, **kwargs):
        """Mock forward pass that returns expected shapes."""
        if self.is_last_block:
            return None, z
        else:
            return m, z

class MockMSAPairWeightedAveraging(nn.Module):
    """Mock implementation of MSAPairWeightedAveraging for testing."""

    def __init__(self, cfg):
        super(MockMSAPairWeightedAveraging, self).__init__()
        self.c_m = cfg.c_m

    def forward(self, m, z):
        """Mock forward pass that returns expected shape."""
        return torch.zeros_like(m)

class MockMSAStack(nn.Module):
    """Mock implementation of MSAStack for testing."""

    def __init__(self, cfg):
        super(MockMSAStack, self).__init__()
        self.c_m = cfg.c_m

    def forward(self, m, z):
        """Mock forward pass that returns expected shape."""
        return torch.zeros_like(m)

class MockPairformerBlock(nn.Module):
    """Mock implementation of PairformerBlock for testing."""

    def __init__(self, cfg):
        super(MockPairformerBlock, self).__init__()
        self.c_z = cfg.c_z

    def forward(self, s, z, pair_mask, **kwargs):
        """Mock forward pass that returns expected shapes."""
        return None, torch.zeros_like(z)

class MockOuterProductMean(nn.Module):
    """Mock implementation of OuterProductMean for testing."""

    def __init__(self, c_m, c_z, c_hidden, eps=1e-3):
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps

        # Mock the layers
        self.layer_norm = nn.LayerNorm(c_m)
        self.linear_1 = nn.Linear(c_m, c_hidden)
        self.linear_2 = nn.Linear(c_m, c_hidden)
        self.linear_out = nn.Linear(c_hidden**2, c_z)

    def forward(self, m, mask=None, chunk_size=None, inplace_safe=False):
        """Mock forward pass that returns expected shape."""
        # Infer shape from m: [batch, n_msa, n_token, c_m] -> [batch, n_token, n_token, c_z]
        n_token = m.shape[-2]
        batch_dims = m.shape[:-2]
        return torch.zeros((*batch_dims, n_token, n_token, self.c_z), dtype=m.dtype, device=m.device)

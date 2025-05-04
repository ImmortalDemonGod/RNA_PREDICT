"""
Dummy implementation of the Pairformer model for testing.
"""

import torch
import torch.nn as nn


class DummyPairformerModel(nn.Module):
    """
    Dummy implementation of the Pairformer model for testing.
    """

    def __init__(self, device):
        super().__init__()
        # Require explicit device; do not default to CPU
        if device is None:
            raise ValueError("DummyPairformerModel requires an explicit device argument; do not use hardcoded defaults.")
        self.device = torch.device(device)

    def forward(self, *args, **kwargs):
        # Return dummy output with correct shape
        return torch.randn(1, 32, 32, 64, device=self.device)

    def to(self, device):
        """
        Move model to specified device.
        """
        self.device = torch.device(device)
        return super().to(device)

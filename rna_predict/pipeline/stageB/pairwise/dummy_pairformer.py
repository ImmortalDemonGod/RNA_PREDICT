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
        """
        Initializes the DummyPairformerModel with a specified device.
        
        Args:
        	device: The device on which the model will operate. Must not be None.
        
        Raises:
        	ValueError: If device is None.
        """
        super().__init__()
        # Require explicit device; do not default to CPU
        if device is None:
            raise ValueError("DummyPairformerModel requires an explicit device argument; do not use hardcoded defaults.")
        self.device = torch.device(device)

    def forward(self, *args, **kwargs):
        # Return dummy output with correct shape
        """
        Returns a dummy tensor with shape (1, 32, 32, 64) on the model's device.
        
        This method ignores all input arguments and is intended for testing purposes.
        """
        return torch.randn(1, 32, 32, 64, device=self.device)

    def to(self, device):
        """
        Move model to specified device.
        """
        self.device = torch.device(device)
        return super().to(device)

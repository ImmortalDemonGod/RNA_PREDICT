import torch
from torch import nn


class DummyTorsionModel(nn.Module):
    """
    A dummy model that returns zeros for torsion angles.
    This is used as a fallback when the actual model fails to load.
    """

    def __init__(self, device, num_angles: int = 7):
        super().__init__()
        if device is None:
            raise ValueError("DummyTorsionModel requires an explicit device argument; do not use hardcoded defaults.")
        self.device = torch.device(device)
        self.num_angles = num_angles

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a tensor of zeros with shape [N, 2*num_angles].
        """
        N = x.size(0)
        return torch.zeros(N, 2 * self.num_angles, device=self.device)

    def predict_angles_from_sequence(self, sequence: str) -> torch.Tensor:
        """
        Returns a tensor of zeros with shape [N, 2*num_angles].
        """
        N = len(sequence)
        return torch.zeros(N, 2 * self.num_angles, device=self.device)

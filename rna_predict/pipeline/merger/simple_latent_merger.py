from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class LatentInputs:
    adjacency: torch.Tensor
    angles: torch.Tensor
    s_emb: torch.Tensor
    z_emb: torch.Tensor
    partial_coords: Optional[torch.Tensor] = None

class SimpleLatentMerger(torch.nn.Module):
    """
    Merges adjacency, angles, single embeddings, pair embeddings,
    plus partial coords, into a single per-residue latent.
    """
    def __init__(self, dim_angles: int, dim_s: int, dim_z: int, dim_out: int):
        super().__init__()
        self.expected_dim_angles = dim_angles
        self.expected_dim_s = dim_s
        self.expected_dim_z = dim_z
        self.dim_out = dim_out
        in_dim = dim_angles + dim_s + dim_z
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, dim_out),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_out, dim_out),
        )

    def forward(self, inputs: LatentInputs):
        angles = inputs.angles
        s_emb = inputs.s_emb
        z_emb = inputs.z_emb
        z_pooled = z_emb.mean(dim=1)
        actual_dim_angles = angles.shape[-1]
        actual_dim_s = s_emb.shape[-1]
        actual_dim_z = z_pooled.shape[-1]
        total_in_dim = actual_dim_angles + actual_dim_s + actual_dim_z
        if self.mlp[0].in_features != total_in_dim:
            print(
                f"[Debug] Creating MLP with dimensions: {total_in_dim} -> {self.dim_out}"
            )
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(total_in_dim, self.dim_out),
                torch.nn.ReLU(),
                torch.nn.Linear(self.dim_out, self.dim_out),
            ).to(angles.device)
        elif self.mlp[0].weight.device != angles.device:
            self.mlp = self.mlp.to(angles.device)
        x = torch.cat([angles, s_emb, z_pooled], dim=-1)
        out = self.mlp(x)
        return out

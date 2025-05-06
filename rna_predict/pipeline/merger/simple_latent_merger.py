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
    def __init__(self, dim_angles: int, dim_s: int, dim_z: int, dim_out: int, device=None):
        """
        Initializes the SimpleLatentMerger module with specified input and output dimensions.
        
        Args:
        	dim_angles: Dimension of the angle features.
        	dim_s: Dimension of the single embeddings.
        	dim_z: Dimension of the pair embeddings.
        	dim_out: Output dimension of the merged latent representation.
        	device: Optional device to move the module to upon initialization.
        """
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
        if device is not None:
            self.to(device)

    def forward(self, inputs: LatentInputs):
        """
        Merges latent input tensors into a unified per-residue representation.
        
        Extracts angle, single, and pair embedding tensors from the input, pools the pair embeddings, concatenates all features, and processes them through an MLP to produce the merged output tensor. Dynamically adjusts the MLP input layer if input dimensions change.
        
        Args:
            inputs: LatentInputs containing angle, single, and pair embedding tensors.
        
        Returns:
            A tensor of merged per-residue latent representations.
        """
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
            )
            # Move to same device as current module
            self.mlp = self.mlp.to(next(self.parameters()).device)
        x = torch.cat([angles, s_emb, z_pooled], dim=-1)
        out = self.mlp(x)
        return out

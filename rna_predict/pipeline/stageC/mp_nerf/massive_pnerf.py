from dataclasses import dataclass
from typing import Tuple, Union, cast

import numpy as np

# diff ml
import torch


def get_axis_matrix(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, norm: bool = True
) -> torch.Tensor:
    """Gets an orthonomal basis as a matrix of [e1, e2, e3].
    Useful for constructing rotation matrices between planes
    according to the first answer here:
    https://math.stackexchange.com/questions/1876615/rotation-matrix-from-plane-a-to-b
    Inputs:
    * a: (batch, 3) or (3, ). point(s) of the plane
    * b: (batch, 3) or (3, ). point(s) of the plane
    * c: (batch, 3) or (3, ). point(s) of the plane
    Outputs: orthonomal basis as a matrix of [e1, e2, e3]. calculated as:
        * e1_ = (c-b)
        * e2_proto = (b-a)
        * e3_ = e1_ ^ e2_proto
        * e2_ = e3_ ^ e1_
        * basis = normalize_by_vectors( [e1_, e2_, e3_] )
    Note: Could be done more by Grahm-Schmidt and extend to N-dimensions
          but this is faster and more intuitive for 3D.
    """
    v1_ = c - b
    v2_ = b - a
    v3_ = torch.cross(v1_, v2_, dim=-1)
    v2_ready = torch.cross(v3_, v1_, dim=-1)
    basis: torch.Tensor = torch.stack([v1_, v2_ready, v3_], dim=-2)
    # normalize if needed
    if norm:
        # Add small epsilon to avoid division by zero
        norm_values = torch.norm(basis, dim=-1, keepdim=True)
        # Clamp to avoid division by zero
        norm_values = torch.clamp(norm_values, min=1e-10)
        result = basis / norm_values
        # Explicitly cast to assure mypy of the type
        return cast(torch.Tensor, result)
    return basis


@dataclass
class MpNerfParams:
    """Parameters for the mp_nerf_torch function."""

    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor
    bond_length: Union[torch.Tensor, float]
    theta: Union[torch.Tensor, float]
    chi: Union[torch.Tensor, float]


def _calculate_perturbed_vectors(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculates ba and cb vectors, perturbing them if norms are near zero."""
    ba = b - a
    cb = c - b

    ba_norm = torch.norm(ba, dim=-1, keepdim=True)  # Keep dim for broadcasting
    cb_norm = torch.norm(cb, dim=-1, keepdim=True)  # Keep dim for broadcasting

    needs_ba_perturb = ba_norm < 1e-10
    needs_cb_perturb = cb_norm < 1e-10

    # Check if any perturbation is needed before creating the tensor
    if torch.any(needs_ba_perturb) or torch.any(needs_cb_perturb):
        perturb = torch.tensor([1e-10, 1e-10, 1e-10], device=ba.device, dtype=ba.dtype)
        # Apply perturbation using torch.where for batch safety
        ba = torch.where(needs_ba_perturb, ba + perturb, ba)
        cb = torch.where(needs_cb_perturb, cb + perturb, cb)

    return ba, cb


def _calculate_rotation_matrix(ba: torch.Tensor, cb: torch.Tensor) -> torch.Tensor:
    """Calculates the normalized rotation matrix based on ba and cb vectors."""
    # calc rotation matrix. based on plane normals and normalized
    n_plane = torch.cross(ba, cb, dim=-1)

    # Check if cross product resulted in zero vector (collinear ba and cb)
    n_plane_norm = torch.norm(
        n_plane, dim=-1, keepdim=True
    )  # Keep dim for broadcasting

    # Use torch.where to handle potential collinearity safely in batches
    needs_perturb = n_plane_norm < 1e-10
    if torch.any(needs_perturb):
        # Add small perturbation to ba to avoid collinearity if needed
        perturb = torch.tensor([0.0, 1e-10, 1e-10], device=ba.device, dtype=ba.dtype)
        safe_ba = torch.where(needs_perturb, ba + perturb, ba)
        # Recalculate cross product only where needed
        n_plane = torch.where(needs_perturb, torch.cross(safe_ba, cb, dim=-1), n_plane)
        # Update n_plane_norm for the perturbed cases
        n_plane_norm = torch.where(
            needs_perturb, torch.norm(n_plane, dim=-1, keepdim=True), n_plane_norm
        )

    # Ensure n_plane_norm is not zero before division (clamp needed after potential recalculation)
    n_plane_norm = torch.clamp(n_plane_norm, min=1e-10)
    n_plane = n_plane / n_plane_norm  # Normalize n_plane safely

    # Calculate the final component of the basis
    n_plane_ = torch.cross(n_plane, cb, dim=-1)
    # Normalize n_plane_ safely
    n_plane_norm_ = torch.norm(n_plane_, dim=-1, keepdim=True)
    n_plane_norm_ = torch.clamp(n_plane_norm_, min=1e-10)
    n_plane_ = n_plane_ / n_plane_norm_

    # Normalize cb safely
    cb_norm = torch.norm(cb, dim=-1, keepdim=True)
    cb_norm = torch.clamp(cb_norm, min=1e-10)
    cb_normalized = cb / cb_norm

    # Stack the normalized basis vectors
    rotate = torch.stack([cb_normalized, n_plane_, n_plane], dim=-1)

    # Final check for rotation matrix validity (optional, but good practice)
    # determinant = torch.det(rotate)
    # if not torch.allclose(determinant, torch.ones_like(determinant)):
    #     print("Warning: Rotation matrix determinant is not close to 1.")

    return rotate


def mp_nerf_torch(params: MpNerfParams) -> torch.Tensor:
    """Custom Natural extension of Reference Frame.
    Inputs:
    * params: MpNerfParams object containing all geometric inputs.
    Outputs: d (batch, 3) or (float). the next point in the sequence, linked to params.c
    """
    # Extract parameters for convenience (optional, but can maintain readability)
    a, b, c = params.a, params.b, params.c
    bond_length, theta, chi = params.bond_length, params.theta, params.chi

    # Ensure inputs that can be float or Tensor are Tensors for torch operations
    if isinstance(bond_length, float):
        bond_length = torch.tensor(bond_length, device=a.device, dtype=a.dtype)
    if isinstance(theta, float):
        theta = torch.tensor(theta, device=a.device, dtype=a.dtype)
    if isinstance(chi, float):
        chi = torch.tensor(chi, device=a.device, dtype=a.dtype)

    # Validate inputs
    if torch.isnan(a).any() or torch.isnan(b).any() or torch.isnan(c).any():
        raise ValueError("Input coordinates contain NaN values")
    if (
        torch.isnan(bond_length).any()
        or torch.isnan(theta).any()
        or torch.isnan(chi).any()
    ):
        raise ValueError("Input angles or bond length contain NaN values")

    # safety check for theta
    if not ((-np.pi <= theta) * (theta <= np.pi)).all().item():
        # Clamp theta to valid range instead of raising error
        theta = torch.clamp(theta, -np.pi, np.pi)

    # Calculate potentially perturbed vectors ba and cb
    ba, cb = _calculate_perturbed_vectors(a, b, c)

    # Calculate the rotation matrix
    rotate = _calculate_rotation_matrix(ba, cb)

    # calc proto point, rotate. add (-1 for sidechainnet convention)
    # https://github.com/jonathanking/sidechainnet/issues/14
    d = torch.stack(
        [
            -torch.cos(theta),
            torch.sin(theta) * torch.cos(chi),
            torch.sin(theta) * torch.sin(chi),
        ],
        dim=-1,
    ).unsqueeze(-1)

    # extend base point, set length
    # bond_length is guaranteed to be a Tensor here due to the conversion above
    result = c + bond_length.unsqueeze(-1) * torch.matmul(rotate, d).squeeze()

    # Final validation
    if torch.isnan(result).any():
        raise ValueError("NaN values in output coordinates")

    return result


def scn_rigid_index_mask(seq: str, c_alpha: bool = False) -> torch.Tensor:
    """
    Returns indices for rigid body frames in sidechainnet format.

    Args:
        seq (str): Amino acid sequence in one-letter code
        c_alpha (bool): If True, use only C-alpha atoms for frames

    Returns:
        torch.Tensor: Indices for selecting three points to define a rigid frame
    """
    seq_len = len(seq)

    if seq_len < 3:
        # For sequences shorter than 3 residues, return an empty tensor
        return torch.empty(0, dtype=torch.long)

    if c_alpha:
        # For C-alpha trace, return indices of the first 3 C-alpha atoms
        # In SCN format, C-alpha is at index 1 within each residue block of 14
        return torch.tensor([1, 15, 29], dtype=torch.long)  # First 3 C-alpha atoms
    else:
        # For non-C-alpha mode, use N, CA, C of the first residue
        return torch.tensor([0, 1, 2], dtype=torch.long)  # N, CA, C of first residue

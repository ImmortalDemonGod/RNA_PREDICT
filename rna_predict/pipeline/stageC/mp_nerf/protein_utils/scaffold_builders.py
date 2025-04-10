"""
Utilities for building and modifying protein structure scaffolds.
"""

from typing import Dict, Optional, Union

import numpy as np
import torch

# Imports needed by the moved functions
# Assuming utils is in the parent mp_nerf directory
from rna_predict.pipeline.stageC.mp_nerf.utils import get_angle, get_dihedral

from .mask_generators import (
    make_theta_mask,
    make_torsion_mask,
    scn_bond_mask,
    scn_cloud_mask,
    scn_index_mask,
)

# Note: BB_BUILD_INFO is used by protein_fold, not directly here, but might be needed if helpers are extracted later.
# from .sidechain_data import BB_BUILD_INFO


def _create_cloud_and_bond_masks(
    seq: str, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create cloud and bond masks from sequence.

    Args:
        seq: String of amino acid one-letter codes
        device: Device to put tensors on
        dtype: Data type for tensors

    Returns:
        Tuple of (cloud_mask, bond_mask)
    """
    cloud_mask_np = scn_cloud_mask(seq)
    cloud_mask = torch.tensor(cloud_mask_np, device=device, dtype=torch.bool)

    bond_mask_np = scn_bond_mask(seq)
    bond_mask = torch.tensor(bond_mask_np, device=device, dtype=dtype)

    return cloud_mask, bond_mask


def _create_point_ref_mask(
    seq: str, seq_len: int, device: torch.device
) -> torch.Tensor:
    """Create point reference mask from sequence.

    Args:
        seq: String of amino acid one-letter codes
        seq_len: Length of sequence
        device: Device to put tensors on

    Returns:
        Point reference mask tensor
    """
    point_ref_mask_np = scn_index_mask(seq)
    point_ref_mask = torch.tensor(point_ref_mask_np, device=device, dtype=torch.long)

    # Handle empty sequence for permute
    if seq_len > 0:
        point_ref_mask = point_ref_mask.permute(2, 0, 1)
    else:
        point_ref_mask = point_ref_mask.reshape(3, 0, 11)

    return point_ref_mask


def _create_angles_mask(
    seq: str, seq_len: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Create angles mask from sequence.

    Args:
        seq: String of amino acid one-letter codes
        seq_len: Length of sequence
        device: Device to put tensors on
        dtype: Data type for tensors

    Returns:
        Angles mask tensor
    """
    theta_masks = [make_theta_mask(aa) for aa in seq]
    torsion_masks = [make_torsion_mask(aa, fill=True) for aa in seq]

    if seq_len > 0:
        theta_mask_stacked = torch.tensor(
            np.array(theta_masks), device=device, dtype=dtype
        )
        torsion_mask_stacked = torch.tensor(
            np.array(torsion_masks), device=device, dtype=dtype
        )
    else:
        theta_mask_stacked = torch.empty((0, 14), device=device, dtype=dtype)
        torsion_mask_stacked = torch.empty((0, 14), device=device, dtype=dtype)

    return torch.stack([theta_mask_stacked, torsion_mask_stacked], dim=0)


def build_scaffolds_from_scn_angles(
    seq: str,
    angles: Optional[torch.Tensor] = None,
    coords: Optional[torch.Tensor] = None,  # Kept for API compatibility, unused
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, torch.Tensor]:
    """Build scaffolds from SCN angles.

    Args:
        seq: String of amino acid one-letter codes
        angles: Optional tensor of angles (shape: L, 12)
        coords: Optional tensor of coordinates (shape: L, 14, 3) - kept for API compatibility, unused
        device: Device to put tensors on

    Returns:
        dict: Dictionary containing cloud_mask, point_ref_mask, angles_mask, and bond_mask
    """
    seq_len = len(seq)
    device = torch.device(device)  # Ensure device is a torch.device

    # Handle angles input - Raise error if None, as tests imply dependency
    if angles is None:
        raise ValueError("Input 'angles' tensor cannot be None for scaffold building.")

    dtype = angles.dtype  # Use dtype from input angles

    # Create masks
    cloud_mask, bond_mask = _create_cloud_and_bond_masks(seq, device, dtype)
    point_ref_mask = _create_point_ref_mask(seq, seq_len, device)
    angles_mask = _create_angles_mask(seq, seq_len, device, dtype)

    # Handle empty sequence for cloud_mask shape
    if seq_len == 0:
        cloud_mask = cloud_mask.reshape(0, 14)

    return {
        "cloud_mask": cloud_mask,
        "point_ref_mask": point_ref_mask,
        "angles_mask": angles_mask,
        "bond_mask": bond_mask,
    }


def modify_scaffolds_with_coords(
    scaffolds: Dict[str, torch.Tensor], coords: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Gets scaffolds and fills in the right data.
    Inputs:
    * scaffolds: dict. as returned by `build_scaffolds_from_scn_angles`
    * coords: (L, 14, 3). sidechainnet tensor. same device as scaffolds
    Outputs: corrected scaffolds
    """

    # calculate distances and update:
    # N, CA, C
    scaffolds["bond_mask"][1:, 0] = torch.norm(
        coords[1:, 0] - coords[:-1, 2], dim=-1
    )  # N
    scaffolds["bond_mask"][:, 1] = torch.norm(coords[:, 1] - coords[:, 0], dim=-1)  # CA
    scaffolds["bond_mask"][:, 2] = torch.norm(coords[:, 2] - coords[:, 1], dim=-1)  # C
    # O, CB, side chain
    selector = np.arange(len(coords))
    for i in range(3, 14):
        # get indexes
        idx_a, idx_b, idx_c = scaffolds["point_ref_mask"][
            :, :, i - 3
        ]  # (3, L, 11) -> 3 * (L, 11)
        # correct distances
        scaffolds["bond_mask"][:, i] = torch.norm(
            coords[:, i] - coords[selector, idx_c], dim=-1
        )
        # get angles
        scaffolds["angles_mask"][0, :, i] = get_angle(
            coords[selector, idx_b], coords[selector, idx_c], coords[:, i]
        )
        # handle C-beta, where the C requested is from the previous aa
        if i == 4:
            # If sequence length is 1, there's no previous residue.
            if len(coords) > 1:
                # for 1st residue, use position of the second residue's N
                first_next_n = coords[1, :1]  # 1, 3
                # the c requested is from the previous residue
                main_c_prev_idxs = coords[selector[:-1], idx_a[1:]]  # (L-1), 3
                # concat
                coords_a = torch.cat([first_next_n, main_c_prev_idxs])
            else:
                # Handle seq_len=1 case for C-beta dihedral calculation
                coords_a = coords[
                    selector, idx_a
                ]  # Fallback, might not be chemically correct but avoids index error
        else:
            coords_a = coords[selector, idx_a]
        # get dihedrals
        scaffolds["angles_mask"][1, :, i] = get_dihedral(
            coords_a, coords[selector, idx_b], coords[selector, idx_c], coords[:, i]
        )
    # correct angles and dihedrals for backbone
    if len(coords) > 1:  # Check length before slicing
        scaffolds["angles_mask"][0, :-1, 0] = get_angle(
            coords[:-1, 1], coords[:-1, 2], coords[1:, 0]
        )  # ca_c_n
        scaffolds["angles_mask"][0, 1:, 1] = get_angle(
            coords[:-1, 2], coords[1:, 0], coords[1:, 1]
        )  # c_n_ca
        # N determined by previous psi = f(n, ca, c, n+1)
        scaffolds["angles_mask"][1, :-1, 0] = get_dihedral(
            coords[:-1, 0], coords[:-1, 1], coords[:-1, 2], coords[1:, 0]
        )
        # CA determined by omega = f(ca, c, n+1, ca+1)
        scaffolds["angles_mask"][1, 1:, 1] = get_dihedral(
            coords[:-1, 1], coords[:-1, 2], coords[1:, 0], coords[1:, 1]
        )
        # C determined by phi = f(c-1, n, ca, c)
        scaffolds["angles_mask"][1, 1:, 2] = get_dihedral(
            coords[:-1, 2], coords[1:, 0], coords[1:, 1], coords[1:, 2]
        )

    # Angle N-CA-C is always present
    scaffolds["angles_mask"][0, :, 2] = get_angle(
        coords[:, 0], coords[:, 1], coords[:, 2]
    )  # n_ca_c

    return scaffolds


def modify_angles_mask_with_torsions(
    angles_mask: torch.Tensor, torsions: torch.Tensor
) -> torch.Tensor:
    """
    Modifies the angles mask with torsion angles.

    Args:
        angles_mask: (2, L, 14) tensor containing bond angles and torsion angles
        torsions: (L, 3) tensor containing backbone torsion angles (phi, psi, omega)

    Returns:
        Modified angles_mask with updated torsion angles
    """
    # Copy the angles mask to avoid modifying the original
    modified_mask = angles_mask.clone()

    seq_len = torsions.shape[0]
    if seq_len == 0:
        return modified_mask  # Handle empty sequence

    # Update torsion angles for backbone atoms
    # N determined by previous psi: angles_mask[1, i, 0] = psi(i)
    if seq_len > 1:
        modified_mask[1, :-1, 0] = torsions[:-1, 1]  # psi

    # CA determined by omega: angles_mask[1, i, 1] = omega(i)
    if seq_len > 1:
        modified_mask[1, 1:, 1] = torsions[1:, 2]  # omega(i) affects CA(i+1)

    # C determined by phi: angles_mask[1, i, 2] = phi(i)
    if seq_len > 1:
        modified_mask[1, 1:, 2] = torsions[1:, 0]  # phi(i) affects C(i)

    return modified_mask

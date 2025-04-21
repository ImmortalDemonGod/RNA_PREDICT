"""
RNA scaffolding functions for MP-NeRF implementation.
"""

import math

import torch

from .rna_constants import BACKBONE_ATOMS, BACKBONE_INDEX_MAP
from ..final_kb_rna import (
    RNA_CONNECT,
    get_bond_angle,
    get_bond_length,
)

def build_scaffolds_rna_from_torsions(
    seq: str,
    torsions: torch.Tensor,
    device: str = "cpu",
    sugar_pucker: str = "C3'-endo",
) -> dict:
    """Build RNA scaffolds from torsion angles.

    Args:
        seq: RNA sequence string
        torsions: Tensor of shape (L, 7) containing torsion angles
        device: Device to place tensors on
        sugar_pucker: Sugar pucker conformation ("C3'-endo" or "C2'-endo")

    Returns:
        Dictionary containing scaffolds information
    """
    # Validate sequence
    valid_bases = set("ACGU")
    if not all(base in valid_bases for base in seq):
        raise ValueError(f"Invalid sequence: {seq}. Must only contain A, C, G, U.")

    # Get sequence length and number of backbone atoms
    L = len(seq)
    B = len(BACKBONE_ATOMS)

    # Validate torsions shape
    if torsions.shape[0] < L:
        raise ValueError(
            f"Not enough torsion angles for sequence. Expected {L}, got {torsions.shape[0]}"
        )

    # Initialize tensors
    bond_mask = torch.zeros((L, B), dtype=torch.float32, device=device)
    angles_mask = torch.zeros((2, L, B), dtype=torch.float32, device=device)
    point_ref_mask = torch.zeros((3, L, B), dtype=torch.long, device=device)
    cloud_mask = torch.ones((L, B), dtype=torch.bool, device=device)

    # Create backbone triplets from backbone connections
    backbone_triplets = []
    for i in range(len(RNA_CONNECT["backbone"]) - 2):
        atom1 = RNA_CONNECT["backbone"][i][0]
        atom2 = RNA_CONNECT["backbone"][i][1]
        atom3 = RNA_CONNECT["backbone"][i + 1][1]
        backbone_triplets.append((atom1, atom2, atom3))

    # Fill bond lengths and angles
    for i, base in enumerate(seq):
        # Fill bond lengths
        for j, (atom1, atom2) in enumerate(RNA_CONNECT["backbone"]):
            if j >= B:  # Skip if we've reached the end of our tensor
                break
            bond_name = f"{atom1}-{atom2}"
            bond_length = get_bond_length(bond_name, sugar_pucker=sugar_pucker)
            if bond_length is not None:
                bond_mask[i, j] = bond_length

        # Fill bond angles
        for j, (atom1, atom2, atom3) in enumerate(backbone_triplets):
            if j >= B:  # Skip if we've reached the end of our tensor
                break
            angle_name = f"{atom1}-{atom2}-{atom3}"
            angle_deg = get_bond_angle(
                angle_name, sugar_pucker=sugar_pucker, degrees=True
            )
            if angle_deg is not None:
                angles_mask[0, i, j] = angle_deg * (math.pi / 180.0)

        # Fill dihedral angles from torsions
        if torsions.size(1) >= 7:
            alpha, beta, gamma, delta, eps, zeta, chi = torsions[i]
            angles_mask[1, i, 1] = alpha * (math.pi / 180.0)  # alpha
            angles_mask[1, i, 2] = beta * (math.pi / 180.0)  # beta
            angles_mask[1, i, 3] = gamma * (math.pi / 180.0)  # gamma
            angles_mask[1, i, 4] = delta * (math.pi / 180.0)  # delta
            angles_mask[1, i, 5] = eps * (math.pi / 180.0)  # epsilon
            angles_mask[1, i, 6] = zeta * (math.pi / 180.0)  # zeta
            angles_mask[1, i, 9] = chi * (math.pi / 180.0)  # chi

        # Fill point reference indices
        for j in range(B):
            if j == 0:  # P atom
                if i == 0:
                    point_ref_mask[:, i, j] = torch.tensor([0, 1, 2], device=device)
                else:
                    point_ref_mask[:, i, j] = torch.tensor(
                        [
                            (i - 1) * B + BACKBONE_INDEX_MAP["C4'"],
                            (i - 1) * B + BACKBONE_INDEX_MAP["C3'"],
                            (i - 1) * B + BACKBONE_INDEX_MAP["O3'"],
                        ],
                        device=device,
                    )
            else:
                point_ref_mask[:, i, j] = torch.tensor(
                    [i * B + (j - 3), i * B + (j - 2), i * B + (j - 1)], device=device
                )

    # Initialize scaffolds dictionary
    scaffolds = {
        "seq": seq,
        "torsions": torsions[:L],  # Only use the torsions we need
        "device": device,
        "sugar_pucker": sugar_pucker,
        "bond_mask": bond_mask,
        "angles_mask": angles_mask,
        "point_ref_mask": point_ref_mask,
        "cloud_mask": cloud_mask,
    }

    # Return scaffolds
    return scaffolds

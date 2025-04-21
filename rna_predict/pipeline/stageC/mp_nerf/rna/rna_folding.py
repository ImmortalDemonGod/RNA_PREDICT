"""
RNA folding functions for MP-NeRF implementation.
"""

import math
import logging
from typing import Any, Dict

import torch

from .rna_constants import BACKBONE_ATOMS
from .rna_atom_positioning import calculate_atom_position
from ..final_kb_rna import (
    get_bond_angle,
    get_bond_length,
)

def rna_fold(
    scaffolds: Dict[str, Any], device: str = "cpu", do_ring_closure: bool = False
) -> torch.Tensor:
    """
    Fold the RNA sequence into 3D coordinates using MP-NeRF method.

    Args:
        scaffolds: Dictionary containing bond masks, angle masks, etc.
        device: Device to place tensors on ('cpu' or 'cuda')
        do_ring_closure: Whether to perform ring closure refinement

    Returns:
        Tensor of shape [L, B, 3] where L is sequence length and B is number of backbone atoms
    """
    logger = logging.getLogger("rna_predict.pipeline.stageC.mp_nerf.rna_fold")
    logger.debug("[rna_fold] Function entry.")
    # Log scaffold input statistics
    for key, value in scaffolds.items():
        if isinstance(value, torch.Tensor):
            # Ensure value is float or complex for .mean() and .min()/.max()
            if value.is_floating_point() or value.is_complex():
                logger.debug(f"scaffolds['{key}']: shape={value.shape}, min={value.min().item() if value.numel() else 'NA'}, max={value.max().item() if value.numel() else 'NA'}, mean={value.mean().item() if value.numel() else 'NA'}, nan={torch.isnan(value).any().item()}")
            else:
                logger.debug(f"scaffolds['{key}']: shape={value.shape}, dtype={value.dtype}, min={value.min().item() if value.numel() else 'NA'}, max={value.max().item() if value.numel() else 'NA'}, nan={(value != value).any().item() if value.numel() else 'NA'} (non-float tensor)")
        else:
            logger.debug(f"scaffolds['{key}']: type={type(value)}")

    # Validate input
    if not isinstance(scaffolds, dict):
        raise ValueError("scaffolds must be a dictionary")

    # Get sequence length and number of backbone atoms
    L = scaffolds["bond_mask"].shape[0]
    B = scaffolds["bond_mask"].shape[1]

    # Initialize coordinates tensor with proper shape
    coords = torch.zeros((L, B, 3), device=device)
    logger.debug(f"Initialized coords: shape={coords.shape}, nan={torch.isnan(coords).any().item()}")

    # Place first residue atoms
    coords[0, 0] = torch.tensor([0.0, 0.0, 0.0], device=device)  # P
    coords[0, 1] = torch.tensor([1.0, 0.0, 0.0], device=device)  # O5'
    coords[0, 2] = torch.tensor([1.5, 1.0, 0.0], device=device)

    # Place remaining atoms
    for i in range(1, L):
        for j in range(B):
            if j == 0:  # P atom
                prev_o3 = coords[i - 1, 8]  # O3' is at index 8
                logger.debug(f"[i={i},j={j}] prev_o3 before nan_to_num: {prev_o3}")
                prev_o3 = torch.nan_to_num(prev_o3, nan=0.0, posinf=1e6, neginf=-1e6)
                logger.debug(f"[i={i},j={j}] prev_o3 after nan_to_num: {prev_o3}")
                coords[i, j] = prev_o3 + torch.tensor([1.0, 0.0, 0.0], device=device)
                logger.debug(f"[i={i},j={j}] coords[i,j] after assignment: {coords[i, j]}")
            else:
                prev_atom = coords[i, j - 1]
                if j >= 2:
                    prev_prev_atom = coords[i, j - 2]
                else:
                    prev_prev_atom = coords[
                        i - 1, -1
                    ]  # Use last atom of previous residue
                # Clamp any NaN/Inf values in reference atoms
                prev_atom = torch.nan_to_num(
                    prev_atom, nan=0.0, posinf=1e6, neginf=-1e6
                )
                prev_prev_atom = torch.nan_to_num(
                    prev_prev_atom, nan=0.0, posinf=1e6, neginf=-1e6
                )
                logger.debug(f"[i={i},j={j}] prev_atom: {prev_atom}, prev_prev_atom: {prev_prev_atom}")
                # Get bond length and angle
                bond_length = get_bond_length(
                    f"{BACKBONE_ATOMS[j - 1]}-{BACKBONE_ATOMS[j]}"
                )
                if bond_length is None or torch.isnan(torch.tensor(bond_length)).any():
                    bond_length = 1.5  # Default bond length
                bond_length = torch.tensor(bond_length, device=device)
                bond_angle = get_bond_angle(
                    f"{BACKBONE_ATOMS[j - 2]}-{BACKBONE_ATOMS[j - 1]}-{BACKBONE_ATOMS[j]}"
                )
                if bond_angle is None or torch.isnan(torch.tensor(bond_angle)).any():
                    bond_angle = 109.5  # Default tetrahedral angle in degrees
                bond_angle = torch.tensor(bond_angle * (math.pi / 180.0), device=device)
                # Get torsion angle
                torsion_angle = (
                    scaffolds["torsions"][i, j - 3]
                    if j >= 3
                    else torch.tensor(0.0, device=device)
                )
                if torch.isnan(torsion_angle).any():
                    torsion_angle = torch.tensor(0.0, device=device)
                torsion_angle = torsion_angle * (math.pi / 180.0)
                logger.debug(f"[i={i},j={j}] bond_length: {bond_length}, bond_angle: {bond_angle}, torsion_angle: {torsion_angle}")
                # Calculate new position
                try:
                    new_pos = calculate_atom_position(
                        prev_prev_atom,
                        prev_atom,
                        bond_length,
                        bond_angle,
                        torsion_angle,
                        device,
                    )
                    # Clamp any NaN/Inf values in new position
                    new_pos = torch.nan_to_num(
                        new_pos, nan=0.0, posinf=1e6, neginf=-1e6
                    )
                    coords[i, j] = new_pos
                    logger.debug(f"[i={i},j={j}] new_pos: {new_pos}, coords[i,j]: {coords[i, j]}")
                except (RuntimeError, ValueError) as e:
                    logger.error(f"[i={i},j={j}] Exception in calculate_atom_position: {e}")
                    coords[i, j] = prev_atom + torch.tensor(
                        [1.0, 0.0, 0.0], device=device
                    )
                    logger.debug(f"[i={i},j={j}] coords[i,j] after fallback: {coords[i, j]}")
    # Final sanitization of coordinates
    coords = torch.nan_to_num(coords, nan=0.0, posinf=1e6, neginf=-1e6)
    logger.debug(f"Final coords: shape={coords.shape}, min={coords.min().item()}, max={coords.max().item()}, mean={coords.mean().item()}, nan={torch.isnan(coords).any().item()}")

    # Apply ring closure refinement if requested
    if do_ring_closure:
        coords = ring_closure_refinement(coords)

    return coords


def ring_closure_refinement(coords: torch.Tensor) -> torch.Tensor:
    """
    Placeholder. We could do a small iterative approach to ensure the
    ribose ring closes properly for the sugar pucker.
    Currently returns coords as-is.
    """
    return coords

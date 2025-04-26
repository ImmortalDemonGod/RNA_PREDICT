"""
RNA folding functions for MP-NeRF implementation.
"""

import math
import logging
from typing import Any, Dict, List
import torch

from .rna_constants import BACKBONE_ATOMS
from .rna_atom_positioning import calculate_atom_position
from ..final_kb_rna import (
    get_bond_angle,
    get_bond_length,
)

#@snoop
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

    # Instrument: print requires_grad and grad_fn for input torsions
    print("[DEBUG-GRAD] scaffolds['torsions'] requires_grad:", getattr(scaffolds['torsions'], 'requires_grad', None))
    print("[DEBUG-GRAD] scaffolds['torsions'] grad_fn:", getattr(scaffolds['torsions'], 'grad_fn', None))

    # Build coordinates as nested lists to avoid in-place ops
    coords_list: List[List[torch.Tensor]] = []
    for i in range(L):
        residue_coords = []
        if i == 0:
            # First residue: hardcoded positions (not differentiable, but unavoidable for first three atoms)
            residue_coords.append(torch.zeros(3, device=device, dtype=scaffolds["torsions"].dtype))
            residue_coords.append(torch.ones(3, device=device, dtype=scaffolds["torsions"].dtype) * torch.tensor([1.0, 0.0, 0.0], device=device, dtype=scaffolds["torsions"].dtype))
            residue_coords.append(torch.tensor([1.5, 1.0, 0.0], device=device, dtype=scaffolds["torsions"].dtype))
            for j in range(3, B):
                residue_coords.append(torch.zeros(3, device=device, dtype=scaffolds["torsions"].dtype))
        else:
            for j in range(B):
                if j == 0:
                    # Instead of copying/offset, always use calculate_atom_position with differentiable torsion for first atom
                    prev_prev_atom = coords_list[i-1][6]  # e.g., use O5' from previous residue
                    prev_atom = coords_list[i-1][8]      # e.g., use O3' from previous residue
                    if prev_prev_atom.device != torch.device(device):
                        prev_prev_atom = prev_prev_atom.to(device)
                    if prev_atom.device != torch.device(device):
                        prev_atom = prev_atom.to(device)
                    bond_length_val = get_bond_length("O5'-P")
                    bond_length = torch.full((), bond_length_val if bond_length_val is not None else 1.5, dtype=scaffolds["torsions"].dtype, device=device)
                    bond_angle_val = get_bond_angle("O3'-O5'-P")
                    bond_angle = torch.full((), (bond_angle_val if bond_angle_val is not None else 109.5) * (math.pi / 180.0), dtype=scaffolds["torsions"].dtype, device=device)
                    torsion_angle = scaffolds["torsions"][i, 0] * (math.pi / 180.0)
                    if torsion_angle.device != torch.device(device):
                        torsion_angle = torsion_angle.to(device)
                    new_coord = calculate_atom_position(
                        prev_prev_atom,
                        prev_atom,
                        bond_length,
                        bond_angle,
                        torsion_angle,
                        device,
                    )
                    print(f"[DEBUG-GRAD] i={i},j={j} (P, differentiable) requires_grad: {new_coord.requires_grad}, grad_fn: {new_coord.grad_fn}")
                    residue_coords.append(new_coord)
                else:
                    prev_atom = residue_coords[j-1]
                    if prev_atom.device != torch.device(device):
                        prev_atom = prev_atom.to(device)
                    if j >= 2:
                        prev_prev_atom = residue_coords[j-2]
                    else:
                        prev_prev_atom = coords_list[i-1][-1]
                    if prev_prev_atom.device != torch.device(device):
                        prev_prev_atom = prev_prev_atom.to(device)
                    print(f"[DEBUG-GRAD] i={i},j={j} input prev_atom.requires_grad: {prev_atom.requires_grad}, prev_prev_atom.requires_grad: {prev_prev_atom.requires_grad}")
                    bond_length_val = get_bond_length(f"{BACKBONE_ATOMS[j-1]}-{BACKBONE_ATOMS[j]}")
                    bond_length = torch.full((), bond_length_val if bond_length_val is not None else 1.5, dtype=scaffolds["torsions"].dtype, device=device)
                    bond_angle_val = get_bond_angle(f"{BACKBONE_ATOMS[j-2]}-{BACKBONE_ATOMS[j-1]}-{BACKBONE_ATOMS[j]}")
                    bond_angle = torch.full((), (bond_angle_val if bond_angle_val is not None else 109.5) * (math.pi / 180.0), dtype=scaffolds["torsions"].dtype, device=device)
                    if j >= 3:
                        torsion_angle = scaffolds["torsions"][i, j-3] * (math.pi / 180.0)
                        if torsion_angle.device != torch.device(device):
                            torsion_angle = torsion_angle.to(device)
                    else:
                        torsion_angle = torch.zeros((), dtype=scaffolds["torsions"].dtype, device=device)
                    print(f"[DEBUG-GRAD] i={i},j={j} input bond_length.requires_grad: {bond_length.requires_grad}, bond_angle.requires_grad: {bond_angle.requires_grad}, torsion_angle.requires_grad: {torsion_angle.requires_grad}")
                    new_pos = calculate_atom_position(
                        prev_prev_atom,
                        prev_atom,
                        bond_length,
                        bond_angle,
                        torsion_angle,
                        device,
                    )
                    print(f"[DEBUG-GRAD] i={i},j={j} (calc) requires_grad: {new_pos.requires_grad}, grad_fn: {new_pos.grad_fn}")
                    residue_coords.append(new_pos)
        coords_list.append(residue_coords)
        # Print after residue
        print(f"[DEBUG-GRAD] residue {i} last atom requires_grad: {residue_coords[-1].requires_grad}, grad_fn: {residue_coords[-1].grad_fn}")
    # Stack to tensor
    coords_tensor = torch.stack([torch.stack(res, dim=0) for res in coords_list], dim=0)
    print("[DEBUG-GRAD] coords_tensor requires_grad:", coords_tensor.requires_grad)
    print("[DEBUG-GRAD] coords_tensor grad_fn:", coords_tensor.grad_fn)

    # Apply ring closure refinement if requested
    if do_ring_closure:
        coords_tensor = ring_closure_refinement(coords_tensor)

    return coords_tensor


def ring_closure_refinement(coords: torch.Tensor) -> torch.Tensor:
    """
    Placeholder. We could do a small iterative approach to ensure the
    ribose ring closes properly for the sugar pucker.
    Currently returns coords as-is.
    """
    return coords

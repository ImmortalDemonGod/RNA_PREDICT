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

##@snoop
def rna_fold(
    scaffolds: Dict[str, Any], device: str = "cpu", do_ring_closure: bool = False, debug_logging: bool = False
) -> torch.Tensor:
    """
    Fold the RNA sequence into 3D coordinates using MP-NeRF method.

    Args:
        scaffolds: Dictionary containing bond masks, angle masks, etc.
        device: Device to place tensors on ('cpu' or 'cuda')
        do_ring_closure: Whether to perform ring closure refinement
        debug_logging: Whether to enable debug logging

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
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("[DEBUG-GRAD] scaffolds['torsions'] requires_grad: %s, grad_fn: %s" % (getattr(scaffolds['torsions'], 'requires_grad', None), getattr(scaffolds['torsions'], 'grad_fn', None)))

    # Build coordinates as nested lists to avoid in-place ops
    coords_list: List[List[torch.Tensor]] = []
    for i in range(L):
        residue_coords = []
        # SYSTEMATIC DEBUGGING: Log start of residue folding for residue 3
        if i == 3 and debug_logging:
            logger.debug(f"[DEBUG-RNAFOLD-RES3] Folding residue 3: torsions={scaffolds['torsions'][3]}")
        if i == 0:
            # First residue: seed enough atoms for geometric propagation
            # Use canonical geometry for first 9 atoms (P, O5', C5', C4', O4', C3', O3', C2', C1')
            # Values from 1a34_1_B.cif (canonical/experimental geometry):
            residue_coords.append(torch.tensor([13.774, 39.026, 15.251], device=device, dtype=scaffolds["torsions"].dtype))  # P
            residue_coords.append(torch.tensor([14.872, 37.934, 15.708], device=device, dtype=scaffolds["torsions"].dtype))   # O5'
            residue_coords.append(torch.tensor([14.938, 37.437, 17.082], device=device, dtype=scaffolds["torsions"].dtype))   # C5'
            residue_coords.append(torch.tensor([16.189, 36.563, 17.364], device=device, dtype=scaffolds["torsions"].dtype))   # C4'
            residue_coords.append(torch.tensor([17.279, 37.002, 16.486], device=device, dtype=scaffolds["torsions"].dtype))   # O4'
            residue_coords.append(torch.tensor([16.731, 36.539, 18.805], device=device, dtype=scaffolds["torsions"].dtype))   # C3'
            residue_coords.append(torch.tensor([16.075, 35.38, 19.521], device=device, dtype=scaffolds["torsions"].dtype))    # O3'
            residue_coords.append(torch.tensor([18.276, 36.491, 18.593], device=device, dtype=scaffolds["torsions"].dtype))   # C2'
            residue_coords.append(torch.tensor([18.527, 37.062, 17.172], device=device, dtype=scaffolds["torsions"].dtype))   # C1'
            for j in range(9, B):
                residue_coords.append(torch.zeros(3, device=device, dtype=scaffolds["torsions"].dtype))
        else:
            for j in range(B):
                # SYSTEMATIC DEBUGGING: Log intermediate atom coordinates for residue 3
                if i == 3 and debug_logging:
                    logger.debug(f"[DEBUG-RNAFOLD-RES3] j={j} (before placement): prev_atom={residue_coords[j-1] if j > 0 else 'N/A'}")
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
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"[DEBUG-GRAD] i={i},j={j} (P, differentiable) requires_grad: {new_coord.requires_grad}, grad_fn: {new_coord.grad_fn}")
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
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"[DEBUG-GRAD] i={i},j={j} input prev_atom.requires_grad: {prev_atom.requires_grad}, prev_prev_atom.requires_grad: {prev_prev_atom.requires_grad}")
                    bond_length_val = get_bond_length(f"{BACKBONE_ATOMS[j-1]}-{BACKBONE_ATOMS[j]}")
                    bond_length = torch.full((), bond_length_val if bond_length_val is not None else 1.5, dtype=scaffolds["torsions"].dtype, device=device)

                    if j >= 2:
                        angle_triplet = (
                            f"{BACKBONE_ATOMS[j-2]}-"
                            f"{BACKBONE_ATOMS[j-1]}-"
                            f"{BACKBONE_ATOMS[j]}"
                        )
                    else:
                        angle_triplet = f"{BACKBONE_ATOMS[0]}-{BACKBONE_ATOMS[j-1]}-{BACKBONE_ATOMS[j]}"

                    bond_angle_val = get_bond_angle(angle_triplet)
                    bond_angle = torch.full((), (bond_angle_val if bond_angle_val is not None else 109.5) * (math.pi / 180.0), dtype=scaffolds["torsions"].dtype, device=device)
                    if j >= 3:
                        torsion_angle = scaffolds["torsions"][i, j-3] * (math.pi / 180.0)
                        if torsion_angle.device != torch.device(device):
                            torsion_angle = torsion_angle.to(device)
                    else:
                        torsion_angle = torch.zeros((), dtype=scaffolds["torsions"].dtype, device=device)
                    # SYSTEMATIC DEBUGGING: Log all inputs to calculate_atom_position for residue 3
                    if i == 3 and debug_logging:
                        logger.debug(f"[DEBUG-RNAFOLD-RES3] j={j} call: prev_atom={prev_atom}, prev_prev_atom={prev_prev_atom}, bond_length={bond_length}, bond_angle={bond_angle}, torsion_angle={torsion_angle}")
                    new_pos = calculate_atom_position(
                        prev_prev_atom,
                        prev_atom,
                        bond_length,
                        bond_angle,
                        torsion_angle,
                        device,
                    )
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"[DEBUG-GRAD] i={i},j={j} (calc) requires_grad: {new_pos.requires_grad}, grad_fn: {new_pos.grad_fn}")
                    residue_coords.append(new_pos)
        # SYSTEMATIC DEBUGGING: Log final residue coordinates for residue 0 and 1
        if i == 0 and debug_logging:
            logger.debug(f"[DEBUG-RNAFOLD-RES0] residue_coords = {residue_coords}")
        if i == 1 and debug_logging:
            logger.debug(f"[DEBUG-RNAFOLD-RES1] residue_coords = {residue_coords}")
        # SYSTEMATIC DEBUGGING: Log final residue coordinates for residue 2
        if i == 2 and debug_logging:
            logger.debug(f"[DEBUG-RNAFOLD-RES2] residue_coords = {residue_coords}")
            if len(residue_coords) > 8 and debug_logging:
                logger.debug(f"[DEBUG-RNAFOLD-RES2] O5' (coords_list[2][6]) = {residue_coords[6]}")
                logger.debug(f"[DEBUG-RNAFOLD-RES2] O3' (coords_list[2][8]) = {residue_coords[8]}")
        # SYSTEMATIC DEBUGGING: Log final residue coordinates for residue 3
        if i == 3 and debug_logging:
            logger.debug(f"[DEBUG-RNAFOLD-RES3] residue_coords = {residue_coords}")
        coords_list.append(residue_coords)
        # Print after residue
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[DEBUG-GRAD] residue {i} last atom requires_grad: {residue_coords[-1].requires_grad}, grad_fn: {residue_coords[-1].grad_fn}")
    # Stack to tensor
    coords_tensor = torch.stack([torch.stack(res, dim=0) for res in coords_list], dim=0)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"[DEBUG-GRAD] coords_tensor (rna_fold output) requires_grad: {getattr(coords_tensor, 'requires_grad', None)}, grad_fn: {getattr(coords_tensor, 'grad_fn', None)}")

    # --- DEBUG: Check requires_grad and grad_fn after folding ---
    if isinstance(coords_tensor, torch.Tensor):
        logger.debug(f"[GRAD-TRACE-FOLDING] coords_tensor.requires_grad: {coords_tensor.requires_grad}, grad_fn: {coords_tensor.grad_fn}")

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

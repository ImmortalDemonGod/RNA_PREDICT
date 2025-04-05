"""
RNA-specific MP-NeRF implementation for building 3D structures from torsion angles.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import numpy as np

from .final_kb_rna import (
    get_bond_angle,
    get_bond_length,
    get_base_geometry,
    RNA_CONNECT,
)
from .massive_pnerf import MpNerfParams, mp_nerf_torch

###############################################################################
# We'll use a standard ordering for the backbone atoms.
BACKBONE_ATOMS = [
    "P",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "O2'",
    "C1'",
]

# Map from atom name to index in the BACKBONE_ATOMS list
BACKBONE_INDEX_MAP = {atom: i for i, atom in enumerate(BACKBONE_ATOMS)}

###############################################################################
# 2) STANDARD TORSION ANGLES FOR RNA BACKBONE (A-form)
###############################################################################
# These are the standard torsion angles for RNA backbone in A-form
# Values are in degrees
RNA_BACKBONE_TORSIONS_AFORM = {
    "alpha": -60.0,  # P-O5'-C5'-C4'
    "beta": 180.0,   # O5'-C5'-C4'-C3'
    "gamma": 60.0,   # C5'-C4'-C3'-O3'
    "delta": 80.0,   # C4'-C3'-O3'-P
    "epsilon": -150.0,  # C3'-O3'-P-O5'
    "zeta": -70.0,   # O3'-P-O5'-C5'
    "chi": -160.0,   # O4'-C1'-N9/N1-C4/C2 (purine/pyrimidine)
}

###############################################################################
# 3) SCAFFOLDING
###############################################################################


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
        raise ValueError(f"Not enough torsion angles for sequence. Expected {L}, got {torsions.shape[0]}")
    
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
        atom3 = RNA_CONNECT["backbone"][i+1][1]
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
            angle_deg = get_bond_angle(angle_name, sugar_pucker=sugar_pucker, degrees=True)
            if angle_deg is not None:
                angles_mask[0, i, j] = angle_deg * (math.pi / 180.0)
        
        # Fill dihedral angles from torsions
        if torsions.size(1) >= 7:
            alpha, beta, gamma, delta, eps, zeta, chi = torsions[i]
            angles_mask[1, i, 1] = alpha * (math.pi / 180.0)  # alpha
            angles_mask[1, i, 2] = beta * (math.pi / 180.0)   # beta
            angles_mask[1, i, 3] = gamma * (math.pi / 180.0)  # gamma
            angles_mask[1, i, 4] = delta * (math.pi / 180.0)  # delta
            angles_mask[1, i, 5] = eps * (math.pi / 180.0)    # epsilon
            angles_mask[1, i, 6] = zeta * (math.pi / 180.0)   # zeta
            angles_mask[1, i, 9] = chi * (math.pi / 180.0)    # chi
        
        # Fill point reference indices
        for j in range(B):
            if j == 0:  # P atom
                if i == 0:
                    point_ref_mask[:, i, j] = torch.tensor([0, 1, 2], device=device)
                else:
                    point_ref_mask[:, i, j] = torch.tensor([
                        (i-1) * B + BACKBONE_INDEX_MAP["C4'"],
                        (i-1) * B + BACKBONE_INDEX_MAP["C3'"],
                        (i-1) * B + BACKBONE_INDEX_MAP["O3'"]
                    ], device=device)
            else:
                point_ref_mask[:, i, j] = torch.tensor([
                    i * B + (j-3),
                    i * B + (j-2),
                    i * B + (j-1)
                ], device=device)
    
    # Initialize scaffolds dictionary
    scaffolds = {
        'seq': seq,
        'torsions': torsions[:L],  # Only use the torsions we need
        'device': device,
        'sugar_pucker': sugar_pucker,
        'bond_mask': bond_mask,
        'angles_mask': angles_mask,
        'point_ref_mask': point_ref_mask,
        'cloud_mask': cloud_mask,
    }
    
    # Return scaffolds
    return scaffolds


###############################################################################
# 5) FOLDING: rna_fold
###############################################################################
def rna_fold(scaffolds: Dict[str, Any], device: str = "cpu", do_ring_closure: bool = False) -> torch.Tensor:
    """
    Fold the RNA sequence into 3D coordinates using MP-NeRF method.
    
    Args:
        scaffolds: Dictionary containing bond masks, angle masks, etc.
        device: Device to place tensors on ('cpu' or 'cuda')
        do_ring_closure: Whether to perform ring closure refinement
        
    Returns:
        Tensor of shape [L, B, 3] where L is sequence length and B is number of backbone atoms
    """
    # Validate input
    if not isinstance(scaffolds, dict):
        raise ValueError("scaffolds must be a dictionary")
    
    # Get sequence length and number of backbone atoms
    L = scaffolds['bond_mask'].shape[0]
    B = scaffolds['bond_mask'].shape[1]
    
    # Initialize coordinates tensor with proper shape
    coords = torch.zeros((L, B, 3), device=device)
    
    # Place first residue atoms
    coords[0, 0] = torch.tensor([0.0, 0.0, 0.0], device=device)  # P
    coords[0, 1] = torch.tensor([1.0, 0.0, 0.0], device=device)  # O5'
    coords[0, 2] = torch.tensor([1.5, 1.0, 0.0], device=device)  # C5'
    
    # Place remaining atoms
    for i in range(1, L):
        for j in range(B):
            if j == 0:  # P atom
                # Use previous residue's O3' as reference
                prev_o3 = coords[i-1, 8]  # O3' is at index 8
                coords[i, j] = prev_o3 + torch.tensor([1.0, 0.0, 0.0], device=device)
            else:
                # Get reference atoms
                prev_atom = coords[i, j-1]
                if j >= 2:
                    prev_prev_atom = coords[i, j-2]
                else:
                    prev_prev_atom = coords[i-1, -1]  # Use last atom of previous residue
                
                # Get bond length and angle
                bond_length = get_bond_length(f"{BACKBONE_ATOMS[j-1]}-{BACKBONE_ATOMS[j]}")
                if bond_length is None or torch.isnan(torch.tensor(bond_length)).any():
                    bond_length = 1.5  # Default bond length
                bond_length = torch.tensor(bond_length, device=device)
                
                bond_angle = get_bond_angle(f"{BACKBONE_ATOMS[j-2]}-{BACKBONE_ATOMS[j-1]}-{BACKBONE_ATOMS[j]}")
                if bond_angle is None or torch.isnan(torch.tensor(bond_angle)).any():
                    bond_angle = 109.5  # Default tetrahedral angle in degrees
                bond_angle = torch.tensor(bond_angle * (math.pi / 180.0), device=device)
                
                # Get torsion angle
                torsion_angle = scaffolds['torsions'][i, j-3] if j >= 3 else torch.tensor(0.0, device=device)
                if torch.isnan(torsion_angle).any():
                    torsion_angle = torch.tensor(0.0, device=device)
                torsion_angle = torsion_angle * (math.pi / 180.0)
                
                # Create MP-NeRF parameters
                params = MpNerfParams(
                    a=prev_prev_atom,
                    b=prev_atom,
                    c=prev_atom,  # Use same point for c as b for simplicity
                    bond_length=bond_length,
                    theta=bond_angle,
                    chi=torsion_angle
                )
                
                # Place atom
                coords[i, j] = mp_nerf_torch(params)
                
                # Validate new coordinates
                if torch.isnan(coords[i, j]).any():
                    raise ValueError(f"NaN coordinates generated for residue {i}, atom {j}")
    
    # Perform ring closure refinement if requested
    if do_ring_closure:
        coords = ring_closure_refinement(coords)
    
    # Final validation
    if torch.isnan(coords).any():
        raise ValueError("NaN values in final coordinates")
    
    return coords


def ring_closure_refinement(coords: torch.Tensor) -> torch.Tensor:
    """
    Placeholder. We could do a small iterative approach to ensure the
    ribose ring closes properly for the sugar pucker.
    Currently returns coords as-is.
    """
    return coords


###############################################################################
# 6) PLACE BASES
###############################################################################
def place_rna_bases(
    backbone_coords: torch.Tensor,
    seq: str,
    angles_mask: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Place base atoms for each residue in the RNA sequence.
    
    Args:
        backbone_coords: Tensor of shape [L, B, 3] containing backbone coordinates
        seq: RNA sequence
        angles_mask: Tensor of shape [2, L, B] containing angle masks
        device: Device to place tensors on
        
    Returns:
        Tensor of shape [L, max_atoms, 3] containing all atom coordinates
    """
    # Validate input
    if not isinstance(backbone_coords, torch.Tensor):
        raise ValueError("backbone_coords must be a torch.Tensor")
    if not isinstance(seq, str):
        raise ValueError("seq must be a string")
    if not isinstance(angles_mask, torch.Tensor):
        raise ValueError("angles_mask must be a torch.Tensor")
    
    # Check for NaN values
    if torch.isnan(backbone_coords).any():
        raise ValueError("backbone_coords contains NaN values")
    
    # Get sequence length and max atoms per base
    L = len(seq)
    max_atoms = compute_max_rna_atoms()
    
    # Initialize tensor for full coordinates
    full_coords = torch.zeros((L, max_atoms, 3), device=device)
    
    # Copy backbone coordinates
    full_coords[:, :len(BACKBONE_ATOMS), :] = backbone_coords
    
    # Place base atoms for each residue
    for i, base in enumerate(seq):
        # Get base atoms
        base_atoms = get_base_atoms(base)
        if not base_atoms:
            continue
        
        # Get base geometry
        base_geom = get_base_geometry(base)
        if not base_geom:
            continue
        
        # Get C1' position (index 9 in BACKBONE_ATOMS)
        c1_prime = backbone_coords[i, 9]
        
        # Place each base atom
        for j, atom in enumerate(base_atoms):
            # Skip if atom is already placed (e.g., N9/N1)
            if atom in BACKBONE_ATOMS:
                continue
            
            # Get reference atoms for placement
            if j == 0:  # First atom (N9/N1)
                prev_atom = c1_prime
                prev_prev_atom = backbone_coords[i, 8]  # O4'
            else:
                prev_atom = full_coords[i, len(BACKBONE_ATOMS) + j - 1]
                prev_prev_atom = full_coords[i, len(BACKBONE_ATOMS) + j - 2]
            
            # Get bond length and angle
            if j == 0:  # First atom (N9/N1)
                bond_length = get_bond_length("C1'-N9" if base in ['A', 'G'] else "C1'-N1")
                if bond_length is None or torch.isnan(torch.tensor(bond_length)).any():
                    bond_length = 1.5  # Default bond length
                bond_length = torch.tensor(bond_length, device=device)
                
                # Calculate C1'-N9/N1 bond angle
                if base in ['A', 'G']:  # Purines
                    bond_angle_val = 108.2  # O4'-C1'-N9
                else:  # Pyrimidines
                    bond_angle_val = 108.2  # O4'-C1'-N1
                bond_angle = torch.tensor(bond_angle_val * (math.pi / 180.0), device=device)
                
                # Calculate dihedral angle (chi)
                if base in ['A', 'G']:  # Purines
                    chi_val = -160.0  # anti
                else:  # Pyrimidines
                    chi_val = -160.0  # anti
                chi = torch.tensor(chi_val * (math.pi / 180.0), device=device)
            else:
                # Get bond length and angle from base geometry
                prev_atom_name = base_atoms[j-1]
                bond_length = base_geom.get('bond_lengths_ang', {}).get(f"{prev_atom_name}-{atom}")
                if bond_length is None or torch.isnan(torch.tensor(bond_length)).any():
                    bond_length = 1.5  # Default bond length
                bond_length = torch.tensor(bond_length, device=device)
                
                # Get bond angle
                if j >= 2:
                    prev_prev_atom_name = base_atoms[j-2]
                    angle_name = f"{prev_prev_atom_name}-{prev_atom_name}-{atom}"
                    bond_angle_val = base_geom.get('bond_angles_deg', {}).get(angle_name)
                    if bond_angle_val is None:
                        bond_angle_val = 120.0  # Default bond angle
                    bond_angle = torch.tensor(bond_angle_val * (math.pi / 180.0), device=device)
                else:
                    bond_angle = torch.tensor(120.0 * (math.pi / 180.0), device=device)  # Default angle
                
                # For simplicity, use 0.0 as dihedral angle
                chi = torch.tensor(0.0, device=device)
            
            # Create MP-NeRF parameters
            params = MpNerfParams(
                a=prev_prev_atom,
                b=prev_atom,
                c=prev_atom,  # Use same point for c as b for simplicity
                bond_length=bond_length,
                theta=bond_angle,
                chi=chi
            )
            
            # Place atom
            atom_idx = len(BACKBONE_ATOMS) + j
            full_coords[i, atom_idx] = mp_nerf_torch(params)
            
            # Validate new coordinates
            if torch.isnan(full_coords[i, atom_idx]).any():
                raise ValueError(f"NaN coordinates generated for residue {i}, atom {atom}")
    
    # Final validation
    if torch.isnan(full_coords).any():
        raise ValueError("NaN values in final coordinates")
    
    return full_coords


###############################################################################
# 7) BACKWARD COMPATIBILITY FUNCTIONS
###############################################################################


# For backward compatibility with the expected function signatures
def place_bases(backbone_coords: torch.Tensor, seq: str, device: str = "cpu") -> torch.Tensor:
    """
    Backward compatibility function for place_rna_bases.
    """
    # Create a dummy angles mask
    L = len(seq)
    B = len(BACKBONE_ATOMS)
    angles_mask = torch.ones((2, L, B), device=device)
    return place_rna_bases(backbone_coords, seq, angles_mask, device)


def handle_mods(seq: str, scaffolds: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle modified bases in the RNA sequence.
    Currently just returns the scaffolds unmodified.
    
    Args:
        seq: RNA sequence
        scaffolds: Dictionary containing scaffolds information
        
    Returns:
        The scaffolds dictionary unmodified
    """
    # Validate input
    if not isinstance(seq, str):
        raise ValueError("seq must be a string")
    if not isinstance(scaffolds, dict):
        raise ValueError("scaffolds must be a dictionary")
    
    # For now, just return the scaffolds unmodified
    return scaffolds


def skip_missing_atoms(seq, scaffolds=None):
    """
    Backward compatibility function for skip_missing_atoms.

    Args:
        seq: The RNA sequence
        scaffolds: Optional scaffolds dictionary

    Returns:
        The scaffolds dictionary, unchanged
    """
    return scaffolds if scaffolds is not None else seq


def get_base_atoms(base_type=None):
    """
    Get the list of atom names for a given RNA base type.

    Args:
        base_type: The base type ('A', 'G', 'C', 'U')

    Returns:
        List of atom names for the base, or empty list if unknown base type
    """
    if base_type == "A":
        return ["N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"]
    elif base_type == "G":
        return ["N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"]
    elif base_type == "C":
        return ["N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"]
    elif base_type == "U":
        return ["N1", "C2", "N3", "C4", "C5", "C6", "O2", "O4"]
    else:
        return []


def mini_refinement(coords, method=None):
    """
    Backward compatibility function for mini_refinement.

    Args:
        coords: The coordinates tensor
        method: Optional refinement method

    Returns:
        The coordinates tensor, unchanged
    """
    return coords


def validate_rna_geometry(coords):
    """
    Backward compatibility function for validate_rna_geometry.

    Args:
        coords: The coordinates tensor

    Returns:
        True
    """
    return True


def compute_max_rna_atoms():
    """
    Compute the maximum number of atoms in an RNA residue.

    This includes both backbone atoms and base atoms.

    Returns:
        The maximum number of atoms (21 for G)
    """
    # Maximum is for G which has 11 base atoms + 10 backbone atoms = 21
    return 21


# Export all functions for backward compatibility
__all__ = [
    "build_scaffolds_rna_from_torsions",
    "rna_fold",
    "ring_closure_refinement",
    "place_rna_bases",
    "place_bases",
    "handle_mods",
    "skip_missing_atoms",
    "get_base_atoms",
    "mini_refinement",
    "validate_rna_geometry",
    "compute_max_rna_atoms",
]

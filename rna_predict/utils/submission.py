"""Helper functions for handling RNA structure submission data."""
from __future__ import annotations

import pandas as pd
import torch
from torch import Tensor


def reshape_coords(coords: Tensor, num_residues: int) -> Tensor:
    """Reshape coordinates to standard [N, atoms, 3] format, or return flat if variable atom counts.
    
    Args:
        coords: Input coordinates tensor of shape [N, 3], [N*atoms, 3], or [N, atoms, 3]
        num_residues: Number of residues (N) expected in reshaped tensor
        
    Returns:
        Tensor of shape [N, atoms, 3] where atoms is inferred from input shape, or [total_atoms, 3] if variable atoms
        
    Raises:
        ValueError: If coords shape cannot be normalized to [N, atoms, 3] and is not flat [total_atoms, 3]
    """
    if coords.dim() == 2 and coords.shape[0] == num_residues:
        # Shape [N, 3] -> [N, 1, 3] (single atom per residue)
        return coords.unsqueeze(1)
        
    if coords.dim() == 2 and coords.shape[0] == num_residues * 3:
        # Legacy shape [N*3, 3] -> [N, 3, 3] (three atoms per residue)
        return coords.view(num_residues, 3, 3)
        
    if coords.dim() == 2:
        atoms_per_res = coords.shape[0] // num_residues
        if coords.shape[0] % num_residues == 0:
            try:
                return coords.view(num_residues, atoms_per_res, 3)
            except RuntimeError:
                pass  # Fall through to ragged
        # Variable atom count per residue: return flat, let downstream handle mapping
        return coords  # shape [total_atoms, 3]
        
    if coords.dim() == 3:
        return coords
        
    raise ValueError(
        f"Shape mismatch: cannot reshape coords {coords.shape} for {num_residues} residues."
    )


def extract_atom(coords: Tensor, atom_idx: int) -> Tensor:
    """Extract coordinates for a specific atom from each residue.
    
    Args:
        coords: Input coordinates tensor of shape [N, atoms, 3]
        atom_idx: Index of atom to extract from each residue
        
    Returns:
        Tensor of shape [N, 3] containing selected atom coordinates
        
    Raises:
        IndexError: If atom_idx is out of bounds
    """
    try:
        return coords[:, atom_idx, :]
    except IndexError as e:
        raise IndexError(
            f"Invalid atom_idx {atom_idx} for coords shape {coords.shape}"
        ) from e


def coords_to_df(
    sequence: str, 
    coords: Tensor, 
    prediction_repeats: int = 5
) -> pd.DataFrame:
    """Convert coordinates to submission DataFrame format.
    
    Args:
        sequence: RNA sequence string
        coords: Coordinates tensor of shape [N, 3]
        prediction_repeats: Number of times to repeat coordinates
        
    Returns:
        DataFrame with columns:
            - ID: 1-based residue index
            - resname: nucleotide character
            - resid: 1-based residue index
            - x_1..x_n, y_1..y_n, z_1..z_n: Repeated coordinates
    """
    if not sequence:
        # Handle empty sequence
        cols = ["ID", "resname", "resid"]
        for i in range(1, prediction_repeats + 1):
            cols.extend([f"x_{i}", f"y_{i}", f"z_{i}"])
        return pd.DataFrame(columns=cols)

    # Create base data
    N = len(sequence)
    base_data = {
        "ID": range(1, N + 1),
        "resname": list(sequence),
        "resid": range(1, N + 1)
    }
    
    # Handle NaN values in coordinates
    is_nan = torch.isnan(coords)
    x_val = torch.where(is_nan[:, 0], float("nan"), coords[:, 0]).tolist()
    y_val = torch.where(is_nan[:, 1], float("nan"), coords[:, 1]).tolist()
    z_val = torch.where(is_nan[:, 2], float("nan"), coords[:, 2]).tolist()
    
    # Add repeated coordinates
    for i in range(1, prediction_repeats + 1):
        base_data[f"x_{i}"] = x_val
        base_data[f"y_{i}"] = y_val
        base_data[f"z_{i}"] = z_val
        
    return pd.DataFrame(base_data) 
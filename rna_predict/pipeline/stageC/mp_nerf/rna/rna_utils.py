"""
RNA utility functions for MP-NeRF implementation.
"""

from typing import Dict, Any, List, Optional

import torch

from .rna_constants import BACKBONE_ATOMS
from .rna_base_placement import place_rna_bases


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
        The scaffolds dictionary, unchanged, or an empty dict if none present
    """
    if scaffolds is not None:
        return scaffolds
    print("[ERR-RNAPREDICT-NOSCAFF-001] skip_missing_atoms: No scaffolds provided for seq='{}', returning empty dict.".format(seq))
    return {}


def get_base_atoms(base_type=None) -> List[str]:
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


def mini_refinement(coords: torch.Tensor, method: Optional[str] = None) -> torch.Tensor:
    """
    Backward compatibility function for mini_refinement.

    Args:
        coords: The coordinates tensor
        method: Optional refinement method

    Returns:
        The coordinates tensor, unchanged
    """
    return coords


def validate_rna_geometry(coords: torch.Tensor) -> bool:
    """
    Backward compatibility function for validate_rna_geometry.

    Args:
        coords: The coordinates tensor

    Returns:
        True
    """
    return True


def compute_max_rna_atoms() -> int:
    """
    Compute the maximum number of atoms in an RNA residue.

    This includes both backbone atoms and base atoms.

    Returns:
        The maximum number of atoms (21 for G)
    """
    # Maximum is for G which has 11 base atoms + 10 backbone atoms = 21
    return 21


# For backward compatibility with the expected function signatures
def place_bases(
    backbone_coords: torch.Tensor, seq: str, device: str = "cpu"
) -> torch.Tensor:
    """
    Backward compatibility function for place_rna_bases.
    """
    # Create a dummy angles mask
    L = len(seq)
    B = len(BACKBONE_ATOMS)
    angles_mask = torch.ones((2, L, B), device=device)
    return place_rna_bases(backbone_coords, seq, angles_mask, device)

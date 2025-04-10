"""
Utilities for accessing amino acid build information from sidechain_data.
"""

import typing
from typing import Any, List

# Assuming sidechain_data is in the same directory or accessible
from .sidechain_data import SC_BUILD_INFO


def get_rigid_frames(aa: str) -> List[List[int]]:
    """Get rigid frame indices for a given amino acid."""
    # SC_BUILD_INFO[aa] might return different list types based on the key
    return typing.cast(List[List[int]], SC_BUILD_INFO[aa]["rigid-frames-idxs"])


def get_atom_names(aa: str) -> List[str]:
    """Get atom names for a given amino acid."""
    return typing.cast(List[str], SC_BUILD_INFO[aa]["atom-names"])


def get_bond_names(aa: str) -> List[str]:
    """Get bond names for a given amino acid."""
    return typing.cast(List[str], SC_BUILD_INFO[aa]["bonds-names"])


def get_bond_types(aa: str) -> List[str]:
    """Get bond types for a given amino acid."""
    return typing.cast(List[str], SC_BUILD_INFO[aa]["bonds-types"])


def get_bond_values(aa: str) -> List[float]:
    """Get bond values for a given amino acid."""
    return typing.cast(List[float], SC_BUILD_INFO[aa]["bonds-vals"])


def get_angle_names(aa: str) -> List[str]:
    """Get angle names for a given amino acid."""
    return typing.cast(List[str], SC_BUILD_INFO[aa]["angles-names"])


def get_angle_types(aa: str) -> List[str]:
    """Get angle types for a given amino acid."""
    return typing.cast(List[str], SC_BUILD_INFO[aa]["angles-types"])


def get_angle_values(aa: str) -> List[float]:
    """Get angle values for a given amino acid."""
    return typing.cast(List[float], SC_BUILD_INFO[aa]["angles-vals"])


def get_torsion_names(aa: str) -> List[str]:
    """Get torsion names for a given amino acid."""
    return typing.cast(List[str], SC_BUILD_INFO[aa]["torsion-names"])


def get_torsion_types(aa: str) -> List[str]:
    """Get torsion types for a given amino acid."""
    return typing.cast(List[str], SC_BUILD_INFO[aa]["torsion-types"])


def get_torsion_values(aa: str) -> List[Any]:
    """Get torsion values for a given amino acid."""
    return SC_BUILD_INFO[aa]["torsion-vals"]


# Placeholder for BB_BUILD_INFO accessors if needed later
# def get_bb_bond_length(name: str, default: float = 0.0) -> float:
#     """Get a backbone bond length."""
#     return BB_BUILD_INFO.get("BONDLENS", {}).get(name, default)

# def get_bb_bond_angle(name: str, default: float = 0.0) -> float:
#     """Get a backbone bond angle in degrees."""
#     return BB_BUILD_INFO.get("BONDANGS", {}).get(name, default)

# def get_bb_dihedral(name: str, default: float = 0.0) -> float:
#     """Get a backbone dihedral angle in degrees."""
#     return BB_BUILD_INFO.get("DIHEDRS", {}).get(name, default)

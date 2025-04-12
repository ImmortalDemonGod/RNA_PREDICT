"""
Tensor utility functions for RNA prediction pipeline.

This module provides utility functions for tensor operations, including
residue-to-atom bridging functions for converting residue-level tensor
representations to atom-level representations.
"""

# Import public types and constants
from .types import ResidueAtomMap, STANDARD_RNA_ATOMS

# Import public functions
from .residue_mapping import derive_residue_atom_map
from .embedding import residue_to_atoms

# For backward compatibility, re-export everything
__all__ = [
    'ResidueAtomMap',
    'STANDARD_RNA_ATOMS',
    'derive_residue_atom_map',
    'residue_to_atoms',
]

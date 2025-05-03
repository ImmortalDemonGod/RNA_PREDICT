"""
RNA module for MP-NeRF implementation.

This file re-exports all the functions from the refactored modules for backward compatibility.
"""

# Re-export constants
from .rna.rna_constants import (
    BACKBONE_ATOMS,
    BACKBONE_INDEX_MAP,
    RNA_BACKBONE_TORSIONS_AFORM,
)

# Re-export core functions
from .rna.rna_scaffolding import build_scaffolds_rna_from_torsions
from .rna.rna_atom_positioning import calculate_atom_position
from .rna.rna_folding import rna_fold, ring_closure_refinement
from .rna.rna_base_placement import place_rna_bases

# Re-export utility functions
from .rna.rna_utils import (
    handle_mods,
    skip_missing_atoms,
    get_base_atoms,
    mini_refinement,
    validate_rna_geometry,
    compute_max_rna_atoms,
    place_bases,
)

# Define __all__ for explicit exports
__all__ = [
    # Constants
    "BACKBONE_ATOMS",
    "BACKBONE_INDEX_MAP",
    "RNA_BACKBONE_TORSIONS_AFORM",

    # Core functions
    "build_scaffolds_rna_from_torsions",
    "calculate_atom_position",
    "rna_fold",
    "ring_closure_refinement",
    "place_rna_bases",

    # Utility functions
    "handle_mods",
    "skip_missing_atoms",
    "get_base_atoms",
    "mini_refinement",
    "validate_rna_geometry",
    "compute_max_rna_atoms",
    "place_bases",
]

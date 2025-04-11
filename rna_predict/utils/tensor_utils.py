"""
Tensor utility functions for RNA prediction pipeline.

This module provides utility functions for tensor operations, including
residue-to-atom bridging functions for converting residue-level tensor
representations to atom-level representations.

Note: This file is maintained for backward compatibility.
      New code should import directly from the tensor_utils package.
"""

# Re-export everything from the tensor_utils package
from rna_predict.utils.tensor_utils.embedding import residue_to_atoms
from rna_predict.utils.tensor_utils.residue_mapping import derive_residue_atom_map
from rna_predict.utils.tensor_utils.types import STANDARD_RNA_ATOMS, ResidueAtomMap

# Define public API
__all__ = [
    "ResidueAtomMap",
    "STANDARD_RNA_ATOMS",
    "derive_residue_atom_map",
    "residue_to_atoms",
]

"""
Protein utilities package for structure manipulation and mask generation.
"""

from rna_predict.pipeline.stageC.mp_nerf.protein_utils.supreme_data import (
    SUPREME_INFO,  # <-- Add import from supreme_data
    SUPREME_MASK,
)

from .mask_generators import (
    make_atom_token_mask,
    make_bond_mask,
    make_cloud_mask,
    make_idx_mask,
    make_theta_mask,
    make_torsion_mask,
    scn_angle_mask,
    scn_bond_mask,
    scn_cloud_mask,
    scn_index_mask,
    scn_rigid_index_mask,
)
from .sidechain_data import (
    AAS2INDEX,
    AMBIGUOUS,
    INDEX2AAS,
    SC_BUILD_INFO,
    SIDECHAIN_ANGLES,
    SIDECHAIN_BONDS,
    SIDECHAIN_MASK,
    # SUPREME_INFO, # <-- Removed import from sidechain_data
)
from .amino_acid_data_utils import (  # Import data accessors from new module
    get_angle_names,
    get_angle_types,
    get_angle_values,
    get_atom_names,
    get_bond_names,
    get_bond_types,
    get_bond_values,
    get_rigid_frames,
    get_torsion_names,
    get_torsion_types,
    get_torsion_values,
)
from .symmetry_utils import get_symmetric_atom_pairs  # Import from new module
from .scaffold_builders import ( # Import scaffold builders from new module
    build_scaffolds_from_scn_angles,
    modify_angles_mask_with_torsions,
    modify_scaffolds_with_coords,
)
from .structure_utils import (  # Keep remaining imports from structure_utils
    protein_fold,
)

__all__ = [
    "SC_BUILD_INFO",
    "AAS2INDEX",
    "INDEX2AAS",
    "AMBIGUOUS",
    "SUPREME_INFO",
    "SIDECHAIN_ANGLES",
    "SIDECHAIN_BONDS",
    "SIDECHAIN_MASK",
    "get_rigid_frames",
    "get_atom_names",
    "get_bond_names",
    "get_bond_types",
    "get_bond_values",
    "get_angle_names",
    "get_angle_types",
    "get_angle_values",
    "get_torsion_names",
    "get_torsion_types",
    "get_torsion_values",
    "build_scaffolds_from_scn_angles", # Now imported directly
    "modify_scaffolds_with_coords", # Now imported directly
    # "get_symmetric_atom_pairs", # Removed from __all__ as it's imported directly
    "modify_angles_mask_with_torsions", # Now imported directly
    "protein_fold", # Remains from structure_utils
    "SUPREME_MASK",
    "make_cloud_mask",
    "make_bond_mask",
    "make_theta_mask",
    "make_torsion_mask",
    "make_idx_mask",
    "make_atom_token_mask",
    "scn_angle_mask",
    "scn_bond_mask",
    "scn_cloud_mask",
    "scn_index_mask",
    "scn_rigid_index_mask",
]

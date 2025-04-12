"""
Machine learning utilities for RNA structure prediction.
This module provides functions for tensor operations, atom manipulation,
angle calculations, coordinate transformations, and loss functions.
"""

# Re-export all functions for backward compatibility
from .tensor_ops import chain2atoms, process_coordinates
from .atom_utils import (
    rename_symmetric_atoms,
    get_symmetric_atom_pairs,
    atom_selector,
    scn_atom_embedd,
)
from .angle_utils import torsion_angle_loss
from .coordinate_transforms import noise_internals_legacy as noise_internals, combine_noise_legacy as combine_noise
from .loss_functions import fape_torch
from .main import _run_main_logic

# For backward compatibility, if this module is run directly
if __name__ == "__main__":
    _run_main_logic()

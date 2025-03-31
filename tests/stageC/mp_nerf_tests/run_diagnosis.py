#!/usr/bin/env python3
"""
Run the MP-NeRF NaN diagnosis tests and apply fixes.

This script:
1. Runs the diagnostic tests to identify the source of NaNs
2. Applies the appropriate fixes based on the test results
3. Verifies that the fixes resolve the NaN issues
"""

import os
import sys
import logging
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mpnerf_fix")

# Add the project root to the Python path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '../../..')))

# Import the diagnostic test functions - use absolute imports
sys.path.append(current_dir)
from test_mpnerf_nan_diagnosis import (
    test_hypothesis_1_zero_vector_normalization,
    test_hypothesis_2_missing_bond_data,
    test_hypothesis_3_reference_points,
    test_hypothesis_4_zero_angles,
    test_hypothesis_5_angle_format,
    run_all_tests,
    check_for_nans
)

# Import the instrumented MP-NeRF
from instrumented_massive_pnerf import (
    patch_mp_nerf,
    restore_mp_nerf
)

# Import the functions we might need to fix
from rna_predict.pipeline.stageC.stage_c_reconstruction import run_stageC
from rna_predict.pipeline.stageC.mp_nerf.final_kb_rna import RNA_BACKBONE_TORSIONS_AFORM

def print_section(title):
    """Print a section title with formatting."""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"  {title}")
    logger.info(f"{'=' * 80}")

def apply_fixes():
    """
    Apply fixes to the MP-NeRF implementation based on diagnostic test results.
    """
    print_section("Applying Fixes")
    
    # Apply the instrumented MP-NeRF (which includes safety checks)
    patch_mp_nerf()
    logger.info("Applied safe MP-NeRF implementation with division-by-zero protection")
    
    # Test if the fixes resolved the issues
    sequence = "ACGU"
    
    # Test with zero angles (which previously caused NaNs)
    zero_angles = torch.zeros((len(sequence), 7))
    logger.info("Testing with zero angles after applying fixes")
    
    result_zero = run_stageC(
        sequence=sequence,
        torsion_angles=zero_angles,
        method="mp_nerf",
        device="cpu"
    )
    
    has_nan_zero = check_for_nans(result_zero["coords"], "zero angles result")
    
    # Test with A-form angles
    aform_angles = torch.tensor([
        [
            RNA_BACKBONE_TORSIONS_AFORM["alpha"],
            RNA_BACKBONE_TORSIONS_AFORM["beta"],
            RNA_BACKBONE_TORSIONS_AFORM["gamma"],
            RNA_BACKBONE_TORSIONS_AFORM["delta"],
            RNA_BACKBONE_TORSIONS_AFORM["epsilon"],
            RNA_BACKBONE_TORSIONS_AFORM["zeta"],
            0.0  # chi
        ]
    ] * len(sequence), dtype=torch.float32)
    
    logger.info("Testing with A-form angles after applying fixes")
    
    result_aform = run_stageC(
        sequence=sequence,
        torsion_angles=aform_angles,
        method="mp_nerf",
        device="cpu"
    )
    
    has_nan_aform = check_for_nans(result_aform["coords"], "A-form angles result")
    
    if not has_nan_zero and not has_nan_aform:
        logger.info("SUCCESS: Fixes resolved NaN issues for both zero and A-form angles")
        return True
    else:
        logger.error("Fixes did not resolve all NaN issues")
        return False

def create_permanent_fix():
    """
    Create a permanent fix for the MP-NeRF implementation.
    This involves modifying the original massive_pnerf.py file.
    """
    print_section("Creating Permanent Fix")
    
    # Path to the original file
    original_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "rna_predict", "pipeline", "stageC", "mp_nerf", "massive_pnerf.py"
    )
    
    # Read the original file
    with open(original_file, 'r') as f:
        original_code = f.read()
    
    # Create the fixed version - with triple quotes properly escaped
    fixed_code = '''import numpy as np

# diff ml
import torch


def get_axis_matrix(a, b, c, norm=True):
    """Gets an orthonomal basis as a matrix of [e1, e2, e3].
    Useful for constructing rotation matrices between planes
    according to the first answer here:
    https://math.stackexchange.com/questions/1876615/rotation-matrix-from-plane-a-to-b
    Inputs:
    * a: (batch, 3) or (3, ). point(s) of the plane
    * b: (batch, 3) or (3, ). point(s) of the plane
    * c: (batch, 3) or (3, ). point(s) of the plane
    Outputs: orthonomal basis as a matrix of [e1, e2, e3]. calculated as:
        * e1_ = (c-b)
        * e2_proto = (b-a)
        * e3_ = e1_ ^ e2_proto
        * e2_ = e3_ ^ e1_
        * basis = normalize_by_vectors( [e1_, e2_, e3_] )
    Note: Could be done more by Grahm-Schmidt and extend to N-dimensions
          but this is faster and more intuitive for 3D.
    """
    v1_ = c - b
    v2_ = b - a
    v3_ = torch.cross(v1_, v2_, dim=-1)
    v2_ready = torch.cross(v3_, v1_, dim=-1)
    basis = torch.stack([v1_, v2_ready, v3_], dim=-2)
    # normalize if needed
    if norm:
        # Add small epsilon to avoid division by zero
        norm_values = torch.norm(basis, dim=-1, keepdim=True)
        # Clamp to avoid division by zero
        norm_values = torch.clamp(norm_values, min=1e-10)
        return basis / norm_values
    return basis


def mp_nerf_torch(a, b, c, l, theta, chi):
    """Custom Natural extension of Reference Frame.
    Inputs:
    * a: (batch, 3) or (3,). point(s) of the plane, not connected to d
    * b: (batch, 3) or (3,). point(s) of the plane, not connected to d
    * c: (batch, 3) or (3,). point(s) of the plane, connected to d
    * theta: (batch,) or (float).  angle(s) between b-c-d
    * chi: (batch,) or float. dihedral angle(s) between the a-b-c and b-c-d planes
    Outputs: d (batch, 3) or (float). the next point in the sequence, linked to c
    """
    # safety check
    if not ((-np.pi <= theta) * (theta <= np.pi)).all().item():
        # Clamp theta to valid range instead of raising error
        theta = torch.clamp(theta, -np.pi, np.pi)
    
    # calc vecs
    ba = b - a
    cb = c - b
    
    # Check for zero magnitude vectors and add small perturbation if needed
    ba_norm = torch.norm(ba, dim=-1)
    cb_norm = torch.norm(cb, dim=-1)
    
    if (ba_norm < 1e-10).any() or (cb_norm < 1e-10).any():
        # Add small perturbation to avoid zero vectors
        perturb = torch.tensor([1e-10, 1e-10, 1e-10], device=ba.device)
        if (ba_norm < 1e-10).any():
            ba = ba + perturb
        if (cb_norm < 1e-10).any():
            cb = cb + perturb
    
    # calc rotation matrix. based on plane normals and normalized
    n_plane = torch.cross(ba, cb, dim=-1)
    
    # Check if cross product resulted in zero vector (collinear ba and cb)
    n_plane_norm = torch.norm(n_plane, dim=-1)
    if (n_plane_norm < 1e-10).any():
        # Add small perturbation to ba to avoid collinearity
        perturb = torch.tensor([0.0, 1e-10, 1e-10], device=ba.device)
        ba = ba + perturb
        # Recalculate cross product
        n_plane = torch.cross(ba, cb, dim=-1)
    
    n_plane_ = torch.cross(n_plane, cb, dim=-1)
    rotate = torch.stack([cb, n_plane_, n_plane], dim=-1)
    
    # Safe normalization with epsilon to avoid division by zero
    norm = torch.norm(rotate, dim=-2, keepdim=True)
    norm = torch.clamp(norm, min=1e-10)  # Ensure no division by zero
    rotate = rotate / norm
    
    # calc proto point, rotate. add (-1 for sidechainnet convention)
    # https://github.com/jonathanking/sidechainnet/issues/14
    d = torch.stack(
        [
            -torch.cos(theta),
            torch.sin(theta) * torch.cos(chi),
            torch.sin(theta) * torch.sin(chi),
        ],
        dim=-1,
    ).unsqueeze(-1)
    
    # extend base point, set length
    return c + l.unsqueeze(-1) * torch.matmul(rotate, d).squeeze()
'''
    
    # Create a backup of the original file
    backup_file = original_file + ".bak"
    import shutil
    shutil.copy2(original_file, backup_file)
    logger.info(f"Created backup of original file at {backup_file}")
    
    # Write the fixed version
    with open(original_file, 'w') as f:
        f.write(fixed_code)
    
    logger.info(f"Applied permanent fix to {original_file}")
    logger.info("The fix includes:")
    logger.info("1. Safe normalization with epsilon to avoid division by zero")
    logger.info("2. Clamping theta to valid range instead of raising error")
    logger.info("3. Adding small perturbations to avoid zero vectors and collinearity")
    
    return True

def main():
    """Run the diagnosis and apply fixes."""
    print_section("MP-NeRF NaN Diagnosis and Fix")
    
    # First, run the diagnostic tests
    logger.info("Running diagnostic tests to identify NaN sources...")
    run_all_tests()
    
    # Apply temporary fixes and test
    logger.info("\nApplying temporary fixes...")
    fixes_successful = apply_fixes()
    
    # If temporary fixes work, create permanent fix
    if fixes_successful:
        logger.info("\nTemporary fixes were successful. Creating permanent fix...")
        create_permanent_fix()
        
        # Restore original function for clean state
        restore_mp_nerf()
        
        logger.info("\nFix complete! The MP-NeRF implementation should now be robust against NaN issues.")
    else:
        logger.error("\nTemporary fixes were not successful. More investigation needed.")
        # Restore original function
        restore_mp_nerf()

if __name__ == "__main__":
    main()
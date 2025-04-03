#!/usr/bin/env python3
"""
Diagnostic Test Script for MP-NeRF NaN Issues

This script systematically tests different hypotheses about what's causing NaN values
in the run_stageC function, with detailed logging to pinpoint the exact source.
"""

import logging
import os
import sys

import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mpnerf_diagnosis")

# Add the project root to the Python path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# Import the functions to test
from rna_predict.pipeline.stageC.mp_nerf.final_kb_rna import (
    RNA_BACKBONE_TORSIONS_AFORM,
    get_bond_angle,
    get_bond_length,
)
from rna_predict.pipeline.stageC.mp_nerf.massive_pnerf import mp_nerf_torch
from rna_predict.pipeline.stageC.mp_nerf.rna import (
    build_scaffolds_rna_from_torsions,
    rna_fold,
)
from rna_predict.pipeline.stageC.stage_c_reconstruction import run_stageC


def print_section(title):
    """Print a section title with formatting."""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"  {title}")
    logger.info(f"{'=' * 80}")


def check_for_nans(tensor, name):
    """Check if a tensor contains NaN values and log details."""
    if tensor is None:
        logger.warning(f"{name} is None")
        return True

    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()

    if has_nan or has_inf:
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        logger.error(f"{name} contains {nan_count} NaNs and {inf_count} Infs")

        # If it's a multi-dimensional tensor, find where the NaNs are
        if tensor.dim() > 1:
            nan_indices = torch.where(torch.isnan(tensor))
            if len(nan_indices[0]) > 0:
                logger.error(
                    f"First few NaN positions: {[(nan_indices[i][j].item() for i in range(len(nan_indices))) for j in range(min(5, len(nan_indices[0])))]}"
                )
        return True
    return False


def test_hypothesis_1_zero_vector_normalization():
    """
    Test Hypothesis 1: Zero vector normalization in MP-NeRF causes NaNs.

    This test directly calls mp_nerf_torch with vectors that would produce
    zero-magnitude vectors during normalization.
    """
    print_section("Testing Hypothesis 1: Zero Vector Normalization")

    # Create vectors that would lead to zero cross products
    a = torch.tensor([0.0, 0.0, 0.0], device="cpu")
    b = torch.tensor([1.0, 0.0, 0.0], device="cpu")
    c = torch.tensor(
        [2.0, 0.0, 0.0], device="cpu"
    )  # Collinear with b-a, will cause zero cross product

    l = torch.tensor(1.5, device="cpu")  # Bond length
    theta = torch.tensor(1.0, device="cpu")  # Bond angle in radians
    chi = torch.tensor(0.5, device="cpu")  # Dihedral angle in radians

    logger.info(
        "Calling mp_nerf_torch with collinear vectors that would produce zero cross products"
    )

    try:
        result = mp_nerf_torch(a, b, c, l, theta, chi)
        logger.info(f"Result shape: {result.shape}")
        has_nan = check_for_nans(result, "mp_nerf_torch result")

        if has_nan:
            logger.info("CONFIRMED: Zero vector normalization causes NaNs")
            return True
        else:
            logger.info(
                "Zero vector normalization did NOT cause NaNs with these inputs"
            )
            return False
    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
        return False


def test_hypothesis_2_missing_bond_data():
    """
    Test Hypothesis 2: Missing bond lengths/angles in knowledge base cause NaNs.

    This test checks if any bond lengths or angles return None and how the code handles it.
    """
    print_section("Testing Hypothesis 2: Missing Bond Data")

    # Test a few key bond lengths and angles
    bonds_to_test = ["P-O5'", "C1'-C2'", "C3'-O3'", "O4'-C1'", "C2'-O2'", "C3'-C4'"]
    angles_to_test = [
        "C1'-C2'-C3'",
        "C2'-C3'-C4'",
        "C3'-C4'-O4'",
        "O4'-C1'-C2'",
        "O3'-P-O5'",
    ]

    missing_bonds = []
    missing_angles = []

    # Check bond lengths
    for bond in bonds_to_test:
        length = get_bond_length(bond, sugar_pucker="C3'-endo")
        if length is None:
            missing_bonds.append(bond)
            logger.warning(f"Bond length for {bond} is None")

    # Check bond angles
    for angle in angles_to_test:
        angle_val = get_bond_angle(angle, sugar_pucker="C3'-endo")
        if angle_val is None:
            missing_angles.append(angle)
            logger.warning(f"Bond angle for {angle} is None")

    # Now test with a simple sequence
    sequence = "ACGU"
    torsion_angles = torch.zeros((len(sequence), 7))

    logger.info("Building scaffolds with zero angles to check for None values")
    scaffolds = build_scaffolds_rna_from_torsions(
        seq=sequence, torsions=torsion_angles, sugar_pucker="C3'-endo"
    )

    # Check scaffold tensors for NaNs
    has_nan_bond = check_for_nans(scaffolds["bond_mask"], "bond_mask")
    has_nan_angles = check_for_nans(scaffolds["angles_mask"], "angles_mask")

    if missing_bonds or missing_angles or has_nan_bond or has_nan_angles:
        logger.info("CONFIRMED: Missing bond data could be causing issues")
        return True
    else:
        logger.info("All tested bond data is present and no NaNs found in scaffolds")
        return False


def test_hypothesis_3_reference_points():
    """
    Test Hypothesis 3: Reference point issues in rna_fold cause NaNs.

    This test examines how reference points are handled, especially for the first residue.
    """
    print_section("Testing Hypothesis 3: Reference Point Issues")

    sequence = "ACGU"
    torsion_angles = torch.zeros((len(sequence), 7))

    # Build scaffolds
    scaffolds = build_scaffolds_rna_from_torsions(
        seq=sequence, torsions=torsion_angles, sugar_pucker="C3'-endo"
    )

    # Log reference points for inspection
    point_refs = scaffolds["point_ref_mask"]
    logger.info(f"Point reference shape: {point_refs.shape}")
    logger.info(f"First residue references: {point_refs[:, 0, :]}")

    # Check if any reference points are out of bounds
    L, B = scaffolds["bond_mask"].shape
    total = L * B

    out_of_bounds = []
    for i in range(L):
        for j in range(B):
            for k in range(3):
                ref = point_refs[k, i, j].item()
                if ref < 0 or ref >= total:
                    out_of_bounds.append((k, i, j, ref))

    if out_of_bounds:
        logger.warning(f"Found {len(out_of_bounds)} out-of-bounds references")
        logger.warning(f"First few: {out_of_bounds[:5]}")

    # Now run rna_fold and check for NaNs
    logger.info("Running rna_fold to check for NaNs from reference issues")
    coords_bb = rna_fold(scaffolds, device="cpu", do_ring_closure=False)

    has_nan = check_for_nans(coords_bb, "backbone coordinates")

    if has_nan or out_of_bounds:
        logger.info("CONFIRMED: Reference point issues could be causing NaNs")
        return True
    else:
        logger.info("No reference point issues detected")
        return False


def test_hypothesis_4_zero_angles():
    """
    Test Hypothesis 4: Zero angles create degenerate geometry causing NaNs.

    This test compares using zero angles vs. typical A-form angles.
    """
    print_section("Testing Hypothesis 4: Zero Angles vs. A-form Angles")

    sequence = "ACGU"

    # Test with zero angles
    zero_angles = torch.zeros((len(sequence), 7))
    logger.info("Testing with zero angles")

    result_zero = run_stageC(
        sequence=sequence, torsion_angles=zero_angles, method="mp_nerf", device="cpu"
    )

    has_nan_zero = check_for_nans(result_zero["coords"], "zero angles result")

    # Test with A-form angles
    aform_angles = torch.tensor(
        [
            [
                RNA_BACKBONE_TORSIONS_AFORM["alpha"],
                RNA_BACKBONE_TORSIONS_AFORM["beta"],
                RNA_BACKBONE_TORSIONS_AFORM["gamma"],
                RNA_BACKBONE_TORSIONS_AFORM["delta"],
                RNA_BACKBONE_TORSIONS_AFORM["epsilon"],
                RNA_BACKBONE_TORSIONS_AFORM["zeta"],
                0.0,  # chi
            ]
        ]
        * len(sequence),
        dtype=torch.float32,
    )

    logger.info("Testing with A-form angles")

    result_aform = run_stageC(
        sequence=sequence, torsion_angles=aform_angles, method="mp_nerf", device="cpu"
    )

    has_nan_aform = check_for_nans(result_aform["coords"], "A-form angles result")

    if has_nan_zero and not has_nan_aform:
        logger.info(
            "CONFIRMED: Zero angles cause NaNs, but A-form angles work correctly"
        )
        return True
    elif has_nan_zero and has_nan_aform:
        logger.info(
            "Both zero and A-form angles produce NaNs, suggesting another issue"
        )
        return False
    else:
        logger.info("Zero angles did not cause NaNs in this test")
        return False


def test_hypothesis_5_angle_format():
    """
    Test Hypothesis 5: Torsion angle format inconsistency causes NaNs.

    This test checks how different angle formats are handled.
    """
    print_section("Testing Hypothesis 5: Angle Format Inconsistency")

    sequence = "ACGU"

    # Test with degrees (as expected by documentation)
    degrees_angles = torch.tensor(
        [
            [300.0, 180.0, 50.0, 85.0, 180.0, 290.0, 0.0]  # A-form in degrees
        ]
        * len(sequence),
        dtype=torch.float32,
    )

    logger.info("Testing with angles in degrees")

    result_degrees = run_stageC(
        sequence=sequence, torsion_angles=degrees_angles, method="mp_nerf", device="cpu"
    )

    has_nan_degrees = check_for_nans(result_degrees["coords"], "degrees result")

    # Test with radians (converted from degrees)
    radians_angles = degrees_angles * (torch.pi / 180.0)

    logger.info("Testing with angles in radians")

    result_radians = run_stageC(
        sequence=sequence, torsion_angles=radians_angles, method="mp_nerf", device="cpu"
    )

    has_nan_radians = check_for_nans(result_radians["coords"], "radians result")

    # Test with sin/cos pairs
    sincos_angles = torch.zeros((len(sequence), 14))  # 7 angles * 2 (sin, cos)
    for i in range(7):
        angle_rad = radians_angles[0, i].item()
        # Convert scalar to tensor before applying sin/cos
        angle_tensor = torch.tensor(angle_rad)
        sincos_angles[:, i * 2] = torch.sin(angle_tensor)
        sincos_angles[:, i * 2 + 1] = torch.cos(angle_tensor)

    logger.info("Testing with sin/cos pairs")

    try:
        result_sincos = run_stageC(
            sequence=sequence,
            torsion_angles=sincos_angles,
            method="mp_nerf",
            device="cpu",
        )

        has_nan_sincos = check_for_nans(result_sincos["coords"], "sin/cos result")

        logger.info(
            f"Degrees NaNs: {has_nan_degrees}, Radians NaNs: {has_nan_radians}, Sin/Cos NaNs: {has_nan_sincos}"
        )

        if has_nan_degrees != has_nan_radians or has_nan_degrees != has_nan_sincos:
            logger.info("CONFIRMED: Angle format inconsistency affects NaN generation")
            return True
        else:
            logger.info("Angle format does not appear to affect NaN generation")
            return False
    except Exception as e:
        logger.error(f"Error with sin/cos format: {str(e)}")
        logger.info("CONFIRMED: Sin/cos format is not compatible with the function")
        return True


def run_all_tests():
    """Run all hypothesis tests and summarize results."""
    print_section("Running All Hypothesis Tests")

    results = {
        "Hypothesis 1 (Zero Vector Normalization)": test_hypothesis_1_zero_vector_normalization(),
        "Hypothesis 2 (Missing Bond Data)": test_hypothesis_2_missing_bond_data(),
        "Hypothesis 3 (Reference Point Issues)": test_hypothesis_3_reference_points(),
        "Hypothesis 4 (Zero Angles)": test_hypothesis_4_zero_angles(),
        "Hypothesis 5 (Angle Format)": test_hypothesis_5_angle_format(),
    }

    print_section("Test Results Summary")

    confirmed = []
    for hypothesis, result in results.items():
        status = "CONFIRMED" if result else "NOT CONFIRMED"
        logger.info(f"{hypothesis}: {status}")
        if result:
            confirmed.append(hypothesis)

    if confirmed:
        logger.info(f"\nConfirmed hypotheses: {', '.join(confirmed)}")
    else:
        logger.info("\nNo hypotheses were confirmed. The issue may be elsewhere.")


if __name__ == "__main__":
    run_all_tests()

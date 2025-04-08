#!/usr/bin/env python3
"""
Verify that the MP-NeRF fix resolves the NaN issues.

This script runs a simple test with both zero angles and A-form angles
to confirm that the fixed MP-NeRF implementation no longer produces NaNs.
"""

import logging
import sys
from pathlib import Path

import torch

# Local/Project imports (moved for E402)
from rna_predict.pipeline.stageC.mp_nerf.final_kb_rna import RNA_BACKBONE_TORSIONS_AFORM
from rna_predict.pipeline.stageC.stage_c_reconstruction import run_stageC


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("verify_fix")

# Add the project root to the Python path
current_dir = str(Path(__file__).parent.absolute())
sys.path.append(current_dir)

# Imports moved to top of file (E402 fix)


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
        return True
    return False


def verify_fix():
    """Verify that the MP-NeRF fix resolves the NaN issues."""
    logger.info("Verifying MP-NeRF fix...")

    # Test with a simple sequence
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

    # Test with a longer sequence
    long_sequence = "GGCGCUAUGCGCCG"  # 14 nucleotides
    long_aform_angles = torch.tensor(
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
        * len(long_sequence),
        dtype=torch.float32,
    )

    logger.info("Testing with a longer sequence")

    result_long = run_stageC(
        sequence=long_sequence,
        torsion_angles=long_aform_angles,
        method="mp_nerf",
        device="cpu",
    )

    has_nan_long = check_for_nans(result_long["coords"], "long sequence result")

    # Report results
    if not has_nan_zero and not has_nan_aform and not has_nan_long:
        logger.info("SUCCESS: All tests passed without NaNs!")
        logger.info("The MP-NeRF fix has successfully resolved the NaN issues.")
        return True
    else:
        logger.error("FAILURE: NaNs still present in some tests.")
        logger.error(f"Zero angles: {'NaNs present' if has_nan_zero else 'OK'}")
        logger.error(f"A-form angles: {'NaNs present' if has_nan_aform else 'OK'}")
        logger.error(f"Long sequence: {'NaNs present' if has_nan_long else 'OK'}")
        return False


if __name__ == "__main__":
    success = verify_fix()
    if success:
        print("\nVerification successful! The MP-NeRF fix has resolved the NaN issues.")
    else:
        print(
            "\nVerification failed. The MP-NeRF fix did not fully resolve the NaN issues."
        )

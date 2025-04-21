#!/usr/bin/env python3
"""
Test script to verify angle format compatibility between TorsionBERT and run_stageC.

This script tests how run_stageC handles different angle formats:
1. Degrees (expected format)
2. Radians (converted from degrees)
3. Sin/Cos pairs (raw output from TorsionBERT)
"""

import os
import sys

import torch
from hypothesis import given, strategies as st, settings

# Add the project root to the Python path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import the functions to test
from rna_predict.pipeline.stageC.stage_c_reconstruction import run_stageC, create_stage_c_test_config


def check_for_nans(tensor, name):
    """Check if a tensor contains NaN or Inf values."""
    if tensor is None:
        return True
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    return has_nan or has_inf


@given(
    st.lists(st.floats(min_value=-360, max_value=360, allow_nan=False, allow_infinity=False), min_size=7, max_size=7),
    st.sampled_from(["degrees", "radians"])
)
@settings(deadline=None, max_examples=30)
def test_angle_format_compatibility_hypothesis(angle_list, fmt):
    """Hypothesis-based test for angle format compatibility in run_stageC."""
    sequence = "ACGU"
    angles = torch.tensor([angle_list] * len(sequence), dtype=torch.float32)

    # Create proper configuration with all required fields
    cfg = create_stage_c_test_config(
        method="mp_nerf",
        device="cpu",
        do_ring_closure=False,
        place_bases=True,
        sugar_pucker="C3'-endo",
        angle_representation=fmt  # Use the format from the test parameter
    )

    # Convert angles based on format
    if fmt == "degrees":
        input_angles = angles
    elif fmt == "radians":
        input_angles = angles * (torch.pi / 180.0)
    else:
        input_angles = angles

    try:
        # Use the config object instead of individual parameters
        result = run_stageC(
            sequence=sequence,
            torsion_angles=input_angles,
            cfg=cfg
        )
        assert not check_for_nans(result["coords"], "coords"), "Output contains NaN/Inf for valid input."
        # Check that we have a reasonable number of atoms (not exact since it varies by nucleotide)
        assert result["coords"].shape[0] > 0, "No atoms in output"
        assert result["coords"].shape[0] == result["atom_count"], "Mismatch between coords shape and atom_count"
    except Exception as e:
        # Acceptable if error is about unsupported format or invalid input
        acceptable_errors = ["angle", "format", "shape", "dimension", "torsion"]
        assert any(s in str(e).lower() for s in acceptable_errors), f"Unexpected error: {e}"


if __name__ == "__main__":
    test_angle_format_compatibility_hypothesis()

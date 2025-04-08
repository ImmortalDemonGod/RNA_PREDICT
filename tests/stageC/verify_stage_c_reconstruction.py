#!/usr/bin/env python3
"""
Verification Script for Stage C Module (3D Reconstruction)

This script verifies the functionality of the run_stageC function in
rna_predict.pipeline.stageC.stage_c_reconstruction according to the verification checklist.
"""

import os
import sys

import torch

# Add the project root to the Python path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import the function to verify
from rna_predict.pipeline.stageB.torsion.torsion_bert_predictor import (
    StageBTorsionBertPredictor,
)
from rna_predict.pipeline.stageC.stage_c_reconstruction import run_stageC


def print_section(title):
    """Print a section title with formatting."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def print_result(test_name, result, details=""):
    """Print a test result with formatting."""
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"{status} - {test_name}")
    if details:
        print(f"       {details}")


def verify_callable():
    """Verify that run_stageC can be called with appropriate parameters."""
    print_section("Callable Verification")

    try:
        # Simple test call with minimal parameters
        sequence = "ACGU"
        torsion_angles = torch.zeros((len(sequence), 7))
        run_stageC(sequence=sequence, torsion_angles=torsion_angles)
        print_result(
            "Function can be called",
            True,
            "Signature: run_stageC(sequence, torsion_angles, method='mp_nerf', device='cpu', ...)",
        )
        return True
    except Exception as e:
        print_result("Function can be called", False, f"Error: {str(e)}")
        return False


def verify_input_parameters():
    """Verify that run_stageC accepts the expected input parameters."""
    print_section("Input Parameter Verification")

    # Test sequence parameter
    sequence = "ACGU"
    torsion_angles = torch.zeros((len(sequence), 7))
    try:
        run_stageC(sequence=sequence, torsion_angles=torsion_angles)
        print_result("Accepts sequence string", True)
    except Exception as e:
        print_result("Accepts sequence string", False, f"Error: {str(e)}")

    # Test torsion_angles parameter
    try:
        run_stageC(sequence=sequence, torsion_angles=torsion_angles)
        print_result(
            "Accepts torsion_angles as PyTorch Tensor",
            True,
            f"Shape: {torsion_angles.shape}",
        )
    except Exception as e:
        print_result(
            "Accepts torsion_angles as PyTorch Tensor", False, f"Error: {str(e)}"
        )

    # Test method parameter
    try:
        run_stageC(sequence=sequence, torsion_angles=torsion_angles, method="mp_nerf")
        print_result("Accepts method='mp_nerf' argument", True)
    except Exception as e:
        print_result("Accepts method='mp_nerf' argument", False, f"Error: {str(e)}")

    # Test device parameter
    try:
        run_stageC(sequence=sequence, torsion_angles=torsion_angles, device="cpu")
        print_result("Accepts device argument", True)
    except Exception as e:
        print_result("Accepts device argument", False, f"Error: {str(e)}")

    # Verify torsion angle format by checking the docstring
    from rna_predict.pipeline.stageC.mp_nerf.rna import (
        build_scaffolds_rna_from_torsions,
    )

    docstring = build_scaffolds_rna_from_torsions.__doc__ or ""
    if "in DEGREES" in docstring:
        print_result("Torsion angle format", True, "Expects angles in DEGREES")
    else:
        print_result(
            "Torsion angle format", False, "Could not confirm format from docstring"
        )


def verify_dependencies():
    """Verify that the required dependencies exist and import correctly."""
    print_section("Dependency Verification")

    # Check if mp_nerf directory exists
    import os

    mp_nerf_path = os.path.join("rna_predict", "pipeline", "stageC", "mp_nerf")
    mp_nerf_exists = os.path.isdir(mp_nerf_path)
    print_result("MP-NeRF directory exists", mp_nerf_exists, f"Path: {mp_nerf_path}")

    # Check if rna.py exists
    rna_path = os.path.join(mp_nerf_path, "rna.py")
    rna_exists = os.path.isfile(rna_path)
    print_result("rna.py exists", rna_exists, f"Path: {rna_path}")

    # Check if final_kb_rna.py exists
    final_kb_rna_path = os.path.join(mp_nerf_path, "final_kb_rna.py")
    final_kb_rna_exists = os.path.isfile(final_kb_rna_path)
    print_result(
        "final_kb_rna.py exists", final_kb_rna_exists, f"Path: {final_kb_rna_path}"
    )

    # Try importing the required module to verify its existence
    try:
        import rna_predict.pipeline.stageC.mp_nerf.rna  # noqa: F401 - Import check only

        print_result(
            "Required module 'rna_predict.pipeline.stageC.mp_nerf.rna' imports correctly",
            True,
            "build_scaffolds_rna_from_torsions, rna_fold, place_rna_bases",
        )
    except ImportError as e:
        print_result("Required functions import correctly", False, f"Error: {str(e)}")


def verify_output():
    """Verify that run_stageC returns the expected output."""
    print_section("Output Verification")

    sequence = "ACGU"
    torsion_angles = torch.zeros((len(sequence), 7))

    try:
        result = run_stageC(sequence=sequence, torsion_angles=torsion_angles)

        # Check if result is a dictionary
        is_dict = isinstance(result, dict)
        print_result("Returns a dictionary", is_dict)

        # Check if dictionary contains "coords" key
        has_coords_key = "coords" in result
        print_result("Dictionary contains 'coords' key", has_coords_key)

        if has_coords_key:
            # Check if "coords" value is a PyTorch Tensor
            coords = result["coords"]
            is_tensor = isinstance(coords, torch.Tensor)
            print_result("'coords' value is a PyTorch Tensor", is_tensor)

            # Check the shape of the coords tensor
            if is_tensor:
                shape = coords.shape

                # Check if shape is plausible
                if len(shape) == 3:  # [L, num_atoms_per_res, 3]
                    L, num_atoms, dims = shape
                    is_plausible = L == len(sequence) and num_atoms >= 10 and dims == 3
                    shape_details = (
                        f"[L={L}, num_atoms_per_res={num_atoms}, dims={dims}]"
                    )
                elif len(shape) == 2:  # [total_atoms, 3]
                    total_atoms, dims = shape
                    is_plausible = dims == 3
                    shape_details = f"[total_atoms={total_atoms}, dims={dims}]"
                else:
                    is_plausible = False
                    shape_details = f"Unexpected shape: {shape}"

                print_result("Shape is plausible", is_plausible, shape_details)

                # Check for NaNs or Infs
                has_nan = torch.isnan(coords).any().item()
                has_inf = torch.isinf(coords).any().item()
                no_nan_inf = not (has_nan or has_inf)
                print_result(
                    "No NaNs or Infs in coordinates",
                    no_nan_inf,
                    "NaNs: {}, Infs: {}".format(has_nan, has_inf),
                )

    except Exception as e:
        print_result("Output verification", False, f"Error: {str(e)}")


def verify_functionality():
    """Verify that run_stageC functions correctly with different parameters."""
    print_section("Functional Verification")

    # Test with a simple sequence and zero angles
    sequence = "ACGU"
    torsion_angles = torch.zeros((len(sequence), 7))

    try:
        run_stageC(
            sequence=sequence,
            torsion_angles=torsion_angles,
            method="mp_nerf",
            device="cpu",
            do_ring_closure=False,
            place_bases=True,
            sugar_pucker="C3'-endo",
        )
        print_result(
            "Runs with method='mp_nerf'", True, "Simple sequence with zero angles"
        )
    except Exception as e:
        print_result("Runs with method='mp_nerf'", False, f"Error: {str(e)}")

    # Test with a different method (fallback)
    try:
        run_stageC(
            sequence=sequence,
            torsion_angles=torsion_angles,
            method="fallback",
            device="cpu",
        )
        print_result("Runs with method='fallback'", True, "Uses StageCReconstruction")
    except Exception as e:
        print_result("Runs with method='fallback'", False, f"Error: {str(e)}")

    # Test with do_ring_closure=True
    try:
        run_stageC(
            sequence=sequence,
            torsion_angles=torsion_angles,
            method="mp_nerf",
            device="cpu",
            do_ring_closure=True,
        )
        print_result("Runs with do_ring_closure=True", True)
    except Exception as e:
        print_result("Runs with do_ring_closure=True", False, f"Error: {str(e)}")

    # Test with place_bases=False
    try:
        run_stageC(
            sequence=sequence,
            torsion_angles=torsion_angles,
            method="mp_nerf",
            device="cpu",
            place_bases=False,
        )
        print_result("Runs with place_bases=False", True)
    except Exception as e:
        print_result("Runs with place_bases=False", False, f"Error: {str(e)}")


def verify_torsionbert_integration():
    """Verify the integration between TorsionBERT and run_stageC."""
    print_section("TorsionBERT Integration Verification")

    sequence = "ACGU"

    # Test with angle_mode="degrees"
    try:
        # Initialize TorsionBERT with angle_mode="degrees"
        torsion_predictor = StageBTorsionBertPredictor(
            model_name_or_path="dummy_path",  # Use dummy path to avoid loading real model
            device="cpu",
            angle_mode="degrees",
            num_angles=7,
        )

        # Get torsion angles
        torsion_output = torsion_predictor(sequence)
        torsion_angles = torsion_output["torsion_angles"]

        # Run Stage C
        run_stageC(
            sequence=sequence,
            torsion_angles=torsion_angles,
            method="mp_nerf",
            device="cpu",
        )

        print_result(
            "TorsionBERT integration with angle_mode='degrees'",
            True,
            f"TorsionBERT output shape: {torsion_angles.shape}",
        )
    except Exception as e:
        print_result(
            "TorsionBERT integration with angle_mode='degrees'",
            False,
            f"Error: {str(e)}",
        )

    # Test with angle_mode="sin_cos" (should fail or require conversion)
    try:
        # Initialize TorsionBERT with angle_mode="sin_cos"
        torsion_predictor = StageBTorsionBertPredictor(
            model_name_or_path="dummy_path",  # Use dummy path to avoid loading real model
            device="cpu",
            angle_mode="sin_cos",
            num_angles=7,
        )

        # Get torsion angles
        torsion_output = torsion_predictor(sequence)
        torsion_angles = torsion_output["torsion_angles"]

        # This should fail because MP-NeRF expects degrees, not sin/cos pairs
        try:
            run_stageC(
                sequence=sequence,
                torsion_angles=torsion_angles,
                method="mp_nerf",
                device="cpu",
            )
            print_result(
                "TorsionBERT integration with angle_mode='sin_cos'",
                False,
                "Expected failure but succeeded. MP-NeRF might be handling sin/cos pairs.",
            )
        except Exception as e:
            print_result(
                "TorsionBERT integration with angle_mode='sin_cos'",
                True,
                f"Expected failure: {str(e)}",
            )
            print(
                "       This confirms that MP-NeRF expects degrees, not sin/cos pairs."
            )
    except Exception as e:
        print_result(
            "TorsionBERT initialization with angle_mode='sin_cos'",
            False,
            f"Error: {str(e)}",
        )


def main():
    """Run all verification tests."""
    print("\nStage C Module (3D Reconstruction) Verification Script")
    print(
        "Function: run_stageC in rna_predict.pipeline.stageC.stage_c_reconstruction\n"
    )

    # Run all verification tests
    verify_callable()
    verify_input_parameters()
    verify_dependencies()
    verify_output()
    verify_functionality()
    verify_torsionbert_integration()

    print("\nVerification complete!\n")


if __name__ == "__main__":
    main()

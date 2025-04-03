# tests/interface/test_interface_mpnerf_nan.py

import unittest

import numpy as np
import pandas as pd
import torch

# Make sure the RNAPredictor class is importable
from rna_predict.interface import RNAPredictor

# Import the specific stage C function to potentially mock or verify settings

# Optional: If tests are slow due to model loading, consider mocking
# from unittest.mock import patch


class TestRNAPredictorMpNerfNaN(unittest.TestCase):
    """
    Tests the RNAPredictor interface specifically when using the 'mp_nerf'
    method for Stage C reconstruction. The primary goal is to ensure that
    this configuration does not produce NaN (Not a Number) coordinates,
    addressing the bug observed in the snoop trace where rna_fold generated NaNs.
    """

    def setUp(self):
        """
        Set up the RNAPredictor with mp_nerf enabled for Stage C.
        Uses the default TorsionBERT model path ('sayby/rna_torsionbert').
        Tests run on CPU for portability unless CUDA is explicitly available and desired.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(
            f"\n[Test Setup] Initializing RNAPredictor with mp_nerf on device: {self.device}"
        )
        try:
            # Initialize with the actual mp_nerf method required for this test
            self.predictor = RNAPredictor(
                model_name_or_path="sayby/rna_torsionbert",  # Attempt to use the real model
                device=self.device,
                angle_mode="degrees",  # Using degrees, but mode shouldn't affect NaN generation
                num_angles=7,  # Standard 7 angles for backbone + chi
                max_length=512,
                stageC_method="mp_nerf",  # CRUCIAL: Ensure mp_nerf is used
            )
            # Verify the stageC method was set correctly
            self.assertEqual(
                self.predictor.stageC_method,
                "mp_nerf",
                "Stage C method not set to mp_nerf during init.",
            )
        except Exception as e:
            # Fail setup if predictor initialization fails (e.g., model download issue)
            self.fail(f"Failed to initialize RNAPredictor: {e}")

    def test_predict_submission_with_mpnerf_no_nan(self):
        """
        Verify that predict_submission using mp_nerf does not produce NaN coordinates
        for a moderately complex RNA sequence.
        """
        # Using a sequence that isn't trivially short or simple.
        # Replace with other sequences if specific ones are known to fail.
        sequence = "GGCGCUAUGCGCCG"  # Example sequence (14 nt)
        prediction_repeats = 1  # Only need 1 prediction to check for NaNs

        print(
            f"[Test Run] Testing predict_submission for sequence: '{sequence}' with mp_nerf..."
        )

        try:
            # Run the prediction pipeline to generate the submission DataFrame
            submission_df = self.predictor.predict_submission(
                sequence,
                prediction_repeats=prediction_repeats,
                residue_atom_choice=0,  # Check coordinates of the first atom (e.g., P)
            )

            self.assertIsInstance(
                submission_df, pd.DataFrame, "Output should be a DataFrame."
            )
            self.assertEqual(
                len(submission_df),
                len(sequence),
                "DataFrame row count should match sequence length.",
            )

            # Define the coordinate columns to check (only for the first prediction)
            coord_cols = (
                [f"x_{i}" for i in range(1, prediction_repeats + 1)]
                + [f"y_{i}" for i in range(1, prediction_repeats + 1)]
                + [f"z_{i}" for i in range(1, prediction_repeats + 1)]
            )

            # Check if all required columns exist
            missing_cols = [
                col for col in coord_cols if col not in submission_df.columns
            ]
            self.assertEqual(
                len(missing_cols),
                0,
                f"Submission DataFrame is missing columns: {missing_cols}",
            )

            # Extract coordinate data as a NumPy array
            coords_data = submission_df[coord_cols].values.astype(
                float
            )  # Ensure numeric type

            # Assert that there are no NaNs or Infs in the coordinate data
            has_nan = np.isnan(coords_data).any()
            has_inf = np.isinf(coords_data).any()

            if has_nan or has_inf:
                print("\nProblematic values found in coordinates DataFrame:")
                problematic_rows = submission_df[
                    np.isnan(coords_data).any(axis=1)
                    | np.isinf(coords_data).any(axis=1)
                ]
                print(problematic_rows)
                # Use math.isnan to handle potential float conversion issues if needed for debugging individual values
                # problematic_values = coords_data[np.isnan(coords_data) | np.isinf(coords_data)]
                # print("Specific NaN/Inf values:", problematic_values)

            self.assertFalse(
                has_nan, "NaN values detected in the output coordinates using mp_nerf."
            )
            self.assertFalse(
                has_inf,
                "Infinite values detected in the output coordinates using mp_nerf.",
            )

        except ValueError as e:
            # Catch potential errors during the process (e.g., shape mismatches)
            self.fail(f"Predict_submission raised ValueError: {e}")
        except IndexError as e:
            # Catch errors related to atom choice index
            self.fail(
                f"Predict_submission raised IndexError (check residue_atom_choice): {e}"
            )
        except Exception as e:
            # Catch any other unexpected errors
            self.fail(f"An unexpected error occurred during submission prediction: {e}")

    def test_predict_3d_structure_with_mpnerf_no_nan(self):
        """
        Verify that predict_3d_structure directly returns a non-NaN coordinate tensor
        when using the mp_nerf method.
        """
        sequence = "GGCGCUAUGCGCCG"  # Same sequence as above
        print(
            f"[Test Run] Testing predict_3d_structure for sequence: '{sequence}' with mp_nerf..."
        )

        try:
            result_dict = self.predictor.predict_3d_structure(sequence)
            coords = result_dict.get("coords")

            self.assertIsNotNone(
                coords, "Coordinates tensor not found in result dictionary."
            )
            self.assertIsInstance(
                coords, torch.Tensor, "Coordinates should be a PyTorch Tensor."
            )
            self.assertGreater(
                coords.numel(), 0, "Coordinates tensor should not be empty."
            )

            # Check for NaNs directly in the output tensor
            has_nan = torch.isnan(coords).any().item()
            has_inf = torch.isinf(coords).any().item()

            if has_nan or has_inf:
                print("\nProblematic values found in coordinates tensor:")
                # Print rows/residues containing NaN or Inf
                problem_mask = torch.isnan(coords).any(dim=-1).any(
                    dim=-1
                ) | torch.isinf(coords).any(dim=-1).any(dim=-1)
                print(coords[problem_mask])

            self.assertFalse(
                has_nan,
                "NaN values detected in the output coordinates tensor from predict_3d_structure with mp_nerf.",
            )
            self.assertFalse(
                has_inf,
                "Infinite values detected in the output coordinates tensor from predict_3d_structure with mp_nerf.",
            )

            # Check atom count consistency
            expected_atoms_approx = len(sequence) * 10  # Rough estimate (backbone only)
            self.assertGreater(
                result_dict.get("atom_count", 0),
                expected_atoms_approx,
                "Atom count seems too low.",
            )

        except ValueError as e:
            self.fail(f"Predict_3d_structure raised ValueError: {e}")
        except Exception as e:
            self.fail(
                f"An unexpected error occurred during 3D structure prediction: {e}"
            )


if __name__ == "__main__":
    unittest.main()

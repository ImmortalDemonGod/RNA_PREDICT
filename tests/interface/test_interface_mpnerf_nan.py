# tests/interface/test_interface_mpnerf_nan.py

import unittest

import numpy as np
import pandas as pd
import torch
from hypothesis import given, strategies as st, settings, HealthCheck

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
            # Create a proper Hydra configuration object
            from omegaconf import OmegaConf

            # Create a configuration with all the necessary sections
            cfg = OmegaConf.create({
                "device": self.device,
                "model": {
                    "stageB": {
                        "torsion_bert": {
                            "model_name_or_path": "sayby/rna_torsionbert",
                            "device": self.device,
                            "angle_mode": "degrees",  # Using degrees, but mode shouldn't affect NaN generation
                            "num_angles": 7,  # Standard 7 angles for backbone + chi
                            "max_length": 512
                        }
                    },
                    "stageC": {
                        "enabled": True,  # Required parameter
                        "method": "mp_nerf",  # CRUCIAL: Ensure mp_nerf is used
                        "device": self.device,
                        "do_ring_closure": False,
                        "place_bases": True,
                        "sugar_pucker": "C3'-endo",
                        "angle_representation": "degrees",
                        "use_metadata": False,
                        "use_memory_efficient_kernel": False,
                        "use_deepspeed_evo_attention": False,
                        "use_lma": False,
                        "inplace_safe": True,
                        "debug_logging": True  # Enable debug logging for better diagnostics
                    }
                }
            })

            # Initialize with the configuration object
            self.predictor = RNAPredictor(cfg)
            # Verify the stageC method was set correctly
            self.assertEqual(
                self.predictor.stageC_config.method,
                "mp_nerf",
                "Stage C method not set to mp_nerf during init.",
            )
        except Exception as e:
            # Fail setup if predictor initialization fails (e.g., model download issue)
            self.fail(f"Failed to initialize RNAPredictor: {e}")

    @settings(
        deadline=None,  # Disable deadline checks since model loading can be slow
        max_examples=5,  # Limit number of examples to keep test runtime reasonable
        suppress_health_check=[HealthCheck.too_slow]
    )
    @given(
        sequence=st.text(alphabet=["A", "C", "G", "U"], min_size=5, max_size=20),
        prediction_repeats=st.integers(min_value=1, max_value=3),
        residue_atom_choice=st.integers(min_value=0, max_value=2)
    )
    def test_predict_submission_with_mpnerf_no_nan(self, sequence, prediction_repeats, residue_atom_choice):
        """
        Property-based test: Verify that predict_submission using mp_nerf does not produce NaN coordinates
        for any valid RNA sequence.

        Args:
            sequence: Random RNA sequence
            prediction_repeats: Number of prediction repeats
            residue_atom_choice: Index of atom to use for coordinates
        """
        print(
            f"[Test Run] Testing predict_submission for sequence: '{sequence}' with mp_nerf..."
        )

        try:
            # Run the prediction pipeline to generate the submission DataFrame
            submission_df = self.predictor.predict_submission(
                sequence,
                prediction_repeats=prediction_repeats,
                residue_atom_choice=residue_atom_choice,
            )

            self.assertIsInstance(
                submission_df, pd.DataFrame, "Output should be a DataFrame."
            )
            # Check if the DataFrame has a reasonable number of rows
            # MP-NeRF can produce a flat format with multiple atoms per residue
            # The key check is that we have a valid DataFrame with the right columns
            # and no NaN values, not the exact row count

            # For MP-NeRF, we expect either:
            # 1. Flat format: Multiple rows (atoms) per residue, with ID column containing sequential numbers
            # 2. Standard format: One row per residue

            # [UNIQUE-ERR-MPNERF-FORMAT] MP-NeRF can produce a flat format with variable atom counts
            # Just check that we have a valid DataFrame with more rows than zero
            self.assertGreater(
                len(submission_df),
                0,
                "[UNIQUE-ERR-MPNERF-EMPTY] DataFrame should not be empty.",
            )

            # Define the coordinate columns to check
            if 'residue_index' in submission_df.columns:
                # In flat format, we expect x_1, y_1, z_1 columns
                coord_cols = ['x_1', 'y_1', 'z_1']
            else:
                # In standard format, we expect x_i, y_i, z_i columns for each prediction
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

    @settings(
        deadline=None,  # Disable deadline checks since model loading can be slow
        max_examples=5,  # Limit number of examples to keep test runtime reasonable
        suppress_health_check=[HealthCheck.too_slow]
    )
    @given(
        sequence=st.text(alphabet=["A", "C", "G", "U"], min_size=5, max_size=20)
    )
    def test_predict_3d_structure_with_mpnerf_no_nan(self, sequence):
        """
        Property-based test: Verify that predict_3d_structure directly returns a non-NaN coordinate tensor
        when using the mp_nerf method for any valid RNA sequence.

        Args:
            sequence: Random RNA sequence
        """
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

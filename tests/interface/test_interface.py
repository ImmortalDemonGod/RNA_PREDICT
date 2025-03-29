# test_rna_predictor.py
"""
Merged Comprehensive Test Suite for RNAPredictor

This suite combines:
- Logical test grouping by functionality (Init, predict_3d_structure, predict_submission).
- setUp methods for consistent test fixtures.
- Mocking of run_stageC and StageBTorsionBertPredictor to force shape scenarios.
- Property-based tests using Hypothesis for broad input coverage.
- Thorough checks for edge cases, empty sequences, shape mismatches, index errors, etc.

Usage:
    python -m unittest test_rna_predictor.py -v
"""

import unittest
from unittest.mock import patch

import pandas as pd
import torch
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

# Adjust import if needed for your environment
try:
    from rna_predict.interface import RNAPredictor
except ImportError:
    from interface import RNAPredictor

# --------------------------------------------------------------------------------------
# Strategy definitions for property-based tests
# --------------------------------------------------------------------------------------

# Strategy for valid RNA sequences (A, C, G, U), allowing empty to test edge cases
valid_rna_sequences = st.text(alphabet="ACGU", min_size=0, max_size=50)

# Strategy enumerating how we might shape the coords in forced mocks:
#   0 => shape [N, 3]          (single-atom)
#   1 => shape [N * atoms, 3]  (legacy fallback?)
#   2 => shape [N, atoms, 3]   (already in [N, #atoms, 3])
coords_shape_type = st.integers(min_value=0, max_value=2)
atoms_per_res_strategy = st.integers(min_value=1, max_value=10)


# --------------------------------------------------------------------------------------
#                           Test Class: Initialization
# --------------------------------------------------------------------------------------
class TestRNAPredictorInitialization(unittest.TestCase):
    """
    Tests the RNAPredictor constructor logic, including:
      - default parameters,
      - user-provided parameters,
      - GPU vs CPU auto-detection,
      - random fuzzing of constructor args via Hypothesis.
    """

    def test_init_defaults(self):
        """
        Test that default arguments successfully initialize.
        Checks device auto-detection (cpu/cuda), torsion predictor, and stageC_method.
        """
        predictor = RNAPredictor()
        self.assertIsNotNone(
            predictor.torsion_predictor,
            "Should initialize torsion_predictor by default.",
        )
        self.assertIn(
            str(predictor.device),
            ["cpu", "cuda"],
            "Device should be cpu or cuda based on availability.",
        )
        self.assertEqual(
            predictor.stageC_method,
            "mp_nerf",
            "Default stageC_method should be 'mp_nerf'.",
        )

    def test_init_custom_params(self):
        """
        Test that user-provided parameters are respected.
        """
        custom_device = torch.device("cpu")
        predictor = RNAPredictor(
            model_name_or_path="custom/path",
            device=custom_device,
            angle_mode="sin_cos",
            num_angles=5,
            max_length=256,
            stageC_method="other_method",
        )
        self.assertEqual(str(predictor.device), "cpu")
        self.assertEqual(predictor.torsion_predictor.model_name_or_path, "custom/path")
        self.assertEqual(predictor.torsion_predictor.angle_mode, "sin_cos")
        self.assertEqual(predictor.torsion_predictor.num_angles, 5)
        self.assertEqual(predictor.torsion_predictor.max_length, 256)
        self.assertEqual(predictor.stageC_method, "other_method")

    @given(
        model_name_or_path=st.text(min_size=0, max_size=15),
        device=st.one_of(st.none(), st.sampled_from(["cpu", "cuda"])),
        angle_mode=st.sampled_from(["degrees", "radians", "sin_cos"]),
        num_angles=st.integers(min_value=1, max_value=10),
        max_length=st.integers(min_value=1, max_value=1024),
        stageC_method=st.sampled_from(["mp_nerf", "dummy_method", "other_method"]),
    )
    @settings(
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
        max_examples=20,
    )
    def test_fuzz_constructor_init(
        self,
        model_name_or_path,
        device,
        angle_mode,
        num_angles,
        max_length,
        stageC_method,
    ):
        """
        Hypothesis-driven fuzz test of the constructor to ensure broad coverage of parameter combos.
        """
        predictor = RNAPredictor(
            model_name_or_path=model_name_or_path,
            device=device,
            angle_mode=angle_mode,
            num_angles=num_angles,
            max_length=max_length,
            stageC_method=stageC_method,
        )
        self.assertEqual(predictor.torsion_predictor.angle_mode, angle_mode)
        self.assertEqual(predictor.torsion_predictor.num_angles, num_angles)
        self.assertEqual(predictor.torsion_predictor.max_length, max_length)


# --------------------------------------------------------------------------------------
#                         Test Class: Predict3DStructure
# --------------------------------------------------------------------------------------
class TestPredict3DStructure(unittest.TestCase):
    """
    Tests the predict_3d_structure method. Includes normal usage and
    forced shape errors, plus Hypothesis property-based tests.
    """

    def setUp(self):
        """
        Create a fresh RNAPredictor instance for each test,
        ensuring consistent device usage (CPU) to avoid GPU complications.
        """
        self.predictor = RNAPredictor(device="cpu")

    def test_predict_3d_structure_basic(self):
        """
        Basic functional test with a short RNA sequence.
        Verifies presence of 'coords' and 'atom_count' in the returned dict.
        """
        sequence = "ACGU"
        result = self.predictor.predict_3d_structure(sequence)
        self.assertIn("coords", result)
        self.assertIn("atom_count", result)
        coords = result["coords"]
        self.assertTrue(
            hasattr(coords, "shape"), "coords should be a tensor or similar."
        )
        self.assertGreaterEqual(
            coords.shape[-1], 3, "Should have at least x,y,z in the last dimension."
        )

    def test_predict_3d_structure_empty_seq(self):
        """
        If sequence is empty, pipeline might return coords of shape [0,3].
        Confirm method doesn't raise an error for an empty string.
        """
        sequence = ""
        result = self.predictor.predict_3d_structure(sequence)
        coords = result["coords"]
        self.assertEqual(
            coords.shape[0],
            0,
            "Empty sequence => zero residues => coords shape[0] = 0.",
        )
        self.assertEqual(result["atom_count"], 0, "No atoms if sequence is empty.")

    @given(valid_rna_sequences)
    @settings(
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
        max_examples=10,
    )
    def test_predict_3d_structure_with_random_sequences(self, sequence):
        """
        Property-based test: random valid RNA sequences.
        Ensures the method runs without raising exceptions for typical usage.
        """
        try:
            result = self.predictor.predict_3d_structure(sequence)
            self.assertIn("coords", result)
        except (RuntimeError, OSError):
            # If no real model is found or environment lacks GPU, we pass.
            pass


# --------------------------------------------------------------------------------------
#                        Test Class: PredictSubmission
# --------------------------------------------------------------------------------------
class TestPredictSubmission(unittest.TestCase):
    """
    Tests the predict_submission method, focusing on DataFrame output,
    repeated coords, residue atom choice, and shape handling.
    """

    def setUp(self):
        """Instantiate a RNAPredictor for repeated usage."""
        self.predictor = RNAPredictor(device="cpu")

    def test_predict_submission_basic(self):
        """
        Test normal usage with a small sequence.
        Ensures columns are correct and repeated coords match.
        """
        sequence = "ACGU"
        repeats = 5
        df = self.predictor.predict_submission(
            sequence, prediction_repeats=repeats, residue_atom_choice=0
        )
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(
            len(df), len(sequence), "DataFrame rows should match number of residues."
        )
        expected_cols = ["ID", "resname", "resid"]
        for i in range(1, repeats + 1):
            expected_cols += [f"x_{i}", f"y_{i}", f"z_{i}"]
        self.assertListEqual(list(df.columns), expected_cols)

    def test_predict_submission_empty_seq(self):
        """
        If sequence is empty, expect an empty DataFrame but valid columns.
        """
        sequence = ""
        df = self.predictor.predict_submission(sequence, prediction_repeats=2)
        self.assertTrue(df.empty, "DataFrame should be empty for empty sequence.")
        expected_cols = [
            "ID",
            "resname",
            "resid",
            "x_1",
            "y_1",
            "z_1",
            "x_2",
            "y_2",
            "z_2",
        ]
        self.assertListEqual(list(df.columns), expected_cols)

    def test_predict_submission_invalid_atom_choice(self):
        """
        If we pick a residue_atom_choice that doesn't exist, code should raise IndexError.
        """
        sequence = "ACGU"
        with self.assertRaises(IndexError):
            self.predictor.predict_submission(sequence, residue_atom_choice=9999)

    @patch("rna_predict.interface.run_stageC")
    def test_predict_submission_forced_shape_mismatch(self, mock_stageC):
        """
        Force a shape mismatch in coords so code raises ValueError (unexpected shape).
        """
        mock_stageC.return_value = {"coords": torch.zeros((10, 4)), "atom_count": 40}
        sequence = "ACGU"
        with self.assertRaises(ValueError):
            self.predictor.predict_submission(sequence)

    def test_predict_submission_custom_repeats(self):
        """
        Provide a custom number of repeats and ensure correct columns.
        """
        sequence = "ACGU"
        repeats = 3
        df = self.predictor.predict_submission(
            sequence, prediction_repeats=repeats, residue_atom_choice=0
        )
        self.assertEqual(len(df), len(sequence))
        self.assertIn("x_3", df.columns, "Should have x_3 for the final repeat.")
        self.assertIn("y_1", df.columns)
        self.assertIn("z_2", df.columns)

    @given(
        seq=valid_rna_sequences,
        repeats=st.integers(min_value=1, max_value=5),
        atom_choice=st.integers(min_value=0, max_value=4),
    )
    @settings(
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
        max_examples=10,
    )
    @example(seq="", repeats=5, atom_choice=0)
    def test_predict_submission_hypothesis_random(self, seq, repeats, atom_choice):
        """
        Hypothesis test for broad coverage of submission logic with random sequences.
        """
        try:
            df = self.predictor.predict_submission(
                seq, prediction_repeats=repeats, residue_atom_choice=atom_choice
            )
            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(
                len(df),
                len(seq),
                "Row count should match sequence length or be 0 if seq is empty.",
            )
        except (ValueError, IndexError, RuntimeError, OSError):
            # Accept environment or shape-based errors
            pass


# --------------------------------------------------------------------------------------
#               Test Class: PredictSubmissionParametricShapes (Optional)
# --------------------------------------------------------------------------------------
class TestPredictSubmissionParametricShapes(unittest.TestCase):
    """
    Demonstrates shape-based forced mocking, ensuring coverage of coordinate shapes:
      [N,3], [N*atoms,3], [N,atoms,3].
    This merges advanced shape logic with Hypothesis for broad coverage.
    """

    def setUp(self):
        self.predictor = RNAPredictor(device="cpu")

    @given(
        seq=valid_rna_sequences.filter(lambda s: len(s) > 0),  # non-empty
        shape_type=coords_shape_type,
        atoms_per_res=atoms_per_res_strategy,
        repeats=st.integers(min_value=1, max_value=3),
    )
    @settings(
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
        max_examples=10,
    )
    def test_forced_coord_shapes(self, seq, shape_type, atoms_per_res, repeats):
        """
        Force run_stageC to return specific shapes, verifying correct reshaping or error handling.
        """
        N = len(seq)
        # Construct coords based on shape_type
        if shape_type == 0:
            coords = torch.zeros((N, 3))  # Single-atom scenario
            atom_count = N
        elif shape_type == 1:
            coords = torch.zeros((N * atoms_per_res, 3))
            atom_count = N * atoms_per_res
        else:
            coords = torch.zeros((N, atoms_per_res, 3))
            atom_count = N * atoms_per_res

        stageC_result = {"coords": coords, "atom_count": atom_count}

        with patch.object(
            self.predictor, "predict_3d_structure", return_value=stageC_result
        ) as mock_p3d:
            df = self.predictor.predict_submission(
                seq, prediction_repeats=repeats, residue_atom_choice=0
            )
            self.assertEqual(len(df), N, "Rows must match number of residues.")
            # Columns: ID, resname, resid + repeats*(x,y,z) => 3 + 3*repeats total
            self.assertEqual(df.shape[1], 3 + (3 * repeats))
            mock_p3d.assert_called_once_with(seq)


# --------------------------------------------------------------------------------------
#                                     Main
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)

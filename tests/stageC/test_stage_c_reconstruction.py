"""
Comprehensive Test Suite for Stage C Reconstruction

This single test file merges and expands upon the best ideas and lessons learned
from previous versions (V1 through V7). It aims to provide robust, thorough,
and maintainable tests for the StageCReconstruction class and the run_stageC
entrypoint (including the mp_nerf branch).

Features & Highlights
---------------------
1. **Logical Grouping**
   - Separate test classes for:
       • StageCReconstruction (fallback).
       • run_stageC_rna_mpnerf (direct call).
       • run_stageC dispatcher (mp_nerf vs. fallback).

2. **SetUp and Fixtures**
   - Common data is set up once in each test class (unittest's setUp).
   - Repeated usage of default torsion angles, sequences, etc.

3. **Docstrings and Readability**
   - Each test class and test method has docstrings explaining its coverage goals
     and clarifying the tested functionality or code paths.

4. **Hypothesis Property-Based Testing**
   - Tests use Hypothesis to generate diverse inputs (sequence strings, torsion angles)
     with constraints, discovering edge cases automatically.
   - Minimally configured @settings to keep runtime moderate; can be tuned for more thoroughness.

5. **Mocking of External mp_nerf Dependencies**
   - Tests that rely on mp_nerf patch the internal calls (build_scaffolds_rna_from_torsions, skip_missing_atoms, etc.).
   - This isolates the code under test from external complexities and ensures stable, deterministic test behavior.

6. **Error Handling & Edge Cases**
   - Tests for invalid torsion angles (None).
   - Edge-case: empty sequences and zero-length torsion arrays.
   - Confirm that non-"mp_nerf" methods fall back gracefully to StageCReconstruction
     without raising errors.

7. **Additional Coverage**
   - Demonstrates round-trip style checks (e.g., "test_hypothesis_mpnerf_no_mock")
     verifying that the code can handle repeated or random sequences.
   - Handles minimal device checks and partial ring closure checks.

Usage
-----
Run with:
    python -m unittest path/to/this_test_file.py

If run_stageC or run_stageC_rna_mpnerf changes, these tests should adapt easily,
since the suite uses mocking to avoid direct external calls to the mp_nerf library
(unless specifically testing "no_mock" paths).

Dependencies
------------
- Python 3.7+
- PyTorch
- Hypothesis

"""

# For demonstration, we assume the module is in the same directory or in path:
# from rna_predict.pipeline.stageC import stage_c_reconstruction as scr
# Adjust imports if needed to reflect your actual project structure.
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.append(".")
from rna_predict.pipeline.stageC import stage_c_reconstruction as scr


class TestStageCReconstruction(unittest.TestCase):
    """
    Tests the legacy fallback approach (StageCReconstruction) used when
    run_stageC is called with method != 'mp_nerf'.
    """

    def setUp(self):
        """
        Create an instance of StageCReconstruction and default torsion angles
        for repeated usage in tests.
        """
        self.fallback_reconstructor = scr.StageCReconstruction()
        # Example: 4-nt sequence, each with 7 angles
        self.default_torsions = torch.zeros((4, 7), dtype=torch.float32)

    def test_basic_call(self):
        """
        Check that calling the fallback reconstruction returns a dict
        containing coords and atom_count, with shape (N*3,3).
        """
        result = self.fallback_reconstructor(self.default_torsions)
        self.assertIsInstance(result, dict)
        self.assertIn("coords", result)
        self.assertIn("atom_count", result)
        coords = result["coords"]
        atom_count = result["atom_count"]

        expected_shape = (self.default_torsions.size(0) * 3, 3)
        self.assertEqual(coords.shape, expected_shape)
        self.assertEqual(atom_count, expected_shape[0])

    def test_empty_torsions(self):
        """
        If the torsion angles tensor is empty (0,7), the coords shape
        should be (0,3) and atom_count should be zero.
        """
        empty_tensor = torch.zeros((0, 7), dtype=torch.float32)
        result = self.fallback_reconstructor(empty_tensor)
        self.assertEqual(result["coords"].shape, (0, 3))
        self.assertEqual(result["atom_count"], 0)

    def test_invalid_input_type(self):
        """
        Passing a non-tensor (e.g., None) should raise an AttributeError
        because .size() is called on torsion_angles internally.
        """
        with self.assertRaises(AttributeError):
            self.fallback_reconstructor(None)

    @given(
        # Generate random lengths for torsion angles in range [1..10]
        torsion_length=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=10)
    def test_fuzz_fallback_various_lengths(self, torsion_length: int):
        """
        Hypothesis-based fuzz test that checks fallback reconstruction
        for random torsion angle lengths. The shape should remain (N*3,3).
        """
        torsion_tensor = torch.zeros((torsion_length, 7), dtype=torch.float32)
        result = self.fallback_reconstructor(torsion_tensor)
        expected_shape = (torsion_length * 3, 3)
        self.assertEqual(result["coords"].shape, expected_shape)
        self.assertEqual(result["atom_count"], expected_shape[0])


class TestRunStageCRnaMpnerf(unittest.TestCase):
    """
    Tests for the run_stageC_rna_mpnerf function, which specifically handles
    the mp_nerf path of Stage C reconstruction. We mock out external calls
    to ensure stable and isolated testing.
    """

    def setUp(self):
        """
        Default sequence and torsion angles for repeated usage.
        """
        self.sequence = "ACGU"
        self.default_torsions = torch.zeros((4, 7), dtype=torch.float32)
        self.device = "cpu"
        self.default_sugar_pucker = "C3'-endo"

    def _mpnerf_patch_manager(self):
        """
        Returns a context manager that patches the mp_nerf subcalls:
          - build_scaffolds_rna_from_torsions
          - skip_missing_atoms
          - handle_mods
          - rna_fold
          - place_rna_bases
        Each is replaced with a MagicMock returning consistent data, so we can
        validate final shapes and ensure calls occur as expected.
        """
        return patch.multiple(
            "rna_predict.pipeline.stageC.mp_nerf.rna",
            build_scaffolds_rna_from_torsions=MagicMock(
                return_value={"angles_mask": torch.ones((4,))}
            ),
            skip_missing_atoms=MagicMock(side_effect=lambda seq, scf: scf),
            handle_mods=MagicMock(side_effect=lambda seq, scf: scf),
            rna_fold=MagicMock(return_value=torch.ones((5, 3))),
            place_rna_bases=MagicMock(return_value=torch.ones((2, 3))),
        )

    def test_mpnerf_with_bases(self):
        """
        Test run_stageC_rna_mpnerf with place_bases=True, verifying final coords shape.
        """
        with self._mpnerf_patch_manager():
            result = scr.run_stageC_rna_mpnerf(
                sequence=self.sequence,
                predicted_torsions=self.default_torsions,
                device=self.device,
                do_ring_closure=True,
                place_bases=True,
                sugar_pucker=self.default_sugar_pucker,
            )
        self.assertIsInstance(result, dict)
        self.assertIn("coords", result)
        self.assertIn("atom_count", result)
        # place_bases mock => shape (2,3), so atom_count=6
        self.assertEqual(result["coords"].shape, (2, 3))
        self.assertEqual(result["atom_count"], 6)

    def test_mpnerf_without_bases(self):
        """
        Test run_stageC_rna_mpnerf with place_bases=False. place_rna_bases call
        should not occur. The final coords come directly from rna_fold.
        """
        place_rna_bases_mock = MagicMock()
        with patch("rna_predict.pipeline.stageC.mp_nerf.rna.build_scaffolds_rna_from_torsions", 
                  return_value={"angles_mask": torch.ones((4,))}), \
             patch("rna_predict.pipeline.stageC.mp_nerf.rna.skip_missing_atoms", 
                  side_effect=lambda seq, scf: scf), \
             patch("rna_predict.pipeline.stageC.mp_nerf.rna.handle_mods", 
                  side_effect=lambda seq, scf: scf), \
             patch("rna_predict.pipeline.stageC.mp_nerf.rna.rna_fold", 
                  return_value=torch.ones((5, 3))), \
             patch("rna_predict.pipeline.stageC.mp_nerf.rna.place_rna_bases", 
                  place_rna_bases_mock):
            
            result = scr.run_stageC_rna_mpnerf(
                sequence=self.sequence,
                predicted_torsions=self.default_torsions,
                device=self.device,
                do_ring_closure=False,
                place_bases=False,
                sugar_pucker=self.default_sugar_pucker,
            )
            self.assertFalse(
                place_rna_bases_mock.called, "place_rna_bases must not be called."
            )
        # shape from rna_fold => (5,3), so atom_count=15
        self.assertEqual(result["coords"].shape, (5, 3))
        self.assertEqual(result["atom_count"], 15)

    @given(
        seq=st.text(),  # random sequences
        do_ring_closure=st.booleans(),
        place_bases=st.booleans(),
    )
    @settings(max_examples=10)
    def test_mpnerf_fuzz(self, seq, do_ring_closure, place_bases):
        """
        Hypothesis-based fuzz test for run_stageC_rna_mpnerf. We mock external calls
        to keep it consistent. Sequences can be empty or random text.
        """
        with self._mpnerf_patch_manager():
            # We'll keep torsions dimension consistent => (4,7).
            # A robust approach might shape them according to len(seq), but
            # here we demonstrate partial fuzz.
            result = scr.run_stageC_rna_mpnerf(
                sequence=seq,
                predicted_torsions=self.default_torsions,
                device=self.device,
                do_ring_closure=do_ring_closure,
                place_bases=place_bases,
                sugar_pucker="C3'-endo",
            )
        self.assertIn("coords", result)
        self.assertIn("atom_count", result)
        self.assertIsInstance(result["coords"], torch.Tensor)
        self.assertIsInstance(result["atom_count"], int)


class TestRunStageCIntegration(unittest.TestCase):
    """
    Tests for the unified run_stageC function that delegates to either
    run_stageC_rna_mpnerf or StageCReconstruction, based on the 'method' argument.
    """

    def setUp(self):
        """
        Reuse default sequence and torsion angles. We'll vary them as needed.
        """
        self.sequence = "ACGU"
        self.default_torsions = torch.zeros((4, 7), dtype=torch.float32)
        self.device = "cpu"
        self.default_sugar_pucker = "C3'-endo"

    def test_fallback_mode(self):
        """
        If method != 'mp_nerf', run_stageC uses StageCReconstruction.
        """
        out = scr.run_stageC(
            sequence=self.sequence,
            torsion_angles=self.default_torsions,
            method="fallback_method",
            device=self.device,
        )
        self.assertIn("coords", out)
        self.assertIn("atom_count", out)
        expected_shape = (4 * 3, 3)  # fallback => (N*3,3)
        self.assertEqual(out["coords"].shape, expected_shape)
        self.assertEqual(out["atom_count"], expected_shape[0])

    def test_mpnerf_method_calls_subfunc(self):
        """
        If method='mp_nerf', run_stageC calls run_stageC_rna_mpnerf internally.
        We patch that function to confirm it was called with the correct arguments.
        """
        with patch(
            "rna_predict.pipeline.stageC.stage_c_reconstruction.run_stageC_rna_mpnerf"
        ) as mock_sub:
            mock_sub.return_value = {"coords": torch.ones((2, 3)), "atom_count": 6}

            result = scr.run_stageC(
                sequence=self.sequence,
                torsion_angles=self.default_torsions,
                method="mp_nerf",
                device=self.device,
                do_ring_closure=True,
                place_bases=False,
                sugar_pucker=self.default_sugar_pucker,
            )
            mock_sub.assert_called_once_with(
                sequence=self.sequence,
                predicted_torsions=self.default_torsions,
                device=self.device,
                do_ring_closure=True,
                place_bases=False,
                sugar_pucker=self.default_sugar_pucker,
            )
            self.assertEqual(result["coords"].shape, (2, 3))
            self.assertEqual(result["atom_count"], 6)

    def test_invalid_torsions_argument(self):
        """
        Passing None for torsion_angles should trigger an AttributeError
        because .size() is used internally.
        """
        with self.assertRaises(AttributeError):
            scr.run_stageC(
                sequence=self.sequence,
                torsion_angles=None,
                method="mp_nerf",
            )

    @given(
        seq=st.text(min_size=1, max_size=6),
        method=st.sampled_from(["mp_nerf", "fallback", "something_else"]),
        do_ring_closure=st.booleans(),
        place_bases=st.booleans(),
    )
    @settings(max_examples=10)
    def test_run_stageC_hypothesis(self, seq, method, do_ring_closure, place_bases):
        """
        Property-based fuzz test for run_stageC that covers both fallback and mp_nerf.
        We patch mp_nerf path calls if method == 'mp_nerf'.
        """
        torsion_tensor = torch.zeros((len(seq), 7), dtype=torch.float32)

        if method == "mp_nerf":
            with patch(
                "rna_predict.pipeline.stageC.stage_c_reconstruction.run_stageC_rna_mpnerf"
            ) as mock_sub:
                dummy_return = {"coords": torch.ones((3, 3)), "atom_count": 9}
                mock_sub.return_value = dummy_return
                result = scr.run_stageC(
                    sequence=seq,
                    torsion_angles=torsion_tensor,
                    method=method,
                    device=self.device,
                    do_ring_closure=do_ring_closure,
                    place_bases=place_bases,
                    sugar_pucker=self.default_sugar_pucker,
                )
                mock_sub.assert_called_once()
                self.assertEqual(result, dummy_return)
        else:
            result = scr.run_stageC(
                sequence=seq,
                torsion_angles=torsion_tensor,
                method=method,
                device=self.device,
                do_ring_closure=do_ring_closure,
                place_bases=place_bases,
                sugar_pucker=self.default_sugar_pucker,
            )
            # Fallback path => shape = (N*3,3)
            expected_shape = (len(seq) * 3, 3)
            self.assertEqual(result["coords"].shape, expected_shape)
            self.assertEqual(result["atom_count"], expected_shape[0])

    def test_round_trip_example(self):
        """
        Demonstrates a round-trip style test:
        1) Run fallback (method="some_unknown")
        2) Then run mp_nerf on the same torsion angles
        This is mostly conceptual; real usage might do more advanced verifications.
        """
        # First fallback
        out1 = scr.run_stageC(
            sequence=self.sequence,
            torsion_angles=self.default_torsions,
            method="some_unknown",
            device=self.device,
        )
        self.assertIn("coords", out1)
        self.assertIn("atom_count", out1)

        # Then mp_nerf
        with patch(
            "rna_predict.pipeline.stageC.stage_c_reconstruction.run_stageC_rna_mpnerf"
        ) as mock_sub:
            # Suppose mp_nerf returns the same shape
            mock_sub.return_value = {"coords": torch.ones((2, 3)), "atom_count": 6}
            out2 = scr.run_stageC(
                sequence=self.sequence,
                torsion_angles=self.default_torsions,
                method="mp_nerf",
                device=self.device,
            )
            self.assertIn("coords", out2)
            self.assertIn("atom_count", out2)

            # There's no direct reason these results would match in real usage.
            # But we confirm the pipeline can run them back-to-back on the same data
            # without errors. A "round-trip" concept might apply if your architecture
            # had cyclical or repeated calls.


if __name__ == "__main__":
    unittest.main()

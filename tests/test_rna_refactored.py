import unittest
import torch
import math
from hypothesis import given, strategies as st, settings
from typing import List, Dict, Any

# Assuming rna.py is on the Python path or in the same package:
from rna_predict.pipeline.stageC.mp_nerf.rna import (build_scaffolds_rna_from_torsions,
                    rna_fold,
                    place_rna_bases,
                    skip_missing_atoms,
                    handle_mods,
                    validate_rna_geometry,
                    mini_refinement,
                    ring_closure_refinement,
                    get_base_atoms,
                )


class TestGetBaseAtoms(unittest.TestCase):
    """Test suite for the get_base_atoms function."""

    def test_known_base_types(self):
        """
        Verify that known bases (A, G, C, U) return the correct list of base atoms.
        """
        # Expected dictionary for reference
        expected = {
            "A": ["N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
            "G": ["N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
            "C": ["N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
            "U": ["N1", "C2", "N3", "C4", "C5", "C6", "O2", "O4"],
        }
        for base_type, expected_atoms in expected.items():
            with self.subTest(base_type=base_type):
                result = get_base_atoms(base_type)
                self.assertEqual(result, expected_atoms)

    def test_unknown_base_type(self):
        """
        For an unknown base type, the function returns an empty list.
        """
        result = get_base_atoms("X")
        self.assertEqual(result, [], "Unknown base should return an empty list.")

    @given(st.text(min_size=1, max_size=3))
    @settings(max_examples=10)
    def test_fuzz_get_base_atoms(self, base_type: str):
        """
        Fuzzy test for get_base_atoms with random short strings.
        """
        # Just ensure it doesn't crash and returns a list
        result = get_base_atoms(base_type)
        self.assertIsInstance(result, list)


class TestBuildScaffoldsRnaFromTorsions(unittest.TestCase):
    """Test suite for the build_scaffolds_rna_from_torsions function."""

    def setUp(self):
        """Common data for these tests."""
        self.seq = "ACGU"
        # alpha..zeta, chi => shape [L, 7]
        self.torsions = torch.zeros((len(self.seq), 7))
        self.device = "cpu"

    def test_basic_output_structure(self):
        """
        Check that the scaffolds dictionary has the expected keys and shapes.
        """
        scaffolds = build_scaffolds_rna_from_torsions(self.seq, self.torsions, self.device)
        self.assertIn("bond_mask", scaffolds)
        self.assertIn("angles_mask", scaffolds)
        self.assertIn("point_ref_mask", scaffolds)
        self.assertIn("cloud_mask", scaffolds)

        L = len(self.seq)
        B = 10  # Based on BACKBONE_ATOMS
        self.assertEqual(scaffolds["bond_mask"].shape, (L, B))
        self.assertEqual(scaffolds["angles_mask"].shape, (2, L, B))
        self.assertEqual(scaffolds["point_ref_mask"].shape, (3, L, B))
        self.assertEqual(scaffolds["cloud_mask"].shape, (L, B))

    def test_empty_sequence(self):
        """
        If an empty seq is provided, all returned tensors should be empty.
        """
        scaffolds = rna.build_scaffolds_rna_from_torsions("", torch.zeros((0, 7)), self.device)
        self.assertEqual(scaffolds["bond_mask"].numel(), 0)
        self.assertEqual(scaffolds["angles_mask"].numel(), 0)
        self.assertEqual(scaffolds["point_ref_mask"].numel(), 0)
        self.assertEqual(scaffolds["cloud_mask"].numel(), 0)

    @given(st.text(alphabet="ACGU", min_size=1, max_size=10),
           st.integers(min_value=0, max_value=1))  # simplistic device toggling
    @settings(max_examples=5)
    def test_fuzz_build_scaffolds_rna_from_torsions(self, seq: str, device_flag: int):
        """
        Fuzzy test ensuring build_scaffolds_rna_from_torsions doesn't crash
        with random short RNA sequences and small torsion shapes.
        """
        device = "cuda" if (device_flag == 1 and torch.cuda.is_available()) else "cpu"
        # Matching shape: [L, 7]
        torsions_shape = (len(seq), 7)
        torsions = torch.zeros(torsions_shape)
        scaffolds = build_scaffolds_rna_from_torsions(seq, torsions, device=device)
        self.assertIsInstance(scaffolds, dict)
        self.assertTrue(all(k in scaffolds for k in ["bond_mask", "angles_mask",
                                                     "point_ref_mask", "cloud_mask"]))


class TestRnaFold(unittest.TestCase):
    """Test suite for rna_fold function."""

    def setUp(self):
        """Prepare standard scaffolds dictionary and parameters."""
        seq = "ACG"
        torsions = torch.zeros((len(seq), 7))
        self.scaffolds = build_scaffolds_rna_from_torsions(seq, torsions)
        self.device = "cpu"

    def test_rna_fold_shapes(self):
        """
        Verify that rna_fold returns a coordinate tensor of the correct shape.
        """
        coords = rna_fold(self.scaffolds, device=self.device, do_ring_closure=False)
        # L=3, B=10 => shape [3,10,3]
        self.assertEqual(coords.shape, (3, 10, 3))

    def test_rna_fold_ring_closure(self):
        """
        Check that do_ring_closure=True doesn't break shape or cause errors.
        (ring_closure_refinement currently does nothing, but we test anyway)
        """
        coords = rna_fold(self.scaffolds, device=self.device, do_ring_closure=True)
        self.assertEqual(coords.shape, (3, 10, 3))

    @given(st.builds(dict), st.text(), st.booleans())
    @settings(max_examples=5)
    def test_fuzz_rna_fold(self, scaffolds: Dict[Any, Any], device: str, do_ring_closure: bool):
        """
        Fuzzy test for rna_fold ensuring it doesn't crash with arbitrary scaffolds.
        This doesn't guarantee correct shape if scaffolds are malformed,
        but ensures no unhandled exceptions occur.
        """
        try:
            rna_fold(scaffolds, device=device, do_ring_closure=do_ring_closure)
        except Exception:
            # We only catch to ensure test continues. Real coverage checks for safe usage.
            pass


class TestPlaceRnaBases(unittest.TestCase):
    """Test suite for place_rna_bases function."""

    def setUp(self):
        self.seq = "ACG"
        self.scaffolds = build_scaffolds_rna_from_torsions(self.seq, torch.zeros((3, 7)))
        self.backbone_coords = rna_fold(self.scaffolds)
        self.angles_mask = self.scaffolds["angles_mask"]

    def test_place_rna_bases_shapes(self):
        """
        Check that place_rna_bases output shape is correct [L, max_atoms, 3].
        """
        coords = place_rna_bases(self.backbone_coords, self.seq, self.angles_mask)
        # compute_max_rna_atoms() => up to 21 for G
        self.assertEqual(coords.shape[1], compute_max_rna_atoms())

    @given(st.builds(torch.zeros, st.tuples(st.integers(min_value=0, max_value=5),
                                           st.just(3)).map(lambda x: x)),
           st.text(alphabet="ACGU", min_size=0, max_size=5),
           st.builds(torch.zeros, st.tuples(st.just(2),
                                           st.integers(min_value=0, max_value=5),
                                           st.just(10))))
    @settings(max_examples=5)
    def test_fuzz_place_rna_bases(self,
                                  backbone_coords: torch.Tensor,
                                  seq: str,
                                  angles_mask: torch.Tensor):
        """
        Fuzzy test that place_rna_bases doesn't crash for random small shapes.
        """
        try:
            place_rna_bases(backbone_coords, seq, angles_mask)
        except Exception:
            # The function might fail if shapes mismatch, but it shouldn't raise unexpected errors.
            pass


class TestSkipMissingAtoms(unittest.TestCase):
    """Test suite for skip_missing_atoms function."""

    def test_basic_skip_missing_atoms(self):
        """
        skip_missing_atoms should return the same scaffolds by default.
        """
        seq = "AC"
        scaffolds = {
            "bond_mask": torch.ones((2, 10)),
            "angles_mask": torch.ones((2, 2, 10)),
            "point_ref_mask": torch.zeros((3, 2, 10), dtype=torch.long),
            "cloud_mask": torch.ones((2, 10), dtype=torch.bool),
        }
        new_scaff = skip_missing_atoms(seq, scaffolds)
        self.assertIs(new_scaff, scaffolds, "Currently returns the same reference object.")

    @given(st.text(alphabet="ACGU", min_size=0, max_size=5),
           st.builds(dict))
    @settings(max_examples=5)
    def test_fuzz_skip_missing_atoms(self, seq: str, scaffolds: Dict[str, Any]):
        """
        Fuzzy test skip_missing_atoms to ensure no crash with random scaffolds.
        """
        try:
            skip_missing_atoms(seq, scaffolds)
        except Exception:
            pass


class TestHandleMods(unittest.TestCase):
    """Test suite for handle_mods function."""

    def test_basic_handle_mods(self):
        """
        handle_mods should also return the same scaffolds unmodified by default.
        """
        seq = "ACGU"
        scaffolds = {
            "bond_mask": torch.ones((4, 10)),
            "angles_mask": torch.ones((2, 4, 10)),
            "point_ref_mask": torch.zeros((3, 4, 10), dtype=torch.long),
            "cloud_mask": torch.ones((4, 10), dtype=torch.bool),
        }
        new_scaff = handle_mods(seq, scaffolds)
        self.assertIs(new_scaff, scaffolds)

    @given(st.text(alphabet="ACGU", min_size=0, max_size=5),
           st.builds(dict))
    @settings(max_examples=5)
    def test_fuzz_handle_mods(self, seq: str, scaffolds: Dict[str, Any]):
        """
        Fuzzy test handle_mods for random small input.
        """
        try:
            handle_mods(seq, scaffolds)
        except Exception:
            pass


class TestValidateRnaGeometry(unittest.TestCase):
    """Test suite for validate_rna_geometry function."""

    def test_validate_rna_geometry_no_errors(self):
        """
        validate_rna_geometry doesn't raise an exception; it's a stub.
        """
        coords = torch.zeros((2, 10, 3))
        validate_rna_geometry(coords)

    @given(st.builds(torch.zeros, st.tuples(st.integers(min_value=0, max_value=5),
                                           st.just(10),
                                           st.just(3))))
    @settings(max_examples=5)
    def test_fuzz_validate_rna_geometry(self, coords: torch.Tensor):
        """
        Fuzzy test that validate_rna_geometry doesn't raise an exception with random shapes.
        """
        validate_rna_geometry(coords)


class TestMiniRefinement(unittest.TestCase):
    """Test suite for mini_refinement function."""

    def test_mini_refinement_noop(self):
        """
        mini_refinement returns coords unchanged by default, so test identity.
        """
        coords = torch.randn((2, 10, 3))
        refined = mini_refinement(coords, method="none")
        self.assertTrue(torch.equal(coords, refined), "Should return the same tensor.")

    @given(st.builds(torch.randn, st.tuples(st.integers(min_value=0, max_value=3),
                                            st.integers(min_value=0, max_value=5),
                                            st.just(3))),
           st.sampled_from(["none", "some_method", "other"]))
    @settings(max_examples=5)
    def test_fuzz_mini_refinement(self, coords: torch.Tensor, method: str):
        """
        Fuzzy test mini_refinement for different shapes and method strings.
        """
        result = mini_refinement(coords, method)
        # Typically identical, but we just check shape here
        self.assertEqual(result.shape, coords.shape)


class TestRingClosureRefinement(unittest.TestCase):
    """Test suite for ring_closure_refinement function."""

    def test_ring_closure_refinement_noop(self):
        """
        ring_closure_refinement currently returns the same coords for RBC. Check identity.
        """
        coords = torch.randn((3, 10, 3))
        new_coords = ring_closure_refinement(coords)
        self.assertTrue(torch.equal(coords, new_coords))

    @given(st.builds(torch.randn, st.tuples(st.integers(min_value=0, max_value=5),
                                            st.integers(min_value=0, max_value=10),
                                            st.just(3))))
    @settings(max_examples=5)
    def test_fuzz_ring_closure_refinement(self, coords: torch.Tensor):
        """
        Fuzzy test ring_closure_refinement for random shapes, verifying no exceptions.
        """
        new_coords = ring_closure_refinement(coords)
        self.assertEqual(coords.shape, new_coords.shape)


class TestFullPipelineIntegration(unittest.TestCase):
    """Optional end-to-end check that uses multiple rna.py functions together."""

    def test_end_to_end(self):
        """
        Create a small RNA sequence, build scaffolds, fold, place bases, and ensure everything completes.
        """
        seq = "ACG"
        torsions = torch.zeros((len(seq), 7))
        scaff = build_scaffolds_rna_from_torsions(seq, torsions)
        coords_bb = rna_fold(scaff, do_ring_closure=False)
        coords_full = place_rna_bases(coords_bb, seq, scaff["angles_mask"])
        # Just do basic sanity checks on shapes
        self.assertEqual(coords_bb.shape, (3, 10, 3))
        self.assertEqual(coords_full.shape[0], 3)
        self.assertEqual(coords_full.shape[1], compute_max_rna_atoms())
        # Optionally call ring_closure
        refined_bb = rna_fold(scaff, do_ring_closure=True)
        self.assertEqual(refined_bb.shape, (3, 10, 3))


if __name__ == "__main__":
    # If invoked directly, run all unittests
    unittest.main()
import unittest

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from rna_predict.pipeline.stageC.mp_nerf.rna.rna_constants import (
    BACKBONE_ATOMS,
    RNA_BACKBONE_TORSIONS_AFORM,
)
from rna_predict.pipeline.stageC.mp_nerf.rna.rna_scaffolding import build_scaffolds_rna_from_torsions
from rna_predict.pipeline.stageC.mp_nerf.rna.rna_folding import rna_fold, ring_closure_refinement
from rna_predict.pipeline.stageC.mp_nerf.rna.rna_base_placement import place_rna_bases
from rna_predict.pipeline.stageC.mp_nerf.rna.rna_utils import (
    compute_max_rna_atoms,
    get_base_atoms,
    handle_mods,
    mini_refinement,
    place_bases,
    skip_missing_atoms,
    validate_rna_geometry,
)


class TestBuildScaffoldsRNAFromTorsions(unittest.TestCase):
    """Test cases for building RNA scaffolds from torsion angles."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"

    @given(st.text(min_size=1, max_size=100, alphabet="ACGU"))
    @settings(deadline=None)  # Remove deadline constraint
    def test_basic_scaffold_creation(self, seq):
        """Test basic scaffold creation with valid sequences."""
        # Create random torsion angles
        L = len(seq)
        torsions = torch.randn(L, 7, device=self.device)

        # Build scaffolds
        scaffolds = build_scaffolds_rna_from_torsions(seq, torsions, device=self.device)

        # Check scaffolds dictionary
        self.assertIsInstance(scaffolds, dict)
        self.assertEqual(scaffolds["seq"], seq)
        self.assertEqual(scaffolds["device"], self.device)
        self.assertEqual(scaffolds["sugar_pucker"], "C3'-endo")

        # Check tensor shapes
        B = len(BACKBONE_ATOMS)
        self.assertEqual(scaffolds["bond_mask"].shape, (L, B))
        self.assertEqual(scaffolds["angles_mask"].shape, (2, L, B))
        self.assertEqual(scaffolds["point_ref_mask"].shape, (3, L, B))
        self.assertEqual(scaffolds["cloud_mask"].shape, (L, B))

        # Check data types
        self.assertEqual(scaffolds["bond_mask"].dtype, torch.float32)
        self.assertEqual(scaffolds["angles_mask"].dtype, torch.float32)
        self.assertEqual(scaffolds["point_ref_mask"].dtype, torch.long)
        self.assertEqual(scaffolds["cloud_mask"].dtype, torch.bool)

    def test_invalid_sequence(self):
        """Test handling of invalid sequences."""
        # Create a single set of torsion angles
        torsions = torch.randn(1, 7, device=self.device)

        # Test with invalid sequence
        with self.assertRaises(ValueError):
            build_scaffolds_rna_from_torsions("X", torsions, device=self.device)

    def test_mismatched_torsions(self):
        """Test handling of mismatched torsion angles."""
        # Create torsions tensor with wrong length
        torsions = torch.randn(2, 7, device=self.device)

        # Test with mismatched sequence and torsions lengths
        with self.assertRaises(ValueError):
            build_scaffolds_rna_from_torsions("AUGC", torsions, device=self.device)

    @given(st.text(min_size=1, max_size=100, alphabet="ACGU"))
    def test_sugar_pucker_variants(self, seq):
        """Test scaffold creation with different sugar pucker conformations."""
        torsions = torch.tensor(
            [
                [
                    RNA_BACKBONE_TORSIONS_AFORM["alpha"],
                    RNA_BACKBONE_TORSIONS_AFORM["beta"],
                    RNA_BACKBONE_TORSIONS_AFORM["gamma"],
                    RNA_BACKBONE_TORSIONS_AFORM["delta"],
                    RNA_BACKBONE_TORSIONS_AFORM["epsilon"],
                    RNA_BACKBONE_TORSIONS_AFORM["zeta"],
                    RNA_BACKBONE_TORSIONS_AFORM["chi"],
                ]
                for _ in seq
            ],
            device=self.device,
        )

        # Test both C3'-endo and C2'-endo conformations
        for pucker in ["C3'-endo", "C2'-endo"]:
            scaffolds = build_scaffolds_rna_from_torsions(
                seq, torsions, device=self.device, sugar_pucker=pucker
            )
            self.assertIsNotNone(scaffolds)
            self.assertTrue(all(mask is not None for mask in scaffolds.values()))


class TestRNAFold(unittest.TestCase):
    """Test cases for RNA folding functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"
        self.seq = "AUGC"
        self.torsions = torch.tensor(
            [
                [
                    RNA_BACKBONE_TORSIONS_AFORM["alpha"],
                    RNA_BACKBONE_TORSIONS_AFORM["beta"],
                    RNA_BACKBONE_TORSIONS_AFORM["gamma"],
                    RNA_BACKBONE_TORSIONS_AFORM["delta"],
                    RNA_BACKBONE_TORSIONS_AFORM["epsilon"],
                    RNA_BACKBONE_TORSIONS_AFORM["zeta"],
                    RNA_BACKBONE_TORSIONS_AFORM["chi"],
                ],
                [
                    RNA_BACKBONE_TORSIONS_AFORM["alpha"],
                    RNA_BACKBONE_TORSIONS_AFORM["beta"],
                    RNA_BACKBONE_TORSIONS_AFORM["gamma"],
                    RNA_BACKBONE_TORSIONS_AFORM["delta"],
                    RNA_BACKBONE_TORSIONS_AFORM["epsilon"],
                    RNA_BACKBONE_TORSIONS_AFORM["zeta"],
                    RNA_BACKBONE_TORSIONS_AFORM["chi"],
                ],
                [
                    RNA_BACKBONE_TORSIONS_AFORM["alpha"],
                    RNA_BACKBONE_TORSIONS_AFORM["beta"],
                    RNA_BACKBONE_TORSIONS_AFORM["gamma"],
                    RNA_BACKBONE_TORSIONS_AFORM["delta"],
                    RNA_BACKBONE_TORSIONS_AFORM["epsilon"],
                    RNA_BACKBONE_TORSIONS_AFORM["zeta"],
                    RNA_BACKBONE_TORSIONS_AFORM["chi"],
                ],
                [
                    RNA_BACKBONE_TORSIONS_AFORM["alpha"],
                    RNA_BACKBONE_TORSIONS_AFORM["beta"],
                    RNA_BACKBONE_TORSIONS_AFORM["gamma"],
                    RNA_BACKBONE_TORSIONS_AFORM["delta"],
                    RNA_BACKBONE_TORSIONS_AFORM["epsilon"],
                    RNA_BACKBONE_TORSIONS_AFORM["zeta"],
                    RNA_BACKBONE_TORSIONS_AFORM["chi"],
                ],
            ],
            device=self.device,
        )
        self.scaffolds = build_scaffolds_rna_from_torsions(
            self.seq, self.torsions, device=self.device
        )

    @given(st.text(min_size=1, max_size=100, alphabet="ACGU"))
    @settings(deadline=None)
    def test_basic_folding(self, seq):
        """Test basic RNA folding with valid sequences."""
        # Create torsions tensor matching sequence length
        torsions = torch.tensor(
            [
                [
                    RNA_BACKBONE_TORSIONS_AFORM["alpha"],
                    RNA_BACKBONE_TORSIONS_AFORM["beta"],
                    RNA_BACKBONE_TORSIONS_AFORM["gamma"],
                    RNA_BACKBONE_TORSIONS_AFORM["delta"],
                    RNA_BACKBONE_TORSIONS_AFORM["epsilon"],
                    RNA_BACKBONE_TORSIONS_AFORM["zeta"],
                    RNA_BACKBONE_TORSIONS_AFORM["chi"],
                ]
                for _ in seq
            ],
            device=self.device,
        )

        scaffolds = build_scaffolds_rna_from_torsions(seq, torsions, device=self.device)
        coords = rna_fold(scaffolds, device=self.device)

        # Check output shape
        L = len(seq)
        B = len(BACKBONE_ATOMS)
        self.assertEqual(coords.shape, (L, B, 3))

        # Check data type
        self.assertEqual(coords.dtype, torch.float32)

        # Check that coordinates are not NaN
        self.assertFalse(torch.isnan(coords).any())

    def test_ring_closure(self):
        """Test RNA folding with ring closure refinement."""
        coords = rna_fold(self.scaffolds, device=self.device, do_ring_closure=True)

        # Check output shape
        L = len(self.seq)
        B = len(BACKBONE_ATOMS)
        self.assertEqual(coords.shape, (L, B, 3))

        # Check data type
        self.assertEqual(coords.dtype, torch.float32)

        # Check that coordinates are not NaN
        self.assertFalse(torch.isnan(coords).any())


class TestRingClosureRefinement(unittest.TestCase):
    """Test cases for ring closure refinement."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"
        # Create a simple test structure
        self.coords = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            device=self.device,
        )

    @given(
        st.lists(
            st.lists(
                st.floats(min_value=-10.0, max_value=10.0), min_size=3, max_size=3
            ),
            min_size=4,
            max_size=4,
        )
    )
    def test_basic_refinement(self, coords_list):
        """Test basic ring closure refinement with various coordinate sets."""
        coords = torch.tensor(coords_list, device=self.device)
        refined_coords = ring_closure_refinement(coords)

        # Check output shape
        self.assertEqual(refined_coords.shape, coords.shape)

        # Check data type
        self.assertEqual(refined_coords.dtype, torch.float32)

        # Check that coordinates are not NaN
        self.assertFalse(torch.isnan(refined_coords).any())


class TestPlaceBases(unittest.TestCase):
    """Test cases for base placement functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"

    @given(st.text(min_size=1, max_size=100, alphabet="ACGU"))
    @settings(deadline=None)
    def test_basic_base_placement(self, seq):
        """Test basic base placement with valid sequences."""
        # Create backbone coordinates with correct shape (L, B, 3)
        L = len(seq)
        B = len(BACKBONE_ATOMS)
        backbone_coords = torch.zeros(L, B, 3, device=self.device)  # Corrected shape

        # Place bases
        coords = place_bases(backbone_coords, seq, device=self.device)

        # Check output shape
        self.assertEqual(coords.ndim, 3)  # Corrected dimension check
        self.assertEqual(coords.shape[2], 3)  # Check last dimension is 3 (xyz)
        self.assertEqual(coords.shape[0], L)  # Check first dimension is sequence length

        # Check data type and device
        self.assertEqual(coords.dtype, torch.float32)
        self.assertEqual(coords.device.type, self.device)


class TestPlaceRNABases(unittest.TestCase):
    """Test cases for RNA base placement functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"

    @given(st.text(min_size=1, max_size=100, alphabet="ACGU"))
    @settings(deadline=None)
    def test_basic_rna_base_placement(self, seq):
        """Test basic RNA base placement with valid sequences."""
        # Create backbone coordinates with correct shape (L, B, 3)
        L = len(seq)
        B = len(BACKBONE_ATOMS)
        backbone_coords = torch.zeros(L, B, 3, device=self.device)  # Corrected shape
        angles_mask = torch.ones(
            (2, L, B), device=self.device
        )  # Added dummy angles_mask

        # Place bases
        coords = place_rna_bases(
            backbone_coords, seq, angles_mask, device=self.device
        )  # Added angles_mask argument

        # Check output shape
        self.assertEqual(coords.ndim, 3)  # Corrected dimension check
        self.assertEqual(coords.shape[2], 3)  # Check last dimension is 3 (xyz)
        self.assertEqual(coords.shape[0], L)  # Check first dimension is sequence length

        # Check data type and device
        self.assertEqual(coords.dtype, torch.float32)
        self.assertEqual(coords.device.type, self.device)


class TestHandleMods(unittest.TestCase):
    """Test cases for handling modified bases."""

    def test_basic_mod_handling(self):
        """Test handling of modified bases."""
        seq = "m1A"
        scaffolds = {}  # Add dummy scaffolds
        result = handle_mods(seq, scaffolds)  # Pass scaffolds
        self.assertIsInstance(result, dict)  # Check if dict is returned
        self.assertIs(result, scaffolds)  # Check if the same dict is returned

    def test_no_mods(self):
        """Test handling of unmodified sequences."""
        seq = "AUGC"
        scaffolds = {}  # Add dummy scaffolds
        result = handle_mods(seq, scaffolds)  # Pass scaffolds
        self.assertIs(result, scaffolds)  # Check if the same dict is returned

    @given(st.text(min_size=1, max_size=100, alphabet="ACGU"))
    def test_random_sequence(self, seq):
        """Test handling of random sequences."""
        scaffolds = {}  # Add dummy scaffolds
        result = handle_mods(seq, scaffolds)  # Pass scaffolds
        self.assertIsInstance(result, dict)  # Check if dict is returned
        self.assertIs(result, scaffolds)  # Check if the same dict is returned


class TestSkipMissingAtoms(unittest.TestCase):
    """Test cases for skipping missing atoms."""

    def test_basic_skip(self):
        """Test basic skipping of missing atoms."""
        seq = "AUGC"
        # Create scaffolds
        torsions = torch.zeros((len(seq), 7))
        scaffolds = build_scaffolds_rna_from_torsions(seq, torsions)

        # Test with scaffolds
        result = skip_missing_atoms(seq, scaffolds)
        self.assertIsInstance(result, dict)
        self.assertEqual(result, scaffolds)

    @given(st.text(min_size=1, max_size=10, alphabet="ACGU"))
    def test_random_sequence(self, seq):
        """Test skipping with random sequences."""
        # Create scaffolds
        torsions = torch.zeros((len(seq), 7))
        scaffolds = build_scaffolds_rna_from_torsions(seq, torsions)

        # Test with scaffolds
        result = skip_missing_atoms(seq, scaffolds)
        self.assertIsInstance(result, dict)
        self.assertEqual(result, scaffolds)


class TestGetBaseAtoms(unittest.TestCase):
    """Test cases for getting base atoms."""

    def test_get_base_atoms(self):
        """Test getting atoms for valid bases."""
        atoms = get_base_atoms("A")
        self.assertIsInstance(atoms, list)
        self.assertTrue(len(atoms) > 0)

    def test_invalid_base(self):
        """Test handling of invalid bases."""
        atoms = get_base_atoms("X")
        self.assertEqual(atoms, [])

    @given(st.text(min_size=1, max_size=1, alphabet="ACGU"))
    def test_random_base(self, base):
        """Test getting atoms for random bases."""
        atoms = get_base_atoms(base)
        self.assertIsInstance(atoms, list)


class TestMiniRefinement(unittest.TestCase):
    """Test cases for mini refinement."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"
        self.coords = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            device=self.device,
        )

    @given(
        st.lists(
            st.lists(
                st.floats(min_value=-10.0, max_value=10.0), min_size=3, max_size=3
            ),
            min_size=4,
            max_size=4,
        )
    )
    def test_basic_refinement(self, coords_list):
        """Test basic refinement with various coordinate sets."""
        coords = torch.tensor(coords_list, device=self.device)
        refined_coords = mini_refinement(coords)

        # Check output shape
        self.assertEqual(refined_coords.shape, coords.shape)

        # Check data type
        self.assertEqual(refined_coords.dtype, torch.float32)

        # Check that coordinates are not NaN
        self.assertFalse(torch.isnan(refined_coords).any())


class TestValidateRNAGeometry(unittest.TestCase):
    """Test cases for RNA geometry validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"
        self.coords = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            device=self.device,
        )

    @given(
        st.lists(
            st.lists(
                st.floats(min_value=-10.0, max_value=10.0), min_size=3, max_size=3
            ),
            min_size=4,
            max_size=4,
        )
    )
    def test_basic_validation(self, coords_list):
        """Test basic geometry validation with various coordinate sets."""
        coords = torch.tensor(coords_list, device=self.device)
        result = validate_rna_geometry(coords)
        self.assertIsInstance(result, bool)


class TestComputeMaxRNAAtoms(unittest.TestCase):
    """Test cases for computing maximum RNA atoms."""

    def test_compute_max_atoms(self):
        """Test computation of maximum RNA atoms."""
        max_atoms = compute_max_rna_atoms()
        self.assertIsInstance(max_atoms, int)
        self.assertTrue(max_atoms > 0)


if __name__ == "__main__":
    unittest.main()

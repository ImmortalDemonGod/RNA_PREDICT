import unittest
import torch
import numpy as np
from hypothesis import given, settings, strategies as st
from rna_predict.pipeline.stageC.mp_nerf.proteins import (
    scn_cloud_mask,
    scn_bond_mask,
    scn_angle_mask,
    scn_index_mask,
    build_scaffolds_from_scn_angles,
    modify_angles_mask_with_torsions,
    modify_scaffolds_with_coords,
    protein_fold,
    sidechain_fold,
)
from rna_predict.pipeline.stageC.mp_nerf.protein_utils.sidechain_data import SUPREME_INFO

class TestScnCloudMask(unittest.TestCase):
    """Test suite for the scn_cloud_mask function."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_seq = "AA"  # Simple sequence for testing
        self.valid_coords = torch.randn(1, 28, 3)  # (batch, length*14, 3)
        self.device = torch.device("cpu")

    @given(seq=st.text(alphabet="ACDEFGHIKLMNPQRSTVWY", min_size=1))
    def test_basic_cloud_mask(self, seq):
        """Test basic functionality of cloud mask generation."""
        mask = scn_cloud_mask(seq)
        self.assertIsInstance(mask, torch.Tensor)
        self.assertEqual(mask.shape[0], len(seq))
        self.assertEqual(mask.shape[1], 14)

    def test_cloud_mask_with_coords(self):
        """Test cloud mask generation with coordinates."""
        mask = scn_cloud_mask(self.valid_seq, coords=self.valid_coords)
        self.assertIsInstance(mask, torch.Tensor)
        self.assertEqual(mask.shape[0], 1)  # batch size
        self.assertEqual(mask.shape[1], 2)  # sequence length
        self.assertEqual(mask.shape[2], 14)  # atoms per residue

    def test_strict_mode(self):
        """Test strict mode of cloud mask generation."""
        coords = torch.zeros(1, 28, 3)  # All zeros
        mask = scn_cloud_mask(self.valid_seq, coords=coords, strict=True)
        self.assertTrue(torch.all(mask == 0))

class TestScnBondMask(unittest.TestCase):
    """Test suite for the scn_bond_mask function."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_seq = "AA"

    @given(seq=st.text(alphabet="ACDEFGHIKLMNPQRSTVWY", min_size=1))
    def test_basic_bond_mask(self, seq):
        """Test basic functionality of bond mask generation."""
        mask = scn_bond_mask(seq)
        self.assertIsInstance(mask, torch.Tensor)
        self.assertEqual(mask.shape[0], len(seq))
        self.assertEqual(mask.shape[1], 14)

    def test_bond_mask_values(self):
        """Test bond mask values for known amino acids."""
        mask = scn_bond_mask(self.valid_seq)
        self.assertTrue(torch.all(mask >= 0))  # Bond lengths should be non-negative

class TestScnAngleMask(unittest.TestCase):
    """Test suite for the scn_angle_mask function."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_seq = "AA"
        self.valid_angles = torch.randn(2, 12)  # (length, 12 angles)
        self.device = torch.device("cpu")

    @given(seq=st.text(alphabet="ACDEFGHIKLMNPQRSTVWY", min_size=1))
    def test_basic_angle_mask(self, seq):
        """Test basic functionality of angle mask generation."""
        mask = scn_angle_mask(seq)
        self.assertIsInstance(mask, torch.Tensor)
        self.assertEqual(mask.shape[0], 2)  # theta and dihedral
        self.assertEqual(mask.shape[1], len(seq))
        self.assertEqual(mask.shape[2], 14)

    def test_angle_mask_with_angles(self):
        """Test angle mask generation with provided angles."""
        mask = scn_angle_mask(self.valid_seq, angles=self.valid_angles)
        self.assertIsInstance(mask, torch.Tensor)
        self.assertEqual(mask.shape[0], 2)
        self.assertEqual(mask.shape[1], 2)
        self.assertEqual(mask.shape[2], 14)

class TestScnIndexMask(unittest.TestCase):
    """Test suite for the scn_index_mask function."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_seq = "AA"

    @given(seq=st.text(alphabet="ACDEFGHIKLMNPQRSTVWY", min_size=1))
    def test_basic_index_mask(self, seq):
        """Test basic functionality of index mask generation."""
        mask = scn_index_mask(seq)
        self.assertIsInstance(mask, torch.Tensor)
        self.assertEqual(mask.shape[0], 3)  # 3 reference points
        self.assertEqual(mask.shape[1], len(seq))
        self.assertEqual(mask.shape[2], 11)  # 11 atoms (excluding N-CA-C)

class TestBuildScaffoldsFromScnAngles(unittest.TestCase):
    """Test suite for the build_scaffolds_from_scn_angles function."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_seq = "AA"
        self.valid_angles = torch.randn(2, 12)  # (length, 12 angles)
        self.valid_coords = torch.randn(2, 14, 3)  # (length, 14 atoms, 3 coords)

    @given(seq=st.text(alphabet="ACDEFGHIKLMNPQRSTVWY", min_size=1))
    @settings(deadline=None)  # Remove deadline constraint
    def test_basic_scaffold_building(self, seq):
        """Test basic functionality of scaffold building."""
        scaffolds = build_scaffolds_from_scn_angles(seq)
        self.assertIsInstance(scaffolds, dict)
        self.assertIn("cloud_mask", scaffolds)
        self.assertIn("point_ref_mask", scaffolds)
        self.assertIn("angles_mask", scaffolds)
        self.assertIn("bond_mask", scaffolds)

    def test_scaffold_building_with_angles(self):
        """Test scaffold building with provided angles."""
        scaffolds = build_scaffolds_from_scn_angles(self.valid_seq, angles=self.valid_angles)
        self.assertIsInstance(scaffolds, dict)
        for key in ["cloud_mask", "point_ref_mask", "angles_mask", "bond_mask"]:
            self.assertIn(key, scaffolds)
            self.assertIsInstance(scaffolds[key], torch.Tensor)

class TestModifyAnglesMaskWithTorsions(unittest.TestCase):
    """Test suite for the modify_angles_mask_with_torsions function."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_seq = "AA"
        self.valid_angles_mask = torch.randn(2, 2, 14)  # (2 angles, length, 14 atoms)
        self.valid_torsions = torch.randn(2, 4)  # (length, 4 torsions)

    def test_basic_torsion_modification(self):
        """Test basic functionality of torsion modification."""
        modified_mask = modify_angles_mask_with_torsions(
            self.valid_seq, self.valid_angles_mask, self.valid_torsions
        )
        self.assertIsInstance(modified_mask, torch.Tensor)
        self.assertEqual(modified_mask.shape, self.valid_angles_mask.shape)

class TestModifyScaffoldsWithCoords(unittest.TestCase):
    """Test suite for the modify_scaffolds_with_coords function."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_seq = "AA"
        self.valid_scaffolds = {
            "cloud_mask": torch.ones(2, 14, dtype=torch.bool),
            "point_ref_mask": torch.randint(0, 14, (3, 2, 11)),
            "angles_mask": torch.randn(2, 2, 14),
            "bond_mask": torch.randn(2, 14),
        }
        self.valid_coords = torch.randn(2, 14, 3)  # (length, 14 atoms, 3 coords)

    def test_basic_scaffold_modification(self):
        """Test basic functionality of scaffold modification."""
        modified_scaffolds = modify_scaffolds_with_coords(
            self.valid_scaffolds, self.valid_coords
        )
        self.assertIsInstance(modified_scaffolds, dict)
        self.assertEqual(modified_scaffolds["bond_mask"].shape, self.valid_scaffolds["bond_mask"].shape)

class TestProteinFold(unittest.TestCase):
    """Test suite for the protein_fold function."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_cloud_mask = torch.ones(2, 14, dtype=torch.bool)
        self.valid_point_ref_mask = torch.randint(0, 14, (3, 2, 11))
        self.valid_angles_mask = torch.randn(2, 2, 14)
        self.valid_bond_mask = torch.randn(2, 14)
        self.device = torch.device("cpu")

    def test_basic_protein_folding(self):
        """Test basic functionality of protein folding."""
        coords, cloud_mask = protein_fold(
            self.valid_cloud_mask,
            self.valid_point_ref_mask,
            self.valid_angles_mask,
            self.valid_bond_mask,
            device=self.device,
        )
        self.assertIsInstance(coords, torch.Tensor)
        self.assertEqual(coords.shape[0], 2)  # length
        self.assertEqual(coords.shape[1], 14)  # atoms
        self.assertEqual(coords.shape[2], 3)  # coordinates
        self.assertEqual(cloud_mask.shape, self.valid_cloud_mask.shape)

class TestSidechainFold(unittest.TestCase):
    """Test suite for the sidechain_fold function."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_wrapper = torch.randn(2, 14, 3)  # (length, 14 atoms, 3 coords)
        self.valid_cloud_mask = torch.ones(2, 14, dtype=torch.bool)
        self.valid_point_ref_mask = torch.randint(0, 14, (3, 2, 11))
        self.valid_angles_mask = torch.randn(2, 2, 14)
        self.valid_bond_mask = torch.randn(2, 14)
        self.device = torch.device("cpu")

    def test_basic_sidechain_folding(self):
        """Test basic functionality of sidechain folding."""
        wrapper, cloud_mask = sidechain_fold(
            self.valid_wrapper,
            self.valid_cloud_mask,
            self.valid_point_ref_mask,
            self.valid_angles_mask,
            self.valid_bond_mask,
            device=self.device,
        )
        self.assertIsInstance(wrapper, torch.Tensor)
        self.assertEqual(wrapper.shape, self.valid_wrapper.shape)
        self.assertEqual(cloud_mask.shape, self.valid_cloud_mask.shape)

    def test_sidechain_folding_with_cbeta(self):
        """Test sidechain folding with c-beta placement."""
        wrapper, cloud_mask = sidechain_fold(
            self.valid_wrapper,
            self.valid_cloud_mask,
            self.valid_point_ref_mask,
            self.valid_angles_mask,
            self.valid_bond_mask,
            device=self.device,
            c_beta=True,
        )
        self.assertIsInstance(wrapper, torch.Tensor)
        self.assertEqual(wrapper.shape, self.valid_wrapper.shape)
        self.assertEqual(cloud_mask.shape, self.valid_cloud_mask.shape) 
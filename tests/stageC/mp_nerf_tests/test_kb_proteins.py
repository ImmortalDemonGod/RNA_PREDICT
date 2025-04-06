"""Tests for protein utilities mask generators, updated for new SUPREME_INFO."""

import unittest

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis import settings

# Adjust this import as needed
import rna_predict.pipeline.stageC.mp_nerf.protein_utils as protein_utils
from rna_predict.pipeline.stageC.mp_nerf.protein_utils.supreme_data import (
    SUPREME_INFO,
)  # Import updated data


class TestMakeCloudMask(unittest.TestCase):
    """Tests for make_cloud_mask."""

    def test_make_cloud_mask_valid_aa(self):
        """Test cloud mask for Alanine ('A')."""
        mask = protein_utils.make_cloud_mask("A")
        self.assertEqual(mask.shape, (14,), "Mask shape should be (14,)")
        self.assertTrue(mask.dtype == bool, "Mask dtype should be bool")
        # Check based on updated SUPREME_INFO['A']['atoms'] = 5
        self.assertEqual(int(mask.sum()), 5, "Expected 5 True values for Alanine")
        self.assertTrue(np.all(mask[:5]), "First 5 elements should be True for Alanine")
        self.assertFalse(
            np.any(mask[5:]), "Elements after index 4 should be False for Alanine"
        )

    def test_make_cloud_mask_underscore(self):
        """Test cloud mask for padding ('_')."""
        mask = protein_utils.make_cloud_mask("_")
        self.assertEqual(mask.shape, (14,))
        self.assertFalse(np.any(mask), "Mask for '_' should be all False")

    def test_make_cloud_mask_invalid_aa(self):
        """Test invalid amino acid code raises KeyError."""
        with self.assertRaises(KeyError):
            protein_utils.make_cloud_mask("X")

    @given(st.sampled_from(list(SUPREME_INFO.keys())))
    @settings(deadline=None)
    def test_make_cloud_mask_hypothesis_valid(self, aa):
        """Hypothesis test for all valid amino acids."""
        mask = protein_utils.make_cloud_mask(aa)
        self.assertEqual(mask.shape, (14,))
        self.assertTrue(mask.dtype == bool)
        # Check consistency with SUPREME_INFO data
        expected_mask = np.array(SUPREME_INFO[aa]["cloud_mask"], dtype=bool)
        np.testing.assert_array_equal(mask, expected_mask)


class TestMakeBondMask(unittest.TestCase):
    """Tests for make_bond_mask."""

    def test_make_bond_mask_valid(self):
        """Test bond mask for Alanine ('A')."""
        mask = protein_utils.make_bond_mask("A")
        self.assertEqual(mask.shape, (14,), "Mask shape should be (14,)")
        self.assertTrue(mask.dtype == float, "Mask dtype should be float")
        # Check based on updated SUPREME_INFO['A']['atoms'] = 5
        self.assertEqual(
            np.count_nonzero(mask), 5, "Expected 5 non-zero values for Alanine"
        )
        self.assertTrue(
            np.all(mask[:5] > 0), "First 5 elements should be non-zero for Alanine"
        )
        self.assertTrue(
            np.all(mask[5:] == 0), "Elements after index 4 should be zero for Alanine"
        )

    def test_make_bond_mask_underscore(self):
        """Test bond mask for padding ('_')."""
        mask = protein_utils.make_bond_mask("_")
        self.assertEqual(mask.shape, (14,))
        self.assertTrue(np.all(mask == 0), "Mask for '_' should be all zeros")

    def test_make_bond_mask_invalid_aa(self):
        """Test invalid amino acid code raises KeyError."""
        with self.assertRaises(KeyError):
            protein_utils.make_bond_mask("X")

    @given(st.sampled_from(list(SUPREME_INFO.keys())))
    def test_make_bond_mask_hypothesis_valid(self, aa):
        """Hypothesis test for all valid amino acids."""
        mask = protein_utils.make_bond_mask(aa)
        self.assertEqual(mask.shape, (14,))
        self.assertTrue(mask.dtype == float)
        # Check consistency with SUPREME_INFO data
        expected_mask = np.array(SUPREME_INFO[aa]["bond_mask"], dtype=float)
        np.testing.assert_array_equal(mask, expected_mask)


class TestMakeThetaMask(unittest.TestCase):
    """Tests for make_theta_mask."""

    def test_make_theta_mask_valid(self):
        """Test theta mask for Alanine ('A')."""
        mask = protein_utils.make_theta_mask("A")
        self.assertEqual(mask.shape, (14,), "Mask shape should be (14,)")
        self.assertTrue(mask.dtype == float, "Mask dtype should be float")
        # Check based on updated SUPREME_INFO['A']['atoms'] = 5
        self.assertEqual(
            np.count_nonzero(mask), 5, "Expected 5 non-zero values for Alanine"
        )
        self.assertTrue(
            np.all(mask[:5] > 0), "First 5 elements should be non-zero for Alanine"
        )
        self.assertTrue(
            np.all(mask[5:] == 0), "Elements after index 4 should be zero for Alanine"
        )

    def test_make_theta_mask_underscore(self):
        """Test theta mask for padding ('_')."""
        mask = protein_utils.make_theta_mask("_")
        self.assertEqual(mask.shape, (14,))
        self.assertTrue(np.all(mask == 0), "Mask for '_' should be all zeros")

    def test_make_theta_mask_invalid_aa(self):
        """Test invalid amino acid code raises KeyError."""
        with self.assertRaises(KeyError):
            protein_utils.make_theta_mask("X")

    @given(st.sampled_from(list(SUPREME_INFO.keys())))
    @settings(deadline=None)
    def test_make_theta_mask_hypothesis_valid(self, aa):
        """Hypothesis test for all valid amino acids."""
        mask = protein_utils.make_theta_mask(aa)
        self.assertEqual(mask.shape, (14,))
        self.assertTrue(mask.dtype == float)
        # Check consistency with SUPREME_INFO data
        expected_mask = np.array(SUPREME_INFO[aa]["theta_mask"], dtype=float)
        np.testing.assert_array_equal(mask, expected_mask)


class TestMakeTorsionMask(unittest.TestCase):
    """Tests for make_torsion_mask."""

    def test_make_torsion_mask_valid(self):
        """Test torsion mask for Alanine ('A')."""
        # Test with fill=False (may contain NaNs)
        mask = protein_utils.make_torsion_mask("A", fill=False)
        self.assertEqual(mask.shape, (14,), "Mask shape should be (14,)")
        self.assertTrue(mask.dtype == float, "Mask dtype should be float")
        # Check based on updated SUPREME_INFO['A']['atoms'] = 5
        self.assertEqual(
            np.count_nonzero(~np.isnan(mask)),
            5,
            "Expected 5 non-NaN values for Alanine",
        )
        self.assertTrue(
            np.all(~np.isnan(mask[:5])),
            "First 5 elements should be non-NaN for Alanine",
        )
        self.assertTrue(
            np.all(np.isnan(mask[5:])),
            "Elements after index 4 should be NaN for Alanine",
        )

        # Test with fill=True (no NaNs)
        mask_filled = protein_utils.make_torsion_mask("A", fill=True)
        self.assertEqual(mask_filled.shape, (14,))
        self.assertTrue(mask_filled.dtype == float)
        self.assertEqual(
            np.count_nonzero(mask_filled), 5, "Expected 5 non-zero values for Alanine"
        )
        self.assertTrue(
            np.all(mask_filled[:5] > 0),
            "First 5 elements should be non-zero for Alanine",
        )
        self.assertTrue(
            np.all(mask_filled[5:] == 0),
            "Elements after index 4 should be zero for Alanine",
        )

    def test_make_torsion_mask_underscore(self):
        """Test torsion mask for padding ('_')."""
        # Test with fill=False (may contain NaNs)
        mask = protein_utils.make_torsion_mask("_", fill=False)
        self.assertEqual(mask.shape, (14,))
        self.assertTrue(np.all(np.isnan(mask)), "Mask for '_' should be all NaNs")

        # Test with fill=True (no NaNs)
        mask_filled = protein_utils.make_torsion_mask("_", fill=True)
        self.assertEqual(mask_filled.shape, (14,))
        self.assertTrue(
            np.all(mask_filled == 0), "Filled mask for '_' should be all zeros"
        )

    def test_make_torsion_mask_invalid_aa(self):
        """Test invalid amino acid code raises KeyError."""
        with self.assertRaises(KeyError):
            protein_utils.make_torsion_mask("X")

    @given(st.sampled_from(list(SUPREME_INFO.keys())))
    def test_make_torsion_mask_hypothesis_valid(self, aa):
        """Hypothesis test for all valid amino acids."""
        # Test with fill=False
        mask = protein_utils.make_torsion_mask(aa, fill=False)
        self.assertEqual(mask.shape, (14,))
        self.assertTrue(mask.dtype == float)
        # Check consistency with SUPREME_INFO data
        expected_mask = np.array(SUPREME_INFO[aa]["torsion_mask"], dtype=float)
        np.testing.assert_array_equal(mask, expected_mask)

        # Test with fill=True
        mask_filled = protein_utils.make_torsion_mask(aa, fill=True)
        self.assertEqual(mask_filled.shape, (14,))
        self.assertTrue(mask_filled.dtype == float)
        expected_mask_filled = np.array(
            SUPREME_INFO[aa]["torsion_mask_filled"], dtype=float
        )
        np.testing.assert_array_equal(mask_filled, expected_mask_filled)


class TestMakeIdxMask(unittest.TestCase):
    """Tests for make_idx_mask."""

    def test_make_idx_mask_valid(self):
        """Test idx mask for Alanine ('A')."""
        mask = protein_utils.make_idx_mask("A")
        self.assertEqual(mask.shape, (11, 3), "Mask shape should be (11, 3)")
        self.assertTrue(mask.dtype == int, "Mask dtype should be int")
        # Check based on updated SUPREME_INFO['A']['sc_torsions'] = 1
        # First row should be [0, 1, 2] for backbone reference
        np.testing.assert_array_equal(mask[0], [0, 1, 2])
        # Second row should be valid indices for the single sidechain torsion
        self.assertTrue(np.all(mask[1] >= 0) and np.all(mask[1] < 5))
        # Remaining rows should be zeros
        self.assertTrue(np.all(mask[2:] == 0))

    def test_make_idx_mask_underscore(self):
        """Test idx mask for padding ('_')."""
        mask = protein_utils.make_idx_mask("_")
        self.assertEqual(mask.shape, (11, 3))
        self.assertTrue(np.all(mask == 0), "Mask for '_' should be all zeros")

    def test_make_idx_mask_invalid_aa(self):
        """Test invalid amino acid code raises KeyError."""
        with self.assertRaises(KeyError):
            protein_utils.make_idx_mask("X")

    @given(st.sampled_from(list(SUPREME_INFO.keys())))
    def test_make_idx_mask_hypothesis_valid(self, aa):
        """Hypothesis test for all valid amino acids."""
        mask = protein_utils.make_idx_mask(aa)
        self.assertEqual(mask.shape, (11, 3))
        self.assertTrue(mask.dtype == int)
        # Check consistency with SUPREME_INFO data
        expected_mask = np.array(SUPREME_INFO[aa]["idx_mask"], dtype=int)
        np.testing.assert_array_equal(mask, expected_mask)


class TestMakeAtomTokenMask(unittest.TestCase):
    """Tests for make_atom_token_mask."""

    def test_make_atom_token_mask_valid(self):
        """Test atom token mask for Alanine ('A')."""
        mask = protein_utils.make_atom_token_mask("A")
        self.assertEqual(mask.shape, (14,), "Mask shape should be (14,)")
        self.assertTrue(mask.dtype == int, "Mask dtype should be int")
        # Check based on updated SUPREME_INFO['A']['atoms'] = 5
        self.assertEqual(
            np.count_nonzero(mask), 5, "Expected 5 non-zero values for Alanine"
        )
        self.assertTrue(
            np.all(mask[:5] > 0), "First 5 elements should be non-zero for Alanine"
        )
        self.assertTrue(
            np.all(mask[5:] == 0), "Elements after index 4 should be zero for Alanine"
        )

    def test_make_atom_token_mask_underscore(self):
        """Test atom token mask for padding ('_')."""
        mask = protein_utils.make_atom_token_mask("_")
        self.assertEqual(mask.shape, (14,))
        self.assertTrue(np.all(mask == 0), "Mask for '_' should be all zeros")

    def test_make_atom_token_mask_invalid_aa(self):
        """Test invalid amino acid code raises KeyError."""
        with self.assertRaises(KeyError):
            protein_utils.make_atom_token_mask("X")

    @given(st.sampled_from(list(SUPREME_INFO.keys())))
    def test_make_atom_token_mask_hypothesis_valid(self, aa):
        """Hypothesis test for all valid amino acids."""
        mask = protein_utils.make_atom_token_mask(aa)
        self.assertEqual(mask.shape, (14,))
        self.assertTrue(mask.dtype == int)
        # Check consistency with SUPREME_INFO data
        expected_mask = np.array(SUPREME_INFO[aa]["atom_token_mask"], dtype=int)
        np.testing.assert_array_equal(mask, expected_mask)


if __name__ == "__main__":
    # Allows direct execution: python -m unittest test_kb_proteins.py
    unittest.main()

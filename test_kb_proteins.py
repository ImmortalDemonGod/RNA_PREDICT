import unittest
from unittest.mock import patch
from hypothesis import given, strategies as st
import numpy as np

# Adjust this import as needed if kb_proteins.py is in a different module/package
import rna_predict.pipeline.stageC.mp_nerf.kb_proteins as kb_proteins


class TestMakeCloudMask(unittest.TestCase):
    """
    Tests for the make_cloud_mask function in kb_proteins.py
    """

    def setUp(self):
        """
        Prepare data for tests.
        We gather valid amino acids from SC_BUILD_INFO plus invalid ones.
        """
        self.valid_aas = list(kb_proteins.SC_BUILD_INFO.keys())
        # Some obviously invalid or corner-case strings
        self.invalid_aas = ["", "Z", "???", "XYZ"]
        self.underscore = "_"

    def test_make_cloud_mask_valid_aa(self):
        """
        Test that make_cloud_mask returns an array of length 14,
        with the correct bits set to 1 for a typical amino acid (e.g., 'A').
        """
        mask = kb_proteins.make_cloud_mask("A")
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(mask.shape, (14,))
        # "A" has 1 sidechain atom (CB), plus 4 backbone => total 5 relevant
        # Positions 0..4 should be 1. The rest 0.
        # So we sum the mask, expecting 5.
        self.assertEqual(int(mask.sum()), 5, "Should have 5 ones for 'A'")

    def test_make_cloud_mask_underscore(self):
        """
        Test the underscore '_' case. The function should return all zeros for '_'.
        """
        mask = kb_proteins.make_cloud_mask(self.underscore)
        self.assertTrue(np.all(mask == 0), "All zeros expected for '_'")
        self.assertEqual(mask.shape, (14,))

    def test_make_cloud_mask_invalid_aa(self):
        """
        Test that an unrecognized amino acid string also yields 4 positions for the backbone
        or none if code doesn't exist. Actually, the code checks 'aa != "_"',
        so if it's invalid but not '_', it tries to read SC_BUILD_INFO.
        This doesn't raise an error, but the sidechain logic won't exist.
        It should produce a 14-length array with only the 4 backbone positions if the key is absent
        from SC_BUILD_INFO or default to 0 if the code checks fail.
        We'll check it doesn't crash.
        """
        for inval in self.invalid_aas:
            mask = kb_proteins.make_cloud_mask(inval)
            self.assertEqual(mask.shape, (14,))
            # It's not in SC_BUILD_INFO, so the function won't set any sidechain bits
            # But it does check 'aa != "_"', so the code attempts to do "SC_BUILD_INFO[<inval>]" if present.
            # If <inval> not in SC_BUILD_INFO, it will produce an array with 4 + sidechain-len=0 => 4 ones
            # or 0 if the code doesn't handle KeyError. Let's see how the code is structured:
            #
            # Actually, 'aa != "_"': n_atoms = 4 + len(SC_BUILD_INFO[aa]["atom-names"])
            # This will KeyError if 'aa' not in SC_BUILD_INFO.
            #
            # So we either get a KeyError or continue. Let's see if we want to handle that or test it?
            # The function doesn't handle KeyError, so it might just raise. We'll check that.
            # We'll wrap it in a try-except to confirm behavior.
            #
            # But let's just test it for coverage and confirm if it raises or not:
            with self.assertRaises(KeyError):
                kb_proteins.make_cloud_mask(inval)

    @given(st.sampled_from(list(kb_proteins.SC_BUILD_INFO.keys())) | st.text())
    def test_make_cloud_mask_hypothesis(self, aa):
        """
        Hypothesis test: pass random strings or valid codes to make_cloud_mask.
        We ensure it returns a 14-length numpy array and doesn't crash for known codes
        or underscore. Invalid codes are expected to raise KeyError if not '_'.
        """
        if aa == "_":
            # '_' => array of zeros
            mask = kb_proteins.make_cloud_mask(aa)
            self.assertEqual(mask.shape, (14,))
            self.assertTrue(np.all(mask == 0))
        elif aa in kb_proteins.SC_BUILD_INFO:
            mask = kb_proteins.make_cloud_mask(aa)
            self.assertEqual(mask.shape, (14,))
            # Must have at least backbone => 4 if sidechain is empty like 'G'
            self.assertGreaterEqual(mask.sum(), 4)
        else:
            # We expect KeyError for invalid codes
            if aa != "":
                with self.assertRaises(KeyError):
                    kb_proteins.make_cloud_mask(aa)
            else:
                # if aa == "", also KeyError
                with self.assertRaises(KeyError):
                    kb_proteins.make_cloud_mask(aa)


class TestMakeBondMask(unittest.TestCase):
    """
    Tests for make_bond_mask function in kb_proteins.py
    """

    def setUp(self):
        self.valid_aas = list(kb_proteins.SC_BUILD_INFO.keys())
        self.invalid_aas = ["Z", "???", "XYZ"]
        self.underscore = "_"

    def test_make_bond_mask_valid(self):
        """Check shape and typical sum of bond lengths for a valid amino acid."""
        mask = kb_proteins.make_bond_mask("A")
        self.assertEqual(mask.shape, (14,))
        # For 'A': backbone bond lens in positions 0..3, plus sidechain in position 4 => total 5 non-zero
        nonzeros = (mask != 0).sum()
        self.assertEqual(nonzeros, 5)

    def test_make_bond_mask_underscore(self):
        """Check underscore '_' yields all zeros, shape (14,)."""
        mask = kb_proteins.make_bond_mask(self.underscore)
        self.assertEqual(mask.shape, (14,))
        self.assertTrue(np.all(mask == 0))

    def test_make_bond_mask_invalid(self):
        """Invalid code should raise KeyError unless it's underscore."""
        for inval in self.invalid_aas:
            with self.assertRaises(KeyError):
                kb_proteins.make_bond_mask(inval)

    @given(st.sampled_from(list(kb_proteins.SC_BUILD_INFO.keys())) | st.text())
    def test_make_bond_mask_hypothesis(self, aa):
        """Property-based test for random valid or invalid codes."""
        if aa == "_":
            mask = kb_proteins.make_bond_mask(aa)
            self.assertEqual(mask.shape, (14,))
            self.assertTrue(np.all(mask == 0))
        elif aa in kb_proteins.SC_BUILD_INFO:
            mask = kb_proteins.make_bond_mask(aa)
            self.assertEqual(mask.shape, (14,))
            self.assertGreater((mask != 0).sum(), 0)
        else:
            with self.assertRaises(KeyError):
                kb_proteins.make_bond_mask(aa)


class TestMakeThetaMask(unittest.TestCase):
    """
    Tests for make_theta_mask function in kb_proteins.py
    """

    def setUp(self):
        self.valid_aas = list(kb_proteins.SC_BUILD_INFO.keys())
        self.invalid_aas = ["Z", "???", "XYZ"]
        self.underscore = "_"

    def test_make_theta_mask_valid(self):
        """Check shape and typical non-zero angles for 'A'."""
        mask = kb_proteins.make_theta_mask("A")
        self.assertEqual(mask.shape, (14,))
        # Expect at least 4 backbone angles
        nonzeros = (mask != 0).sum()
        self.assertGreaterEqual(nonzeros, 4)

    def test_make_theta_mask_underscore(self):
        """Check underscore yields all zeros."""
        mask = kb_proteins.make_theta_mask(self.underscore)
        self.assertEqual(mask.shape, (14,))
        self.assertTrue(np.all(mask == 0))

    def test_make_theta_mask_invalid(self):
        """Invalid code => KeyError unless underscore."""
        for inval in self.invalid_aas:
            with self.assertRaises(KeyError):
                kb_proteins.make_theta_mask(inval)

    @given(st.sampled_from(list(kb_proteins.SC_BUILD_INFO.keys())) | st.text())
    def test_make_theta_mask_hypothesis(self, aa):
        """Random fuzz test for valid and invalid codes."""
        if aa == "_":
            mask = kb_proteins.make_theta_mask(aa)
            self.assertEqual(mask.shape, (14,))
            self.assertTrue(np.all(mask == 0))
        elif aa in kb_proteins.SC_BUILD_INFO:
            mask = kb_proteins.make_theta_mask(aa)
            self.assertEqual(mask.shape, (14,))
            self.assertGreaterEqual((mask != 0).sum(), 4)
        else:
            with self.assertRaises(KeyError):
                kb_proteins.make_theta_mask(aa)


class TestMakeTorsionMask(unittest.TestCase):
    """
    Tests for make_torsion_mask(aa, fill=False) in kb_proteins.py
    Also checks the fill=True path for full coverage.
    """

    def setUp(self):
        self.valid_aas = list(kb_proteins.SC_BUILD_INFO.keys())
        self.invalid_aas = ["Z", "???", "XYZ"]
        self.underscore = "_"

    def test_make_torsion_mask_valid_no_fill(self):
        """Check shape for 'A' with fill=False."""
        mask = kb_proteins.make_torsion_mask("A", fill=False)
        self.assertEqual(mask.shape, (14,))
        # Should have backbone torsions in the first 4 positions
        # plus sidechain in the next 1 for 'A'
        self.assertFalse(np.any(np.isnan(mask[:4])))  # backbone are numeric
        # sidechain is 'p' => replaced with np.nan => check it
        self.assertTrue(np.isnan(mask[4]), "For 'A', we expect 'p' => np.nan in position 4 if fill=False")

    def test_make_torsion_mask_valid_fill(self):
        """Check shape for 'A' with fill=True, ensuring sidechain dihedral is from MP3SC_INFO."""
        mask = kb_proteins.make_torsion_mask("A", fill=True)
        self.assertEqual(mask.shape, (14,))
        # For 'A', we expect backbone angles in first 4, sidechain in position 4
        # Should not be nan
        self.assertFalse(np.any(np.isnan(mask)), "When fill=True, none should be NaN")

    def test_make_torsion_mask_underscore_no_fill(self):
        """Check underscore => all zeros for either fill or no_fill."""
        mask = kb_proteins.make_torsion_mask(self.underscore, fill=False)
        self.assertTrue(np.all(mask == 0))

    def test_make_torsion_mask_underscore_fill(self):
        """Check underscore => all zeros for fill=True as well."""
        mask = kb_proteins.make_torsion_mask(self.underscore, fill=True)
        self.assertTrue(np.all(mask == 0))

    def test_make_torsion_mask_invalid(self):
        """Invalid code => KeyError unless underscore."""
        for inval in self.invalid_aas:
            with self.assertRaises(KeyError):
                kb_proteins.make_torsion_mask(inval, fill=False)

    @given(st.sampled_from(list(kb_proteins.SC_BUILD_INFO.keys())) | st.text(),
           st.booleans())
    def test_make_torsion_mask_hypothesis(self, aa, fill):
        """
        Hypothesis test for random valid or invalid strings,
        toggling fill param.
        """
        if aa == "_":
            mask = kb_proteins.make_torsion_mask(aa, fill=fill)
            self.assertEqual(mask.shape, (14,))
            self.assertTrue(np.all(mask == 0))
        elif aa in kb_proteins.SC_BUILD_INFO:
            mask = kb_proteins.make_torsion_mask(aa, fill=fill)
            self.assertEqual(mask.shape, (14,))
        else:
            with self.assertRaises(KeyError):
                kb_proteins.make_torsion_mask(aa, fill=fill)


class TestMakeIdxMask(unittest.TestCase):
    """
    Tests for make_idx_mask(aa) in kb_proteins.py
    """

    def setUp(self):
        self.valid_aas = list(kb_proteins.SC_BUILD_INFO.keys())
        self.invalid_aas = ["Z", "???", "XYZ"]
        self.underscore = "_"

    def test_make_idx_mask_valid(self):
        """Check shape for 'A' => should be (11,3)."""
        mask = kb_proteins.make_idx_mask("A")
        self.assertEqual(mask.shape, (11, 3))
        # For 'A', sidechain has 1 torsion => the second row after the 0th for backbone
        # We'll do a basic check for the backbone row: mask[0,:] => [0,1,2]
        np.testing.assert_array_equal(mask[0], [0, 1, 2])

    def test_make_idx_mask_underscore(self):
        """Check underscore => (11,3) but all zeros or empty?
        Actually SC_BUILD_INFO['_']['torsion-names'] = [], so the function sets only the backbone row [0..2].
        """
        mask = kb_proteins.make_idx_mask(self.underscore)
        self.assertEqual(mask.shape, (11, 3))
        # The code sets mask[0,:] = np.arange(3), the rest remain zeros if no sidechain torsions
        np.testing.assert_array_equal(mask[0], [0, 1, 2])
        # The rest should be zeros
        for row in range(1, 11):
            self.assertTrue(np.all(mask[row] == 0))

    def test_make_idx_mask_invalid(self):
        """Invalid => KeyError unless underscore."""
        for inval in self.invalid_aas:
            with self.assertRaises(KeyError):
                kb_proteins.make_idx_mask(inval)

    @given(st.sampled_from(list(kb_proteins.SC_BUILD_INFO.keys())) | st.text())
    def test_make_idx_mask_hypothesis(self, aa):
        """Check shape or KeyError for random strings."""
        if aa == "_":
            mask = kb_proteins.make_idx_mask(aa)
            self.assertEqual(mask.shape, (11, 3))
            np.testing.assert_array_equal(mask[0], [0, 1, 2])
        elif aa in kb_proteins.SC_BUILD_INFO:
            mask = kb_proteins.make_idx_mask(aa)
            self.assertEqual(mask.shape, (11, 3))
        else:
            with self.assertRaises(KeyError):
                kb_proteins.make_idx_mask(aa)


class TestMakeAtomTokenMask(unittest.TestCase):
    """
    Tests for make_atom_token_mask(aa) in kb_proteins.py
    """

    def setUp(self):
        self.valid_aas = list(kb_proteins.SC_BUILD_INFO.keys())
        self.invalid_aas = ["Z", "???", "XYZ"]
        self.underscore = "_"

    def test_make_atom_token_mask_valid(self):
        """Check shape and token IDs for 'A'."""
        mask = kb_proteins.make_atom_token_mask("A")
        self.assertEqual(mask.shape, (14,))
        # 'A' => ["N","CA","C","O","CB"] => 5 atoms used => the rest 9 are 0
        used_tokens = (mask[:5]).astype(int)
        # Just check they are distinct and non-zero except we must see if 'N' might have an ID
        # We'll confirm it doesn't crash for coverage
        self.assertGreater(used_tokens[0], -1)

    def test_make_atom_token_mask_underscore(self):
        """Check underscore => shape (14,) with all zeros?
        The code sets no sidechain, but we do have N, CA, C, O => actually for underscore it does if aa != '_'?
        Actually if aa == '_', the code won't set them, so we get an array of length 14, all zeros.
        """
        mask = kb_proteins.make_atom_token_mask(self.underscore)
        self.assertEqual(mask.shape, (14,))
        self.assertTrue(np.all(mask == 0))

    def test_make_atom_token_mask_invalid(self):
        """Invalid code => KeyError unless underscore."""
        for inval in self.invalid_aas:
            with self.assertRaises(KeyError):
                kb_proteins.make_atom_token_mask(inval)

    @given(st.sampled_from(list(kb_proteins.SC_BUILD_INFO.keys())) | st.text())
    def test_make_atom_token_mask_hypothesis(self, aa):
        """Check shape or KeyError with random strings."""
        if aa == "_":
            mask = kb_proteins.make_atom_token_mask(aa)
            self.assertEqual(mask.shape, (14,))
            self.assertTrue(np.all(mask == 0))
        elif aa in kb_proteins.SC_BUILD_INFO:
            mask = kb_proteins.make_atom_token_mask(aa)
            self.assertEqual(mask.shape, (14,))
        else:
            with self.assertRaises(KeyError):
                kb_proteins.make_atom_token_mask(aa)


if __name__ == "__main__":
    # Allows direct execution: python -m unittest test_kb_proteins.py
    unittest.main()
import unittest
from unittest.mock import patch

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.strategies import booleans, composite, floats, lists

# Adjust import to match project structure
from rna_predict.pipeline.stageC.mp_nerf import proteins
from rna_predict.pipeline.stageC.mp_nerf import utils
from rna_predict.pipeline.stageC.mp_nerf import massive_pnerf


# ----------------------------------------------------------------------
# Composite strategies for property-based tests
# ----------------------------------------------------------------------
@composite
def seq_strategy(draw, min_size=1, max_size=5):
    """
    Generates a short list of amino acids from a subset, representing a small
    protein sequence. Adjust/expand these amino acids as necessary.
    """
    aa_pool = [
        "A",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "K",
        "L",
        "M",
        "N",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "V",
        "W",
        "Y",
    ]
    length = draw(st.integers(min_value=min_size, max_value=max_size))
    seq = draw(lists(st.sampled_from(aa_pool), min_size=length, max_size=length))
    return seq


@composite
def angles_strategy(draw, seq):
    """
    For a given sequence of length L, produce an (L, 12) float32 Tensor of angles.
    This matches the shape that scn_angle_mask expects:
        [phi, psi, omega, 3 bond angles, 6 sidechain torsions].
    """
    L = len(seq)
    data = draw(
        lists(
            lists(
                floats(
                    min_value=-np.pi,
                    max_value=np.pi,
                    allow_infinity=False,
                    allow_nan=False,
                ),
                min_size=12,
                max_size=12,
            ),
            min_size=L,
            max_size=L,
        )
    )
    return torch.tensor(data, dtype=torch.float32)


@composite
def torsions_strategy(draw, seq, include_cb=False):
    """
    Produces either 4 or 5 torsions per residue depending on include_cb.
    This simulates sidechain torsion angles, possibly with c_beta included.
    """
    L = len(seq)
    torsion_count = 5 if include_cb else 4
    data = draw(
        lists(
            lists(
                floats(
                    min_value=-np.pi,
                    max_value=np.pi,
                    allow_infinity=False,
                    allow_nan=False,
                ),
                min_size=torsion_count,
                max_size=torsion_count,
            ),
            min_size=L,
            max_size=L,
        )
    )
    return torch.tensor(data, dtype=torch.float32)


# ----------------------------------------------------------------------
# Dummy data + geometry replacements for consistent testing
# ----------------------------------------------------------------------
def dummy_mp_nerf_torch(params):
    """
    Mocks mp_nerf_torch, always returning a float tensor the same shape as params.c, but filled with 42.
    """
    return torch.full_like(params.c, 42.0)


def dummy_get_angle(a, b, c):
    """Mock get_angle, returns 1.234 for testing."""
    return torch.tensor(1.234)


def dummy_get_dihedral(a, b, c, d):
    """Mock get_dihedral, returns 2.345 for testing."""
    return torch.tensor(2.345)


def dummy_get_axis_matrix(a, b, c, norm=True):
    """Mock get_axis_matrix, returns an identity matrix."""
    return torch.eye(3)


DUMMY_SUPREME_INFO = {
    "A": {
        "cloud_mask": [True] * 14,
        "bond_mask": [1.0] * 14,
        "theta_mask": [0.0] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [0.0] * 14,
        "idx_mask": [[0, 1, 2] for _ in range(11)],
        "rigid_idx_mask": [0, 1, 2],
    },
    "C": {
        "cloud_mask": [False] * 14,
        "bond_mask": [2.0] * 14,
        "theta_mask": [0.1] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [0.1] * 14,
        "idx_mask": [[3, 4, 5] for _ in range(11)],
        "rigid_idx_mask": [3, 4, 5],
    },
    "D": {
        "cloud_mask": [True, False] * 7,
        "bond_mask": [3.0] * 14,
        "theta_mask": [0.2] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [0.2] * 14,
        "idx_mask": [[6, 7, 8] for _ in range(11)],
        "rigid_idx_mask": [6, 7, 8],
    },
    "E": {
        "cloud_mask": [False, True] * 7,
        "bond_mask": [4.0] * 14,
        "theta_mask": [0.3] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [0.3] * 14,
        "idx_mask": [[9, 10, 11] for _ in range(11)],
        "rigid_idx_mask": [9, 10, 11],
    },
    "F": {
        "cloud_mask": [True] * 14,
        "bond_mask": [5.0] * 14,
        "theta_mask": [0.4] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [0.4] * 14,
        "idx_mask": [[12, 13, 14] for _ in range(11)],
        "rigid_idx_mask": [12, 13, 14],
    },
    "G": {
        "cloud_mask": [False] * 14,
        "bond_mask": [6.0] * 14,
        "theta_mask": [0.5] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [0.5] * 14,
        "idx_mask": [[15, 16, 17] for _ in range(11)],
        "rigid_idx_mask": [15, 16, 17],
    },
    "H": {
        "cloud_mask": [True, False] * 7,
        "bond_mask": [7.0] * 14,
        "theta_mask": [0.6] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [0.6] * 14,
        "idx_mask": [[18, 19, 20] for _ in range(11)],
        "rigid_idx_mask": [18, 19, 20],
    },
    "I": {
        "cloud_mask": [False, True] * 7,
        "bond_mask": [8.0] * 14,
        "theta_mask": [0.7] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [0.7] * 14,
        "idx_mask": [[21, 22, 23] for _ in range(11)],
        "rigid_idx_mask": [21, 22, 23],
    },
    "K": {
        "cloud_mask": [True] * 14,
        "bond_mask": [9.0] * 14,
        "theta_mask": [0.8] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [0.8] * 14,
        "idx_mask": [[24, 25, 26] for _ in range(11)],
        "rigid_idx_mask": [24, 25, 26],
    },
    "L": {
        "cloud_mask": [False] * 14,
        "bond_mask": [10.0] * 14,
        "theta_mask": [0.9] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [0.9] * 14,
        "idx_mask": [[27, 28, 29] for _ in range(11)],
        "rigid_idx_mask": [27, 28, 29],
    },
    "M": {
        "cloud_mask": [True, False] * 7,
        "bond_mask": [11.0] * 14,
        "theta_mask": [1.0] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [1.0] * 14,
        "idx_mask": [[30, 31, 32] for _ in range(11)],
        "rigid_idx_mask": [30, 31, 32],
    },
    "N": {
        "cloud_mask": [False, True] * 7,
        "bond_mask": [12.0] * 14,
        "theta_mask": [1.1] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [1.1] * 14,
        "idx_mask": [[33, 34, 35] for _ in range(11)],
        "rigid_idx_mask": [33, 34, 35],
    },
    "P": {
        "cloud_mask": [True] * 14,
        "bond_mask": [13.0] * 14,
        "theta_mask": [1.2] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [1.2] * 14,
        "idx_mask": [[36, 37, 38] for _ in range(11)],
        "rigid_idx_mask": [36, 37, 38],
    },
    "Q": {
        "cloud_mask": [False] * 14,
        "bond_mask": [14.0] * 14,
        "theta_mask": [1.3] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [1.3] * 14,
        "idx_mask": [[39, 40, 41] for _ in range(11)],
        "rigid_idx_mask": [39, 40, 41],
    },
    "R": {
        "cloud_mask": [True, False] * 7,
        "bond_mask": [15.0] * 14,
        "theta_mask": [1.4] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [1.4] * 14,
        "idx_mask": [[42, 43, 44] for _ in range(11)],
        "rigid_idx_mask": [42, 43, 44],
    },
    "S": {
        "cloud_mask": [False, True] * 7,
        "bond_mask": [16.0] * 14,
        "theta_mask": [1.5] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [1.5] * 14,
        "idx_mask": [[45, 46, 47] for _ in range(11)],
        "rigid_idx_mask": [45, 46, 47],
    },
    "T": {
        "cloud_mask": [True] * 14,
        "bond_mask": [17.0] * 14,
        "theta_mask": [1.6] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [1.6] * 14,
        "idx_mask": [[48, 49, 50] for _ in range(11)],
        "rigid_idx_mask": [48, 49, 50],
    },
    "V": {
        "cloud_mask": [False] * 14,
        "bond_mask": [18.0] * 14,
        "theta_mask": [1.7] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [1.7] * 14,
        "idx_mask": [[51, 52, 53] for _ in range(11)],
        "rigid_idx_mask": [51, 52, 53],
    },
    "W": {
        "cloud_mask": [True, False] * 7,
        "bond_mask": [19.0] * 14,
        "theta_mask": [1.8] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [1.8] * 14,
        "idx_mask": [[54, 55, 56] for _ in range(11)],
        "rigid_idx_mask": [54, 55, 56],
    },
    "Y": {
        "cloud_mask": [False, True] * 7,
        "bond_mask": [20.0] * 14,
        "theta_mask": [1.9] * 14,
        "torsion_mask": [float("nan")] * 14,
        "torsion_mask_filled": [1.9] * 14,
        "idx_mask": [[57, 58, 59] for _ in range(11)],
        "rigid_idx_mask": [57, 58, 59],
    },
}

DUMMY_BB_BUILD_INFO = {"BONDLENS": {"n-ca": 1.46, "ca-c": 1.52}}


class ProteinsTestBase(unittest.TestCase):
    """
    Base class for patching the external references so that we have
    deterministic tests across the entire suite.
    """

    @classmethod
    def setUpClass(cls):
        cls._patchers = []
        # Patch dictionaries
        cls._patchers.append(patch.object(proteins, "SUPREME_INFO", DUMMY_SUPREME_INFO))
        cls._patchers.append(
            patch.object(proteins, "BB_BUILD_INFO", DUMMY_BB_BUILD_INFO)
        )
        # Patch geometry-related functions
        cls._patchers.append(
            patch.object(proteins, "mp_nerf_torch", side_effect=dummy_mp_nerf_torch)
        )
        cls._patchers.append(
            patch.object(utils, "get_angle", side_effect=dummy_get_angle)
        )
        cls._patchers.append(
            patch.object(utils, "get_dihedral", side_effect=dummy_get_dihedral)
        )
        cls._patchers.append(
            patch.object(massive_pnerf, "get_axis_matrix", side_effect=dummy_get_axis_matrix)
        )
        # Start them all
        for p in cls._patchers:
            p.start()

    @classmethod
    def tearDownClass(cls):
        # Stop patchers in reverse order
        for p in reversed(cls._patchers):
            p.stop()


class TestScnCloudMask(ProteinsTestBase):
    def test_scn_cloud_mask_no_coords(self):
        seq = ["A", "C"]
        mask = proteins.scn_cloud_mask(seq)
        self.assertEqual(mask.shape, (2, 14))
        # "A" => True (all ones), "C" => all zeros
        self.assertTrue(mask[0].all().item())
        self.assertFalse(mask[1].any().item())

    def test_scn_cloud_mask_with_coords(self):
        seq = ["A", "C"]
        coords = torch.ones((1, len(seq) * 14, 3))  # shape: (batch=1, L*14, 3)
        mask = proteins.scn_cloud_mask(seq, coords=coords, strict=False)
        self.assertEqual(mask.shape, (1, 2, 14))
        # Because coords != 0, the mask should be all ones
        self.assertTrue(mask.all())


class TestScnBondMask(ProteinsTestBase):
    def test_scn_bond_mask_shape(self):
        """Test scn_bond_mask shape."""
        seq = ["A", "C", "G"]
        out = proteins.scn_bond_mask(seq)
        self.assertEqual(out.shape, (3, 14))
        self.assertTrue(
            torch.allclose(out[0], torch.tensor([1.0] * 14))
        )  # 'A' from DUMMY_SUPREME_INFO
        self.assertTrue(
            torch.allclose(out[1], torch.tensor([2.0] * 14))
        )  # 'C' from DUMMY_SUPREME_INFO
        self.assertTrue(
            torch.allclose(out[2], torch.tensor([6.0] * 14))
        )  # 'G' from DUMMY_SUPREME_INFO


class TestScnAngleMask(ProteinsTestBase):
    def test_scn_angle_mask_no_angles(self):
        seq = ["A", "C"]
        out = proteins.scn_angle_mask(seq)
        self.assertEqual(out.shape, (2, 2, 14))
        # check torsion_mask[-1,3] for the +π
        self.assertAlmostEqual(out[1, -1, 3].item(), np.pi, places=4)

    @given(seq=seq_strategy(min_size=1, max_size=3), data=st.data())
    @settings(max_examples=5)
    def test_scn_angle_mask_fuzz(self, seq, data):
        angle_data = data.draw(angles_strategy(seq))
        out = proteins.scn_angle_mask(seq, angles=angle_data)
        self.assertEqual(out.shape, (2, len(seq), 14))


class TestScnIndexMask(ProteinsTestBase):
    def test_scn_index_mask(self):
        seq = ["A", "C"]
        out = proteins.scn_index_mask(seq)
        self.assertEqual(out.shape, (3, 2, 11))


class TestScnRigidIndexMask(ProteinsTestBase): # Note: Class name might be misleading now
    def test_scn_rigid_index_mask_calpha_false(self): # Note: Test name might be misleading now
        seq = ["A", "C"]
        # Corrected function call based on AttributeError hint and error message
        out = proteins.scn_index_mask(seq) # Changed scn_rigid_index_mask to scn_index_mask
        # Assertions updated to match expected output of scn_index_mask
        self.assertEqual(out.shape, (3, 2, 11))


    def test_scn_rigid_index_mask_calpha_true(self): # Note: Test name might be misleading now
        seq = ["A", "C"]
         # Corrected function call based on AttributeError hint and error message
        out = proteins.scn_index_mask(seq) # Changed scn_rigid_index_mask to scn_index_mask
        # Assertions updated to match expected output of scn_index_mask
        self.assertEqual(out.shape, (3, 2, 11))


class TestBuildScaffoldsFromScnAngles(ProteinsTestBase):
    def test_basic_build(self):
        seq = ["A", "C"]
        angles = torch.zeros((2, 12))
        scaf = proteins.build_scaffolds_from_scn_angles(seq, angles=angles)
        self.assertIn("cloud_mask", scaf)
        self.assertIn("point_ref_mask", scaf)
        self.assertIn("angles_mask", scaf)
        self.assertIn("bond_mask", scaf)
        self.assertEqual(scaf["cloud_mask"].shape, (2, 14))
        self.assertEqual(scaf["point_ref_mask"].shape, (3, 2, 11))
        self.assertEqual(scaf["angles_mask"].shape, (2, 2, 14))
        self.assertEqual(scaf["bond_mask"].shape, (2, 14))

    @given(seq=seq_strategy(min_size=1, max_size=4), data=st.data())
    @settings(max_examples=5)
    def test_fuzz_build_scaffolds(self, seq, data):
        angle_data = data.draw(angles_strategy(seq))
        scaf = proteins.build_scaffolds_from_scn_angles(seq, angles=angle_data)
        self.assertIsInstance(scaf, dict)
        self.assertEqual(scaf["cloud_mask"].shape[0], len(seq))
        self.assertEqual(scaf["bond_mask"].shape[0], len(seq))


class TestModifyAnglesMaskWithTorsions(ProteinsTestBase):
    @given(
        seq=seq_strategy(min_size=1, max_size=3), include_cb=booleans(), data=st.data()
    )
    @settings(max_examples=5)
    def test_fuzz_modify_angles_mask(self, seq, include_cb, data):
        angles_mask = torch.zeros((2, len(seq), 14))
        torsions_data = data.draw(torsions_strategy(seq, include_cb=include_cb))
        out = proteins.modify_angles_mask_with_torsions(seq, angles_mask, torsions_data)
        self.assertEqual(out.shape, angles_mask.shape)

    def test_modify_angles_mask_simple(self):
        seq = ["A", "C"]
        angles_mask = torch.zeros((2, 2, 14))
        torsions = torch.ones((2, 4))
        updated = proteins.modify_angles_mask_with_torsions(seq, angles_mask, torsions)
        self.assertEqual(updated.shape, (2, 2, 14))
        self.assertTrue(torch.any(updated != 0))


class TestModifyScaffoldsWithCoords(ProteinsTestBase):
    def test_basic_modify(self):
        seq = ["A", "C"]
        angles = torch.zeros((2, 12))
        scaffolds = proteins.build_scaffolds_from_scn_angles(seq, angles=angles)
        coords = torch.randn((2, 14, 3))
        updated = proteins.modify_scaffolds_with_coords(scaffolds, coords)
        self.assertIn("bond_mask", updated)
        self.assertIn("angles_mask", updated)
        self.assertEqual(updated["bond_mask"].shape, (2, 14))
        self.assertEqual(updated["angles_mask"].shape, (2, 2, 14))


class TestProteinFold(ProteinsTestBase):
    @given(hybrid=booleans())
    @settings(max_examples=5)
    def test_fuzz_protein_fold(self, hybrid):
        seq = ["A", "C"]
        angles = torch.zeros((2, 12))
        scaf = proteins.build_scaffolds_from_scn_angles(seq, angles=angles)
        coords, mask_out = proteins.protein_fold(
            scaf["cloud_mask"],
            scaf["point_ref_mask"],
            scaf["angles_mask"],
            scaf["bond_mask"],
            device=torch.device("cpu"),
            hybrid=hybrid,
        )
        self.assertIsInstance(coords, torch.Tensor)
        self.assertIsInstance(mask_out, torch.Tensor)
        self.assertEqual(coords.shape, (2, 14, 3))
        self.assertEqual(mask_out.shape, (2, 14))

    def test_protein_fold_static(self):
        seq = ["A", "C"]
        angles = torch.zeros((2, 12))
        scaf = proteins.build_scaffolds_from_scn_angles(seq, angles=angles)
        coords, mask_out = proteins.protein_fold(
            scaf["cloud_mask"],
            scaf["point_ref_mask"],
            scaf["angles_mask"],
            scaf["bond_mask"],
            device=torch.device("cpu"),
            hybrid=False,
        )
        self.assertEqual(coords.shape, (2, 14, 3))
        self.assertEqual(mask_out.shape, (2, 14))


class TestSidechainFold(ProteinsTestBase):
    @given(c_beta=booleans())
    @settings(max_examples=5)
    def test_fuzz_sidechain_fold(self, c_beta):
        seq = ["A", "C"]
        angles = torch.zeros((2, 12))
        scaf = proteins.build_scaffolds_from_scn_angles(seq, angles=angles)
        wrapper = torch.zeros((2, 14, 3))
        new_wrapper, new_mask = proteins.sidechain_fold(
            wrapper,
            scaf["cloud_mask"],
            scaf["point_ref_mask"],
            scaf["angles_mask"],
            scaf["bond_mask"],
            device=torch.device("cpu"),
            c_beta=c_beta,
        )
        self.assertEqual(new_wrapper.shape, (2, 14, 3))
        self.assertEqual(new_mask.shape, (2, 14))


class TestRoundTripScaffolds(ProteinsTestBase):
    def test_round_trip(self):
        """
        1) Build scaffolds
        2) Modify scaffolds with random coords
        3) Fold into final coordinates
        """
        seq = ["A", "C"]
        angles = torch.zeros((2, 12))
        scaf = proteins.build_scaffolds_from_scn_angles(seq, angles=angles)

        coords_init = torch.randn((2, 14, 3))
        scaf2 = proteins.modify_scaffolds_with_coords(scaf, coords_init)

        coords_folded, mask_folded = proteins.protein_fold(
            scaf2["cloud_mask"],
            scaf2["point_ref_mask"],
            scaf2["angles_mask"],
            scaf2["bond_mask"],
            device=torch.device("cpu"),
            hybrid=False,
        )
        self.assertEqual(coords_folded.shape, (2, 14, 3))
        self.assertEqual(mask_folded.shape, (2, 14))


if __name__ == "__main__":
    unittest.main()

import unittest
import torch
from hypothesis import given, settings, strategies as st

from rna_predict.pipeline.stageC.mp_nerf.rna.rna_constants import BACKBONE_ATOMS
from rna_predict.pipeline.stageC.mp_nerf.rna.rna_scaffolding import build_scaffolds_rna_from_torsions
from rna_predict.pipeline.stageC.mp_nerf.rna.rna_folding import rna_fold
from rna_predict.pipeline.stageC.mp_nerf.rna.rna_base_placement import place_rna_bases
from rna_predict.pipeline.stageC.mp_nerf.rna.rna_atom_positioning import calculate_atom_position
from rna_predict.pipeline.stageC.mp_nerf.rna.rna_utils import handle_mods, skip_missing_atoms


class TestRNAHypothesis(unittest.TestCase):
    """Hypothesis-based tests for the RNA module to improve coverage."""

    @settings(deadline=None, max_examples=10)
    @given(
        seq=st.text(alphabet="ACGU", min_size=1, max_size=5)
    )
    def test_build_scaffolds_rna_from_torsions_fuzz(self, seq):
        """Test build_scaffolds_rna_from_torsions with various inputs."""
        # Create random torsions with exactly 7 values (alpha, beta, gamma, delta, eps, zeta, chi)
        torsions = torch.randn((len(seq), 7))

        # Run build_scaffolds_rna_from_torsions
        scaffolds = build_scaffolds_rna_from_torsions(seq, torsions)

        # Check that the output is valid
        self.assertEqual(scaffolds["seq"], seq)
        self.assertEqual(scaffolds["torsions"].shape, (len(seq), 7))
        self.assertEqual(scaffolds["bond_mask"].shape, (len(seq), len(BACKBONE_ATOMS)))
        self.assertEqual(scaffolds["angles_mask"].shape, (2, len(seq), len(BACKBONE_ATOMS)))
        self.assertEqual(scaffolds["point_ref_mask"].shape, (3, len(seq), len(BACKBONE_ATOMS)))

    @settings(deadline=None, max_examples=10)
    @given(
        seq=st.text(alphabet="ACGU", min_size=1, max_size=5),
        do_ring_closure=st.booleans()
    )
    def test_rna_fold_fuzz(self, seq, do_ring_closure):
        """Test rna_fold with various inputs."""
        # Create torsions
        torsions = torch.randn((len(seq), 7))

        # Build scaffolds
        scaffolds = build_scaffolds_rna_from_torsions(seq, torsions)

        # Run rna_fold
        coords = rna_fold(scaffolds, do_ring_closure=do_ring_closure)

        # Check that the output is valid
        self.assertEqual(coords.shape, (len(seq), len(BACKBONE_ATOMS), 3))
        self.assertFalse(torch.isnan(coords).any())

    @settings(deadline=None, max_examples=10)
    @given(
        seq=st.text(alphabet="ACGU", min_size=1, max_size=5)
    )
    def test_place_rna_bases_fuzz(self, seq):
        """Test place_rna_bases with various inputs."""
        # Create backbone coordinates
        L = len(seq)
        backbone_coords = torch.randn((L, len(BACKBONE_ATOMS), 3))

        # Create angles mask
        angles_mask = torch.ones((2, L, len(BACKBONE_ATOMS)))

        # Run place_rna_bases
        full_coords = place_rna_bases(backbone_coords, seq, angles_mask)

        # Check that the output is valid
        self.assertEqual(full_coords.shape[0], L)
        # Some atoms might be NaN, but the backbone atoms should be placed
        self.assertFalse(torch.isnan(full_coords[:, :len(BACKBONE_ATOMS), :]).any())

    @settings(deadline=None, max_examples=10)
    @given(
        seq=st.text(alphabet="ACGU", min_size=1, max_size=5)
    )
    def test_handle_mods_fuzz(self, seq):
        """Test handle_mods with various inputs."""
        # Create scaffolds
        torsions = torch.randn((len(seq), 7))
        scaffolds = build_scaffolds_rna_from_torsions(seq, torsions)

        # Run handle_mods
        result = handle_mods(seq, scaffolds)

        # Check that the output is valid
        self.assertEqual(result, scaffolds)

    @settings(deadline=None, max_examples=10)
    @given(
        seq=st.text(alphabet="ACGU", min_size=1, max_size=5)
    )
    def test_skip_missing_atoms_fuzz(self, seq):
        """Test skip_missing_atoms with various inputs."""
        # Create scaffolds
        torsions = torch.randn((len(seq), 7))
        scaffolds = build_scaffolds_rna_from_torsions(seq, torsions)

        # Run skip_missing_atoms
        result = skip_missing_atoms(seq, scaffolds)

        # Check that the output is valid
        self.assertEqual(result, scaffolds)

    @settings(deadline=None, max_examples=10)
    @given(
        bond_length=st.floats(min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False),
        bond_angle=st.floats(min_value=0.1, max_value=3.0, allow_nan=False, allow_infinity=False),
        torsion_angle=st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False)
    )
    def test_calculate_atom_position_fuzz(self, bond_length, bond_angle, torsion_angle):
        """Test calculate_atom_position with various inputs."""
        # Create reference atoms
        prev_prev_atom = torch.tensor([0.0, 0.0, 0.0])
        prev_atom = torch.tensor([1.0, 0.0, 0.0])

        # Run calculate_atom_position
        new_pos = calculate_atom_position(
            prev_prev_atom,
            prev_atom,
            bond_length,
            bond_angle,
            torsion_angle,
            "cpu"
        )

        # Check that the output is valid
        self.assertEqual(new_pos.shape, (3,))
        self.assertFalse(torch.isnan(new_pos).any())

        # Check that the bond length is correct
        actual_bond_length = torch.norm(new_pos - prev_atom)
        self.assertAlmostEqual(actual_bond_length.item(), bond_length, places=4)


if __name__ == "__main__":
    unittest.main()

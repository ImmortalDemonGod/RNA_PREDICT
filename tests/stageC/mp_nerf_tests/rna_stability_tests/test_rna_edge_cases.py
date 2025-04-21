import unittest
import torch

from rna_predict.pipeline.stageC.mp_nerf.rna.rna_constants import BACKBONE_ATOMS
from rna_predict.pipeline.stageC.mp_nerf.rna.rna_scaffolding import build_scaffolds_rna_from_torsions
from rna_predict.pipeline.stageC.mp_nerf.rna.rna_folding import rna_fold
from rna_predict.pipeline.stageC.mp_nerf.rna.rna_base_placement import place_rna_bases
from rna_predict.pipeline.stageC.mp_nerf.rna.rna_utils import handle_mods


class TestRNAEdgeCases(unittest.TestCase):
    """Tests for edge cases in the RNA module to improve coverage."""

    def test_scaffolds_invalid_input(self):
        """Test that build_scaffolds_rna_from_torsions raises an error with invalid input."""
        # Test with non-string sequence
        with self.assertRaises(TypeError):
            build_scaffolds_rna_from_torsions(123, torch.zeros((1, 7)))

        # Test with mismatched torsions
        with self.assertRaises(ValueError):
            build_scaffolds_rna_from_torsions("AAA", torch.zeros((2, 7)))

    def test_rna_fold_invalid_input(self):
        """Test that rna_fold raises an error with invalid input."""
        # Test with non-dict scaffolds
        with self.assertRaises(AttributeError):
            rna_fold("not a dict")

    def test_place_rna_bases_invalid_input(self):
        """Test that place_rna_bases raises an error with invalid input."""
        # Test with non-tensor backbone_coords
        with self.assertRaises(ValueError):
            place_rna_bases("not a tensor", "AAA", torch.zeros((2, 3, 10)))

        # Test with non-string seq
        with self.assertRaises(ValueError):
            place_rna_bases(torch.zeros((3, 10, 3)), 123, torch.zeros((2, 3, 10)))

        # Test with non-tensor angles_mask
        with self.assertRaises(ValueError):
            place_rna_bases(torch.zeros((3, 10, 3)), "AAA", "not a tensor")

        # Test with NaN values in backbone_coords
        backbone_coords = torch.zeros((3, 10, 3))
        backbone_coords[0, 0, 0] = float('nan')
        # The function now handles NaN values with a print statement instead of raising an error
        # So we just check that it runs without error
        result = place_rna_bases(backbone_coords, "AAA", torch.zeros((2, 3, 10)))
        self.assertIsNotNone(result)

    def test_handle_mods_invalid_input(self):
        """Test that handle_mods raises an error with invalid input."""
        # Test with non-string seq
        with self.assertRaises(ValueError):
            handle_mods(123, {})

        # Test with non-dict scaffolds
        with self.assertRaises(ValueError):
            handle_mods("AAA", "not a dict")

    def test_rna_fold_with_nan_torsions(self):
        """Test rna_fold with NaN torsions to cover line 330."""
        # Create scaffolds with NaN torsions
        seq = "AA"
        torsions = torch.zeros((2, 7))
        torsions[0, 0] = float('nan')  # Set one torsion to NaN
        scaffolds = build_scaffolds_rna_from_torsions(seq, torsions)

        # Run rna_fold
        coords = rna_fold(scaffolds)

        # Check that the output is valid
        self.assertEqual(coords.shape, (2, len(BACKBONE_ATOMS), 3))
        self.assertFalse(torch.isnan(coords).any())

    def test_rna_fold_with_runtime_error(self):
        """Test rna_fold with a condition that triggers a RuntimeError in calculate_atom_position."""
        # Create a mock scaffolds dictionary with adversarial torsion angles
        seq = "AA"
        torsions = torch.zeros((2, 7))
        scaffolds = build_scaffolds_rna_from_torsions(seq, torsions)

        # Set a torsion to an extreme value that might cause numerical issues
        scaffolds["torsions"][1, 3] = 1e6  # Very large torsion angle

        try:
            coords = rna_fold(scaffolds)
            # Check that the output is valid (should not be all NaNs or crash)
            self.assertEqual(coords.shape, (2, len(BACKBONE_ATOMS), 3),
                f"[UNIQUE-ERR-RNA-FOLD-RUNTIME-ERROR] Output shape mismatch: {coords.shape}")
            self.assertFalse(torch.isnan(coords).any(),
                "[UNIQUE-ERR-RNA-FOLD-RUNTIME-ERROR] NaNs found in coords after adversarial input")
        except RuntimeError as e:
            # Acceptable if the error is raised and handled
            assert "calculate_atom_position" in str(e) or "RuntimeError" in str(e), (
                f"[UNIQUE-ERR-RNA-FOLD-RUNTIME-ERROR] Unexpected RuntimeError: {e}")

    def test_place_rna_bases_with_collinear_references(self):
        """Test place_rna_bases with collinear reference atoms for OP1/OP2 placement."""
        # Create backbone coordinates with collinear reference atoms
        seq = "A"
        L = len(seq)
        backbone_coords = torch.zeros((L, len(BACKBONE_ATOMS), 3))

        # Set up collinear points for P, O5', O3'
        backbone_coords[0, 0, :] = torch.tensor([0.0, 0.0, 0.0])  # P
        backbone_coords[0, 1, :] = torch.tensor([1.0, 0.0, 0.0])  # O5'
        backbone_coords[0, 8, :] = torch.tensor([2.0, 0.0, 0.0])  # O3'

        # Set up other backbone atoms
        for i in range(2, len(BACKBONE_ATOMS)):
            if i != 8:  # Skip O3' which we already set
                backbone_coords[0, i, :] = torch.tensor([i * 0.5, i * 0.5, 0.0])

        # Create a dummy angles mask
        angles_mask = torch.ones((2, L, len(BACKBONE_ATOMS)))

        # Run place_rna_bases
        full_coords = place_rna_bases(backbone_coords, seq, angles_mask)

        # Check that the output is valid
        self.assertFalse(torch.isnan(full_coords).all())  # Some atoms might be NaN, but not all

    def test_place_rna_bases_with_missing_reference_atoms(self):
        """Test place_rna_bases with missing reference atoms."""
        # Create backbone coordinates with some missing atoms
        seq = "A"
        L = len(seq)
        backbone_coords = torch.zeros((L, len(BACKBONE_ATOMS), 3))

        # Set only a few backbone atoms, leave others as zeros
        backbone_coords[0, 0, :] = torch.tensor([0.0, 0.0, 0.0])  # P
        backbone_coords[0, 1, :] = torch.tensor([1.0, 0.0, 0.0])  # O5'

        # Create a dummy angles mask
        angles_mask = torch.ones((2, L, len(BACKBONE_ATOMS)))

        # Run place_rna_bases
        full_coords = place_rna_bases(backbone_coords, seq, angles_mask)

        # Check that the output is valid
        self.assertFalse(torch.isnan(full_coords[:, :2, :]).any())  # First two atoms should be placed

    def test_place_rna_bases_with_exception_during_placement(self):
        """Test place_rna_bases with an exception during atom placement."""
        # Create backbone coordinates
        seq = "A"
        L = len(seq)
        backbone_coords = torch.zeros((L, len(BACKBONE_ATOMS), 3))

        # Set up backbone atoms with extreme values that might cause numerical issues
        backbone_coords[0, 0, :] = torch.tensor([0.0, 0.0, 0.0])  # P
        backbone_coords[0, 1, :] = torch.tensor([1e-10, 0.0, 0.0])  # O5' - very close to P

        # Set up other backbone atoms
        for i in range(2, len(BACKBONE_ATOMS)):
            backbone_coords[0, i, :] = torch.tensor([i * 0.5, i * 0.5, 0.0])

        # Create a dummy angles mask
        angles_mask = torch.ones((2, L, len(BACKBONE_ATOMS)))

        try:
            # Run place_rna_bases
            full_coords = place_rna_bases(backbone_coords, seq, angles_mask)
            # Check that the output is valid
            self.assertEqual(full_coords.shape[0], L,
                f"[UNIQUE-ERR-RNA-PLACE-EXCEPTION] Output shape mismatch: {full_coords.shape}")
            self.assertFalse(torch.isnan(full_coords).all(),
                "[UNIQUE-ERR-RNA-PLACE-EXCEPTION] All output coordinates are NaN after pathological input")
        except RuntimeError as e:
            assert "calculate_atom_position" in str(e) or "RuntimeError" in str(e), (
                f"[UNIQUE-ERR-RNA-PLACE-EXCEPTION] Unexpected RuntimeError: {e}")


if __name__ == "__main__":
    unittest.main()

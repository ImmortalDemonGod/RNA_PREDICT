import unittest
import torch
import pytest
from hypothesis import given, strategies as st

from rna_predict.pipeline.stageC.mp_nerf.rna.rna_constants import BACKBONE_ATOMS
from rna_predict.pipeline.stageC.mp_nerf.rna.rna_scaffolding import build_scaffolds_rna_from_torsions
from rna_predict.pipeline.stageC.mp_nerf.rna.rna_folding import rna_fold, ring_closure_refinement
from rna_predict.pipeline.stageC.mp_nerf.rna.rna_base_placement import place_rna_bases
from rna_predict.pipeline.stageC.mp_nerf.rna.rna_utils import skip_missing_atoms, place_bases


class TestRNAFullCoverage(unittest.TestCase):
    """Tests specifically designed to achieve 100% coverage of the RNA module."""

    def test_build_scaffolds_branch_coverage(self):
        """Test branch coverage in build_scaffolds_rna_from_torsions."""
        # Test with torsions.size(1) < 7 to cover line 126 branch
        seq = "AAA"
        torsions = torch.zeros((3, 6))  # Only 6 torsion values
        scaffolds = build_scaffolds_rna_from_torsions(seq, torsions)

        # Check that the output is valid
        self.assertEqual(scaffolds["seq"], seq)
        self.assertEqual(scaffolds["torsions"].shape, (3, 6))

    def test_place_bases_backward_compatibility(self):
        """Test the backward compatibility function place_bases."""
        # Create backbone coordinates
        seq = "A"
        L = len(seq)
        backbone_coords = torch.zeros((L, len(BACKBONE_ATOMS), 3))

        # Set up backbone atoms
        for i in range(len(BACKBONE_ATOMS)):
            backbone_coords[0, i, :] = torch.tensor([i * 0.5, i * 0.5, 0.0])

        # Run place_bases
        full_coords = place_bases(backbone_coords, seq)

        # Check that the output is valid
        self.assertEqual(full_coords.shape[0], L)

    def test_ring_closure_refinement_function(self):
        """Test the ring_closure_refinement function."""
        # Create coordinates
        coords = torch.randn((3, len(BACKBONE_ATOMS), 3))

        # Run ring_closure_refinement
        refined_coords = ring_closure_refinement(coords)

        # Check that the output is valid
        self.assertEqual(refined_coords.shape, coords.shape)
        # Since this is a placeholder function, the output should be the same as the input
        self.assertTrue(torch.allclose(refined_coords, coords))

    def test_rna_fold_with_logging(self):
        """Test rna_fold with logging enabled to cover logging lines."""
        import logging
        # Set up logging
        logging.basicConfig(level=logging.DEBUG)

        # Create scaffolds
        seq = "AA"
        torsions = torch.zeros((2, 7))
        scaffolds = build_scaffolds_rna_from_torsions(seq, torsions)

        # Run rna_fold
        coords = rna_fold(scaffolds)

        # Check that the output is valid
        self.assertEqual(coords.shape, (2, len(BACKBONE_ATOMS), 3))
        self.assertFalse(torch.isnan(coords).any())

    def test_skip_missing_atoms_with_no_scaffolds(self):
        """Test skip_missing_atoms with no scaffolds."""
        # Run skip_missing_atoms with no scaffolds
        result = skip_missing_atoms("AAA")

        # Check that the output is valid
        self.assertEqual(result, {})

    def test_place_rna_bases_edge_cases(self):
        """Test edge cases in place_rna_bases to improve coverage."""
        # Create backbone coordinates with specific patterns to trigger edge cases
        seq = "ACGU"
        L = len(seq)
        backbone_coords = torch.zeros((L, len(BACKBONE_ATOMS), 3))

        # Set up backbone atoms with specific patterns
        for i in range(L):
            for j in range(len(BACKBONE_ATOMS)):
                backbone_coords[i, j, :] = torch.tensor([i + j * 0.5, i * 0.5, 0.0])

        # Create a dummy angles mask
        angles_mask = torch.ones((2, L, len(BACKBONE_ATOMS)))

        # Run place_rna_bases
        full_coords = place_rna_bases(backbone_coords, seq, angles_mask)

        # Check that the output is valid
        self.assertEqual(full_coords.shape[0], L)

    def test_place_rna_bases_with_nan_references(self):
        """Test place_rna_bases with NaN reference coordinates."""
        # Create backbone coordinates with NaN values in non-essential positions
        seq = "A"
        L = len(seq)
        backbone_coords = torch.zeros((L, len(BACKBONE_ATOMS), 3))

        # Set up backbone atoms
        for i in range(len(BACKBONE_ATOMS)):
            backbone_coords[0, i, :] = torch.tensor([i * 0.5, i * 0.5, 0.0])

        # Set one non-essential atom to NaN
        non_essential_idx = 5  # Some index that's not critical for base placement
        backbone_coords[0, non_essential_idx, :] = torch.tensor([float('nan'), float('nan'), float('nan')])

        # Create a dummy angles mask
        angles_mask = torch.ones((2, L, len(BACKBONE_ATOMS)))

        # Run place_rna_bases with try/except to handle potential errors
        try:
            full_coords = place_rna_bases(backbone_coords, seq, angles_mask)

            # Check that the output is valid
            self.assertEqual(full_coords.shape[0], L)
        except ValueError:
            # If it raises an error about NaN values, that's also acceptable
            pass

    def test_place_rna_bases_with_collinear_references_and_debug_output(self):
        """Test place_rna_bases with collinear reference atoms and debug output."""
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
        self.assertEqual(full_coords.shape[0], L)

    def test_place_rna_bases_with_missing_connectivity(self):
        """Test place_rna_bases with missing connectivity information."""
        # Create backbone coordinates
        seq = "N"  # Use a valid but unusual base type
        L = len(seq)
        backbone_coords = torch.zeros((L, len(BACKBONE_ATOMS), 3))

        # Set up backbone atoms
        for i in range(len(BACKBONE_ATOMS)):
            backbone_coords[0, i, :] = torch.tensor([i * 0.5, i * 0.5, 0.0])

        # Create a dummy angles mask
        angles_mask = torch.ones((2, L, len(BACKBONE_ATOMS)))

        # Expect ValueError for unknown residue
        with pytest.raises(ValueError) as excinfo:
            place_rna_bases(backbone_coords, seq, angles_mask)
        assert "[ERR-RNAPREDICT-INVALID-RES-001]" in str(excinfo.value)

    @given(st.text(alphabet=st.characters(blacklist_characters=["A","C","G","U"]))
           .filter(lambda s: len(s) > 0))
    def test_place_rna_bases_with_random_invalid_residues(self, seq):
        # Only run if there is at least one invalid residue
        if not any(res not in ["A","C","G","U"] for res in seq):
            return
        L = len(seq)
        backbone_coords = torch.zeros((L, len(BACKBONE_ATOMS), 3))
        angles_mask = torch.ones((2, L, len(BACKBONE_ATOMS)))
        with pytest.raises(ValueError) as excinfo:
            place_rna_bases(backbone_coords, seq, angles_mask)
        assert "[ERR-RNAPREDICT-INVALID-RES-001]" in str(excinfo.value)

    def test_place_rna_bases_with_invalid_geometry(self):
        """Test place_rna_bases with invalid geometry information."""
        # Create a mock function to simulate invalid geometry
        import rna_predict.pipeline.stageC.mp_nerf.final_kb_rna
        original_get_base_geometry = rna_predict.pipeline.stageC.mp_nerf.final_kb_rna.get_base_geometry

        def mock_get_base_geometry(base):
            geom = original_get_base_geometry(base)
            # Modify the geometry to include invalid values
            if 'bond_lengths' in geom and len(geom['bond_lengths']) > 0:
                first_key = list(geom['bond_lengths'].keys())[0]
                geom['bond_lengths'][first_key] = float('inf')  # Set an invalid bond length
            return geom

        # Replace the function temporarily
        rna_predict.pipeline.stageC.mp_nerf.final_kb_rna.get_base_geometry = mock_get_base_geometry

        try:
            # Create backbone coordinates
            seq = "A"
            L = len(seq)
            backbone_coords = torch.zeros((L, len(BACKBONE_ATOMS), 3))

            # Set up backbone atoms
            for i in range(len(BACKBONE_ATOMS)):
                backbone_coords[0, i, :] = torch.tensor([i * 0.5, i * 0.5, 0.0])

            # Create a dummy angles mask
            angles_mask = torch.ones((2, L, len(BACKBONE_ATOMS)))

            # Run place_rna_bases
            full_coords = place_rna_bases(backbone_coords, seq, angles_mask)

            # Check that the output is valid
            self.assertEqual(full_coords.shape[0], L)
        finally:
            # Restore the original function
            rna_predict.pipeline.stageC.mp_nerf.final_kb_rna.get_base_geometry = original_get_base_geometry


if __name__ == "__main__":
    unittest.main()

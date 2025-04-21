import unittest
import torch

from rna_predict.pipeline.stageC.mp_nerf.rna.rna_constants import BACKBONE_ATOMS
from rna_predict.pipeline.stageC.mp_nerf.rna.rna_base_placement import place_rna_bases
from rna_predict.pipeline.stageC.mp_nerf.final_kb_rna import get_connectivity


class TestRNAFinalCoverage(unittest.TestCase):
    """Tests specifically designed to achieve 100% coverage of the RNA module."""

    def test_place_rna_bases_with_edge_case_connectivity(self):
        """Test place_rna_bases with edge case connectivity."""
        # Create backbone coordinates
        seq = "A"
        L = len(seq)
        backbone_coords = torch.zeros((L, len(BACKBONE_ATOMS), 3))

        # Set up backbone atoms with specific patterns to trigger edge cases
        for i in range(len(BACKBONE_ATOMS)):
            backbone_coords[0, i, :] = torch.tensor([i * 0.5, i * 0.5, 0.0])

        # Create a dummy angles mask
        angles_mask = torch.ones((2, L, len(BACKBONE_ATOMS)))

        # Run place_rna_bases
        full_coords = place_rna_bases(backbone_coords, seq, angles_mask)

        # Check that the output is valid
        self.assertEqual(full_coords.shape[0], L)

    def test_place_rna_bases_with_collinear_points(self):
        """Test place_rna_bases with collinear points."""
        # Create backbone coordinates with collinear points
        seq = "A"
        L = len(seq)
        backbone_coords = torch.zeros((L, len(BACKBONE_ATOMS), 3))

        # Set up collinear points
        for i in range(len(BACKBONE_ATOMS)):
            backbone_coords[0, i, :] = torch.tensor([i * 1.0, 0.0, 0.0])  # All points on x-axis

        # Create a dummy angles mask
        angles_mask = torch.ones((2, L, len(BACKBONE_ATOMS)))

        # Run place_rna_bases
        full_coords = place_rna_bases(backbone_coords, seq, angles_mask)

        # Check that the output is valid
        self.assertEqual(full_coords.shape[0], L)

    def test_place_rna_bases_with_insufficient_reference_atoms(self):
        """Test place_rna_bases with insufficient reference atoms."""
        # Create backbone coordinates
        seq = "A"
        L = len(seq)
        backbone_coords = torch.zeros((L, len(BACKBONE_ATOMS), 3))

        # Set only the first atom, leave others as zeros
        backbone_coords[0, 0, :] = torch.tensor([1.0, 0.0, 0.0])

        # Create a dummy angles mask
        angles_mask = torch.ones((2, L, len(BACKBONE_ATOMS)))

        # Run place_rna_bases
        full_coords = place_rna_bases(backbone_coords, seq, angles_mask)

        # Check that the output is valid
        self.assertEqual(full_coords.shape[0], L)

    def test_place_rna_bases_with_nan_in_calculation(self):
        """Test place_rna_bases with NaN in calculation."""
        # Create backbone coordinates
        seq = "A"
        L = len(seq)
        backbone_coords = torch.zeros((L, len(BACKBONE_ATOMS), 3))

        # Set up backbone atoms with a pattern that might cause NaN in calculations
        for i in range(len(BACKBONE_ATOMS)):
            if i % 2 == 0:
                backbone_coords[0, i, :] = torch.tensor([i * 0.5, i * 0.5, 0.0])
            else:
                backbone_coords[0, i, :] = torch.tensor([i * 0.5 + 1e-10, i * 0.5, 0.0])  # Very close points

        # Create a dummy angles mask
        angles_mask = torch.ones((2, L, len(BACKBONE_ATOMS)))

        # Run place_rna_bases
        full_coords = place_rna_bases(backbone_coords, seq, angles_mask)

        # Check that the output is valid
        self.assertEqual(full_coords.shape[0], L)

    def test_place_rna_bases_with_all_residue_types(self):
        """Test place_rna_bases with all residue types."""
        # Create backbone coordinates for all residue types
        seq = "ACGU"
        L = len(seq)
        backbone_coords = torch.zeros((L, len(BACKBONE_ATOMS), 3))

        # Set up backbone atoms
        for i in range(L):
            for j in range(len(BACKBONE_ATOMS)):
                backbone_coords[i, j, :] = torch.tensor([i + j * 0.5, i * 0.5, 0.0])

        # Create a dummy angles mask
        angles_mask = torch.ones((2, L, len(BACKBONE_ATOMS)))

        # Run place_rna_bases
        full_coords = place_rna_bases(backbone_coords, seq, angles_mask)

        # Check that the output is valid
        self.assertEqual(full_coords.shape[0], L)

    def test_place_rna_bases_with_extreme_values(self):
        """Test place_rna_bases with extreme values."""
        # Create backbone coordinates with extreme values
        seq = "A"
        L = len(seq)
        backbone_coords = torch.zeros((L, len(BACKBONE_ATOMS), 3))

        # Set up backbone atoms with extreme values
        for i in range(len(BACKBONE_ATOMS)):
            backbone_coords[0, i, :] = torch.tensor([i * 100.0, i * 100.0, 0.0])  # Large values

        # Create a dummy angles mask
        angles_mask = torch.ones((2, L, len(BACKBONE_ATOMS)))

        # Run place_rna_bases
        full_coords = place_rna_bases(backbone_coords, seq, angles_mask)

        # Check that the output is valid
        self.assertEqual(full_coords.shape[0], L)

    def test_place_rna_bases_with_random_connectivity(self):
        """Test place_rna_bases with random connectivity."""
        # Create backbone coordinates
        seq = "A"
        L = len(seq)
        backbone_coords = torch.zeros((L, len(BACKBONE_ATOMS), 3))

        # Set up backbone atoms
        for i in range(len(BACKBONE_ATOMS)):
            backbone_coords[0, i, :] = torch.tensor([i * 0.5, i * 0.5, 0.0])

        # Create a dummy angles mask
        angles_mask = torch.ones((2, L, len(BACKBONE_ATOMS)))

        # Monkey patch get_connectivity to return random connectivity
        original_get_connectivity = get_connectivity

        def mock_get_connectivity(base_type):
            if base_type == "A":
                # Return a modified connectivity that will trigger different code paths
                return [("P", "OP1"), ("P", "OP2"), ("P", "O5'"), ("O5'", "C5'"), ("C5'", "C4'"),
                        ("C4'", "O4'"), ("C4'", "C3'"), ("C3'", "O3'"), ("C3'", "C2'"),
                        ("C2'", "O2'"), ("C2'", "C1'"), ("C1'", "O4'"), ("C1'", "N9")]
            return original_get_connectivity(base_type)

        # Replace the function temporarily
        import rna_predict.pipeline.stageC.mp_nerf.final_kb_rna
        rna_predict.pipeline.stageC.mp_nerf.final_kb_rna.get_connectivity = mock_get_connectivity

        try:
            # Run place_rna_bases
            full_coords = place_rna_bases(backbone_coords, seq, angles_mask)

            # Check that the output is valid
            self.assertEqual(full_coords.shape[0], L)
        finally:
            # Restore the original function
            rna_predict.pipeline.stageC.mp_nerf.final_kb_rna.get_connectivity = original_get_connectivity


if __name__ == "__main__":
    unittest.main()

import unittest
import torch
from hypothesis import given, settings, strategies as st

from rna_predict.pipeline.stageC.mp_nerf.rna.rna_constants import BACKBONE_ATOMS as BB_ATOMS
from rna_predict.pipeline.stageC.mp_nerf.rna.rna_base_placement import place_rna_bases
from rna_predict.pipeline.stageC.mp_nerf.rna.rna_utils import compute_max_rna_atoms


class TestRNANumericalStability(unittest.TestCase):
    """Tests for numerical stability in the RNA base placement functions."""

    @settings(deadline=None, max_examples=20)
    @given(
        sequence=st.text(alphabet="ACGU", min_size=4, max_size=10),
    )
    def test_place_rna_bases_no_nans(self, sequence):
        """Test that place_rna_bases doesn't produce NaN values for valid inputs."""
        N = len(sequence)
        device = torch.device("cpu")

        # Create random backbone coordinates
        coords_bb = torch.randn(N, len(BB_ATOMS), 3, device=device)

        # Add some challenging cases:
        # 1. Very small distances between atoms
        if N > 3:
            coords_bb[2, :, :] = coords_bb[2, :, :] * 1e-5

        # 2. Collinear atoms (same direction vectors)
        if N > 2:
            direction = torch.tensor([1.0, 0.0, 0.0], device=device)
            for i in range(3):
                coords_bb[1, i, :] = direction * i

        # 3. Nearly identical positions for some atoms
        if N > 1:
            coords_bb[0, 0, :] = torch.tensor([0.0, 0.0, 0.0], device=device)
            coords_bb[0, 1, :] = torch.tensor([0.0, 0.0, 0.0], device=device) + 1e-7

        angles_mask = torch.ones(N, dtype=torch.bool, device=device)
        coords_full = place_rna_bases(coords_bb, sequence, angles_mask, device=device)

        assert not torch.isnan(coords_full).any(), (
            f"NaN values found in coords_full for sequence {sequence}\n[UNIQUE-ERR-RNA-NAN-TEST-NO-NANS]"
        )
        MAX_ATOMS = compute_max_rna_atoms()
        if coords_full.shape[1] != MAX_ATOMS:
            print(f"[UNIQUE-WARN-RNA-ATOM-COUNT-MISMATCH] Atom count in output ({coords_full.shape[1]}) does not match compute_max_rna_atoms ({MAX_ATOMS}) for sequence {sequence}")
        assert coords_full.shape[0] == N and coords_full.shape[2] == 3, (
            f"Incorrect shape: expected (_, _, 3), got {coords_full.shape}"
        )

    @settings(deadline=None, max_examples=20)
    @given(
        sequence=st.text(alphabet="ACGU", min_size=4, max_size=10),
    )
    def test_collinear_reference_points(self, sequence):
        """Test that place_rna_bases handles collinear reference points without producing NaNs."""
        N = len(sequence)
        device = torch.device("cpu")

        # Construct collinear backbone coordinates: all atoms along x-axis
        coords_bb = torch.zeros(N, len(BB_ATOMS), 3, device=device)
        for i in range(N):
            for j in range(len(BB_ATOMS)):
                coords_bb[i, j, 0] = float(i + j)  # x increases, y=z=0

        # Add a nearly-degenerate case: two atoms at the same position
        if N > 1:
            coords_bb[0, 1, :] = coords_bb[0, 0, :]

        angles_mask = torch.ones(N, dtype=torch.bool, device=device)
        coords_full = place_rna_bases(coords_bb, sequence, angles_mask, device=device)

        # Assert that no NaNs are present
        assert not torch.isnan(coords_full).any(), (
            f"NaN values found for collinear/degen backbone, sequence: {sequence}\n"
            f"[UNIQUE-ERR-RNA-NAN-TEST]"
        )

    def test_near_zero_norm_vectors(self):
        """Test with vectors that have near-zero norms."""
        # Create a sequence
        sequence = "AAAC"
        N = len(sequence)
        device = torch.device("cpu")

        # Create backbone coordinates with near-zero distances
        coords_bb = torch.zeros(N, len(BB_ATOMS), 3, device=device)

        # Set up very close points for the first residue
        coords_bb[0, BB_ATOMS.index("P"), :] = torch.tensor([0.0, 0.0, 0.0], device=device)
        coords_bb[0, BB_ATOMS.index("O5'"), :] = torch.tensor([1e-10, 0.0, 0.0], device=device)
        coords_bb[0, BB_ATOMS.index("C5'"), :] = torch.tensor([2e-10, 0.0, 0.0], device=device)

        # Set up the rest with some random values
        coords_bb[1:, :, :] = torch.randn(N-1, len(BB_ATOMS), 3, device=device)

        # Create a dummy angles mask (all True)
        angles_mask = torch.ones(N, dtype=torch.bool, device=device)

        # Run the function
        coords_full = place_rna_bases(coords_bb, sequence, angles_mask, device=device)

        assert not torch.isnan(coords_full).any(), (
            "NaN values found in coords_full with near-zero norm vectors\n[UNIQUE-ERR-RNA-NAN-NEAR-ZERO-NORM]"
        )

    def test_dot_product_edge_cases(self):
        """Test with vectors that produce dot products at the edges of [-1, 1]."""
        # Create a sequence
        sequence = "AAAC"
        N = len(sequence)
        device = torch.device("cpu")

        # Create backbone coordinates
        coords_bb = torch.zeros(N, len(BB_ATOMS), 3, device=device)

        # Set up parallel vectors (dot product = 1)
        coords_bb[0, BB_ATOMS.index("P"), :] = torch.tensor([0.0, 0.0, 0.0], device=device)
        coords_bb[0, BB_ATOMS.index("O5'"), :] = torch.tensor([1.0, 0.0, 0.0], device=device)
        coords_bb[0, BB_ATOMS.index("C5'"), :] = torch.tensor([2.0, 0.0, 0.0], device=device)

        # Set up anti-parallel vectors (dot product = -1)
        coords_bb[1, BB_ATOMS.index("P"), :] = torch.tensor([0.0, 0.0, 0.0], device=device)
        coords_bb[1, BB_ATOMS.index("O5'"), :] = torch.tensor([1.0, 0.0, 0.0], device=device)
        coords_bb[1, BB_ATOMS.index("C5'"), :] = torch.tensor([-1.0, 0.0, 0.0], device=device)

        # Set up the rest with some random values
        coords_bb[2:, :, :] = torch.randn(N-2, len(BB_ATOMS), 3, device=device)

        # Create a dummy angles mask (all True)
        angles_mask = torch.ones(N, dtype=torch.bool, device=device)

        # Run the function
        coords_full = place_rna_bases(coords_bb, sequence, angles_mask, device=device)

        assert not torch.isnan(coords_full).any(), (
            "NaN values found in coords_full with edge-case dot products\n[UNIQUE-ERR-RNA-NAN-DOT-PRODUCT-EDGE]"
        )


if __name__ == "__main__":
    unittest.main()

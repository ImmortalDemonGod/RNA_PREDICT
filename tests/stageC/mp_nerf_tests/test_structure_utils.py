# tests/stageC/mp_nerf_tests/test_structure_utils.py
import pytest
import torch
from rna_predict.pipeline.stageC.mp_nerf.protein_utils.structure_utils import (
    modify_angles_mask_with_torsions,
)


def test_modify_angles_mask_with_torsions_basic():
    """
    Tests if modify_angles_mask_with_torsions correctly updates the
    torsion angle part of the angles_mask based on the input torsions tensor.

    Covers lines 428-438 in structure_utils.py.
    """
    seq_len = 5
    num_atoms = 14
    dtype = torch.float32

    # Create a dummy angles_mask (2, L, 14) - index 0 is bond angles, index 1 is torsions
    # Initialize with easily identifiable values (e.g., all zeros)
    angles_mask = torch.zeros((2, seq_len, num_atoms), dtype=dtype)

    # Create a dummy torsions tensor (L, 3) - phi, psi, omega
    # Use distinct values to easily track assignments
    torsions = torch.arange(seq_len * 3, dtype=dtype).reshape(seq_len, 3)
    # Example:
    # tensor([[ 0.,  1.,  2.],  # phi[0], psi[0], omega[0]
    #         [ 3.,  4.,  5.],  # phi[1], psi[1], omega[1]
    #         [ 6.,  7.,  8.],  # phi[2], psi[2], omega[2]
    #         [ 9., 10., 11.],  # phi[3], psi[3], omega[3]
    #         [12., 13., 14.]]) # phi[4], psi[4], omega[4]

    # Keep a copy of the original mask to check it wasn't modified in place
    original_angles_mask = angles_mask.clone() # Line 428 implicitly tested by checking original later

    # Call the function under test
    modified_mask = modify_angles_mask_with_torsions(angles_mask, torsions) # Lines 432, 434, 436 executed

    # --- Assertions ---

    # 1. Check that the original angles_mask was not modified
    assert torch.equal(angles_mask, original_angles_mask), "Original angles_mask should not be modified."

    # 2. Check that the bond angle part (index 0) remains unchanged
    assert torch.equal(modified_mask[0], original_angles_mask[0]), "Bond angles (index 0) should not be modified."

    # 3. Check the specific torsion angle updates
    # N determined by previous psi: modified_mask[1, :-1, 0] = torsions[:-1, 1]
    expected_psi = torsions[:-1, 1] # psi[0] to psi[3] -> [1., 4., 7., 10.]
    assert torch.equal(modified_mask[1, :-1, 0], expected_psi), "Psi angles update failed." # Line 432 check

    # CA determined by omega: modified_mask[1, 1:, 1] = torsions[1:, 2]
    expected_omega = torsions[1:, 2] # omega[1] to omega[4] -> [5., 8., 11., 14.]
    assert torch.equal(modified_mask[1, 1:, 1], expected_omega), "Omega angles update failed." # Line 434 check

    # C determined by phi: modified_mask[1, 1:, 2] = torsions[1:, 0]
    expected_phi = torsions[1:, 0] # phi[1] to phi[4] -> [3., 6., 9., 12.]
    assert torch.equal(modified_mask[1, 1:, 2], expected_phi), "Phi angles update failed." # Line 436 check

    # 4. Check that other torsion angles (columns 3 onwards) remain unchanged (should still be 0)
    assert torch.all(modified_mask[1, :, 3:] == 0), "Other torsion angles should remain unchanged."

    # 5. Check boundary elements that were *not* updated by the slicing remain unchanged
    # modified_mask[1, -1, 0] (last row, col 0) should be original value (0)
    assert modified_mask[1, -1, 0] == original_angles_mask[1, -1, 0], "Boundary element [1, -1, 0] changed unexpectedly."
    # modified_mask[1, 0, 1] (first row, col 1) should be original value (0)
    assert modified_mask[1, 0, 1] == original_angles_mask[1, 0, 1], "Boundary element [1, 0, 1] changed unexpectedly."
    # modified_mask[1, 0, 2] (first row, col 2) should be original value (0)
    assert modified_mask[1, 0, 2] == original_angles_mask[1, 0, 2], "Boundary element [1, 0, 2] changed unexpectedly."

    # 6. Check the shape of the output
    assert modified_mask.shape == (2, seq_len, num_atoms), "Output shape is incorrect." # Line 438 implicitly tested by returning
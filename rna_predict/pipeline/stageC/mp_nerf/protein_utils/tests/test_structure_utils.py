import torch
import pytest
import numpy as np
from rna_predict.pipeline.stageC.mp_nerf.protein_utils.structure_utils import modify_angles_mask_with_torsions

def test_modify_angles_mask_with_torsions_empty_sequence():
    """Test modify_angles_mask_with_torsions with empty sequence."""
    # Create empty tensors
    angles_mask = torch.zeros((2, 0, 3))  # (2, L, 3) where L=0
    torsions = torch.zeros((0, 3))  # (L, 3) where L=0
    
    # Call function
    result = modify_angles_mask_with_torsions(angles_mask, torsions)
    
    # Check result
    assert torch.equal(result, angles_mask)
    assert result.shape == (2, 0, 3)

def test_modify_angles_mask_with_torsions_single_residue():
    """Test modify_angles_mask_with_torsions with single residue."""
    # Create tensors for single residue
    angles_mask = torch.zeros((2, 1, 3))
    torsions = torch.tensor([[1.0, 2.0, 3.0]])  # Single residue with phi, psi, omega
    
    # Call function
    result = modify_angles_mask_with_torsions(angles_mask, torsions)
    
    # For single residue, no updates should occur as we need at least 2 residues
    # to update torsion angles
    assert torch.equal(result, angles_mask)

def test_modify_angles_mask_with_torsions_two_residues():
    """Test modify_angles_mask_with_torsions with two residues."""
    # Create tensors for two residues
    angles_mask = torch.zeros((2, 2, 3))
    torsions = torch.tensor([
        [1.0, 2.0, 3.0],  # First residue: phi, psi, omega
        [4.0, 5.0, 6.0]   # Second residue: phi, psi, omega
    ])
    
    # Call function
    result = modify_angles_mask_with_torsions(angles_mask, torsions)
    
    # Check updates
    # N determined by previous psi
    assert result[1, 0, 0] == 2.0  # psi of first residue
    # CA determined by omega
    assert result[1, 1, 1] == 6.0  # omega of second residue
    # C determined by phi
    assert result[1, 1, 2] == 4.0  # phi of second residue

def test_modify_angles_mask_with_torsions_multiple_residues():
    """Test modify_angles_mask_with_torsions with multiple residues."""
    # Create tensors for three residues
    angles_mask = torch.zeros((2, 3, 3))
    torsions = torch.tensor([
        [1.0, 2.0, 3.0],  # First residue: phi, psi, omega
        [4.0, 5.0, 6.0],  # Second residue: phi, psi, omega
        [7.0, 8.0, 9.0]   # Third residue: phi, psi, omega
    ])
    
    # Call function
    result = modify_angles_mask_with_torsions(angles_mask, torsions)
    
    # Check updates for all residues
    # N determined by previous psi
    assert result[1, 0, 0] == 2.0  # psi of first residue
    assert result[1, 1, 0] == 5.0  # psi of second residue
    # CA determined by omega
    assert result[1, 1, 1] == 6.0  # omega of second residue
    assert result[1, 2, 1] == 9.0  # omega of third residue
    # C determined by phi
    assert result[1, 1, 2] == 4.0  # phi of second residue
    assert result[1, 2, 2] == 7.0  # phi of third residue

def test_modify_angles_mask_with_torsions_preserves_original_mask():
    """Test that modify_angles_mask_with_torsions preserves the original mask values where no updates occur."""
    # Create tensors with non-zero initial values
    angles_mask = torch.ones((2, 3, 3)) * 0.5  # Fill with 0.5
    torsions = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    
    # Call function
    result = modify_angles_mask_with_torsions(angles_mask, torsions)
    
    # Check that non-updated positions retain original value
    assert result[0, 0, 0] == 0.5  # First dimension unchanged
    assert result[1, 0, 1] == 0.5  # CA of first residue unchanged
    assert result[1, 0, 2] == 0.5  # C of first residue unchanged
    assert result[1, 2, 0] == 0.5  # N of last residue unchanged 
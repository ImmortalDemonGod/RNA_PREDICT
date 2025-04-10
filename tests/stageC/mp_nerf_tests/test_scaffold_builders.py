"""
Tests for scaffold_builders.py module.
"""

import pytest
import torch
import numpy as np

from rna_predict.pipeline.stageC.mp_nerf.protein_utils.scaffold_builders import (
    build_scaffolds_from_scn_angles,
    modify_scaffolds_with_coords,
    modify_angles_mask_with_torsions,
    _create_cloud_and_bond_masks,
    _create_point_ref_mask,
    _create_angles_mask,
)


@pytest.fixture
def cpu_device():
    """Fixture to provide CPU device."""
    return torch.device("cpu")


@pytest.fixture
def single_residue_data(cpu_device):
    """Fixture to provide data for a single residue."""
    seq = "A"  # Alanine
    angles = torch.zeros((1, 12), device=cpu_device)
    coords = torch.ones((1, 14, 3), device=cpu_device)
    return {"seq": seq, "angles": angles, "coords": coords, "device": cpu_device}


@pytest.fixture
def two_residue_data(cpu_device):
    """Fixture to provide data for two residues."""
    seq = "AC"  # Alanine, Cysteine
    angles = torch.zeros((2, 12), device=cpu_device)
    coords = torch.ones((2, 14, 3), device=cpu_device)
    # Make the second residue's coordinates different to test angle calculations
    coords[1] = coords[1] * 2.0
    return {"seq": seq, "angles": angles, "coords": coords, "device": cpu_device}


@pytest.fixture
def empty_seq_data(cpu_device):
    """Fixture to provide data for an empty sequence."""
    seq = ""
    angles = torch.zeros((0, 12), device=cpu_device)
    coords = torch.zeros((0, 14, 3), device=cpu_device)
    return {"seq": seq, "angles": angles, "coords": coords, "device": cpu_device}


# --- Tests for helper functions ---

def test_create_cloud_and_bond_masks(single_residue_data):
    """Test _create_cloud_and_bond_masks function."""
    data = single_residue_data
    cloud_mask, bond_mask = _create_cloud_and_bond_masks(
        data["seq"], data["device"], data["angles"].dtype
    )
    
    assert isinstance(cloud_mask, torch.Tensor)
    assert isinstance(bond_mask, torch.Tensor)
    assert cloud_mask.shape[1] == 14  # 14 atoms
    assert bond_mask.shape[1] == 14  # 14 atoms
    assert cloud_mask.dtype == torch.bool
    assert bond_mask.dtype == data["angles"].dtype


def test_create_point_ref_mask(single_residue_data):
    """Test _create_point_ref_mask function."""
    data = single_residue_data
    point_ref_mask = _create_point_ref_mask(
        data["seq"], len(data["seq"]), data["device"]
    )
    
    assert isinstance(point_ref_mask, torch.Tensor)
    assert point_ref_mask.shape == (3, 1, 11)  # 3 coordinates, 1 residue, 11 atoms
    assert point_ref_mask.dtype == torch.long


def test_create_point_ref_mask_empty_seq(empty_seq_data):
    """Test _create_point_ref_mask function with empty sequence."""
    data = empty_seq_data
    point_ref_mask = _create_point_ref_mask(
        data["seq"], len(data["seq"]), data["device"]
    )
    
    assert isinstance(point_ref_mask, torch.Tensor)
    assert point_ref_mask.shape == (3, 0, 11)  # 3 coordinates, 0 residues, 11 atoms
    assert point_ref_mask.dtype == torch.long


def test_create_angles_mask(single_residue_data):
    """Test _create_angles_mask function."""
    data = single_residue_data
    angles_mask = _create_angles_mask(
        data["seq"], len(data["seq"]), data["device"], data["angles"].dtype
    )
    
    assert isinstance(angles_mask, torch.Tensor)
    assert angles_mask.shape == (2, 1, 14)  # 2 types (theta, torsion), 1 residue, 14 atoms
    assert angles_mask.dtype == data["angles"].dtype


def test_create_angles_mask_empty_seq(empty_seq_data):
    """Test _create_angles_mask function with empty sequence."""
    data = empty_seq_data
    angles_mask = _create_angles_mask(
        data["seq"], len(data["seq"]), data["device"], data["angles"].dtype
    )
    
    assert isinstance(angles_mask, torch.Tensor)
    assert angles_mask.shape == (2, 0, 14)  # 2 types (theta, torsion), 0 residues, 14 atoms
    assert angles_mask.dtype == data["angles"].dtype


# --- Tests for modify_scaffolds_with_coords ---

def test_modify_scaffolds_with_coords_single_residue(single_residue_data):
    """Test modify_scaffolds_with_coords with a single residue."""
    data = single_residue_data
    
    # Build scaffolds
    scaffolds = build_scaffolds_from_scn_angles(
        data["seq"], data["angles"], device=data["device"]
    )
    
    # Make a copy of original scaffolds for comparison
    original_bond_mask = scaffolds["bond_mask"].clone()
    original_angles_mask = scaffolds["angles_mask"].clone()
    
    # Modify scaffolds with coordinates
    modified_scaffolds = modify_scaffolds_with_coords(scaffolds, data["coords"])
    
    # Check that modified_scaffolds has the same keys as scaffolds
    assert set(modified_scaffolds.keys()) == set(scaffolds.keys())
    
    # Check that bond_mask and angles_mask have been modified
    assert not torch.equal(modified_scaffolds["bond_mask"], original_bond_mask)
    assert not torch.equal(modified_scaffolds["angles_mask"], original_angles_mask)
    
    # Check that N-CA-C angle is set (line 222-224)
    n_ca_c_angle = modified_scaffolds["angles_mask"][0, 0, 2]
    assert n_ca_c_angle is not None
    assert not torch.isnan(n_ca_c_angle)


def test_modify_scaffolds_with_coords_two_residues(two_residue_data):
    """Test modify_scaffolds_with_coords with two residues."""
    data = two_residue_data
    
    # Build scaffolds
    scaffolds = build_scaffolds_from_scn_angles(
        data["seq"], data["angles"], device=data["device"]
    )
    
    # Make a copy of original scaffolds for comparison
    original_bond_mask = scaffolds["bond_mask"].clone()
    original_angles_mask = scaffolds["angles_mask"].clone()
    
    # Modify scaffolds with coordinates
    modified_scaffolds = modify_scaffolds_with_coords(scaffolds, data["coords"])
    
    # Check that modified_scaffolds has the same keys as scaffolds
    assert set(modified_scaffolds.keys()) == set(scaffolds.keys())
    
    # Check that bond_mask and angles_mask have been modified
    assert not torch.equal(modified_scaffolds["bond_mask"], original_bond_mask)
    assert not torch.equal(modified_scaffolds["angles_mask"], original_angles_mask)
    
    # Check that backbone angles and dihedrals are set (lines 202-217)
    # ca_c_n angle
    ca_c_n_angle = modified_scaffolds["angles_mask"][0, 0, 0]
    assert ca_c_n_angle is not None
    assert not torch.isnan(ca_c_n_angle)
    
    # c_n_ca angle
    c_n_ca_angle = modified_scaffolds["angles_mask"][0, 1, 1]
    assert c_n_ca_angle is not None
    assert not torch.isnan(c_n_ca_angle)
    
    # N dihedral
    n_dihedral = modified_scaffolds["angles_mask"][1, 0, 0]
    assert n_dihedral is not None
    assert not torch.isnan(n_dihedral)
    
    # CA dihedral
    ca_dihedral = modified_scaffolds["angles_mask"][1, 1, 1]
    assert ca_dihedral is not None
    assert not torch.isnan(ca_dihedral)
    
    # C dihedral
    c_dihedral = modified_scaffolds["angles_mask"][1, 1, 2]
    assert c_dihedral is not None
    assert not torch.isnan(c_dihedral)


# --- Tests for modify_angles_mask_with_torsions ---

def test_modify_angles_mask_with_torsions_empty_seq(empty_seq_data):
    """Test modify_angles_mask_with_torsions with empty sequence."""
    data = empty_seq_data
    
    # Create empty angles_mask and torsions
    angles_mask = torch.zeros((2, 0, 14), device=data["device"])
    torsions = torch.zeros((0, 3), device=data["device"])
    
    # Modify angles_mask with torsions
    modified_mask = modify_angles_mask_with_torsions(angles_mask, torsions)
    
    # Check that modified_mask is the same as angles_mask for empty sequence
    assert torch.equal(modified_mask, angles_mask)
    assert modified_mask.shape == (2, 0, 14)


def test_modify_angles_mask_with_torsions_single_residue(single_residue_data):
    """Test modify_angles_mask_with_torsions with a single residue."""
    data = single_residue_data
    
    # Create angles_mask and torsions
    angles_mask = torch.zeros((2, 1, 14), device=data["device"])
    torsions = torch.ones((1, 3), device=data["device"]) * np.pi/2  # 90 degrees
    
    # Modify angles_mask with torsions
    modified_mask = modify_angles_mask_with_torsions(angles_mask, torsions)
    
    # Check that modified_mask has the same shape as angles_mask
    assert modified_mask.shape == angles_mask.shape
    
    # For a single residue, no backbone torsions should be modified
    # because they depend on previous/next residues
    assert torch.equal(modified_mask, angles_mask)


def test_modify_angles_mask_with_torsions_two_residues(two_residue_data):
    """Test modify_angles_mask_with_torsions with two residues."""
    data = two_residue_data
    
    # Create angles_mask and torsions
    angles_mask = torch.zeros((2, 2, 14), device=data["device"])
    torsions = torch.ones((2, 3), device=data["device"]) * np.pi/2  # 90 degrees
    
    # Modify angles_mask with torsions
    modified_mask = modify_angles_mask_with_torsions(angles_mask, torsions)
    
    # Check that modified_mask has the same shape as angles_mask
    assert modified_mask.shape == angles_mask.shape
    
    # Check that backbone torsions have been modified
    # N determined by previous psi: angles_mask[1, i, 0] = psi(i)
    assert torch.isclose(modified_mask[1, 0, 0], torsions[0, 1])
    
    # CA determined by omega: angles_mask[1, i, 1] = omega(i)
    assert torch.isclose(modified_mask[1, 1, 1], torsions[1, 2])
    
    # C determined by phi: angles_mask[1, i, 2] = phi(i)
    assert torch.isclose(modified_mask[1, 1, 2], torsions[1, 0])

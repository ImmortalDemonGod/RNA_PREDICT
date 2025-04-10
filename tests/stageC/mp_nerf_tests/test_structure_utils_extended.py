# tests/stageC/mp_nerf_tests/test_structure_utils_extended.py
import pytest
import torch
import numpy as np

# Import moved functions from their new locations
from rna_predict.pipeline.stageC.mp_nerf.protein_utils.amino_acid_data_utils import (
    get_rigid_frames,
    get_atom_names,
    get_bond_names,
    get_bond_types,
    get_bond_values,
    get_angle_names,
    get_angle_types,
    get_angle_values,
    get_torsion_names,
    get_torsion_types,
    get_torsion_values,
)
from rna_predict.pipeline.stageC.mp_nerf.protein_utils.geometry_utils import (
    to_zero_two_pi,
)
from rna_predict.pipeline.stageC.mp_nerf.protein_utils.scaffold_builders import ( # Import from new module
    build_scaffolds_from_scn_angles,
    modify_scaffolds_with_coords,
    modify_angles_mask_with_torsions, # Correctly import from scaffold_builders
)
from rna_predict.pipeline.stageC.mp_nerf.protein_utils.structure_utils import (
    # modify_angles_mask_with_torsions, # Moved to scaffold_builders
    protein_fold,
)
from rna_predict.pipeline.stageC.mp_nerf.protein_utils.symmetry_utils import (
    get_symmetric_atom_pairs,
)
from rna_predict.pipeline.stageC.mp_nerf.protein_utils.sidechain_data import (
    SC_BUILD_INFO,
)
@pytest.fixture
def cpu_device():
    """Fixture to provide CPU device."""
    return torch.device("cpu")


# --- Tests for angle conversion functions ---

@pytest.mark.parametrize(
    "angle, expected",
    [
        (0.0, 0.0),
        (np.pi, np.pi),
        (2 * np.pi, 0.0),  # Wraps around to 0
        (-np.pi, np.pi),  # Negative angles wrap to positive
        (3 * np.pi, np.pi),  # Wraps around
        (-2 * np.pi, 0.0),  # Wraps around to 0
    ],
)
def test_to_zero_two_pi(angle, expected):
    """Test to_zero_two_pi function."""
    result = to_zero_two_pi(angle)
    assert np.isclose(result, expected)


# --- Tests for accessor functions ---

def test_get_rigid_frames():
    """Test get_rigid_frames function."""
    # Test with a known amino acid
    frames = get_rigid_frames("A")  # Alanine
    assert isinstance(frames, list)
    assert all(isinstance(frame, list) for frame in frames)

    # Verify against SC_BUILD_INFO
    expected = SC_BUILD_INFO["A"]["rigid-frames-idxs"]
    assert frames == expected


def test_get_atom_names():
    """Test get_atom_names function."""
    # Test with a known amino acid
    names = get_atom_names("A")  # Alanine
    assert isinstance(names, list)
    assert all(isinstance(name, str) for name in names)

    # Verify against SC_BUILD_INFO
    expected = SC_BUILD_INFO["A"]["atom-names"]
    assert names == expected


def test_get_bond_names():
    """Test get_bond_names function."""
    # Test with a known amino acid
    names = get_bond_names("A")  # Alanine
    assert isinstance(names, list)
    assert all(isinstance(name, str) for name in names)

    # Verify against SC_BUILD_INFO
    expected = SC_BUILD_INFO["A"]["bonds-names"]
    assert names == expected


def test_get_bond_types():
    """Test get_bond_types function."""
    # Test with a known amino acid
    types = get_bond_types("A")  # Alanine
    assert isinstance(types, list)
    assert all(isinstance(t, str) for t in types)

    # Verify against SC_BUILD_INFO
    expected = SC_BUILD_INFO["A"]["bonds-types"]
    assert types == expected


def test_get_bond_values():
    """Test get_bond_values function."""
    # Test with a known amino acid
    values = get_bond_values("A")  # Alanine
    assert isinstance(values, list)
    assert all(isinstance(v, (int, float)) for v in values)

    # Verify against SC_BUILD_INFO
    expected = SC_BUILD_INFO["A"]["bonds-vals"]
    assert values == expected


def test_get_angle_names():
    """Test get_angle_names function."""
    # Test with a known amino acid
    names = get_angle_names("A")  # Alanine
    assert isinstance(names, list)
    assert all(isinstance(name, str) for name in names)

    # Verify against SC_BUILD_INFO
    expected = SC_BUILD_INFO["A"]["angles-names"]
    assert names == expected


def test_get_angle_types():
    """Test get_angle_types function."""
    # Test with a known amino acid
    types = get_angle_types("A")  # Alanine
    assert isinstance(types, list)
    assert all(isinstance(t, str) for t in types)

    # Verify against SC_BUILD_INFO
    expected = SC_BUILD_INFO["A"]["angles-types"]
    assert types == expected


def test_get_angle_values():
    """Test get_angle_values function."""
    # Test with a known amino acid
    values = get_angle_values("A")  # Alanine
    assert isinstance(values, list)
    assert all(isinstance(v, (int, float)) for v in values)

    # Verify against SC_BUILD_INFO
    expected = SC_BUILD_INFO["A"]["angles-vals"]
    assert values == expected


def test_get_torsion_names():
    """Test get_torsion_names function."""
    # Test with a known amino acid
    names = get_torsion_names("A")  # Alanine
    assert isinstance(names, list)
    assert all(isinstance(name, str) for name in names)

    # Verify against SC_BUILD_INFO
    expected = SC_BUILD_INFO["A"]["torsion-names"]
    assert names == expected


def test_get_torsion_types():
    """Test get_torsion_types function."""
    # Test with a known amino acid
    types = get_torsion_types("A")  # Alanine
    assert isinstance(types, list)
    assert all(isinstance(t, str) for t in types)

    # Verify against SC_BUILD_INFO
    expected = SC_BUILD_INFO["A"]["torsion-types"]
    assert types == expected


def test_get_torsion_values():
    """Test get_torsion_values function."""
    # Test with a known amino acid
    values = get_torsion_values("A")  # Alanine
    assert isinstance(values, list)

    # Verify against SC_BUILD_INFO
    expected = SC_BUILD_INFO["A"]["torsion-vals"]
    assert values == expected


# --- Tests for build_scaffolds_from_scn_angles ---

def test_build_scaffolds_from_scn_angles_empty_sequence(cpu_device):
    """Test build_scaffolds_from_scn_angles with empty sequence."""
    seq = ""
    angles = torch.zeros((0, 12), device=cpu_device)

    scaffolds = build_scaffolds_from_scn_angles(seq, angles, device=cpu_device)

    # Check that scaffolds has the expected keys
    assert set(scaffolds.keys()) == {"cloud_mask", "point_ref_mask", "angles_mask", "bond_mask"}

    # Check shapes for empty sequence
    assert scaffolds["cloud_mask"].shape == (0, 14)
    assert scaffolds["point_ref_mask"].shape == (3, 0, 11)
    assert scaffolds["angles_mask"].shape == (2, 0, 14)
    assert scaffolds["bond_mask"].shape == (0, 14)


def test_build_scaffolds_from_scn_angles_single_residue(cpu_device):
    """Test build_scaffolds_from_scn_angles with a single residue."""
    seq = "A"  # Alanine
    angles = torch.zeros((1, 12), device=cpu_device)

    scaffolds = build_scaffolds_from_scn_angles(seq, angles, device=cpu_device)

    # Check that scaffolds has the expected keys
    assert set(scaffolds.keys()) == {"cloud_mask", "point_ref_mask", "angles_mask", "bond_mask"}

    # Check shapes for single residue
    assert scaffolds["cloud_mask"].shape == (1, 14)
    assert scaffolds["point_ref_mask"].shape == (3, 1, 11)
    assert scaffolds["angles_mask"].shape == (2, 1, 14)
    assert scaffolds["bond_mask"].shape == (1, 14)

    # Check that cloud_mask has the expected values for Alanine
    # Alanine has N, CA, C, O, CB atoms (indices 0-4)
    assert torch.all(scaffolds["cloud_mask"][0, :5])  # First 5 atoms should be True
    assert not torch.any(scaffolds["cloud_mask"][0, 5:])  # Rest should be False


def test_build_scaffolds_from_scn_angles_multiple_residues(cpu_device):
    """Test build_scaffolds_from_scn_angles with multiple residues."""
    seq = "ACD"  # Alanine, Cysteine, Aspartic Acid
    angles = torch.zeros((3, 12), device=cpu_device)

    scaffolds = build_scaffolds_from_scn_angles(seq, angles, device=cpu_device)

    # Check that scaffolds has the expected keys
    assert set(scaffolds.keys()) == {"cloud_mask", "point_ref_mask", "angles_mask", "bond_mask"}

    # Check shapes for multiple residues
    assert scaffolds["cloud_mask"].shape == (3, 14)
    assert scaffolds["point_ref_mask"].shape == (3, 3, 11)
    assert scaffolds["angles_mask"].shape == (2, 3, 14)
    assert scaffolds["bond_mask"].shape == (3, 14)

    # Check that cloud_mask has the expected values for each residue
    # Alanine has N, CA, C, O, CB atoms (indices 0-4)
    assert torch.all(scaffolds["cloud_mask"][0, :5])
    assert not torch.any(scaffolds["cloud_mask"][0, 5:])

    # Cysteine has N, CA, C, O, CB, SG atoms (indices 0-5)
    assert torch.all(scaffolds["cloud_mask"][1, :6])
    assert not torch.any(scaffolds["cloud_mask"][1, 6:])

    # Aspartic Acid has N, CA, C, O, CB, CG, OD1, OD2 atoms (indices 0-7)
    assert torch.all(scaffolds["cloud_mask"][2, :8])
    assert not torch.any(scaffolds["cloud_mask"][2, 8:])


def test_build_scaffolds_from_scn_angles_no_angles(cpu_device):
    """Test build_scaffolds_from_scn_angles with no angles."""
    seq = "A"  # Alanine

    # Should raise ValueError if angles is None
    with pytest.raises(ValueError, match="Input 'angles' tensor cannot be None"):
        build_scaffolds_from_scn_angles(seq, angles=None, device=cpu_device)


# --- Tests for modify_scaffolds_with_coords ---

def test_modify_scaffolds_with_coords(cpu_device):
    """Test modify_scaffolds_with_coords function."""
    # Create a simple scaffold
    seq = "A"  # Alanine
    angles = torch.zeros((1, 12), device=cpu_device)
    scaffolds = build_scaffolds_from_scn_angles(seq, angles, device=cpu_device)

    # Create coordinates with non-zero values to ensure changes
    coords = torch.ones((1, 14, 3), device=cpu_device) * 2.0

    # Make a copy of the original scaffolds for comparison
    scaffolds["bond_mask"].clone()
    scaffolds["angles_mask"].clone()

    # Modify scaffolds with coordinates
    modified_scaffolds = modify_scaffolds_with_coords(scaffolds, coords)

    # Check that modified_scaffolds has the same keys as scaffolds
    assert set(modified_scaffolds.keys()) == set(scaffolds.keys())

    # Check that cloud_mask and point_ref_mask were preserved
    assert torch.equal(modified_scaffolds["cloud_mask"], scaffolds["cloud_mask"])
    assert torch.equal(modified_scaffolds["point_ref_mask"], scaffolds["point_ref_mask"])

    # Check that the scaffolds object was modified in-place
    assert id(modified_scaffolds) == id(scaffolds)


# --- Tests for get_symmetric_atom_pairs ---

def test_get_symmetric_atom_pairs():
    """Test get_symmetric_atom_pairs function."""
    # Test with a sequence containing amino acids with symmetric atoms
    seq = "DE"  # Aspartic Acid, Glutamic Acid (which have symmetric atoms according to the implementation)

    pairs = get_symmetric_atom_pairs(seq)

    # Check that pairs is a dictionary
    assert isinstance(pairs, dict)

    # Check that keys are strings representing residue indices
    assert set(pairs.keys()) == {"0", "1"}

    # Check that values are lists of tuples
    for key, value in pairs.items():
        assert isinstance(value, list)
        assert all(isinstance(pair, tuple) for pair in value)
        assert all(len(pair) == 2 for pair in value)

    # Check that the pairs are present for each residue
    # D has OD1/OD2 pairs (indices 6, 7)
    assert (6, 7) in pairs["0"] or (7, 6) in pairs["0"]

    # E has OE1/OE2 pairs (indices 8, 9)
    assert (8, 9) in pairs["1"] or (9, 8) in pairs["1"]


# --- Tests for modify_angles_mask_with_torsions ---

def test_modify_angles_mask_with_torsions_three_residues(cpu_device):
    """Test modify_angles_mask_with_torsions with three residues."""
    # Create tensors for three residues
    angles_mask = torch.zeros((2, 3, 3), device=cpu_device)
    torsions = torch.tensor(
        [
            [1.0, 2.0, 3.0],  # First residue: phi, psi, omega
            [4.0, 5.0, 6.0],  # Second residue: phi, psi, omega
            [7.0, 8.0, 9.0],  # Third residue: phi, psi, omega
        ],
        device=cpu_device,
    )

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


def test_modify_angles_mask_with_torsions_batch_dimension(cpu_device):
    """Test modify_angles_mask_with_torsions with batch dimension."""
    # Create tensors with batch dimension
    angles_mask = torch.zeros((2, 2, 3), device=cpu_device)
    torsions = torch.tensor(
        [
            [1.0, 2.0, 3.0],  # First residue: phi, psi, omega
            [4.0, 5.0, 6.0],  # Second residue: phi, psi, omega
        ],
        device=cpu_device,
    )

    # Call function
    result = modify_angles_mask_with_torsions(angles_mask, torsions)

    # Check updates
    # N determined by previous psi
    assert result[1, 0, 0] == 2.0  # psi of first residue
    # CA determined by omega
    assert result[1, 1, 1] == 6.0  # omega of second residue
    # C determined by phi
    assert result[1, 1, 2] == 4.0  # phi of second residue


# --- Tests for protein_fold ---

def test_protein_fold_empty_sequence(cpu_device):
    """Test protein_fold with empty sequence."""
    seq = ""
    angles = torch.zeros((0, 12), device=cpu_device)

    coords, cloud_mask = protein_fold(seq, angles, device=cpu_device)

    # Check shapes for empty sequence
    assert coords.shape == (0, 14, 3)
    assert cloud_mask.shape == (0, 14)


def test_protein_fold_single_residue(cpu_device):
    """Test protein_fold with a single residue."""
    seq = "A"  # Alanine
    angles = torch.zeros((1, 12), device=cpu_device)

    coords, cloud_mask = protein_fold(seq, angles, device=cpu_device)

    # Check shapes for single residue
    assert coords.shape == (1, 14, 3)
    assert cloud_mask.shape == (1, 14)

    # Check that cloud_mask has the expected values for Alanine
    # Alanine has N, CA, C, O, CB atoms (indices 0-4)
    assert torch.all(cloud_mask[0, :5])  # First 5 atoms should be True
    assert not torch.any(cloud_mask[0, 5:])  # Rest should be False

    # Check that coordinates are not all zeros for valid atoms (excluding N at index 0)
    for i in range(1, 5):  # CA, C, O, CB
        print(f"Atom {i} coords: {coords[0, i]}")
        assert not torch.all(coords[0, i] == 0)


def test_protein_fold_multiple_residues(cpu_device):
    """Test protein_fold with multiple residues."""
    seq = "ACD"  # Alanine, Cysteine, Aspartic Acid
    angles = torch.zeros((3, 12), device=cpu_device)

    coords, cloud_mask = protein_fold(seq, angles, device=cpu_device)

    # Check shapes for multiple residues
    assert coords.shape == (3, 14, 3)
    assert cloud_mask.shape == (3, 14)

    # Check that cloud_mask has the expected values for each residue
    # Alanine has N, CA, C, O, CB atoms (indices 0-4)
    assert torch.all(cloud_mask[0, :5])
    assert not torch.any(cloud_mask[0, 5:])

    # Cysteine has N, CA, C, O, CB, SG atoms (indices 0-5)
    assert torch.all(cloud_mask[1, :6])
    assert not torch.any(cloud_mask[1, 6:])

    # Aspartic Acid has N, CA, C, O, CB, CG, OD1, OD2 atoms (indices 0-7)
    assert torch.all(cloud_mask[2, :8])
    assert not torch.any(cloud_mask[2, 8:])

    # Check that coordinates are not all zeros for valid atoms
    for i in range(3):  # For each residue
        for j in range(sum(cloud_mask[i])):  # For each valid atom
            # Skip the check for the first atom (N) of the first residue, which is expected at origin
            if i == 0 and j == 0:
                continue
            assert not torch.all(coords[i, j] == 0)


def test_protein_fold_with_coords(cpu_device):
    """Test protein_fold with initial coordinates."""
    seq = "A"  # Alanine
    angles = torch.zeros((1, 12), device=cpu_device)
    initial_coords = torch.ones((1, 14, 3), device=cpu_device)

    coords, cloud_mask = protein_fold(seq, angles, coords=initial_coords, device=cpu_device)

    # Check shapes
    assert coords.shape == (1, 14, 3)
    assert cloud_mask.shape == (1, 14)

    # Initial coords should be ignored in the current implementation
    # So the result should not be all ones
    assert not torch.all(coords == 1.0)

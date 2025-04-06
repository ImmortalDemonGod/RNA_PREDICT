# tests/stageC/mp_nerf_tests/test_structure_utils.py
import pytest
import torch
import numpy as np
from rna_predict.pipeline.stageC.mp_nerf.protein_utils.structure_utils import (
    modify_angles_mask_with_torsions,
    to_zero_two_pi,
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
    get_symmetric_atom_pairs,
    build_scaffolds_from_scn_angles,
    modify_scaffolds_with_coords,
    protein_fold,
)
# Import mask generators locally like the source file to avoid potential circular deps
# Note: Ideally, these mask generators should have their own dedicated tests.
from rna_predict.pipeline.stageC.mp_nerf.protein_utils.mask_generators import (
    make_bond_mask,
    make_cloud_mask,
    make_idx_mask,
    make_theta_mask,
    make_torsion_mask,
)
from rna_predict.pipeline.stageC.mp_nerf.protein_utils.sidechain_data import SC_BUILD_INFO, BB_BUILD_INFO
# Import angle/dihedral functions used in modify_scaffolds_with_coords and protein_fold
from rna_predict.pipeline.stageC.mp_nerf.utils import get_angle, get_dihedral


# --- Fixtures ---
@pytest.fixture
def cpu_device():
    """Provides the CPU device."""
    return torch.device("cpu")

@pytest.fixture
def default_angles_coords(cpu_device):
    """Provides default angles and coordinates for testing."""
    seq_len = 4
    seq = "ARND" # Sample sequence
    angles = torch.randn(seq_len, 12, dtype=torch.float32).to(cpu_device) # (L, 12)
    coords = torch.randn(seq_len, 14, 3, dtype=torch.float32).to(cpu_device) # (L, 14, 3)
    # Ensure coords are not NaN where cloud_mask would be True
    cloud_mask = make_cloud_mask(seq, coords=coords).bool().to(cpu_device)
    coords[~cloud_mask] = 0 # Zero out masked coords for simplicity in tests

    return {"seq": seq, "angles": angles, "coords": coords, "seq_len": seq_len}

@pytest.fixture
def sample_scaffolds(default_angles_coords, cpu_device):
    """Provides sample scaffolds generated from default angles."""
    data = default_angles_coords
    # Use coords=None initially to get scaffolds based on standard geometry + angles
    scaffolds = build_scaffolds_from_scn_angles(
        data["seq"], angles=data["angles"], coords=None, device=cpu_device
    )
    # Add cloud_mask based on actual coords for modify/fold tests
    scaffolds["cloud_mask"] = make_cloud_mask(data["seq"], coords=data["coords"]).bool().to(cpu_device)
    return scaffolds


# --- Existing Test ---

def test_modify_angles_mask_with_torsions_basic():
    """
    Tests if modify_angles_mask_with_torsions correctly updates the
    torsion angle part of the angles_mask based on the input torsions tensor.
    """
    seq_len = 5
    num_atoms = 14
    dtype = torch.float32

    angles_mask = torch.zeros((2, seq_len, num_atoms), dtype=dtype)
    torsions = torch.arange(seq_len * 3, dtype=dtype).reshape(seq_len, 3)
    original_angles_mask = angles_mask.clone()
    modified_mask = modify_angles_mask_with_torsions(angles_mask, torsions)

    assert torch.equal(angles_mask, original_angles_mask), "Original angles_mask should not be modified."
    assert torch.equal(modified_mask[0], original_angles_mask[0]), "Bond angles (index 0) should not be modified."

    expected_psi = torsions[:-1, 1]
    assert torch.equal(modified_mask[1, :-1, 0], expected_psi), "Psi angles update failed."

    expected_omega = torsions[1:, 2]
    assert torch.equal(modified_mask[1, 1:, 1], expected_omega), "Omega angles update failed."

    expected_phi = torsions[1:, 0]
    assert torch.equal(modified_mask[1, 1:, 2], expected_phi), "Phi angles update failed."

    assert torch.all(modified_mask[1, :, 3:] == 0), "Other torsion angles should remain unchanged."
    assert modified_mask[1, -1, 0] == original_angles_mask[1, -1, 0], "Boundary element [1, -1, 0] changed unexpectedly."
    assert modified_mask[1, 0, 1] == original_angles_mask[1, 0, 1], "Boundary element [1, 0, 1] changed unexpectedly."
    assert modified_mask[1, 0, 2] == original_angles_mask[1, 0, 2], "Boundary element [1, 0, 2] changed unexpectedly."
    assert modified_mask.shape == (2, seq_len, num_atoms), "Output shape is incorrect."


# --- New Tests ---

@pytest.mark.parametrize(
    "input_angle, expected_angle",
    [
        (0.0, 0.0),                 # Angle already in range
        (np.pi / 2, np.pi / 2),     # Angle already in range
        (np.pi, np.pi),             # Angle already in range
        (3 * np.pi / 2, 3 * np.pi / 2), # Angle already in range
        (2 * np.pi, 0.0),           # Boundary case
        (2.5 * np.pi, 0.5 * np.pi), # Angle > 2pi
        (4 * np.pi, 0.0),           # Multiple of 2pi
        (-np.pi / 2, 3 * np.pi / 2), # Negative angle
        (-np.pi, np.pi),             # Negative angle
        (-3 * np.pi / 2, np.pi / 2), # Negative angle
        (-2 * np.pi, 0.0),           # Negative boundary
        (-2.5 * np.pi, 1.5 * np.pi), # Negative angle < -2pi
    ],
)
def test_to_zero_two_pi(input_angle, expected_angle):
    """Tests converting angles to the [0, 2π] range."""
    input_tensor = torch.tensor([input_angle])
    expected_tensor = torch.tensor([expected_angle])
    result_tensor = to_zero_two_pi(input_tensor)
    assert torch.allclose(result_tensor, expected_tensor, atol=1e-6)

# --- Tests for Getter Functions ---

VALID_AMINO_ACIDS = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
INVALID_AMINO_ACID = "X"

@pytest.mark.parametrize("aa", VALID_AMINO_ACIDS)
def test_get_rigid_frames_valid(aa):
    """Tests get_rigid_frames for valid amino acids."""
    expected = SC_BUILD_INFO[aa]["rigid-frames-idxs"]
    assert get_rigid_frames(aa) == expected

def test_get_rigid_frames_invalid():
    """Tests get_rigid_frames raises KeyError for invalid amino acids."""
    with pytest.raises(KeyError):
        get_rigid_frames(INVALID_AMINO_ACID)

@pytest.mark.parametrize("aa", VALID_AMINO_ACIDS)
def test_get_atom_names_valid(aa):
    """Tests get_atom_names for valid amino acids."""
    expected = SC_BUILD_INFO[aa]["atom-names"]
    assert get_atom_names(aa) == expected

def test_get_atom_names_invalid():
    """Tests get_atom_names raises KeyError for invalid amino acids."""
    with pytest.raises(KeyError):
        get_atom_names(INVALID_AMINO_ACID)

@pytest.mark.parametrize("aa", VALID_AMINO_ACIDS)
def test_get_bond_names_valid(aa):
    """Tests get_bond_names for valid amino acids."""
    expected = SC_BUILD_INFO[aa]["bonds-names"]
    assert get_bond_names(aa) == expected

def test_get_bond_names_invalid():
    """Tests get_bond_names raises KeyError for invalid amino acids."""
    with pytest.raises(KeyError):
        get_bond_names(INVALID_AMINO_ACID)

@pytest.mark.parametrize("aa", VALID_AMINO_ACIDS)
def test_get_bond_types_valid(aa):
    """Tests get_bond_types for valid amino acids."""
    expected = SC_BUILD_INFO[aa]["bonds-types"]
    assert get_bond_types(aa) == expected

def test_get_bond_types_invalid():
    """Tests get_bond_types raises KeyError for invalid amino acids."""
    with pytest.raises(KeyError):
        get_bond_types(INVALID_AMINO_ACID)

@pytest.mark.parametrize("aa", VALID_AMINO_ACIDS)
def test_get_bond_values_valid(aa):
    """Tests get_bond_values for valid amino acids."""
    expected = SC_BUILD_INFO[aa]["bonds-vals"]
    assert get_bond_values(aa) == expected

def test_get_bond_values_invalid():
    """Tests get_bond_values raises KeyError for invalid amino acids."""
    with pytest.raises(KeyError):
        get_bond_values(INVALID_AMINO_ACID)

@pytest.mark.parametrize("aa", VALID_AMINO_ACIDS)
def test_get_angle_names_valid(aa):
    """Tests get_angle_names for valid amino acids."""
    expected = SC_BUILD_INFO[aa]["angles-names"]
    assert get_angle_names(aa) == expected

def test_get_angle_names_invalid():
    """Tests get_angle_names raises KeyError for invalid amino acids."""
    with pytest.raises(KeyError):
        get_angle_names(INVALID_AMINO_ACID)

@pytest.mark.parametrize("aa", VALID_AMINO_ACIDS)
def test_get_angle_types_valid(aa):
    """Tests get_angle_types for valid amino acids."""
    expected = SC_BUILD_INFO[aa]["angles-types"]
    assert get_angle_types(aa) == expected

def test_get_angle_types_invalid():
    """Tests get_angle_types raises KeyError for invalid amino acids."""
    with pytest.raises(KeyError):
        get_angle_types(INVALID_AMINO_ACID)

@pytest.mark.parametrize("aa", VALID_AMINO_ACIDS)
def test_get_angle_values_valid(aa):
    """Tests get_angle_values for valid amino acids."""
    expected = SC_BUILD_INFO[aa]["angles-vals"]
    assert get_angle_values(aa) == expected

def test_get_angle_values_invalid():
    """Tests get_angle_values raises KeyError for invalid amino acids."""
    with pytest.raises(KeyError):
        get_angle_values(INVALID_AMINO_ACID)

@pytest.mark.parametrize("aa", VALID_AMINO_ACIDS)
def test_get_torsion_names_valid(aa):
    """Tests get_torsion_names for valid amino acids."""
    expected = SC_BUILD_INFO[aa]["torsion-names"]
    assert get_torsion_names(aa) == expected

def test_get_torsion_names_invalid():
    """Tests get_torsion_names raises KeyError for invalid amino acids."""
    with pytest.raises(KeyError):
        get_torsion_names(INVALID_AMINO_ACID)

@pytest.mark.parametrize("aa", VALID_AMINO_ACIDS)
def test_get_torsion_types_valid(aa):
    """Tests get_torsion_types for valid amino acids."""
    expected = SC_BUILD_INFO[aa]["torsion-types"]
    assert get_torsion_types(aa) == expected

def test_get_torsion_types_invalid():
    """Tests get_torsion_types raises KeyError for invalid amino acids."""
    with pytest.raises(KeyError):
        get_torsion_types(INVALID_AMINO_ACID)

@pytest.mark.parametrize("aa", VALID_AMINO_ACIDS)
def test_get_torsion_values_valid(aa):
    """Tests get_torsion_values for valid amino acids."""
    expected = SC_BUILD_INFO[aa]["torsion-vals"]
    assert get_torsion_values(aa) == expected

def test_get_torsion_values_invalid():
    """Tests get_torsion_values raises KeyError for invalid amino acids."""
    with pytest.raises(KeyError):
        get_torsion_values(INVALID_AMINO_ACID)

# --- Test for get_symmetric_atom_pairs ---

@pytest.mark.parametrize(
    "sequence, expected_dict",
    [
        ("A", {"0": []}),
        ("G", {"0": []}),
        ("AG", {"0": [], "1": []}),
        ("D", {"0": [(4, 5), (6, 7)]}), # Aspartic acid
        ("Y", {"0": [(4, 5), (6, 7), (8, 9), (10, 11)]}), # Tyrosine
        ("DAY", {"0": [(4, 5), (6, 7)], "1": [], "2": [(4, 5), (6, 7), (8, 9), (10, 11)]}),
        ("X", {}), # Invalid AA should be skipped
        ("AXG", {"0": [], "2": []}), # Skip invalid 'X'
        ("", {}), # Empty sequence
    ]
)
def test_get_symmetric_atom_pairs(sequence, expected_dict):
    """Tests get_symmetric_atom_pairs for various sequences."""
    result = get_symmetric_atom_pairs(sequence)
    assert result == expected_dict


# --- Tests for build_scaffolds_from_scn_angles ---

def test_build_scaffolds_angles_only(default_angles_coords, cpu_device):
    """Tests scaffold building using only angles."""
    seq = default_angles_coords["seq"]
    angles = default_angles_coords["angles"].to(cpu_device)
    seq_len = default_angles_coords["seq_len"]

    scaffolds = build_scaffolds_from_scn_angles(seq, angles=angles, coords=None, device=cpu_device)

    assert isinstance(scaffolds, dict)
    assert "cloud_mask" in scaffolds
    assert "point_ref_mask" in scaffolds
    assert "angles_mask" in scaffolds
    assert "bond_mask" in scaffolds

    assert scaffolds["cloud_mask"].shape == (seq_len, 14)
    assert scaffolds["cloud_mask"].dtype == torch.bool
    assert scaffolds["cloud_mask"].device == cpu_device

    assert scaffolds["point_ref_mask"].shape == (3, seq_len, 11)
    assert scaffolds["point_ref_mask"].dtype == torch.long
    assert scaffolds["point_ref_mask"].device == cpu_device

    assert scaffolds["angles_mask"].shape == (2, seq_len, 14)
    assert scaffolds["angles_mask"].dtype == angles.dtype
    assert scaffolds["angles_mask"].device == cpu_device

    assert scaffolds["bond_mask"].shape == (seq_len, 14)
    assert scaffolds["bond_mask"].dtype == angles.dtype
    assert scaffolds["bond_mask"].device == cpu_device

    expected_theta = make_theta_mask(seq, angles).to(cpu_device, angles.dtype)
    expected_torsion = make_torsion_mask(seq, angles).to(cpu_device, angles.dtype)
    assert torch.allclose(scaffolds["angles_mask"][0], expected_theta)
    assert torch.allclose(scaffolds["angles_mask"][1], expected_torsion)


def test_build_scaffolds_coords_and_angles(default_angles_coords, cpu_device):
    """Tests scaffold building using both coords and angles."""
    seq = default_angles_coords["seq"]
    angles = default_angles_coords["angles"].to(cpu_device)
    coords = default_angles_coords["coords"].to(cpu_device)
    seq_len = default_angles_coords["seq_len"]

    scaffolds = build_scaffolds_from_scn_angles(seq, angles=angles, coords=coords, device=cpu_device)

    assert isinstance(scaffolds, dict)
    assert "cloud_mask" in scaffolds
    assert scaffolds["cloud_mask"].shape == (seq_len, 14)
    assert scaffolds["cloud_mask"].dtype == torch.bool

    expected_cloud_mask = make_cloud_mask(seq, coords=coords).bool().to(cpu_device)
    assert torch.equal(scaffolds["cloud_mask"], expected_cloud_mask)

    assert "point_ref_mask" in scaffolds
    assert scaffolds["point_ref_mask"].shape == (3, seq_len, 11)
    assert "angles_mask" in scaffolds
    assert scaffolds["angles_mask"].shape == (2, seq_len, 14)
    assert "bond_mask" in scaffolds
    assert scaffolds["bond_mask"].shape == (seq_len, 14)


def test_build_scaffolds_angles_none_raises(default_angles_coords, cpu_device):
    """Tests that providing None for angles raises an error."""
    seq = default_angles_coords["seq"]
    coords = default_angles_coords["coords"].to(cpu_device)

    with pytest.raises(AttributeError): # Expects angles.dtype
        build_scaffolds_from_scn_angles(seq, angles=None, coords=coords, device=cpu_device)

def test_build_scaffolds_empty_seq(cpu_device):
    """Tests scaffold building with an empty sequence."""
    seq = ""
    angles = torch.empty((0, 12), device=cpu_device)
    coords = torch.empty((0, 14, 3), device=cpu_device)

    scaffolds = build_scaffolds_from_scn_angles(seq, angles=angles, coords=coords, device=cpu_device)

    assert scaffolds["cloud_mask"].shape == (0, 14)
    assert scaffolds["point_ref_mask"].shape == (3, 0, 11)
    assert scaffolds["angles_mask"].shape == (2, 0, 14)
    assert scaffolds["bond_mask"].shape == (0, 14)

# --- Tests for modify_scaffolds_with_coords ---

def test_modify_scaffolds_updates_in_place(sample_scaffolds, default_angles_coords):
    """Tests that the function modifies the input scaffold dictionary."""
    scaffolds_copy = {k: v.clone() for k, v in sample_scaffolds.items()}
    coords = default_angles_coords["coords"]

    returned_scaffolds = modify_scaffolds_with_coords(sample_scaffolds, coords)

    assert returned_scaffolds is sample_scaffolds
    assert not torch.equal(sample_scaffolds["bond_mask"], scaffolds_copy["bond_mask"])
    assert not torch.equal(sample_scaffolds["angles_mask"], scaffolds_copy["angles_mask"])


def test_modify_scaffolds_bond_lengths(sample_scaffolds, default_angles_coords):
    """Tests if bond lengths in the scaffold are correctly updated from coords."""
    coords = default_angles_coords["coords"]
    seq_len = default_angles_coords["seq_len"]
    modified_scaffolds = modify_scaffolds_with_coords(sample_scaffolds, coords)
    bond_mask = modified_scaffolds["bond_mask"]

    # Check N-CA bond length (atom index 1)
    expected_n_ca = torch.norm(coords[:, 1] - coords[:, 0], dim=-1)
    assert torch.allclose(bond_mask[:, 1], expected_n_ca)

    # Check CA-C bond length (atom index 2)
    expected_ca_c = torch.norm(coords[:, 2] - coords[:, 1], dim=-1)
    assert torch.allclose(bond_mask[:, 2], expected_ca_c)

    # Check C-N+1 bond length (atom index 0 of next residue)
    if seq_len > 1:
        expected_c_n = torch.norm(coords[1:, 0] - coords[:-1, 2], dim=-1)
        assert torch.allclose(bond_mask[1:, 0], expected_c_n)

    # Check a sidechain bond length, e.g., CB (atom index 4) from CA (atom index 1)
    cloud_mask_cb = modified_scaffolds["cloud_mask"][:, 4]
    if torch.any(cloud_mask_cb):
        idx_a, idx_b, idx_c = modified_scaffolds["point_ref_mask"][:, cloud_mask_cb, 4 - 3]
        selector = cloud_mask_cb.nonzero().squeeze(-1) # Ensure selector is 1D
        expected_cb_ca = torch.norm(coords[selector, 4] - coords[selector, idx_c], dim=-1)
        assert torch.allclose(bond_mask[cloud_mask_cb, 4], expected_cb_ca)


def test_modify_scaffolds_angles_dihedrals(sample_scaffolds, default_angles_coords):
    """Tests if angles and dihedrals in the scaffold are correctly updated."""
    coords = default_angles_coords["coords"]
    seq_len = default_angles_coords["seq_len"]
    modified_scaffolds = modify_scaffolds_with_coords(sample_scaffolds, coords)
    angles_mask = modified_scaffolds["angles_mask"]
    point_ref_mask = modified_scaffolds["point_ref_mask"]
    cloud_mask = modified_scaffolds["cloud_mask"]
    selector = np.arange(seq_len)

    # Check backbone angle N-CA-C (atom index 2)
    expected_n_ca_c = get_angle(coords[:, 0], coords[:, 1], coords[:, 2])
    assert torch.allclose(angles_mask[0, :, 2], expected_n_ca_c)

    # Check backbone dihedral phi (C-1, N, CA, C) (atom index 2)
    if seq_len > 1:
        expected_phi = get_dihedral(coords[:-1, 2], coords[1:, 0], coords[1:, 1], coords[1:, 2])
        assert torch.allclose(angles_mask[1, 1:, 2], expected_phi)

    # Check a sidechain angle, e.g., N-CA-CB (atom index 4)
    atom_idx = 4
    mask_i = cloud_mask[:, atom_idx]
    if torch.any(mask_i):
        idx_a, idx_b, idx_c = point_ref_mask[:, mask_i, atom_idx - 3]
        selector_i = mask_i.nonzero().squeeze(-1) # Ensure selector is 1D
        expected_angle_sc = get_angle(coords[selector_i, idx_b], coords[selector_i, idx_c], coords[selector_i, atom_idx])
        assert torch.allclose(angles_mask[0, mask_i, atom_idx], expected_angle_sc)

    # Check a sidechain dihedral, e.g., N-CA-CB-CG (atom index 5, depends on 4)
    atom_idx = 5
    mask_i = cloud_mask[:, atom_idx]
    if torch.any(mask_i):
        idx_a, idx_b, idx_c = point_ref_mask[:, mask_i, atom_idx - 3]
        selector_i = mask_i.nonzero().squeeze(-1) # Ensure selector is 1D

        # Handle C-beta special case for atom 'a' in dihedral calculation
        coords_a_dihedral = coords[selector_i, idx_a]
        # Check if the first residue in the selection needs special handling
        if selector_i.numel() > 0 and selector_i[0] == 0:
             # If the first residue has this atom, its 'a' coord needs special handling for CB
             if atom_idx == 4: # CB depends on C(i-1) which doesn't exist for i=0
                 # This case is tricky, the original code might have edge case issues here.
                 # For now, we might skip the check for the first residue if atom_idx is 4
                 # or use a placeholder like the next N. Let's assume the test setup avoids this.
                 pass # Or add specific handling if needed
        # For other residues or other atoms, use the standard logic
        # Note: The original logic for atom 5 (CG) depending on atom 4 (CB)
        # implicitly handled the C-beta case because idx_a for CG points to N(i),
        # not C(i-1). Let's recalculate expected dihedral carefully.
        expected_dihedral_sc = get_dihedral(coords[selector_i, idx_a], coords[selector_i, idx_b], coords[selector_i, idx_c], coords[selector_i, atom_idx])

        assert torch.allclose(angles_mask[1, mask_i, atom_idx], expected_dihedral_sc)


# --- Tests for protein_fold ---

def test_protein_fold_basic_shape_type(sample_scaffolds, cpu_device):
    """Tests the basic output shape and type of protein_fold."""
    seq_len = sample_scaffolds["cloud_mask"].shape[0]
    coords, cloud_mask_out = protein_fold(
        sample_scaffolds["cloud_mask"],
        sample_scaffolds["point_ref_mask"],
        sample_scaffolds["angles_mask"],
        sample_scaffolds["bond_mask"],
        device=cpu_device
    )

    assert isinstance(coords, torch.Tensor)
    assert isinstance(cloud_mask_out, torch.Tensor)
    assert coords.shape == (seq_len, 14, 3)
    assert cloud_mask_out.shape == (seq_len, 14)
    assert coords.dtype == sample_scaffolds["bond_mask"].dtype # Should match precision
    assert cloud_mask_out.dtype == torch.bool
    assert coords.device == cpu_device
    assert cloud_mask_out.device == cpu_device
    assert torch.equal(cloud_mask_out, sample_scaffolds["cloud_mask"])


def test_protein_fold_short_sequence(cpu_device):
    """Tests protein_fold with a single amino acid sequence."""
    seq = "A"
    seq_len = 1
    angles = torch.randn(seq_len, 12, dtype=torch.float32).to(cpu_device)
    scaffolds = build_scaffolds_from_scn_angles(seq, angles=angles, device=cpu_device)

    coords, cloud_mask_out = protein_fold(
        scaffolds["cloud_mask"],
        scaffolds["point_ref_mask"],
        scaffolds["angles_mask"],
        scaffolds["bond_mask"],
        device=cpu_device
    )

    assert coords.shape == (seq_len, 14, 3)
    assert cloud_mask_out.shape == (seq_len, 14)
    assert torch.all(~torch.isnan(coords[0, :3]))
    assert torch.allclose(coords[0, 0], torch.zeros(3, device=cpu_device))


def test_protein_fold_glycine(cpu_device):
    """Tests protein_fold with Glycine (no CB)."""
    seq = "AGA"
    seq_len = 3
    angles = torch.randn(seq_len, 12, dtype=torch.float32).to(cpu_device)
    scaffolds = build_scaffolds_from_scn_angles(seq, angles=angles, device=cpu_device)

    coords, cloud_mask_out = protein_fold(
        scaffolds["cloud_mask"],
        scaffolds["point_ref_mask"],
        scaffolds["angles_mask"],
        scaffolds["bond_mask"],
        device=cpu_device
    )

    assert coords.shape == (3, 14, 3)
    assert not cloud_mask_out[1, 4] # Glycine CB is masked
    assert torch.all(coords[1, 4] == 0)


def test_protein_fold_geometry(cpu_device):
    """Tests some basic geometric properties of the folded structure."""
    seq = "ARND"
    seq_len = 4
    angles = torch.zeros(seq_len, 12, dtype=torch.float32).to(cpu_device)
    angles[:, 2] = np.pi # Set omega to pi (trans peptide bond)
    scaffolds = build_scaffolds_from_scn_angles(seq, angles=angles, device=cpu_device)

    coords, _ = protein_fold(
        scaffolds["cloud_mask"],
        scaffolds["point_ref_mask"],
        scaffolds["angles_mask"],
        scaffolds["bond_mask"],
        device=cpu_device
    )

    # Check peptide bond length (C-N distance between residues)
    c_coords = coords[:-1, 2]
    n_coords = coords[1:, 0]
    peptide_bond_lengths = torch.norm(n_coords - c_coords, dim=-1)
    expected_peptide_bond = BB_BUILD_INFO["BONDLENS"]["c-n"]
    assert torch.allclose(peptide_bond_lengths, torch.tensor(expected_peptide_bond, device=cpu_device), atol=1e-3)

    # Check omega dihedral angle (should be close to pi)
    if seq_len > 1:
        ca_i = coords[:-1, 1]
        c_i = coords[:-1, 2]
        n_ip1 = coords[1:, 0]
        ca_ip1 = coords[1:, 1]
        omega_angles = get_dihedral(ca_i, c_i, n_ip1, ca_ip1)
        omega_angles = torch.atan2(torch.sin(omega_angles), torch.cos(omega_angles))
        assert torch.allclose(torch.abs(omega_angles), torch.tensor(np.pi, device=cpu_device), atol=1e-2)


def test_protein_fold_first_three_atoms(cpu_device):
    """Tests the coordinates of the first three atoms (N, CA, C) of the first residue."""
    seq = "A"
    seq_len = 1
    # Use zero angles for predictable geometry, except N-CA-C angle
    angles = torch.zeros(seq_len, 12, dtype=torch.float32).to(cpu_device)
    n_ca_c_angle = np.radians(BB_BUILD_INFO["ANGLES"]["n-ca-c"]) # Get standard angle in radians
    angles[0, 5] = n_ca_c_angle # Index 5 corresponds to N-CA-C in the (L,12) angles tensor

    scaffolds = build_scaffolds_from_scn_angles(seq, angles=angles, device=cpu_device)
    # Manually set the N-CA-C angle in the angles_mask used by protein_fold
    scaffolds["angles_mask"][0, 0, 2] = n_ca_c_angle # Bond angle theta, residue 0, atom C (index 2)

    coords, _ = protein_fold(
        scaffolds["cloud_mask"],
        scaffolds["point_ref_mask"],
        scaffolds["angles_mask"],
        scaffolds["bond_mask"],
        device=cpu_device
    )

    # Expected coordinates based on standard bond lengths and N-CA-C angle
    n_coord = torch.tensor([0.0, 0.0, 0.0], device=cpu_device)
    ca_coord_expected = n_coord + torch.tensor([BB_BUILD_INFO["BONDLENS"]["n-ca"], 0.0, 0.0], device=cpu_device)
    # C coord calculation: rotate vector [bond_len, 0, 0] by (pi - angle) around z-axis and add to CA
    ca_c_bond = BB_BUILD_INFO["BONDLENS"]["ca-c"]
    angle_rad = np.pi - n_ca_c_angle
    c_coord_relative = torch.tensor([ca_c_bond * np.cos(angle_rad), ca_c_bond * np.sin(angle_rad), 0.0], device=cpu_device)
    # Need to align this relative vector based on the N-CA vector
    # Since N-CA is along x-axis, the relative vector is already in the correct frame
    c_coord_expected = ca_coord_expected + c_coord_relative

    assert torch.allclose(coords[0, 0], n_coord, atol=1e-5)
    assert torch.allclose(coords[0, 1], ca_coord_expected, atol=1e-5)
    assert torch.allclose(coords[0, 2], c_coord_expected, atol=1e-5)

# TODO: Add test for hybrid=True if GPU available/mockable
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
    scn_cloud_mask,
)
from rna_predict.pipeline.stageC.mp_nerf.protein_utils.mask_generators import make_cloud_mask
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
    cloud_mask = scn_cloud_mask(seq, coords=coords).astype(bool)
    coords[~cloud_mask] = 0 # Zero out masked coords for simplicity in tests

    return {"seq": seq, "angles": angles, "coords": coords, "seq_len": seq_len}

@pytest.fixture
def sample_scaffolds_and_data(default_angles_coords, cpu_device):
    """Provides sample scaffolds generated from default angles, plus original data."""
    data = default_angles_coords
    # Use coords=None initially to get scaffolds based on standard geometry + angles
    scaffolds = build_scaffolds_from_scn_angles(
        data["seq"], angles=data["angles"], coords=None, device=cpu_device
    )
    # Add cloud_mask based on actual coords for modify/fold tests if needed,
    # but protein_fold recalculates this internally now.
    # Let's return the original seq and angles needed for the direct protein_fold call.
    return {
        "scaffolds": scaffolds, # Keep original scaffolds if other tests use them
        "seq": data["seq"],
        "angles": data["angles"].to(cpu_device), # Ensure angles are on the right device
        "seq_len": data["seq_len"]
    }


# Update the test function to use the new fixture
def test_protein_fold_basic_shape_type(sample_scaffolds_and_data, cpu_device):
    """Tests the basic output shape and type of protein_fold."""
    fixture_data = sample_scaffolds_and_data
    seq_len = fixture_data["seq_len"]
    seq = fixture_data["seq"]
    angles = fixture_data["angles"] # Already on cpu_device from fixture

    # Call protein_fold directly with seq and angles
    coords, cloud_mask_out = protein_fold(
        seq=seq,
        angles=angles,
        coords=None, # No initial coords for this test
        device=cpu_device
    )

    assert isinstance(coords, torch.Tensor)
    assert isinstance(cloud_mask_out, torch.Tensor)
    assert coords.shape == (seq_len, 14, 3)
    assert cloud_mask_out.shape == (seq_len, 14)
    assert coords.dtype == angles.dtype # Should match precision of input angles
    assert cloud_mask_out.dtype == torch.bool
    assert coords.device == cpu_device
    assert cloud_mask_out.device == cpu_device
    # Check output cloud mask consistency (protein_fold generates its own)
    expected_cloud_mask_np = np.array([make_cloud_mask(aa) for aa in seq]) # Generate expected mask
    expected_cloud_mask = torch.tensor(expected_cloud_mask_np, device=cpu_device, dtype=torch.bool)
    # Note: protein_fold's internal build_scaffolds call uses scn_cloud_mask,
    # which delegates to make_cloud_mask. So this check should be valid.
    assert torch.equal(cloud_mask_out, expected_cloud_mask)


def test_protein_fold_short_sequence(cpu_device):
    """Tests protein_fold with a single amino acid sequence."""
    seq = "A"
    seq_len = 1
    angles = torch.randn(seq_len, 12, dtype=torch.float32).to(cpu_device)
    # coords = None # No initial coords needed for this test case

    # Call protein_fold with the correct arguments: seq, angles, coords, device
    coords, cloud_mask_out = protein_fold(
        seq=seq,
        angles=angles,
        coords=None, # Pass None explicitly if not using initial coords
        device=cpu_device
    )

    assert coords.shape == (seq_len, 14, 3)
    assert cloud_mask_out.shape == (seq_len, 14)
    # Check that backbone atoms (N, CA, C - indices 0, 1, 2) are not NaN
    assert torch.all(~torch.isnan(coords[0, :3]))
    # Check if the first atom (N) is at the origin (standard starting point)
    assert torch.allclose(coords[0, 0], torch.zeros(3, device=cpu_device), atol=1e-6)


def test_protein_fold_glycine(cpu_device):
    """Tests protein_fold with Glycine (no CB)."""
    seq = "AGA"
    seq_len = 3
    angles = torch.randn(seq_len, 12, dtype=torch.float32).to(cpu_device)
    # No need to build scaffolds separately if calling protein_fold directly
    # scaffolds = build_scaffolds_from_scn_angles(seq, angles=angles, device=cpu_device)

    # Call protein_fold with keyword arguments
    coords, cloud_mask_out = protein_fold(
        seq=seq,
        angles=angles,
        coords=None,
        device=cpu_device
    )

    assert coords.shape == (3, 14, 3)
    assert not cloud_mask_out[1, 4] # Glycine CB is masked
    # Check if masked coordinates are zero (or NaN depending on implementation)
    # Assuming protein_fold zeros out masked coords
    assert torch.all(coords[1, 4] == 0)


def test_protein_fold_geometry(cpu_device):
    """Tests some basic geometric properties of the folded structure."""
    seq = "ARND"
    seq_len = 4
    angles = torch.zeros(seq_len, 12, dtype=torch.float32).to(cpu_device)
    angles[:, 2] = np.pi # Set omega to pi (trans peptide bond) - Index 2 in (L, 12) is omega
    # No need to build scaffolds separately
    # scaffolds = build_scaffolds_from_scn_angles(seq, angles=angles, device=cpu_device)

    # Call protein_fold with keyword arguments
    coords, _ = protein_fold(
        seq=seq,
        angles=angles,
        coords=None,
        device=cpu_device
    )

    # Check peptide bond length (C-N distance between residues)
    c_coords = coords[:-1, 2]
    n_coords = coords[1:, 0]
    peptide_bond_lengths = torch.norm(n_coords - c_coords, dim=-1)
    expected_peptide_bond = BB_BUILD_INFO.get("BONDLENS", {}).get("c-n", 1.329) # Use .get with default
    assert torch.allclose(peptide_bond_lengths, torch.tensor(expected_peptide_bond, device=cpu_device, dtype=coords.dtype), atol=1e-3)

    # Check omega dihedral angle (should be close to pi)
    if seq_len > 1:
        ca_i = coords[:-1, 1]
        c_i = coords[:-1, 2]
        n_ip1 = coords[1:, 0]
        ca_ip1 = coords[1:, 1]
        omega_angles = get_dihedral(ca_i, c_i, n_ip1, ca_ip1)
        # Wrap angles to [-pi, pi] for comparison
        omega_angles_wrapped = torch.atan2(torch.sin(omega_angles), torch.cos(omega_angles))
        assert torch.allclose(torch.abs(omega_angles_wrapped), torch.tensor(np.pi, device=cpu_device, dtype=coords.dtype), atol=1e-2)


def test_protein_fold_first_three_atoms(cpu_device):
    """Tests the coordinates of the first three atoms (N, CA, C) of the first residue."""
    seq = "A"
    seq_len = 1
    # Use zero angles for predictable geometry, except N-CA-C angle
    angles = torch.zeros(seq_len, 12, dtype=torch.float32).to(cpu_device)
    # Access angle using the structure observed in mask_generators.py
    n_ca_c_angle = np.radians(BB_BUILD_INFO.get("BONDANGS", {}).get("n-ca-c", 1.939)) # Use .get with default
    angles[0, 5] = n_ca_c_angle # Index 5 corresponds to N-CA-C in the (L,12) angles tensor

    # Build scaffolds using the corrected function
    # scaffolds = build_scaffolds_from_scn_angles(seq, angles=angles, device=cpu_device) # No longer needed if calling protein_fold directly
    # Manually set the N-CA-C angle in the angles_mask used by protein_fold
    # This override might be redundant if build_scaffolds uses make_theta_mask correctly,
    # and protein_fold uses the angles tensor directly. Let's call protein_fold directly.
    # scaffolds["angles_mask"][0, 0, 2] = n_ca_c_angle # Bond angle theta, residue 0, atom C (index 2)

    # Call protein_fold with the correct arguments
    coords, _ = protein_fold(
        seq=seq,
        angles=angles, # Pass the angles tensor used to build scaffolds
        coords=None,
        device=cpu_device
    )

    # Expected coordinates based on standard bond lengths and N-CA-C angle
    n_coord = torch.tensor([0.0, 0.0, 0.0], device=cpu_device)
    n_ca_bond = BB_BUILD_INFO.get("BONDLENS", {}).get("n-ca", 1.458) # Use .get with default
    ca_coord_expected = n_coord + torch.tensor([n_ca_bond, 0.0, 0.0], device=cpu_device)

    # C coord calculation: rotate vector [bond_len, 0, 0] by (pi - angle) around z-axis and add to CA
    ca_c_bond = BB_BUILD_INFO.get("BONDLENS", {}).get("ca-c", 1.525) # Use .get with default
    angle_rad = np.pi - n_ca_c_angle
    c_coord_relative = torch.tensor([ca_c_bond * np.cos(angle_rad), ca_c_bond * np.sin(angle_rad), 0.0], device=cpu_device, dtype=torch.float32)
    # Need to align this relative vector based on the N-CA vector
    # Since N-CA is along x-axis, the relative vector is already in the correct frame
    c_coord_expected = ca_coord_expected + c_coord_relative

    assert torch.allclose(coords[0, 0], n_coord, atol=1e-5)
    assert torch.allclose(coords[0, 1], ca_coord_expected, atol=1e-5)
    assert torch.allclose(coords[0, 2], c_coord_expected, atol=1e-5)

# TODO: Add test for hybrid=True if GPU available/mockable
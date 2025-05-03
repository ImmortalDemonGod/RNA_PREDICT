# tests/stageC/mp_nerf_tests/test_utils.py
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Assuming the file is in rna_predict.pipeline.stageC.mp_nerf.utils
# Adjust the import path if necessary based on your project structure and PYTHONPATH
# Try importing from the expected location first
try:
    from rna_predict.pipeline.stageC.mp_nerf import utils

    # Mock Bio.PDB for tests where file parsing is not the focus or might fail
    # We primarily want to test *our* logic around Bio.PDB calls
    mock_pdb_parser = MagicMock()
    mock_cif_parser = MagicMock()
    mock_structure = MagicMock()
    mock_model = MagicMock()
    mock_chain = MagicMock()  # Added mock for chain
    mock_residue = MagicMock()
    mock_atom = MagicMock()
    mock_atom.get_coord.return_value = np.array(
        [1.0, 2.0, 3.0]
    )  # Use numpy array as BioPython returns

    # Configure mocks to simulate structure hierarchy
    mock_residue.get_atoms.return_value = [
        mock_atom
    ] * 3  # Simulate 3 atoms per residue
    mock_residue.id = (" ", 1, " ")  # Example residue ID
    mock_residue.get_resname.return_value = "ALA"  # Example residue name
    mock_chain.get_residues.return_value = [mock_residue] * 2  # Simulate 2 residues
    mock_model.get_chains.return_value = [mock_chain]  # Simulate 1 chain
    mock_structure.get_models.return_value = [mock_model]  # Simulate 1 model
    mock_pdb_parser.get_structure.return_value = mock_structure
    mock_cif_parser.get_structure.return_value = mock_structure

    # Patch Bio.PDB where it's imported in the utils module
    patch_pdb_parser = patch(
        "rna_predict.pipeline.stageC.mp_nerf.utils.PDBParser",
        return_value=mock_pdb_parser,
    )
    patch_cif_parser = patch(
        "rna_predict.pipeline.stageC.mp_nerf.utils.MMCIFParser",
        return_value=mock_cif_parser,
    )
    # Patch the IO classes used in save_structure
    patch_pdb_io = patch("rna_predict.pipeline.stageC.mp_nerf.utils.PDBIO")
    patch_cif_io = patch("rna_predict.pipeline.stageC.mp_nerf.utils.MMCIFIO")
    # Patch Atom class where it's imported/used in utils.py
    patch_atom_cls = patch("rna_predict.pipeline.stageC.mp_nerf.utils.Atom")
    # Remove patches for classes not directly imported/used by name in utils.save_structure
    # patch_residue_cls = patch("Bio.PDB.Residue.Residue") # Removed
    # patch_chain_cls = patch("Bio.PDB.Chain.Chain")       # Removed
    # patch_model_cls = patch("Bio.PDB.Model.Model")       # Removed
    # Patch StructureBuilder where it's imported/used in utils.py
    patch_builder_cls = patch(
        "rna_predict.pipeline.stageC.mp_nerf.utils.StructureBuilder"
    )


except ImportError as e:
    print(f"Import Error: {e}")
    pytest.skip(
        "Skipping tests because rna_predict module or its dependencies not found.",
        allow_module_level=True,
    )

    # Define dummy functions if import fails to avoid collection errors
    class DummyUtils:  # Dummy class to hold functions
        @staticmethod
        def get_coords_from_pdb(*args, **kwargs):
            pass
        @staticmethod
        def get_coords_from_cif(*args, **kwargs):
            pass
        @staticmethod
        def get_coords_from_file(*args, **kwargs):
            pass
        @staticmethod
        def get_device(*args, **kwargs):
            pass
        @staticmethod
        def save_structure(*args, **kwargs):
            pass
        @staticmethod
        def get_prot(*args, **kwargs):
            pass

    utils = DummyUtils

# --- Fixtures for Dummy Files ---


@pytest.fixture
def dummy_pdb_content() -> str:
    """Provides content for a simple dummy PDB file."""
    # Minimal PDB format with N, CA, C atoms for two residues
    return """ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.500   2.500   3.500  1.00  0.00           C
ATOM      3  C   ALA A   1       2.000   3.000   4.000  1.00  0.00           C
ATOM      4  N   GLY A   2       3.000   4.000   5.000  1.00  0.00           N
ATOM      5  CA  GLY A   2       3.500   4.500   5.500  1.00  0.00           C
ATOM      6  C   GLY A   2       4.000   5.000   6.000  1.00  0.00           C
"""


@pytest.fixture
def dummy_cif_content() -> str:
    """Provides content for a simple dummy CIF file."""
    # Minimal CIF format with N, CA, C atoms for two residues
    return """data_test
_entry.id test_cif

loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_asym_id
_atom_site.auth_seq_id
ATOM   1  N N   ALA A 1 10.000 11.000 12.000 1.00 0.00 A 1
ATOM   2  C CA  ALA A 1 10.500 11.500 12.500 1.00 0.00 A 1
ATOM   3  C C   ALA A 1 11.000 12.000 13.000 1.00 0.00 A 1
ATOM   4  N N   GLY A 2 12.000 13.000 14.000 1.00 0.00 A 2
ATOM   5  C CA  GLY A 2 12.500 13.500 14.500 1.00 0.00 A 2
ATOM   6  C C   GLY A 2 13.000 14.000 15.000 1.00 0.00 A 2
#
"""


@pytest.fixture
def pdb_file(tmp_path: Path, dummy_pdb_content: str) -> Path:
    """Creates a dummy PDB file in a temporary directory."""
    p = tmp_path / "test.pdb"
    p.write_text(dummy_pdb_content)
    return p


@pytest.fixture
def cif_file(tmp_path: Path, dummy_cif_content: str) -> Path:
    """Creates a dummy CIF file in a temporary directory."""
    p = tmp_path / "test.cif"
    p.write_text(dummy_cif_content)
    return p


# --- Test Functions ---


# Test get_coords_from_pdb (Lines 33-51)
# Use mocks to isolate the function's logic from Bio.PDB parsing details
@patch_pdb_parser
def test_get_coords_from_pdb(mock_parser, tmp_path: Path):
    """
    Tests get_coords_from_pdb logic using mocked Bio.PDB.
    Covers lines: 33-51
    """
    pdb_file = tmp_path / "dummy.pdb"
    pdb_file.touch()  # File just needs to exist

    # Expected coords based on mock_atom.get_coord() = [1,2,3]
    # 2 residues * 3 atoms/residue = 6 atoms
    expected_coords = torch.tensor([[1.0, 2.0, 3.0]] * 6, dtype=torch.float32)

    coords = utils.get_coords_from_pdb(str(pdb_file))

    # Assert on the INSTANCE returned by the mocked class, not the class mock itself
    mock_pdb_parser.get_structure.assert_called_once_with("structure", str(pdb_file))
    assert mock_atom.get_coord.call_count == 6  # 2 residues * 3 atoms
    assert isinstance(coords, torch.Tensor)
    assert coords.shape == (6, 3)
    assert torch.allclose(coords, expected_coords)
    assert coords.dtype == torch.float32


# Test get_coords_from_cif (Lines 54-72)
@patch_cif_parser
def test_get_coords_from_cif(mock_parser, tmp_path: Path):
    """
    Tests get_coords_from_cif logic using mocked Bio.PDB.
    Covers lines: 54-72
    """
    # ... (rest of file unchanged)

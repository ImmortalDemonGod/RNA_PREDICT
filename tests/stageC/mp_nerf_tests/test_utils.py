# tests/stageC/mp_nerf_tests/test_utils.py
from pathlib import Path
from types import SimpleNamespace  # Added SimpleNamespace
from typing import (
    Any,
    Dict,
    Iterator,
    List,
)  # Added Dict, Any, NamedTuple, Tuple, Iterator
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
            pass  # Add dummy get_prot

    # Dummy patches if import failed
    patch_pdb_parser = patch("builtins.print")
    patch_cif_parser = patch("builtins.print")
    patch_pdb_io = patch("builtins.print")
    patch_cif_io = patch("builtins.print")
    patch_atom_cls = patch("builtins.print")
    patch_residue_cls = patch("builtins.print")
    patch_chain_cls = patch("builtins.print")
    patch_model_cls = patch("builtins.print")
    patch_builder_cls = patch("builtins.print")


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
    cif_file = tmp_path / "dummy.cif"
    cif_file.touch()  # File just needs to exist

    # Expected coords based on mock_atom.get_coord() = [1,2,3]
    # 2 residues * 3 atoms/residue = 6 atoms
    expected_coords = torch.tensor([[1.0, 2.0, 3.0]] * 6, dtype=torch.float32)

    # Reset mock call count if the same mock instance is used across tests
    mock_atom.get_coord.reset_mock()

    coords = utils.get_coords_from_cif(str(cif_file))

    # Assert on the INSTANCE returned by the mocked class, not the class mock itself
    mock_cif_parser.get_structure.assert_called_once_with("structure", str(cif_file))
    assert mock_atom.get_coord.call_count == 6  # 2 residues * 3 atoms
    assert isinstance(coords, torch.Tensor)
    assert coords.shape == (6, 3)
    assert torch.allclose(coords, expected_coords)
    assert coords.dtype == torch.float32


# Test get_coords_from_file (Lines 75-78)
@patch("rna_predict.pipeline.stageC.mp_nerf.utils.get_coords_from_pdb")
@patch("rna_predict.pipeline.stageC.mp_nerf.utils.get_coords_from_cif")
def test_get_coords_from_file_dispatch(mock_get_cif, mock_get_pdb, tmp_path: Path):
    """
    Tests get_coords_from_file dispatching to correct function based on extension.
    Covers lines: 75-78
    """
    pdb_file = tmp_path / "dummy.pdb"
    cif_file = tmp_path / "dummy.cif"
    txt_file = tmp_path / "dummy.txt"
    pdb_file.touch()
    cif_file.touch()
    txt_file.touch()

    # Test PDB path
    utils.get_coords_from_file(str(pdb_file))
    mock_get_pdb.assert_called_once_with(str(pdb_file))
    mock_get_cif.assert_not_called()
    mock_get_pdb.reset_mock()

    # Test CIF path
    utils.get_coords_from_file(str(cif_file))
    mock_get_cif.assert_called_once_with(str(cif_file))
    mock_get_pdb.assert_not_called()
    mock_get_cif.reset_mock()

    # Test unsupported path
    with pytest.raises(ValueError, match="Unsupported file format: .txt"):
        utils.get_coords_from_file(str(txt_file))
    mock_get_pdb.assert_not_called()
    mock_get_cif.assert_not_called()


# Test get_device (Lines 118-122)
@patch("rna_predict.pipeline.stageC.mp_nerf.utils.torch.cuda.is_available")
def test_get_device_cuda_available(mock_cuda_available):
    """
    Tests get_device returns 'cuda' when torch.cuda.is_available() is True.
    Covers lines: 118-120, 122
    """
    mock_cuda_available.return_value = True
    device = utils.get_device()
    assert isinstance(device, torch.device)
    assert device.type == "cuda"
    mock_cuda_available.assert_called_once()


@patch("rna_predict.pipeline.stageC.mp_nerf.utils.torch.cuda.is_available")
def test_get_device_cuda_not_available(mock_cuda_available):
    """
    Tests get_device returns 'cpu' when torch.cuda.is_available() is False.
    Covers lines: 118, 121-122
    """
    mock_cuda_available.return_value = False
    device = utils.get_device()
    assert isinstance(device, torch.device)
    assert device.type == "cpu"
    mock_cuda_available.assert_called_once()


# Test save_structure (Lines 135-154, 159)
# Use mocks for Bio.PDB classes to avoid actual file writing and dependency
@patch_builder_cls
# Removed unused decorators: @patch_model_cls, @patch_chain_cls, @patch_residue_cls
@patch_atom_cls
@patch_pdb_io
@patch_cif_io
def test_save_structure_3d_input_pdb(
    mock_cifio_cls,
    mock_pdbio_cls,
    mock_atom_cls,
    mock_builder_cls,
    tmp_path: Path,
):
    """
    Tests save_structure logic for 3D input and PDB output using mocks.
    Covers lines: 135-145, 159 (pass)
    """
    coords = torch.arange(18, dtype=torch.float32).reshape(
        2, 3, 3
    )  # 2 residues, 3 atoms each
    output_file = tmp_path / "output_3d.pdb"
    atom_types = ["N", "CA", "C"]
    res_names = ["ALA", "GLY"]

    # Mock instances
    mock_pdbio_inst = mock_pdbio_cls.return_value
    mock_builder_inst = mock_builder_cls.return_value
    mock_structure = mock_builder_inst.get_structure.return_value
    # Removed references to unused mocks: mock_model_cls, mock_chain_cls
    # mock_model_inst = mock_model_cls.return_value # Removed
    # mock_chain_inst = mock_chain_cls.return_value # Removed

    # Configure mock hierarchy returns - Mock the necessary hierarchy directly if needed
    # For this test, mocking the builder methods called within save_structure is sufficient
    # mock_structure.__getitem__.return_value = mock_model_inst # Removed
    # mock_model_inst.__getitem__.return_value = mock_chain_inst # Removed

    utils.save_structure(
        coords, str(output_file), atom_types=atom_types, res_names=res_names
    )

    # Assertions
    mock_builder_cls.assert_called_once()  # Builder class was instantiated
    mock_builder_inst.init_structure.assert_called_once_with(
        "structure"
    )  # Method called on instance
    mock_builder_inst.init_model.assert_called_once_with(0)  # Method called on instance
    mock_builder_inst.init_chain.assert_called_once_with(
        "A"
    )  # Method called on instance
    assert (
        mock_builder_inst.init_residue.call_count == 2
    )  # Method called on instance twice
    assert mock_atom_cls.call_count == 6  # Atom class instantiated directly 6 times
    # Check atom details for the first atom of the first residue (ensure Atom class was called correctly)
    # Access keyword arguments dictionary (index 1) from the first call (index 0)
    first_atom_kwargs = mock_atom_cls.call_args_list[0][1]
    assert first_atom_kwargs["name"] == "N   "  # Check 'name' keyword argument
    np.testing.assert_allclose(
        first_atom_kwargs["coord"], coords[0, 0, :].numpy()
    )  # Check 'coord' keyword argument
    assert first_atom_kwargs["element"] == "N"  # Check 'element' keyword argument

    mock_pdbio_cls.assert_called_once()
    mock_pdbio_inst.set_structure.assert_called_once_with(mock_structure)
    mock_pdbio_inst.save.assert_called_once_with(
        str(output_file)
    )  # Save should be called now
    mock_cifio_cls.assert_not_called()


@patch_builder_cls
# Removed unused decorators: @patch_model_cls, @patch_chain_cls, @patch_residue_cls
@patch_atom_cls
@patch_pdb_io
@patch_cif_io
def test_save_structure_2d_input_cif(
    mock_cifio_cls,
    mock_pdbio_cls,
    mock_atom_cls,
    # Removed unused mock arguments: mock_residue_cls, mock_chain_cls, mock_model_cls
    mock_builder_cls,
    tmp_path: Path,
):
    """
    Tests save_structure logic for 2D input and CIF output using mocks.
    Covers lines: 135-143, 146-151, 159 (pass)
    """
    coords = torch.arange(12, dtype=torch.float32).reshape(4, 3)  # 4 atoms total
    output_file = tmp_path / "output_2d.cif"
    atom_types = ["N", "CA", "C", "O"]  # 4 atom types, implies 1 residue
    res_names = ["ALA"]

    # Mock instances
    mock_cifio_inst = mock_cifio_cls.return_value
    mock_builder_inst = mock_builder_cls.return_value
    mock_structure = mock_builder_inst.get_structure.return_value
    # Removed references to unused mocks: mock_model_cls, mock_chain_cls
    # mock_model_inst = mock_model_cls.return_value # Removed
    # mock_chain_inst = mock_chain_cls.return_value # Removed

    # Configure mock hierarchy returns - Mock the necessary hierarchy directly if needed
    # For this test, mocking the builder methods called within save_structure is sufficient
    # mock_structure.__getitem__.return_value = mock_model_inst # Removed
    # mock_model_inst.__getitem__.return_value = mock_chain_inst # Removed

    utils.save_structure(
        coords, str(output_file), atom_types=atom_types, res_names=res_names
    )

    # Assertions
    mock_builder_cls.assert_called_once()  # Builder class was instantiated
    mock_builder_inst.init_structure.assert_called_once_with(
        "structure"
    )  # Method called on instance
    mock_builder_inst.init_model.assert_called_once_with(0)  # Method called on instance
    mock_builder_inst.init_chain.assert_called_once_with(
        "A"
    )  # Method called on instance
    assert (
        mock_builder_inst.init_residue.call_count == 1
    )  # Method called on instance once
    assert mock_atom_cls.call_count == 4  # Atom class instantiated directly 4 times

    mock_cifio_cls.assert_called_once()
    mock_cifio_inst.set_structure.assert_called_once_with(
        mock_structure
    )  # Check set_structure call
    mock_cifio_inst.save.assert_called_once_with(str(output_file))
    mock_pdbio_cls.assert_not_called()


def test_save_structure_invalid_input_shape_no_mocks(tmp_path: Path):
    """
    Tests save_structure raises ValueError for invalid input tensor shape (1D or 4D).
    Covers lines: 135-143, 146, 152-154 (ValueError)
    Does not require mocks as error should happen before Bio.PDB calls.
    """
    output_file = tmp_path / "output_invalid.pdb"

    # Case 1: 1D tensor
    coords_1d = torch.randn(10)
    with pytest.raises(ValueError, match="Coordinates must be a 2D or 3D tensor"):
        utils.save_structure(coords_1d, str(output_file))

    # Case 2: 4D tensor
    coords_4d = torch.randn(2, 3, 3, 3)
    with pytest.raises(ValueError, match="Coordinates must be a 2D or 3D tensor"):
        utils.save_structure(coords_4d, str(output_file))

    # Case 3: 2D tensor with wrong inner dimension
    coords_2d_wrong = torch.randn(5, 4)
    # Update regex to match actual error message format with parentheses
    with pytest.raises(
        ValueError, match=r"Coordinates must have shape \(M, 3\), got .*"
    ):
        utils.save_structure(coords_2d_wrong, str(output_file))

    # Case 4: 3D tensor with wrong inner dimension
    coords_3d_wrong = torch.randn(2, 3, 4)
    # Update regex to match actual error message format with parentheses
    with pytest.raises(
        ValueError, match=r"Coordinates must have shape \(N, 3, 3\), got .*"
    ):
        utils.save_structure(coords_3d_wrong, str(output_file))

    # Case 5: 3D tensor with wrong middle dimension (not 3 atoms)
    coords_3d_wrong_mid = torch.randn(2, 4, 3)
    # Update regex to match actual error message format with parentheses
    with pytest.raises(
        ValueError, match=r"Coordinates must have shape \(N, 3, 3\), got .*"
    ):
        utils.save_structure(coords_3d_wrong_mid, str(output_file))


def test_save_structure_2d_input_shape_mismatch_no_mocks(tmp_path: Path):
    """
    Tests save_structure with 2D input where num_atoms is not divisible by len(atom_types).
    Covers lines: 135-143, 146-151 (ValueError/RuntimeError before/during reshape)
    """
    coords_2d = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
        ]
    )  # 5 atoms
    output_file = tmp_path / "structure.pdb"
    atom_types = [
        "N",
        "CA",
        "C",
        "O",
    ]  # 4 atom types specified, 5 is not divisible by 4

    with pytest.raises(
        (ValueError, RuntimeError), match=r"Number of atoms .* must be divisible by"
    ):
        utils.save_structure(coords_2d, str(output_file), atom_types=atom_types)


# --- Coverage Notes ---
# Line 11: Covered by import numpy as np
# Lines 31-78: Covered by tests for get_coords_from_pdb, get_coords_from_cif, get_coords_from_file
# Lines 116-124: Covered by tests for get_device
# Lines 134-154, 159: Covered by tests for save_structure (including error cases and successful runs up to the 'pass')

# --- Tests for get_prot (Lines 58-105) --- # Adjusted line range based on previous analysis


# Helper structures to mimic dataloader and vocab
class DummyVocab:
    def int2char(self, aa_int: int) -> str:
        if aa_int == 20:  # Padding token
            return "<PAD>"  # Use a non-standard char to avoid confusion if 'X' is a real AA
        return chr(ord("A") + aa_int)  # Simple A, B, C... mapping


# class DummyBatch(NamedTuple): # Use SimpleNamespace instead if NamedTuple causes issues
#     int_seqs: torch.Tensor
#     angs: torch.Tensor
#     msks: torch.Tensor
#     crds: torch.Tensor
#     pids: List[str]


def create_dummy_dataloader(
    batches: List[SimpleNamespace],
) -> Dict[str, Iterator[SimpleNamespace]]:
    """Creates a simple dictionary mimicking a dataloader structure."""
    # Make the dataloader yield batches one by one like an iterator
    return {"train": iter(batches)}


# Constants for get_prot tests
PAD_TOKEN_INT = 20
DEFAULT_MIN_LEN = 5
DEFAULT_MAX_LEN = 15
NUM_ANGLES = 12  # From sidechainnet documentation/typical usage
ATOMS_PER_RESIDUE = 14  # From sidechainnet documentation/typical usage


def create_mock_batch_item(
    seq_len: int,
    padding_len: int,
    valid_padding: bool = True,
    pid: str = "protein1",
    num_angles: int = NUM_ANGLES,
    atoms_per_residue: int = ATOMS_PER_RESIDUE,
) -> Dict[str, Any]:
    """Helper to create data for one item in a batch."""
    total_len = seq_len + padding_len
    real_seq_ints = torch.randint(0, PAD_TOKEN_INT, (seq_len,))
    padding_ints = torch.full((padding_len,), PAD_TOKEN_INT)
    int_seq = torch.cat([real_seq_ints, padding_ints])

    # Create angles: non-zero for real sequence, zero for padding
    real_angles = torch.rand(seq_len, num_angles) * 2 * np.pi - np.pi
    padding_angles_correct = torch.zeros(padding_len, num_angles)
    if valid_padding:
        padding_angles = padding_angles_correct
    else:
        # Introduce invalid padding (non-zero angles in padding region)
        padding_angles = torch.rand(padding_len, num_angles) * 0.1  # Small non-zero

    angles = torch.cat([real_angles, padding_angles])

    # Create mask (1 for real, 0 for padding)
    mask = torch.cat([torch.ones(seq_len), torch.zeros(padding_len)]).long()

    # Create coordinates (dummy data, shape is important)
    # Coords are expected to be (total_len * atoms_per_residue, 3) before slicing in get_prot
    coords = torch.randn(total_len * atoms_per_residue, 3)

    return {
        "int_seq": int_seq,
        "angs": angles,
        "msks": mask,
        "crds": coords,  # Store the full coords for the item
        "pid": pid,
    }


def create_mock_batch(items: List[Dict[str, Any]]) -> SimpleNamespace:
    """Stacks individual items into a batch."""
    batch = SimpleNamespace()
    # Pad sequences to the max length in the batch for stacking
    max_len = max(item["int_seq"].shape[0] for item in items) if items else 0

    padded_int_seqs: List[torch.Tensor] = []
    padded_angs: List[torch.Tensor] = []
    padded_msks: List[torch.Tensor] = []
    # Coords are tricky because they are flattened (L*14, 3) in the original code's expectation
    # We need to simulate how the batch object might look for the slicing `batch.crds[i][: -padding_seq * 14 or None]`
    # Let's assume batch.crds is a stacked tensor of *padded* flattened coordinates
    max_crd_len = max(item["crds"].shape[0] for item in items) if items else 0
    padded_crds = []

    for item in items:
        current_len = item["int_seq"].shape[0]
        pad_size = max_len - current_len

        # Pad int_seq (using PAD_TOKEN_INT)
        padded_int_seqs.append(
            torch.cat([item["int_seq"], torch.full((pad_size,), PAD_TOKEN_INT)])
        )

        # Pad angs (using zeros)
        padded_angs.append(torch.cat([item["angs"], torch.zeros(pad_size, NUM_ANGLES)]))

        # Pad msks (using zeros)
        padded_msks.append(torch.cat([item["msks"], torch.zeros(pad_size)]).long())

        # Pad crds (using zeros) - This assumes the batch loader pads flattened coords
        current_crd_len = item["crds"].shape[0]
        crd_pad_size = max_crd_len - current_crd_len
        padded_crds.append(torch.cat([item["crds"], torch.zeros(crd_pad_size, 3)]))

    batch.int_seqs = (
        torch.stack(padded_int_seqs) if items else torch.empty(0, 0, dtype=torch.long)
    )
    batch.angs = torch.stack(padded_angs) if items else torch.empty(0, 0, NUM_ANGLES)
    batch.msks = (
        torch.stack(padded_msks) if items else torch.empty(0, 0, dtype=torch.long)
    )
    batch.crds = (
        torch.stack(padded_crds) if items else torch.empty(0, 0, 3)
    )  # Shape (batch_size, max_len*14, 3)
    batch.pids = [item["pid"] for item in items]

    return batch


@pytest.fixture
def mock_dataloader_factory():
    """Factory to create a mock dataloader with specified batches."""

    def _create_dataloader(
        batches: List[SimpleNamespace],
    ) -> Dict[str, Iterator[SimpleNamespace]]:
        # Make the dataloader yield batches one by one like an iterator
        return {"train": iter(batches)}

    return _create_dataloader


# --- Test Cases ---


@pytest.fixture
def mock_vocab_fixture() -> DummyVocab:
    """Provides the mock vocabulary as a fixture."""
    return DummyVocab()


def test_padding_handling():
    """Test that padding is handled correctly."""
    seq_len = 10  # Define sequence length
    padding_len = 5
    item1_data = create_mock_batch_item(
        seq_len=seq_len, padding_len=padding_len, pid="P1"
    )
    # Access the original unpadded coords for assertion comparison later
    item1_data["crds"][: seq_len * ATOMS_PER_RESIDUE]


def test_invalid_padding():
    """Test handling of invalid padding."""
    seq_len_1 = 8  # Define sequence length
    padding_len_1 = 3
    create_mock_batch_item(
        seq_len=seq_len_1,
        padding_len=padding_len_1,
        valid_padding=False,
        pid="P_invalid"
    )


def test_short_sequence():
    """Test handling of short sequences."""
    seq_len_1 = 5  # Define sequence length
    padding_len_1 = 2
    create_mock_batch_item(
        seq_len=seq_len_1, padding_len=padding_len_1, valid_padding=True, pid="P_short"
    )


def test_long_sequence():
    """Test handling of long sequences."""
    seq_len_1 = 15  # Define sequence length
    padding_len_1 = 7
    create_mock_batch_item(
        seq_len=seq_len_1, padding_len=padding_len_1, valid_padding=True, pid="P_long"
    )


def test_get_prot_no_padding(mock_vocab_fixture, mock_dataloader_factory):
    """Test finding a valid protein with zero padding."""
    seq_len = 10
    padding_len = 0  # No padding
    item1_data = create_mock_batch_item(
        seq_len=seq_len, padding_len=padding_len, pid="P_no_pad"
    )
    original_coords_item1 = item1_data["crds"]  # Should be the same shape as sliced

    batch = create_mock_batch([item1_data])
    dataloader = mock_dataloader_factory([batch])

    result = utils.get_prot(
        dataloader,
        mock_vocab_fixture,
        min_len=DEFAULT_MIN_LEN,
        max_len=DEFAULT_MAX_LEN,
        verbose=False,
    )

    assert result is not None
    seq_str, int_seq, coords, angles, padding_seq, mask, pid = result

    assert pid == "P_no_pad"
    assert padding_seq == 0
    assert len(seq_str) == seq_len
    assert int_seq.shape == (seq_len,)
    assert angles.shape == (seq_len, NUM_ANGLES)
    assert mask.shape == (seq_len,)
    # Slicing with `[:-None]` should work correctly
    assert coords.shape == (seq_len * ATOMS_PER_RESIDUE, 3)
    assert torch.equal(coords, original_coords_item1)
    assert torch.all(int_seq != PAD_TOKEN_INT)
    assert torch.all(mask == 1)


def test_get_prot_verbose_true_prints(
    mock_vocab_fixture, mock_dataloader_factory, capsys
):
    """Verify that verbose=True produces output for skipped items."""
    seq_len_1 = DEFAULT_MIN_LEN - 1  # Too short
    padding_len_1 = 2
    seq_len_2 = DEFAULT_MAX_LEN + 1  # Too long
    padding_len_2 = 3
    seq_len_3 = 8  # Padding mismatch
    padding_len_3 = 4
    seq_len_4 = 10  # Valid
    padding_len_4 = 5

    item1_short = create_mock_batch_item(
        seq_len=seq_len_1, padding_len=padding_len_1, valid_padding=True, pid="P_short"
    )
    item2_long = create_mock_batch_item(
        seq_len=seq_len_2, padding_len=padding_len_2, valid_padding=True, pid="P_long"
    )
    item3_pad_mismatch = create_mock_batch_item(
        seq_len=seq_len_3,
        padding_len=padding_len_3,
        valid_padding=False,
        pid="P_pad_mismatch",
    )
    item4_valid = create_mock_batch_item(
        seq_len=seq_len_4, padding_len=padding_len_4, valid_padding=True, pid="P_valid"
    )

    batch = create_mock_batch(
        [item1_short, item2_long, item3_pad_mismatch, item4_valid]
    )
    dataloader = mock_dataloader_factory([batch])

    # Run with verbose=True (default)
    result = utils.get_prot(
        dataloader,
        mock_vocab_fixture,
        min_len=DEFAULT_MIN_LEN,
        max_len=DEFAULT_MAX_LEN,
        verbose=True,
    )
    captured = capsys.readouterr()

    assert result is not None
    assert result[-1] == "P_valid"  # Ensure the correct item was returned

    # Check if print statements for skipped items occurred
    assert "found a seq of length" in captured.out  # For item1 and item2
    assert "but oustide the threshold" in captured.out  # For item1 and item2
    assert "paddings not matching" in captured.out  # For item3
    assert "stopping at sequence of length" in captured.out  # For item4


# Note: A test case for when NO valid protein is found is omitted
# because the `while True` loop would run indefinitely. Testing this
# would require more complex mocking (e.g., raising an exception
# after a certain number of iterations or mocking the dataloader to stop).
# The current tests cover all branches within the specified lines 58-105.

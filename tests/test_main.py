# tests/test_main.py
import os

import pytest

from rna_predict.main import (
    demo_compute_torsions_for_bprna,
    demo_run_input_embedding,
    demo_stream_bprna,
    show_full_bprna_structure,
)


@pytest.fixture
def mock_pdb_dir(tmp_path):
    """
    Fixture to create a temporary 'pdbs' directory to simulate local PDB files.
    This lets us test the torsion computation without requiring real PDB files.
    """
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    # Optionally, you could populate it with small dummy files if needed
    return pdb_dir


def test_demo_run_input_embedding(capfd):
    """
    Test that demo_run_input_embedding() runs without error and prints
    the expected output regarding the shape of the resulting embedding.
    """
    demo_run_input_embedding()
    captured = capfd.readouterr()
    # Check some known substring in the print
    assert (
        "Output single-token embedding shape:" in captured.out
    ), "Expected output shape info not found in stdout."


def test_demo_stream_bprna(capfd):
    """
    Test that demo_stream_bprna() runs without error and prints some rows
    from the streamed dataset. We don't test the actual data beyond verifying
    print statements, since it depends on external sources.
    """
    demo_stream_bprna()
    captured = capfd.readouterr()
    # We expect at least one line mentioning 'sequence length='
    assert "sequence length=" in captured.out, (
        "Expected 'sequence length=' info not found in stdout. "
        "Make sure streaming printed something."
    )


def test_show_full_bprna_structure(capfd):
    """
    Test that show_full_bprna_structure() prints column names and
    the full first sample in the dataset.
    """
    show_full_bprna_structure()
    captured = capfd.readouterr()
    # Check for mention of 'Column names:' and a non-empty set of keys
    assert (
        "Column names:" in captured.out
    ), "Expected 'Column names:' info not found in stdout."
    # We expect some sample data after that
    assert (
        "Full first sample:" in captured.out
    ), "Expected 'Full first sample:' info not found in stdout."


@pytest.mark.skipif(
    not os.path.exists("pdbs"),
    reason="No local 'pdbs' directory found. Provide PDB files or mock them for a full test.",
)
def test_demo_compute_torsions_for_bprna_with_real_pdbs(capfd):
    """
    Test that demo_compute_torsions_for_bprna() runs correctly when a real 'pdbs'
    folder exists. If the user has actual PDB files matching the dataset IDs,
    it will compute torsions. Otherwise, it will skip and print a message.
    """
    demo_compute_torsions_for_bprna()
    captured = capfd.readouterr()
    # We at least expect the initial message to appear
    assert "Computing torsion angles for a few entries in bprna-spot..." in captured.out


def test_demo_compute_torsions_for_bprna_mocked_pdbs(capfd, mock_pdb_dir):
    """
    Test that demo_compute_torsions_for_bprna() gracefully reports 'No local PDB file found'
    if the user does not have matching PDB files. We point it to a temporary folder
    that won't contain any real PDB files matching the dataset IDs.
    """
    # Temporarily change the working directory so that it sees our mock 'pdbs'.
    old_cwd = os.getcwd()
    try:
        os.chdir(mock_pdb_dir.parent)  # parent of 'pdbs'
        demo_compute_torsions_for_bprna()
        captured = capfd.readouterr()
        # The code prints a skip message if pdb files aren't found
        assert (
            "No local PDB file found at" in captured.out
        ), "Expected a skip message when no real PDB files exist."
    finally:
        os.chdir(old_cwd)

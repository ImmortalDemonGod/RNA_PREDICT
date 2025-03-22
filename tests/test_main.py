# tests/test_main.py

import pytest

from rna_predict.main import (
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

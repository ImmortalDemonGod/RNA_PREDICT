import os
import pytest
import torch
from rna_predict.pipeline.run_stageA import (
    build_predictor,
    main,
    run_stageA,
    visualize_with_varna
)
from rna_predict.pipeline.stageA.rfold_predictor import StageARFoldPredictor

@pytest.fixture
def temp_checkpoint_folder(tmp_path) -> str:
    """
    Creates a temp checkpoints folder with an empty 'RNAStralign_trainset_pretrained.pth' file.
    """
    folder = tmp_path / "checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "RNAStralign_trainset_pretrained.pth").touch()  # empty checkpoint file
    return str(folder)

def test_build_predictor_valid(temp_checkpoint_folder: str):
    """
    Test that build_predictor returns a StageARFoldPredictor instance for a valid checkpoint folder.
    """
    config = {"num_hidden": 128, "dropout": 0.3}
    device = torch.device("cpu")
    predictor = build_predictor(temp_checkpoint_folder, config, device)
    assert isinstance(predictor, StageARFoldPredictor)
    # Basic smoke test calling predict_adjacency
    adj = predictor.predict_adjacency("ACGU")
    assert adj.shape == (4, 4)

def test_run_stageA(temp_checkpoint_folder: str):
    """
    Test that run_stageA uses the predictor to produce a correct adjacency shape.
    """
    config = {"num_hidden": 128, "dropout": 0.3}
    device = torch.device("cpu")
    predictor = build_predictor(temp_checkpoint_folder, config, device)

    seq = "ACGUACGU"
    adjacency = run_stageA(seq, predictor)
    assert adjacency.shape == (8, 8)
    # Check that it's presumably 0 or 1. Implementation might vary, so we do a sanity check.
    assert (adjacency >= 0).all() and (adjacency <= 1).all()

@pytest.mark.parametrize("seq", ["A", "ACG", "ACGUACGUA"])
def test_predictor_different_sequences(temp_checkpoint_folder: str, seq: str):
    """
    Check adjacency sizes scale with the input sequence length for multiple test sequences.
    """
    config = {"num_hidden": 128, "dropout": 0.3}
    device = torch.device("cpu")
    predictor = build_predictor(temp_checkpoint_folder, config, device)
    adjacency = run_stageA(seq, predictor)
    assert adjacency.shape == (len(seq), len(seq))

def test_visualize_with_varna_missing_files(tmp_path):
    """
    If the CT file or the jar is missing, the function should warn and return gracefully.
    """
    ct_file = str(tmp_path / "non_existent.ct")
    jar_path = str(tmp_path / "non_existent.jar")
    out_png = str(tmp_path / "out.png")

    # We expect the function to not raise an error but warn and return
    visualize_with_varna(ct_file, jar_path, out_png)
    assert not os.path.exists(out_png), "No output image should be generated"

def test_main_end_to_end(temp_checkpoint_folder, monkeypatch):
    """
    Run the main() function in an environment that has a mock checkpoint folder.
    We don't do extra mocking, so the predictor is real.
    The jar won't exist, so visualize_with_varna will skip gracefully.
    """
    # Temporarily rename the real folder so main won't fail
    # Then put our temp folder in its place
    real_folder = "RFold/checkpoints"
    if os.path.exists(real_folder):
        backup_folder = "RFold/checkpoints_backup"
        os.rename(real_folder, backup_folder)
    else:
        backup_folder = None

    os.makedirs("RFold", exist_ok=True)
    os.symlink(temp_checkpoint_folder, real_folder)  # link or rename
    try:
        main()
        # We can check that the adjacency message was printed, or that a 'test_seq.ct' was created
        # But we let it pass as an integration test
        assert os.path.exists("test_seq.ct"), "main() should have written a test_seq.ct"
    finally:
        # Clean up
        if os.path.islink(real_folder):
            os.unlink(real_folder)
        if backup_folder:
            os.rename(backup_folder, real_folder)
import subprocess
import os
import tempfile
import shutil
import pytest

@pytest.mark.slow
def test_train_with_angle_supervision(tmp_path):
    """
    Integration test: Runs the full training pipeline with angle supervision enabled.
    Verifies that angles_true are loaded, batched, and used in loss computation.
    Checks logs for correct loss debug output and checkpointing.
    """
    # Paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    script = os.path.join(project_root, "rna_predict/training/train.py")
    index_csv = os.path.join(project_root, "rna_predict/dataset/examples/kaggle_minimal_index.csv")
    log_file = tmp_path / "dev_run_output.txt"

    # Build command (use uv as CLI, not as a Python module)
    cmd = [
        "uv", "run", script,
        "model.stageA.debug_logging=False",
        "model.stageB.torsion_bert.debug_logging=False",
        "model.stageB.pairformer.debug_logging=False",
        "model.stageC.debug_logging=False",
        "model.stageD.debug_logging=False",
        "model.stageD.diffusion.diffusion.debug_logging=True",
        "device=cpu",
        f"data.index_csv={index_csv}",
        "data.num_workers=0"
    ]

    # Run and capture output
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=project_root  # Hydra config path is correct only from project root
    )
    log_file.write_text(result.stdout)

    # Assert process success
    assert result.returncode == 0, f"Training failed! Log:\n{result.stdout}"
    # Check for debug lines indicating angle loss computation
    assert "[LOSS-DEBUG] loss_angle after division" in result.stdout, "Angle loss computation not found in output"
    assert "Checkpoints saved to" in result.stdout, "Checkpoint not saved"

    # Optionally: parse checkpoint directory and verify file exists
    for line in result.stdout.splitlines():
        if line.startswith("[DEBUG] Checkpoints saved to:"):
            ckpt_dir = line.split(":", 1)[-1].strip()
            if os.path.isdir(ckpt_dir):
                ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
                assert ckpt_files, f"No checkpoint file found in {ckpt_dir}"
            break
    else:
        pytest.fail("Checkpoint directory not found in logs")

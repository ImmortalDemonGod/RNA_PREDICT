import os
import tempfile
import torch
import pytest
from subprocess import run

# Reference project root, then dataset directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
dataset_dir = os.path.join(project_root, 'rna_predict', 'dataset')
EXAMPLES_DIR = os.path.join(dataset_dir, 'examples')
# Path to compute_ground_truth_angles script in codebase
SCRIPT = os.path.join(dataset_dir, 'preprocessing', 'compute_ground_truth_angles.py')

@pytest.mark.parametrize("filename,chain_id", [
    ("1a34_1_B.cif", "B"),
    ("1a9n_1_R.cif", "R"),
    ("RNA_NET_1a9n_1_Q.cif", "Q"),
    ("synthetic_cppc_0000001.pdb", "B"),  # Use correct chain 'B' for this file
])
def test_compute_ground_truth_angles_cli(filename, chain_id):
    """
    Tests the compute_ground_truth_angles.py CLI for correct output tensor file creation.
    
    Runs the script with a given input file and chain ID, then verifies that output files
    ending with '_angles.pt' are generated. Asserts that each output tensor is 2D with 7
    columns and contains at least one non-NaN angle value.
    """
    with tempfile.TemporaryDirectory() as outdir:
        print(f"[DEBUG] Using SCRIPT path: {SCRIPT}")
        print(f"[DEBUG] Using EXAMPLES_DIR: {EXAMPLES_DIR}")
        result = run([
            "/Users/tomriddle1/.local/bin/uv", "run", SCRIPT,
            "--input_dir", EXAMPLES_DIR,
            "--output_dir", outdir,
            "--chain_id", chain_id,
            "--backend", "mdanalysis"
        ], capture_output=True, text=True)
        if result.returncode != 0:
            print("\n[DEBUG] CLI STDOUT:\n", result.stdout)
            print("[DEBUG] CLI STDERR:\n", result.stderr)
        assert result.returncode == 0, f"CLI failed: {result.stderr}\nSTDOUT: {result.stdout}"
        out_files = [f for f in os.listdir(outdir) if f.endswith("_angles.pt")]
        assert len(out_files) > 0, "No output files generated"
        for f in out_files:
            t = torch.load(os.path.join(outdir, f))
            assert t.ndim == 2 and t.shape[1] == 7
            # At least one angle not nan
            assert torch.any(~torch.isnan(t)), f"All angles nan in {f}"

@pytest.mark.parametrize("filename,chain_id,angle_set,expected_dim", [
    ("1a34_1_B.cif", "B", "canonical", 7),
    ("1a34_1_B.cif", "B", "full", 14),
    ("1a9n_1_R.cif", "R", "canonical", 7),
    ("1a9n_1_R.cif", "R", "full", 14),
    ("RNA_NET_1a9n_1_Q.cif", "Q", "canonical", 7),
    ("RNA_NET_1a9n_1_Q.cif", "Q", "full", 14),
    ("synthetic_cppc_0000001.pdb", "B", "canonical", 7),
    ("synthetic_cppc_0000001.pdb", "B", "full", 14),
])
def test_compute_ground_truth_angles_cli_parametrized(filename, chain_id, angle_set, expected_dim):
    """
    Tests the CLI of compute_ground_truth_angles.py with different angle sets and validates output tensors.
    
    Runs the script with the specified input file, chain ID, and angle set, then checks that output files are generated. Verifies that each output tensor has the expected number of columns and contains at least one non-NaN value.
    """
    with tempfile.TemporaryDirectory() as outdir:
        result = run([
            "/Users/tomriddle1/.local/bin/uv", "run", SCRIPT,
            "--input_dir", EXAMPLES_DIR,
            "--output_dir", outdir,
            "--chain_id", chain_id,
            "--backend", "mdanalysis",
            "--angle_set", angle_set
        ], capture_output=True, text=True)
        assert result.returncode == 0, f"CLI failed: {result.stderr}\nSTDOUT: {result.stdout}"
        out_files = [f for f in os.listdir(outdir) if f.endswith("_angles.pt")]
        assert len(out_files) > 0, "No output files generated"
        for f in out_files:
            t = torch.load(os.path.join(outdir, f))
            assert t.ndim == 2 and t.shape[1] == expected_dim
            # At least one angle not nan
            assert torch.any(~torch.isnan(t)), f"All angles nan in {f} ({angle_set})"

def test_main_empty_input(tmp_path):
    # Prepare empty input dir
    """
    Tests the main function with an empty input directory to ensure no output files are created and no errors occur.
    
    Creates empty input and output directories, simulates CLI arguments, and verifies that the output directory remains empty after running the main function.
    """
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    import sys
    sys_argv_backup = sys.argv[:]
    sys.argv = ["script", "--input_dir", str(input_dir), "--output_dir", str(output_dir)]
    from rna_predict.dataset.preprocessing import compute_ground_truth_angles
    compute_ground_truth_angles.main()
    sys.argv = sys_argv_backup
    # Should not raise, output dir should be empty
    assert not any(output_dir.iterdir())

def test_main_invalid_args(monkeypatch):
    """
    Tests that the main function exits with an error when required CLI arguments are missing.
    
    This test simulates invoking the script with insufficient command-line arguments and asserts
    that a SystemExit exception is raised.
    """
    import sys
    sys_argv_backup = sys.argv[:]
    sys.argv = ["script"] # missing required args
    from rna_predict.dataset.preprocessing import compute_ground_truth_angles
    try:
        with pytest.raises(SystemExit):
            compute_ground_truth_angles.main()
    finally:
        sys.argv = sys_argv_backup

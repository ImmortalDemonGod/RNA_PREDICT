import subprocess
import sys # To ensure using the correct python executable
import pytest # Assuming pytest is the test runner

# Define the path to the script relative to the project root
SCRIPT_PATH = "rna_predict/pipeline/stageA/run_stageA.py"

@pytest.mark.integration
def test_run_stageA_default_config():
    """
    Smoke test: Run run_stageA.py with default Hydra config.
    Checks if the script executes without raising an error (return code 0).
    """
    # Run the script as a module to ensure it can find the rna_predict package
    command = [sys.executable, "-m", "rna_predict.pipeline.stageA.run_stageA"]

    print(f"\nRunning command: {' '.join(command)}") # Log the command being run

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False # Don't raise exception on non-zero exit
    )

    print(f"Return Code: {result.returncode}")
    print(f"Stdout:\n{result.stdout}")
    print(f"Stderr:\n{result.stderr}")

    assert result.returncode == 0, f"Script failed with stderr:\n{result.stderr}"
    # Optionally, add more specific checks on stdout/stderr if needed
    # e.g., assert "Adjacency shape" in result.stdout


@pytest.mark.integration
def test_run_stageA_with_override():
    """
    Test run_stageA.py with a command-line override.
    Checks if the script runs successfully and if the override is reflected
    in the script's output (assuming the script logs the loaded config).
    """
    # The config is under 'model' as shown by --help
    override_key = "model.dropout"
    override_value = "0.1" # Use string for subprocess arg
    override_arg = f"{override_key}={override_value}"
    # Run the script as a module to ensure it can find the rna_predict package
    command = [sys.executable, "-m", "rna_predict.pipeline.stageA.run_stageA", override_arg]

    print(f"\nRunning command: {' '.join(command)}")

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False
    )

    print(f"Return Code: {result.returncode}")
    print(f"Stdout:\n{result.stdout}")
    print(f"Stderr:\n{result.stderr}")

    assert result.returncode == 0, f"Script failed with override '{override_arg}'. Stderr:\n{result.stderr}"

    # Check if the override value appears in the logged config output
    # Adjust check based on actual logging format in run_stageA.py
    # run_stageA.py prints the stage_cfg object, which uses OmegaConf formatting.
    # The output is in Python dict format, so we need to check for 'dropout': 0.1
    expected_log_fragment = f"'dropout': {float(override_value):.1f}"
    # Combine stdout and stderr for checking as Hydra/logging might write to either
    output_to_check = result.stdout + result.stderr
    assert expected_log_fragment in output_to_check, \
        f"Override value '{override_value}' for key '{override_key}' not found in script output (stdout/stderr). Output:\n{output_to_check}"

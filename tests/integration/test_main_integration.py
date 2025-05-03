# tests/test_main_integration.py
import subprocess
import sys
import pytest


@pytest.mark.integration
def test_main_py_subprocess_execution() -> None:
    """
    Test that running rna_predict/main.py as a subprocess covers the
    'if __name__ == "__main__":' lines and prints expected output.

    This ensures we exercise the script's entry point logic and increase coverage
    for lines 101–106 in rna_predict/main.py.
    """
    # Execute main.py in a separate process, capturing stdout/stderr
    result = subprocess.run(
        [sys.executable, "rna_predict/main.py"], capture_output=True, text=True
    )
    # The script should exit with a success code (0)
    assert (
        result.returncode == 0
    ), f"main.py returned non-zero exit code. stderr: {result.stderr}"

    # Check that the script printed the lines from __main__ block
    out = result.stdout
    assert (
        "Running demo_run_input_embedding()..." in out
    ), "Expected script to print the embedding demo start message."
    assert (
        "Now streaming the bprna-spot dataset..." in out
    ), "Expected script to print the bprna streaming message."
    assert (
        "Showing the full dataset structure for the first row..." in out
    ), "Expected script to print the bprna structure message."

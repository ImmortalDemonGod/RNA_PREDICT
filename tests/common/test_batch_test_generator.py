# test_batch_test_generator.py
"""
Comprehensive pytest-based test suite for batch_test_generator.py.

This file tests:
1. The `process_folder` function, which recursively processes Python files
   and generates wrapped test files unless they already exist or are in
   the output directory.
2. The `main` function, which handles command-line arguments and invokes
   `process_folder`.

Test Cases:
- Normal operation with multiple *.py files.
- Skipping already processed files or files within the output directory.
- Handling failure from run_test_generation.
- Edge cases: empty folder, invalid folder paths, insufficient CLI arguments, etc.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Import the module under test using importlib
import importlib.util
import os

# Get the absolute path to the batch_test_generator.py file
script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                          "scripts", "test_utils", "batch_test_generator.py")

# Load the module dynamically
spec = importlib.util.spec_from_file_location("batch_test_generator", script_path)
batch_test_generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch_test_generator)

#################
# FIXTURES
#################


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """
    Creates a temporary directory structure for testing.
    Yields a pathlib.Path to the directory.
    """
    # We can place some .py files in a subfolder
    test_subdir = tmp_path / "subdir"
    test_subdir.mkdir(parents=True, exist_ok=True)

    # Create a couple of Python files
    (test_subdir / "script1.py").write_text("# script1 content")
    (test_subdir / "script2.py").write_text("# script2 content")

    # Create a non-Python file
    (test_subdir / "notes.txt").write_text("some notes")

    yield tmp_path
    # Cleanup handled by pytest tmp_path fixture automatically.


#################
# TESTS: process_folder
#################


def test_process_folder_creates_wrapped_files(temp_dir: Path) -> None:
    """
    Test that process_folder correctly processes *.py files and
    creates 'test_wrapped_<filename>.md' in the output directory
    for each processed file.
    """
    # Arrange
    folder_path = temp_dir / "subdir"
    output_dir = temp_dir / "generated_tests"
    output_dir.mkdir(exist_ok=True)

    # We'll patch `run_test_generation` to return True to simulate successful test generation.
    with patch(
        "scripts.test_utils.batch_test_generator.run_test_generation",
        return_value=True,
    ) as mock_run_gen:
        # Act
        batch_test_generator.process_folder(
            folder_path=folder_path, output_dir=output_dir
        )

    # Assert
    # We had 2 py files in subdir, so we expect 2 calls to run_test_generation
    assert mock_run_gen.call_count == 2

    # Check that test_wrapped_<filename>.md files exist
    wrapped1 = output_dir / "test_wrapped_script1.md"
    wrapped2 = output_dir / "test_wrapped_script2.md"
    assert wrapped1.exists(), "Wrapped test file for script1 not created."
    assert wrapped2.exists(), "Wrapped test file for script2 not created."


def test_process_folder_skips_existing_wrapped_files(temp_dir: Path) -> None:
    """
    Test that process_folder skips processing a file if the wrapped test file
    already exists in the output directory.
    """
    # Arrange
    folder_path = temp_dir / "subdir"
    output_dir = temp_dir / "generated_tests"
    output_dir.mkdir(exist_ok=True)

    # Pre-create a wrapped file for 'script1.py' to simulate "already processed"
    pre_wrapped = output_dir / "test_wrapped_script1.md"
    pre_wrapped.write_text("# pre-existing test")

    with patch(
        "scripts.test_utils.batch_test_generator.run_test_generation",
        return_value=True,
    ) as mock_run_gen:
        # Act
        batch_test_generator.process_folder(
            folder_path=folder_path, output_dir=output_dir
        )

    # Assert
    # Only 'script2.py' should be processed because 'script1.py' is considered "already processed".
    assert (
        mock_run_gen.call_count == 1
    ), "run_test_generation should only have been called for script2.py."

    # We expect 'test_wrapped_script2.md' to exist; 'test_wrapped_script1.md' was skipped.
    assert (output_dir / "test_wrapped_script2.md").exists()
    assert (output_dir / "test_wrapped_script1.md").read_text() == "# pre-existing test"


def test_process_folder_skips_files_in_output_dir(temp_dir: Path) -> None:
    """
    Test that process_folder skips any *.py files located inside the output directory itself.
    """
    # Arrange
    output_dir = temp_dir / "generated_tests"
    output_dir.mkdir(exist_ok=True)

    # Create a *.py file in output_dir
    (output_dir / "insider.py").write_text("# insider file")

    # Make sure the test subdir exists to ensure we're not testing an empty directory
    test_subdir = temp_dir / "subdir"
    test_subdir.mkdir(exist_ok=True)

    # Create Python files in the subdir
    (test_subdir / "script1.py").write_text("# script1 content")
    (test_subdir / "script2.py").write_text("# script2 content")

    with patch(
        "scripts.test_utils.batch_test_generator.run_test_generation",
        return_value=True,
    ) as mock_run_gen:
        # Act - use a different root folder than temp_dir to avoid processing the subdir files
        batch_test_generator.process_folder(
            folder_path=output_dir, output_dir=output_dir
        )

    # Assert
    # The insider.py is inside output_dir, so it should be skipped.
    mock_run_gen.assert_not_called()


def test_process_folder_failure_handling(temp_dir: Path, capsys) -> None:
    """
    Test that process_folder prints a failure message when run_test_generation returns False.
    """
    # Arrange
    folder_path = temp_dir / "subdir"
    output_dir = temp_dir / "generated_tests"
    output_dir.mkdir(exist_ok=True)

    with patch(
        "scripts.test_utils.batch_test_generator.run_test_generation",
        return_value=False,
    ) as mock_run_gen:
        # Act
        batch_test_generator.process_folder(
            folder_path=folder_path, output_dir=output_dir
        )

    # Assert
    captured = capsys.readouterr()
    # We expect a printout about "Failed to generate tests" for each .py file.
    assert "Failed to generate tests for" in captured.out
    # There are 2 .py files in subdir
    assert mock_run_gen.call_count == 2


def test_process_folder_empty_directory(temp_dir: Path, capsys) -> None:
    """
    Test that process_folder prints nothing and processes no files if none are found.
    """
    # Arrange: an empty directory
    empty_subdir = temp_dir / "empty"
    empty_subdir.mkdir(exist_ok=True)
    output_dir = temp_dir / "generated_tests"
    output_dir.mkdir(exist_ok=True)

    with patch(
        "scripts.test_utils.batch_test_generator.run_test_generation",
        return_value=True,
    ) as mock_run_gen:
        batch_test_generator.process_folder(
            folder_path=empty_subdir, output_dir=output_dir
        )

    # Assert
    captured = capsys.readouterr()
    # No python files, so no lines about "Processing" or "Failed".
    assert "Processing" not in captured.out
    assert "Failed" not in captured.out
    mock_run_gen.assert_not_called()


#################
# TESTS: main
#################


def test_main_usage_missing_argument(capsys) -> None:
    """
    Test that main prints usage and exits when no arguments are provided.
    """
    test_argv = ["batch_test_generator.py"]  # no folder path argument
    with patch.object(sys, "argv", test_argv), pytest.raises(SystemExit) as exit_info:
        batch_test_generator.main()
    captured = capsys.readouterr()
    assert "Usage: python batch_test_generator.py <folder_path>" in captured.out
    assert exit_info.value.code == 1


def test_main_invalid_folder(temp_dir: Path, capsys) -> None:
    """
    Test that main prints an error and exits when an invalid folder path is provided.
    """
    # Provide a non-existing path
    invalid_path = str(temp_dir / "not_a_dir")
    test_argv = ["batch_test_generator.py", invalid_path]

    with patch.object(sys, "argv", test_argv), pytest.raises(SystemExit) as exit_info:
        batch_test_generator.main()

    captured = capsys.readouterr()
    assert f"Invalid folder path: {invalid_path}" in captured.out
    assert exit_info.value.code == 1


def test_main_happy_path(temp_dir: Path) -> None:
    """
    Test that main calls process_folder with correct arguments and does not exit
    when provided a valid folder path.
    """
    # We'll create a valid folder path and patch `process_folder` to confirm the call.
    test_argv = ["batch_test_generator.py", str(temp_dir)]
    with (
        patch.object(sys, "argv", test_argv),
        patch("scripts.test_utils.batch_test_generator.process_folder") as mock_pf,
    ):
        batch_test_generator.main()

    # The function should be called exactly once with the valid path and
    # the default output_dir set to "generated_tests" in the same parent folder.
    # However, main sets output_dir to "generated_tests" in the current directory,
    # so we check just the call with folder_path = temp_dir, output_dir = Path("generated_tests").
    mock_pf.assert_called_once()
    args, kwargs = mock_pf.call_args
    assert args[0] == Path(temp_dir)
    assert str(args[1]) == "generated_tests"

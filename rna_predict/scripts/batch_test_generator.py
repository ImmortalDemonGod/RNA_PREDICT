import sys
from pathlib import Path

from rna_predict.scripts.hypot_test_gen import (
    run_test_generation,
)  # Adjust import if necessary


def process_folder(folder_path: Path, output_dir: Path):
    """
    Recursively processes all Python files in the folder.
    Skips files if the corresponding test_wrapped file already exists in output_dir.
    """
    # Iterate recursively over Python files
    for py_file in folder_path.rglob("*.py"):
        # Skip files inside the output directory to avoid reprocessing generated files
        if output_dir in py_file.parents:
            continue

        # Construct the expected wrapped test file name based on the Python file stem
        wrapped_file = output_dir / f"test_wrapped_{py_file.stem}.md"
        if wrapped_file.exists():
            print(f"Skipping {py_file} (already processed)")
            continue

        print(f"Processing {py_file}")
        success = run_test_generation(py_file)
        if not success:
            print(f"Failed to generate tests for {py_file}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python batch_test_generator.py <folder_path>")
        sys.exit(1)

    folder_path = Path(sys.argv[1])
    if not folder_path.is_dir():
        print(f"Invalid folder path: {folder_path}")
        sys.exit(1)

    output_dir = Path("generated_tests")
    output_dir.mkdir(exist_ok=True)

    process_folder(folder_path, output_dir)


if __name__ == "__main__":
    main()

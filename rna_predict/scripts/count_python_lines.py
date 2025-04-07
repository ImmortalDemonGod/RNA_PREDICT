import os
from typing import List


def count_lines_in_file(filepath: str) -> int:
    """Count the total number of lines in a file.

    Args:
        filepath (str): Path to the file to count lines in

    Returns:
        int: Number of lines in the file, or 0 if there was an error
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as file:
            return sum(1 for line in file)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return 0


def find_python_files(directory: str) -> List[str]:
    """Recursively find all .py files in the given directory, excluding specific directories.

    Args:
        directory (str): Directory to search for Python files

    Returns:
        List[str]: List of paths to Python files
    """
    python_files = []
    # List of directory names to exclude from the search.
    exclude_dirs = {".venv", "mkdocs-env", "env", "venv"}

    for root, dirs, files in os.walk(directory):
        # Modify dirs in-place to skip excluded directories.
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files


def main(directory: str = ".") -> None:
    """Count lines in all Python files in a directory and save results.

    Args:
        directory (str): Directory to analyze. Defaults to current directory.
    """
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory")
        return

    # Gather all Python files while skipping excluded directories.
    python_files = find_python_files(directory)
    if not python_files:
        print(f"No Python files found in {directory}")
        return

    file_line_counts = []

    # Count lines for each Python file.
    for filepath in python_files:
        line_count = count_lines_in_file(filepath)
        file_line_counts.append((filepath, line_count))

    # Sort the results in descending order by line count.
    file_line_counts.sort(key=lambda x: x[1], reverse=True)

    # Write the sorted results to an output file.
    output_filename = "python_file_line_counts.txt"
    try:
        with open(output_filename, "w", encoding="utf-8") as output_file:
            for filepath, count in file_line_counts:
                output_file.write(f"{filepath}: {count} lines\n")
    except Exception as e:
        print(f"Error writing to {output_filename}: {e}")
        return

    # Print the results to the console.
    for filepath, count in file_line_counts:
        print(f"{filepath}: {count} lines")

    print(f"\nResults saved to {output_filename}")


if __name__ == "__main__":
    import sys

    # Use the provided directory or default to the current directory.
    target_directory = sys.argv[1] if len(sys.argv) > 1 else "."
    main(target_directory)

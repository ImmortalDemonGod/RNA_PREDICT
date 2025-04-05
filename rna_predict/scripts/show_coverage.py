#!/usr/bin/env python3
import subprocess
import sys
import argparse
import os
import re
from typing import Tuple, List, Optional

def run_command(command, check=True, capture_output=False, text=True):
    """Runs a command using subprocess and handles errors."""
    print(f"Running command: {' '.join(command)}")
    try:
        result = subprocess.run(
            command,
            check=check,
            capture_output=capture_output,
            text=text
            # stderr is handled implicitly by capture_output=True
            # For capture_output=False, stderr defaults to None (inherits), which is fine.
            # If we wanted explicit control for False case: stderr=sys.stderr if not capture_output else None
        )
        if capture_output and result.returncode != 0:
             # Print stderr if capture_output was true and there was an error
            print(f"Error running command: {' '.join(command)}", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            sys.exit(result.returncode)
        elif result.returncode != 0:
             # If not capturing output, CalledProcessError would have been raised if check=True
             # This handles the case where check=False might be used later, though default is True
             print(f"Command failed: {' '.join(command)}", file=sys.stderr)
             sys.exit(result.returncode)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(command)}", file=sys.stderr)
        # stderr is already printed by subprocess on failure when capture_output=False
        if capture_output:
             print(e.stderr, file=sys.stderr) # Ensure stderr is printed if captured
        sys.exit(e.returncode)
    except FileNotFoundError:
        print(f"Error: Command not found - {command[0]}. Is it installed and in PATH?", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


def parse_coverage_report(report_output: str):
    """Parses the coverage report into header, separator, data lines, and total line."""
    lines = report_output.strip().split('\n')
    if len(lines) < 3: # Need at least header, separator, and total
        print("Warning: Coverage report format unexpected.", file=sys.stderr)
        return None, None, [], None # Return structure indicating failure

    header = lines[0]
    separator = lines[1]
    total_line = None
    data_lines = []

    # Find the TOTAL line and separate data lines
    for line in lines[2:]:
        if line.strip().startswith("TOTAL"):
            total_line = line
            break
        else:
            data_lines.append(line)

    if not total_line: # If TOTAL line wasn't found
         print("Warning: TOTAL line not found in coverage report.", file=sys.stderr)
         # Still return the parsed parts, the caller can decide how to handle it

    return header, separator, data_lines, total_line


def filter_coverage_report(report_output: str, file_pattern: str) -> str:
    """Filters the coverage report output based on a file pattern."""
    header, separator, data_lines, total_line = parse_coverage_report(report_output)

    if header is None:  # Parsing failed
        return report_output  # Return original

    # Filter data lines based on the pattern matching the first column (filename)
    filtered_lines = []
    for line in data_lines:
        parts = line.split()
        if parts and file_pattern in parts[0]:
            filtered_lines.append(line)  # 4 spaces indent

    if not filtered_lines:
        no_match_msg = f"(No lines matched the pattern '{file_pattern}')"
        if total_line:
            # 8 spaces indent for the return statement
            return f"{header}\n{separator}\n{no_match_msg}\n{total_line}"
        else:
            # 8 spaces indent for the return statement
            return f"{header}\n{separator}\n{no_match_msg}"

    # Construct the filtered report
    # 4 spaces indent for this block
    filtered_report = f"{header}\n{separator}\n" + "\n".join(filtered_lines)
    if total_line:
        # 8 spaces indent for the modification
        filtered_report += f"\n{total_line}"

    # 4 spaces indent for the final return
    return filtered_report


def get_least_covered_report(report_output: str) -> Tuple[str, Optional[str]]:
    """
    Finds the file with the least coverage (Stmts > 0) and returns a minimal report
    string and the path of the least covered file.
    """
    header, separator, data_lines, total_line = parse_coverage_report(report_output)

    if header is None: # Parsing failed
        return "Error: Could not parse coverage report.", None

    least_covered_line = None
    least_covered_path = None
    min_coverage = 101 # Start higher than possible percentage

    # Regex to find the coverage percentage (last number followed by '%')
    # Handles potential missing columns by looking from the end
    coverage_regex = re.compile(r'(\d+)\s*%\s*$')

    for line in data_lines:
        parts = line.split()
        if len(parts) < 4: # Need at least Name, Stmts, Miss, Cover
            continue

        try:
            # Extract filename (first column)
            file_path = parts[0]

            # Extract statements count (second column)
            stmts = int(parts[1])
            if stmts <= 0: # Ignore files with no statements
                continue

            # Extract coverage percentage using regex from the end of the line
            match = coverage_regex.search(line)
            if not match:
                print(f"Warning: Could not parse coverage percentage for line: {line}", file=sys.stderr)
                continue # Skip lines where coverage can't be parsed

            coverage = int(match.group(1))

            if coverage < min_coverage:
                min_coverage = coverage
                least_covered_line = line
                least_covered_path = file_path # Store the path

        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse line: {line} - {e}", file=sys.stderr)
            continue # Skip lines that don't parse correctly

    if least_covered_line is None:
        no_valid_lines_msg = "(No files with statements > 0 found in the report)"
        if total_line:
            minimal_report = f"{header}\n{separator}\n{no_valid_lines_msg}\n{total_line}"
        else:
            minimal_report = f"{header}\n{separator}\n{no_valid_lines_msg}"
        return minimal_report, None # Return None for path

    # Construct the minimal report
    minimal_report = f"{header}\n{separator}\n{least_covered_line}"
    if total_line:
        minimal_report += f"\n{total_line}"

    return minimal_report, least_covered_path # Return report and path


def main():
    parser = argparse.ArgumentParser(
        description="Run pytest with coverage and show the report. Can filter by file path/pattern or show the least covered file."
    )
    # Group for mutually exclusive arguments
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "file_pattern",
        nargs="?", # Makes it optional
        type=str,
        default=None,
        help="Optional file path or pattern to filter the coverage report.",
    )
    group.add_argument(
        "--least-covered",
        action="store_true",
        help="Show only the header, total, and the file with the lowest coverage percentage (with >0 statements).",
    )
    args = parser.parse_args()

    # Define commands
    # Using -n auto for parallel, requires combine. Using tests/ as requested.
    pytest_command = ["pytest", "--cov=rna_predict", "--cov-branch", "-n", "auto", "tests/"]
    combine_command = ["coverage", "combine"]
    report_command = ["coverage", "report"]

    # Run tests under coverage
    # Run without capturing output so user sees pytest progress/results directly
    run_command(pytest_command, capture_output=False)

    # Combine coverage data from parallel runs (necessary if -n auto was used)
    # Check if .coverage files exist before combining
    if any(fname.startswith('.coverage.') for fname in os.listdir('.')):
        run_command(combine_command, capture_output=False)
    else:
        # If no parallel files, maybe it ran sequentially or failed early.
        # Check if the main .coverage file exists.
        if not os.path.exists(".coverage"):
             print("Warning: No .coverage data file found after pytest run.", file=sys.stderr)
             # Attempt to run report anyway, it might use older data or fail clearly
        else:
             print("No parallel coverage files found, skipping 'coverage combine'.")


    # Generate and capture the text report
    report_result = run_command(report_command, capture_output=True)
    report_output = report_result.stdout

    # Print the report based on arguments
    if args.least_covered:
        print("\n--- Least Covered File Report ---")
        least_covered_report = get_least_covered_report(report_output)
        print(least_covered_report)
    elif args.file_pattern:
        print("\n--- Filtered Coverage Report ---")
        filtered_report = filter_coverage_report(report_output, args.file_pattern)
        print(filtered_report)
    else:
        print("\n--- Full Coverage Report ---")
        print(report_output)

    print("\nCoverage report finished.")


if __name__ == "__main__":
    main()
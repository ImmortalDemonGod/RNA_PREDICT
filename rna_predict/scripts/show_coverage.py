#!/usr/bin/env python3
import subprocess
import sys
import argparse
import os
import re
from typing import Set, Tuple, List, Optional

def run_command(cmd: str, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    """Run a command and return its result."""
    try:
        if capture_output:
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
        else:
            result = subprocess.run(cmd.split(), check=True, text=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if capture_output:
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
        raise


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


def parse_missing_lines(missing_str: str) -> Set[int]:
    """Parses a coverage 'Missing' string (e.g., '5-10, 15, 22-24') into a set of line numbers."""
    lines: Set[int] = set()
    if not missing_str:
        return lines
    # Remove any surrounding whitespace that might interfere
    missing_str = missing_str.strip()
    parts = missing_str.split(',')
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            try:
                # Handle potential empty strings from split if range is malformed like '5-'
                range_parts = [p for p in part.split('-') if p]
                if len(range_parts) == 2:
                    start, end = map(int, range_parts)
                    if start > end: # Handle cases like '10-5' if they occur
                        print(f"Warning: Correcting inverted range '{part}' to '{end}-{start}'.", file=sys.stderr)
                        start, end = end, start
                    lines.update(range(start, end + 1))
                else:
                     print(f"Warning: Could not parse range '{part}' in missing lines string.", file=sys.stderr)

            except ValueError:
                print(f"Warning: Could not parse range '{part}' in missing lines string.", file=sys.stderr)
        else:
            try:
                lines.add(int(part))
            except ValueError:
                 print(f"Warning: Could not parse line number '{part}' in missing lines string.", file=sys.stderr)
    return lines


def parse_memory_usage(csv_file: str) -> List[Tuple[str, float]]:
    """Parse memory usage data from the memprof CSV file."""
    if not os.path.exists(csv_file):
        return []
    
    memory_data = []
    with open(csv_file, 'r') as f:
        # Skip header
        next(f)
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                test_name = parts[0]
                # Convert bytes to MB
                memory_usage = float(parts[1]) / (1024 * 1024)  # Convert bytes to MB
                memory_data.append((test_name, memory_usage))
    
    return sorted(memory_data, key=lambda x: x[1], reverse=True)


def main():
    """Run pytest with coverage and memory profiling."""
    parser = argparse.ArgumentParser(description='Run tests with coverage and memory profiling')
    parser.add_argument('--least-covered', action='store_true', help='Show least covered files')
    parser.add_argument('--filter', type=str, help='Filter coverage report by file pattern')
    args = parser.parse_args()

    # Create a temporary file for memory profiling data
    memprof_csv = 'memory_usage.csv'
    
    print("--- Running Pytest with Coverage and Memory Profiling ---")
    pytest_cmd = "pytest --cov=rna_predict --cov-branch --memprof-top-n=10 --memprof-csv-file=memory_usage.csv -n auto tests/"
    print(f"Running command: {pytest_cmd}")
    run_command(pytest_cmd)

    print("\n--- Combining Coverage Data ---")
    print("Running command: coverage combine --append")
    run_command("coverage combine --append")

    print("\n--- Generating Full Coverage Report ---")
    print("Running command: coverage report -m")
    run_command("coverage report -m")

    print("\n--- Memory Usage Summary ---")
    memory_data = parse_memory_usage(memprof_csv)
    if memory_data:
        print("\nTop 10 tests by memory usage:")
        for test_name, memory_usage in memory_data[:10]:
            print(f"{test_name}: {memory_usage:.2f} MB")
    else:
        print("No memory usage data found. Make sure pytest-memprof is installed.")

    # Clean up the temporary file
    if os.path.exists(memprof_csv):
        os.remove(memprof_csv)

    print("\n--- Full Coverage Report ---")
    if args.least_covered:
        show_least_covered()
    elif args.filter:
        filter_coverage(args.filter)
    else:
        run_command("coverage report -m")


def show_least_covered():
    """Show the file with the lowest coverage and its untested lines."""
    print("\n--- Least Covered File Summary ---")
    result = run_command("coverage report -m", capture_output=True)
    report_output = result.stdout

    # Find the file with the lowest coverage
    lowest_coverage = 100
    least_covered_file = None
    least_covered_lines = None

    for line in report_output.split('\n'):
        if not line.strip() or line.startswith('Name') or line.startswith('---') or line.startswith('TOTAL'):
            continue
        
        parts = line.split()
        if len(parts) >= 4:  # Need at least Name, Stmts, Miss, Cover
            try:
                file_name = parts[0]
                stmts = int(parts[1])
                miss = int(parts[2])
                if stmts > 0:  # Only consider files with statements
                    coverage = 100 - (miss * 100 // stmts)
                    if coverage < lowest_coverage:
                        lowest_coverage = coverage
                        least_covered_file = file_name
                        # Get the missing lines from the last column if it exists
                        if len(parts) > 4 and parts[-1].strip():
                            least_covered_lines = parts[-1]
            except (ValueError, IndexError):
                continue

    if least_covered_file:
        print(f"\nFile with lowest coverage: {least_covered_file}")
        print(f"Coverage: {lowest_coverage}%")
        if least_covered_lines:
            print(f"Missing lines: {least_covered_lines}")

            # Read and display the untested lines
            try:
                with open(least_covered_file, 'r') as f:
                    source_lines = f.readlines()
                
                if least_covered_lines:
                    missing_line_nums = set()
                    for part in least_covered_lines.split(','):
                        part = part.strip()
                        if '-' in part:
                            start, end = map(int, part.split('-'))
                            missing_line_nums.update(range(start, end + 1))
                        else:
                            missing_line_nums.add(int(part))

                    print("\nUntested lines:")
                    for line_num in sorted(missing_line_nums):
                        if 1 <= line_num <= len(source_lines):
                            print(f"{line_num:4d} | {source_lines[line_num-1].rstrip()}")
            except Exception as e:
                print(f"Error reading file {least_covered_file}: {e}")
    else:
        print("No files with statements found in coverage report.")


def filter_coverage(pattern: str):
    """Filter the coverage report by file pattern."""
    print("\n--- Filtered Coverage Report ---")
    result = run_command("coverage report -m", capture_output=True)
    report_output = result.stdout

    filtered_lines = []
    for line in report_output.split('\n'):
        if pattern.lower() in line.lower():
            filtered_lines.append(line)

    if filtered_lines:
        print('\n'.join(filtered_lines))
    else:
        print(f"No files matching pattern '{pattern}' found in coverage report.")


if __name__ == "__main__":
    main()
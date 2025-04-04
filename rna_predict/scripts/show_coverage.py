#!/usr/bin/env python3
import subprocess
import json
import os
from typing import List, Tuple

def group_consecutive_lines(line_numbers: List[int]) -> List[Tuple[int, int]]:
    """
    Given a sorted list of line numbers, return a list of consecutive line ranges.
    E.g. [3,4,5,7,8,10] -> [(3,5), (7,8), (10,10)]
    """
    if not line_numbers:
        return []
    grouped = []
    start = line_numbers[0]
    prev = start
    for i in line_numbers[1:]:
        if i == prev + 1:
            prev = i
        else:
            grouped.append((start, prev))
            start = i
            prev = i
    grouped.append((start, prev))
    return grouped

def read_file_contents(filepath: str) -> List[str]:
    """
    Safely read file contents and return as list of lines (1-based index).
    If file not found, return an empty list.
    """
    if not os.path.isfile(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return lines

def main():
    # Run tests under coverage with pytest in parallel using auto workers
    subprocess.run(["pytest", "--cov=rna_predict", "--cov-branch", "-n", "auto"], check=True)
    
    # Combine coverage data from parallel subprocesses
    subprocess.run(["coverage", "combine"], check=True)
    
    # Generate JSON coverage report from the combined data
    subprocess.run(["coverage", "json", "-o", "coverage.json"], check=True)
    
    # Check if coverage.json exists
    if not os.path.exists("coverage.json"):
        print("Error: coverage.json not found.")
        return
    
    # Load the JSON report
    with open("coverage.json", "r") as f:
        cov_data = json.load(f)
    
    # Load previous coverage data if available
    prev_cov_file = "prev_coverage.json"
    if os.path.exists(prev_cov_file):
        with open(prev_cov_file, "r") as f_prev:
            prev_cov_data = json.load(f_prev)
    else:
        prev_cov_data = {}

    # Print header for detailed file coverage
    header = f"{'File':50} {'Stmts':>6} {'Miss':>6} {'Cover%':>8}"
    print("Detailed File Coverage:")
    print(header)
    print("-" * len(header))
    
    # Get and sort file entries from the JSON report
    files = cov_data.get("files", {})
    sorted_files = sorted(files.items(), key=lambda x: x[0])
    
    for filepath, info in sorted_files:
        summary = info.get("summary", {})
        stmts = summary.get("num_statements", 0)
        missed = summary.get("missing_lines", 0)
        cover = summary.get("percent_covered", 0.0)
        print(f"{filepath:50} {stmts:6} {missed:6} {cover:8.2f}")
    
    # Display missing lines with corresponding source code
    print("\n--- Missing Lines Detail ---")
    for filepath, info in sorted_files:
        missing_lines = info.get("missing_lines", [])
        if not missing_lines:
            continue  # Skip files with 100% coverage or no statements
        
        # Group the missing lines for easier reading
        grouped_ranges = group_consecutive_lines(missing_lines)
        
        # Read file contents
        file_lines = read_file_contents(filepath)
        if not file_lines:
            print(f"\nFile: {filepath}")
            print("  Unable to read file contents or file not found.")
            print(f"  Missing lines: {grouped_ranges}\n")
            continue
        
        print(f"\nFile: {filepath} | Total Missing Lines: {len(missing_lines)}")
        for (start, end) in grouped_ranges:
            if start == end:
                print(f"  Line {start}:")
                line_idx = start - 1  # Convert to 0-based index
                if 0 <= line_idx < len(file_lines):
                    code_str = file_lines[line_idx].rstrip("\n")
                    print(f"    {start}: {code_str}")
            else:
                print(f"  Lines {start}-{end}:")
                for line_num in range(start, end + 1):
                    line_idx = line_num - 1
                    if 0 <= line_idx < len(file_lines):
                        code_str = file_lines[line_idx].rstrip("\n")
                        print(f"    {line_num}: {code_str}")
    
    # Compare current coverage with previous run
    print("\n--- Coverage Comparison ---")
    prev_files = prev_cov_data.get("files", {}) if prev_cov_data else {}
    for filepath, info in sorted_files:
        curr_summary = info.get("summary", {})
        curr_percent = curr_summary.get("percent_covered", 0.0)
        curr_missing_count = len(info.get("missing_lines", []))
        prev_info = prev_files.get(filepath, {})
        prev_summary = prev_info.get("summary", {}) if prev_info else {}
        prev_percent = prev_summary.get("percent_covered", None)
        prev_missing_count = len(prev_info.get("missing_lines", [])) if prev_info.get("missing_lines") is not None else None

        if prev_percent is not None:
            diff_percent = curr_percent - prev_percent
            diff_missing = curr_missing_count - prev_missing_count
            print(f"{filepath:50} Prev: {prev_percent:6.2f}% ({prev_missing_count} missing)  Curr: {curr_percent:6.2f}% ({curr_missing_count} missing)  Diff: {diff_percent:+6.2f}%, {diff_missing:+} missing")
        else:
            print(f"{filepath:50} New file, Curr: {curr_percent:6.2f}% ({curr_missing_count} missing)")
    
    # Save current coverage data for next run
    with open(prev_cov_file, "w") as f_prev:
        json.dump(cov_data, f_prev, indent=4)
    
    print("\nDone displaying coverage details.")

if __name__ == '__main__':
    main()
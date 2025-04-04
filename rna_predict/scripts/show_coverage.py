#!/usr/bin/env python3
import subprocess
import json
import os

def main():
    # Run tests under coverage with pytest in parallel using auto workers
    subprocess.run(["coverage", "run", "--parallel-mode", "-m", "pytest", "-n", "auto"], check=True)
    
    # Combine coverage data from parallel subprocesses
    subprocess.run(["coverage", "combine"], check=True)
    
    # Generate JSON coverage report
    subprocess.run(["coverage", "json", "-o", "coverage.json"], check=True)
    
    # Check if coverage.json exists
    if not os.path.exists("coverage.json"):
        print("Error: coverage.json not found.")
        return
    
    # Load the JSON report
    with open("coverage.json", "r") as f:
        cov_data = json.load(f)
    
    # Extract overall summary data
    totals = cov_data.get("totals", {})
    total_statements = totals.get("statements", "N/A")
    total_missing = totals.get("missing", "N/A")
    total_coverage = totals.get("percent_covered", "N/A")
    
    # Print overall summary
    print("\nCoverage Summary:")
    print(f"Total Statements: {total_statements}")
    print(f"Total Missing:    {total_missing}")
    print(f"Overall Coverage: {total_coverage}%\n")
    
    # Print header for detailed file coverage
    header = f"{'File':50} {'Stmts':>6} {'Miss':>6} {'Cover%':>8}"
    print("Detailed File Coverage:")
    print(header)
    print("-" * len(header))
    
    # Get and sort file entries from the JSON report
    files = cov_data.get("files", {})
    for filepath, info in sorted(files.items()):
        stmts = info.get("statements", 0)
        missed = info.get("missing", 0)
        cover = info.get("percent_covered", 0.0)
        print(f"{filepath:50} {stmts:6} {missed:6} {cover:8.2f}")

if __name__ == '__main__':
    main()
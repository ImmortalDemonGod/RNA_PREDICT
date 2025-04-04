#!/usr/bin/env python3
import subprocess

def main():
    # Run tests under coverage
    subprocess.run(["coverage", "run", "-m", "pytest", "-n", "auto"], check=True)
    # Generate and print detailed coverage report (with missing line numbers)
    result = subprocess.run(["coverage", "report", "-m"], check=True, capture_output=True, text=True)
    print(result.stdout)

if __name__ == '__main__':
    main()
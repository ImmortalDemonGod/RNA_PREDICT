#!/bin/bash

# Automatically find all Python test files in the tests directory
test_files=$(find tests -type f -name "test_*.py")

# Run all tests in parallel using pytest-xdist
# -n auto: automatically determine number of workers based on CPU cores
# --dist loadfile: distribute tests by file to minimize inter-process communication
echo "Running all tests in parallel..."
test_output=$(uv run pytest $test_files -v -n auto --dist loadfile 2>&1)
exit_code=$?

if [ $exit_code -ne 0 ]; then
    echo "$test_output"
    echo "Some tests failed!"
    exit 1
fi

echo "All tests passed successfully!" 
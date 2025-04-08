#!/bin/bash

# Get the number of CPU cores
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    NUM_CORES=$(sysctl -n hw.ncpu)
else
    # Linux
    NUM_CORES=$(nproc)
fi

# Use 7 workers
NUM_WORKERS=8

# Create coverage directory if it doesn't exist
mkdir -p coverage

# Automatically find all Python test files in the tests directory
test_files=$(find tests -type f -name "test_*.py")

# Run tests in parallel using pytest-xdist with increased timeout, memory profiling, and coverage
echo "Running tests in parallel using $NUM_WORKERS workers with memory profiling and coverage..."
pytest $test_files -v -n $NUM_WORKERS --dist=loadfile --timeout=300 \
    --memray --most-allocations=10 --stacks=5 \
    --cov=rna_predict \
    --cov-report=html \
    --cov-report=term-missing \
    --cov-fail-under=77

exit_code=$?

if [ $exit_code -ne 0 ]; then
    echo "Some tests failed or coverage is below threshold!"
    exit 1
fi

echo "All tests passed successfully!"
echo "Coverage report generated in coverage/html/index.html" 
#!/bin/bash

# Default values
NUM_MUTATIONS=1
MUTATION_MODE="s"
REPORTS_DIR="reports/mutation_tests"
OUTPUT_FILE="$REPORTS_DIR/mutation_report_$(date +%Y%m%d_%H%M%S).rst"
LOG_FILE="$REPORTS_DIR/mutation_test.log"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--num-mutations)
            NUM_MUTATIONS="$2"
            shift 2
            ;;
        -m|--mode)
            MUTATION_MODE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$REPORTS_DIR/$(basename $2)"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Ensure reports directory exists
mkdir -p "$REPORTS_DIR"

echo "Running mutation tests with:"
echo "- Number of mutations: $NUM_MUTATIONS"
echo "- Mode: $MUTATION_MODE"
echo "- Categories: all"
echo "- Output file: $OUTPUT_FILE"

# Run mutatest and capture all output
CMD="mutatest -n $NUM_MUTATIONS -m $MUTATION_MODE -o $OUTPUT_FILE"
echo "Running: $CMD"
if ! $CMD 2>&1 | tee "$LOG_FILE"; then
    echo "Error during mutation testing"
    
    # Check for common error patterns
    if grep -q "Population must be a sequence" "$LOG_FILE"; then
        echo "Error: Issue with mutation selection"
        echo "Suggestion: Try running again with fewer mutations or target specific directories"
    elif grep -q "No locations found to mutate" "$LOG_FILE"; then
        echo "Error: No valid mutation targets found"
        echo "Suggestion: Check the target directories and file patterns"
    elif grep -q "timeout" "$LOG_FILE"; then
        echo "Error: Tests timed out"
        echo "Suggestion: Consider increasing the test timeout or running fewer mutations"
    elif grep -q "MemoryError" "$LOG_FILE"; then
        echo "Error: Out of memory"
        echo "Suggestion: Try running with fewer parallel workers or reduce mutation count"
    else
        echo "Unknown error occurred. Check $LOG_FILE for details"
    fi
    
    # Keep the log file for debugging
    echo "Full logs available at: $LOG_FILE"
    exit 1
fi

# If we get here, the command succeeded
echo "Mutation testing completed successfully!"
echo "Results saved to $OUTPUT_FILE"

# Move the log file to have same base name as output for reference
mv "$LOG_FILE" "${OUTPUT_FILE%.*}.log" 
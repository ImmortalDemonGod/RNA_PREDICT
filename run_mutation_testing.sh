#!/bin/bash
# Script to run mutation testing with mutatest

# Ensure compatible versions of dependencies are installed
pip install coverage==5.5 pytest-cov==2.12.1

# Default values
NLOCATIONS=5
MODE="s"
OUTPUT_FILE="mutation_report.rst"
ONLY_CATEGORIES=""
SKIP_CATEGORIES=""
PARALLEL=""

# Help function
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h, --help                 Show this help message"
    echo "  -n, --nlocations NUMBER    Number of locations to mutate (default: 20)"
    echo "  -m, --mode MODE            Running mode: f, s, d, sd (default: s)"
    echo "  -o, --output FILE          Output file for report (default: mutation_report.rst)"
    echo "  -y, --only CATEGORIES      Only use these mutation categories (space separated)"
    echo "  -k, --skip CATEGORIES      Skip these mutation categories (space separated)"
    echo "  # Note: Parallel execution is not available in this version of mutatest"
    echo ""
    echo "Mutation Categories:"
    echo "  aa - AugAssign (e.g., +=, -=, *=)"
    echo "  bn - BinOp (e.g., +, -, *, /)"
    echo "  bc - BinOpBC (e.g., &, |, ^)"
    echo "  bs - BinOpBS (e.g., >>, <<)"
    echo "  bl - BoolOp (e.g., and, or)"
    echo "  cp - Compare (e.g., <, >, <=, >=, ==, !=)"
    echo "  cn - CompareIn (e.g., in, not in)"
    echo "  cs - CompareIs (e.g., is, is not)"
    echo "  if - If (e.g., if statements)"
    echo "  ix - Index (e.g., list indexing)"
    echo "  nc - NameConstant (e.g., True, False, None)"
    echo "  su - SliceUS (e.g., list slicing)"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -n|--nlocations)
            NLOCATIONS="$2"
            shift 2
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -y|--only)
            shift
            ONLY_CATEGORIES=""
            while [[ $# -gt 0 && ! $1 =~ ^- ]]; do
                ONLY_CATEGORIES="$ONLY_CATEGORIES $1"
                shift
            done
            ;;
        -k|--skip)
            shift
            SKIP_CATEGORIES=""
            while [[ $# -gt 0 && ! $1 =~ ^- ]]; do
                SKIP_CATEGORIES="$SKIP_CATEGORIES $1"
                shift
            done
            ;;
        # Parallel option removed as it's not available in this version
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Build the command
CMD="mutatest -n $NLOCATIONS -m $MODE -o $OUTPUT_FILE"

# Add optional arguments
if [ ! -z "$ONLY_CATEGORIES" ]; then
    CMD="$CMD -y $ONLY_CATEGORIES"
fi

if [ ! -z "$SKIP_CATEGORIES" ]; then
    CMD="$CMD -k $SKIP_CATEGORIES"
fi

# Parallel option removed as it's not available in this version

# Print the command
echo "Running: $CMD"

# Run the command
eval $CMD

# Check the exit code
if [ $? -eq 0 ]; then
    echo "Mutation testing completed successfully!"
    echo "Results saved to $OUTPUT_FILE"
else
    echo "Mutation testing failed with exit code $?"
fi

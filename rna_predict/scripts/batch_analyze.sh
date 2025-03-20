#!/bin/bash
set -eo pipefail

###############################################################################
# Script: batch_analyze.sh
# Description: Recursively find all Python files in the given directory and run
#              analyze_code.sh on each.
###############################################################################

# Validate argument
if [ $# -lt 1 ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

TARGET_DIR="$1"

# Create timestamped folder for batch analysis
TIMESTAMP=$(date +'%Y-%m-%d_%H-%M-%S')
OUTPUT_DIR="github_automation_output/code_analysis/$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"

if [ ! -d "$TARGET_DIR" ]; then
  echo "Error: $TARGET_DIR is not a directory or does not exist."
  exit 1
fi

# Ensure analyze_code.sh is in PATH or is referenced by absolute path
ANALYZE_SCRIPT="$(dirname "$0")/analyze_code.sh"
if [ ! -x "$ANALYZE_SCRIPT" ]; then
  echo "Error: analyze_code.sh not found or not executable at $ANALYZE_SCRIPT"
  exit 1
fi

echo "Recursively analyzing Python files under '$TARGET_DIR'..."

# Find all .py files under TARGET_DIR
find "$TARGET_DIR" -type f -name "*.py" | while read -r pyfile; do
  echo "--------------------------------------------------"
  echo "Analyzing: $pyfile"
  echo "--------------------------------------------------"
  
  # Call our analyze_code.sh script with output-dir
  "$ANALYZE_SCRIPT" --output-dir "$OUTPUT_DIR" "$pyfile"

  # Optionally, if you want a small wait between each analysis:
  # sleep 1
done

echo "All done."
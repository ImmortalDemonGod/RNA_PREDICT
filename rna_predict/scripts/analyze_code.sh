#!/bin/bash
set -eo pipefail

# Script Name: analyze_code.sh
# Description: Runs CodeScene (cs), Mypy, and Ruff on a user-provided script file
# and copies combined results to clipboard.
#
# New Features:
# - Test file detection (test_<filename>.py)
# - Coverage run if test file found
# - Auto-switch to --test prompt if coverage < 100%
# - Robust coverage line parsing to avoid integer expression errors

###############################################################################
# 0. Usage & Flag Parsing
###############################################################################

usage() {
    echo "Usage: $0 [--test] [--output-dir DIR] /path/to/your/script.py"
    exit 1
}

# Debugging helper
debug_log() {
    echo "DEBUG: $*" >&2
}

# Default: Not in test mode
TEST_MODE=false

# Simple argument parsing
if [ "$#" -eq 0 ]; then
    usage
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test)
            TEST_MODE=true
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            ;;
        *)
            FILE_PATH="$1"
            shift
            ;;
    esac
done

# Validate we have a file path
if [[ -z "$FILE_PATH" ]]; then
    echo "Error: A script file path is required."
    usage
fi

if [ ! -f "$FILE_PATH" ]; then
    echo "Error: The file \"$FILE_PATH\" does not exist."
    exit 1
fi
###############################################################################
# 0x. Ensure uv is installed before proceeding
###############################################################################
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Attempting to install..."
    # Option A: Use the official standalone installer (macOS/Linux):
    #   curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Option B: Use pip to install uv:
    echo "Installing uv using pip..."
    if pip install uv; then
        echo "uv installed via pip."
    else
        echo "Failed to install uv. Exiting."
        exit 1
    fi

    # If uv is still not in PATH, consider:
    #   export PATH="$(python -m site --user-base)/bin:$PATH"
fi

###############################################################################
# 0y. Ensure CodeScene CLI (cs) is installed before proceeding
###############################################################################
if ! command -v cs >/dev/null 2>&1; then
    echo "cs (CodeScene CLI) not found. Attempting to install..."
    echo "Installing CodeScene CLI using the official script..."
    # Per CodeScene docs: this script attempts to place 'cs' in ~/.local/bin
    # If ~/.local/bin isn’t in PATH, it tries to add it for common shells.
    # On Windows or other OS, user might need a manual approach.
    
    if command -v curl >/dev/null 2>&1; then
        if ! curl -sSf https://downloads.codescene.io/enterprise/cli/install-codescene-cli.sh | sh; then
            echo "Failed to install CodeScene CLI with the official script. Exiting."
            exit 1
        fi
    else
        echo "curl not found or not available. Cannot auto-install CodeScene CLI."
        echo "Please install it manually, or ensure 'cs' is in PATH."
        exit 1
    fi

    # If 'cs' still isn't in PATH, we can attempt to export:
    #   export PATH="$HOME/.local/bin:$PATH"
fi

###############################################################################
# 0y-1. Ensure CodeScene Access Token is set
###############################################################################
# If CS_ACCESS_TOKEN is not already exported in the environment, set a default.
# Replace the placeholder token below with your real CodeScene token if needed.
# If the token is valid, CodeScene analysis should pass the license check.

if [ -z "$CS_ACCESS_TOKEN" ]; then
    echo "CS_ACCESS_TOKEN not set. Setting a default..."
    export CS_ACCESS_TOKEN="Njk0NjM-MjAyNi0wMS0xN1QxOTo1Mzo1Mw-I3sicmVmYWN0b3IuYWNjZXNzIiAiY2xpLmFjY2VzcyJ9.30-SmgU-Ybio83czYew_WCtu_QvyPWyWlSQQD63_gZA"
    # If you have a different token, you can replace the line above or override
    # in your shell environment.
fi

###############################################################################
# 0z. Ensure ruff is installed before proceeding
###############################################################################
# We'll install ruff into the environment that uv manages. So we use "uv run pip".
# This ensures that "uv run ruff" will work properly without needing a system-level install.
if ! uv run which ruff >/dev/null 2>&1; then
    echo "ruff not found in uv environment. Attempting to install..."
    if uv run pip install ruff; then
        echo "ruff installed via uv environment."
    else
        echo "Failed to install ruff. Exiting."
        exit 1
    fi
fi

# Function to find the project root
find_project_root() {
    local path="$1"
    while [[ "$path" != "/" ]]; do
        if [[ -d "$path/.git" ]] || [[ -f "$path/pyproject.toml" ]] || [[ -f "$path/setup.py" ]]; then
            echo "$path"
            return
        fi
        path=$(dirname "$path")
    done
    echo ""
}

PROJECT_ROOT=$(find_project_root "$(dirname "$FILE_PATH")")

if [[ -z "$PROJECT_ROOT" ]]; then
    echo "Error: Could not determine the project root. Ensure your project has a .git directory, pyproject.toml, or setup.py."
    exit 1
fi

debug_log "Project root determined as '$PROJECT_ROOT'."

cd "$PROJECT_ROOT" || { echo "Error: Failed to change directory to '$PROJECT_ROOT'."; exit 1; }


###############################################################################
# 0a. Debug info about the environment and tools
###############################################################################
debug_log "which python -> $(which python || echo 'not found')"
debug_log "Python version -> $(uv run python --version 2>&1 || echo 'Failed to get Python version')"
debug_log "which mypy -> $(uv run which mypy || echo 'Mypy not found')"
debug_log "Mypy version -> $(uv run mypy --version 2>&1 || echo 'Mypy not found')"
debug_log "which cs -> $(uv run which cs || echo 'cs not found')"
debug_log "which ruff -> $(uv run which ruff || echo 'ruff not found')"

# Warn if VIRTUAL_ENV doesn't match .venv
if [ -n "$VIRTUAL_ENV" ] && [[ "$VIRTUAL_ENV" != *".venv"* ]]; then
    echo "warning: \`VIRTUAL_ENV=$VIRTUAL_ENV\` does not match the project environment path \`.venv\` and will be ignored"
fi

# Initialize PYTHONPATH if not set
PYTHONPATH=${PYTHONPATH:-}
# Add PROJECT_ROOT to PYTHONPATH
if [ -z "$PYTHONPATH" ]; then
    PYTHONPATH="$PROJECT_ROOT"
else
    PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
fi
export PYTHONPATH

debug_log "PYTHONPATH set to '$PYTHONPATH'."
debug_log "Current Directory: $(pwd)"

###############################################################################
# 0b. Create temporary directory for outputs
###############################################################################
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

# Final combined output file
COMBINED_OUTPUT="$TEMP_DIR/combined_analysis.txt"
touch "$COMBINED_OUTPUT"

echo "Starting analysis on '$FILE_PATH'..."

###############################################################################
# 1. Run CodeScene
###############################################################################
echo "Running CodeScene..."

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is not installed or not in PATH. Please install uv and ensure it's accessible."
    exit 1
fi

echo -e "\n=== CODESCENE ANALYSIS ===\nPlease systematically review the following CodeScene analysis results and provide a detailed summary of the findings. Then plan out the necessary refactoring steps based on the CodeScene analysis.\n" >> "$COMBINED_OUTPUT"
if ! uv run cs review "$FILE_PATH" --output-format json > "$TEMP_DIR/codescene_results.json"; then
    echo "Warning: CodeScene command failed. Continuing with other checks..."
else
    cat "$TEMP_DIR/codescene_results.json" >> "$COMBINED_OUTPUT"
fi

###############################################################################
# 2. Ensure lxml is installed before running Mypy with --xml-report
###############################################################################
if ! uv run python -c 'import lxml' 2>/dev/null; then
    echo "lxml is not installed. Attempting to install..."
    if ! python -m pip install lxml; then
        echo "Warning: Could not install lxml. Mypy XML reporting might fail."
    fi
fi

###############################################################################
# 3. Run Mypy
###############################################################################
echo "Running Mypy..."
echo -e "\n=== MYPY ANALYSIS ===\n" >> "$COMBINED_OUTPUT"

RELATIVE_PATH=${FILE_PATH#"$PROJECT_ROOT/"}
MYPY_XML_PATH="$TEMP_DIR/mypy_report.xml"

uv run mypy "$FILE_PATH" --pretty --xml-report "$TEMP_DIR" || {
    echo "Note: Mypy found issues (this is normal)."
}

# Extract the specific file's XML content if it exists
if [ -f "$TEMP_DIR/index.xml" ]; then
    cat "$TEMP_DIR/index.xml" >> "$COMBINED_OUTPUT"
fi

###############################################################################
# 4. Run Ruff
###############################################################################
echo "Running Ruff checks and formatting..."
echo -e "\n=== RUFF FIX OUTPUT ===\n" >> "$COMBINED_OUTPUT"

uv run ruff check "$FILE_PATH" > /dev/null 2>&1 || true
uv run ruff check --fix "$FILE_PATH" 2>&1 | tee -a "$COMBINED_OUTPUT" || true
uv run ruff format "$FILE_PATH" > /dev/null 2>&1 || true
uv run ruff check --select I --fix "$FILE_PATH" > /dev/null 2>&1 || true
uv run ruff format "$FILE_PATH" > /dev/null 2>&1 || true

# 4.4.1: Ensure coverage is installed in uv environment
if ! uv run coverage --version >/dev/null 2>&1; then
    echo "'coverage' not found in uv environment. Attempting to install coverage..."
    if uv run pip install coverage; then
        echo "coverage installed via uv environment."
    else
        echo "Failed to install coverage. Exiting or skipping coverage steps."
    fi
fi

###############################################################################
# 4.5 Coverage Check for a Matching Test File
###############################################################################
BASENAME="$(basename "$FILE_PATH" .py)"
TEST_FILE=""

# Use 'find' to locate a matching test file anywhere under the project.
FOUND_TEST="$(find "$PROJECT_ROOT" -type f \( -name "test_${BASENAME}.py" -o -name "${BASENAME}_test.py" \) | head -n 1)"

if [ -n "$FOUND_TEST" ]; then
    TEST_FILE="$FOUND_TEST"
fi

COVERAGE_UNDER_100=false

if [ -z "$TEST_FILE" ]; then
    echo "No test file found for $FILE_PATH. Skipping coverage." | tee -a "$COMBINED_OUTPUT"
else
    echo "=== TEST FILE DETECTED ===" >> "$COMBINED_OUTPUT"
    echo "Found test file at $TEST_FILE" | tee -a "$COMBINED_OUTPUT"
    if ! command -v coverage >/dev/null 2>&1; then
        echo "Warning: 'coverage' is not installed or not in PATH. Please install coverage." | tee -a "$COMBINED_OUTPUT"
    else
        echo "Running coverage on $TEST_FILE..." | tee -a "$COMBINED_OUTPUT"
        # Run coverage, do not exit on test failures
        # Compute a relative path via Python (instead of realpath --relative-to, which isn't on macOS)
        RELATIVE_PATH="$(uv run python -c "import os; print(os.path.relpath('$FILE_PATH', '$PROJECT_ROOT'))")"
        echo "Using Python-based relpath: $RELATIVE_PATH" | tee -a "$COMBINED_OUTPUT"
        
        # Run coverage from project root, including our source tree
        coverage run --source="$PROJECT_ROOT" -m pytest "$TEST_FILE" || {
            echo "Tests failed under coverage. Proceeding with other steps." | tee -a "$COMBINED_OUTPUT"
        }
        
        # Limit coverage report to that single file
        coverage report -m --include="$RELATIVE_PATH" > "$TEMP_DIR/coverage_report.txt" || true
        coverage xml -o "$TEMP_DIR/coverage_report.xml" --include="$RELATIVE_PATH" || true
        
        # Also generate JSON coverage to extract exact missing lines
        coverage json -o "$TEMP_DIR/coverage.json" --include="$RELATIVE_PATH" || true
        
        # Parse missing lines from coverage.json and dump them with code references
        echo -e "\\n=== UNCOVERED LINES DETAIL ===" >> "$TEMP_DIR/coverage_report.txt"
        uv run python -c "
import json, sys, os

json_path = r'$TEMP_DIR/coverage.json'
rel_file = r'$RELATIVE_PATH'
report_file = r'$TEMP_DIR/coverage_report.txt'
if not os.path.exists(json_path):
    print('No coverage.json found; skipping uncovered lines detail.')
    sys.exit(0)

with open(json_path, 'r') as f:
    data = json.load(f)

files_info = data.get('files', {})
file_info = files_info.get(rel_file, {})
missing = file_info.get('missing_lines', [])

if not missing:
    print('No missing lines found for', rel_file)
    sys.exit(0)

# Read the original file's code lines
try:
    with open(rel_file, 'r', encoding='utf-8') as src:
        all_lines = src.readlines()
except FileNotFoundError:
    print(f'Could not open {rel_file}, skipping detail')
    sys.exit(0)

# Write uncovered line detail to coverage_report.txt
with open(report_file, 'a', encoding='utf-8') as rf:
    rf.write(f'File: {rel_file}\\n')
    rf.write(f'Uncovered lines: {missing}\\n')
    rf.write('--- Begin missing lines detail ---\\n')
    for line_num in missing:
        # Coverage lines are 1-based index. Python array is 0-based.
        idx = line_num - 1
        if 0 <= idx < len(all_lines):
            code_line = all_lines[idx].rstrip('\\n')
            rf.write(f'Line {line_num}: {code_line}\\n')
        else:
            rf.write(f'Line {line_num}: [Index out of range]\\n')
    rf.write('--- End missing lines detail ---\\n')
        "
        
        # Append coverage text to combined output
        if [ -f "$TEMP_DIR/coverage_report.txt" ]; then
            echo -e "\\n=== TEST COVERAGE REPORT ===" >> "$COMBINED_OUTPUT"
            cat "$TEMP_DIR/coverage_report.txt" >> "$COMBINED_OUTPUT"
        fi
        
        # Attempt to parse coverage for the specific file by looking for $RELATIVE_PATH
        COVERAGE_LINE=$(grep "$RELATIVE_PATH" "$TEMP_DIR/coverage_report.txt" || true)

        # Example coverage line format (depends on coverage version):
        # quick-fixes/tasks/lifelines_models.py  163  21  88%  91
        # Name    Stmts   Miss  Cover   Missing
        # lifelines_models.py  20  2  90%   10-11
        if [ -n "$COVERAGE_LINE" ]; then
            # Break the line into an array of fields
            read -ra FIELDS <<< "$COVERAGE_LINE"

            # We want to find the field that ends with '%'
            COVERAGE_PERCENT_RAW=""
            for field in "${FIELDS[@]}"; do
                if [[ "$field" == *"%"* ]]; then
                    COVERAGE_PERCENT_RAW="$field"
                    break
                fi
            done

            if [ -n "$COVERAGE_PERCENT_RAW" ]; then
                COVERAGE_INT=$(echo "$COVERAGE_PERCENT_RAW" | tr -d '%')
                # Remove decimals, if any
                COVERAGE_INT=$(echo "$COVERAGE_INT" | sed 's/\..*//')

                echo "Coverage for $TARGET_BASENAME is ${COVERAGE_PERCENT_RAW}." \
                     "Parsed integer coverage: $COVERAGE_INT%" >> "$COMBINED_OUTPUT"

                if [[ "$COVERAGE_INT" =~ ^[0-9]+$ ]]; then
                    if [ "$COVERAGE_INT" -lt 100 ]; then
                        COVERAGE_UNDER_100=true
                    fi
                else
                    echo "Warning: coverage percentage not parseable. Auto-flagging coverage < 100%." >> "$COMBINED_OUTPUT"
                    COVERAGE_UNDER_100=true
                fi
            else
                echo "No percentage field found in coverage line." >> "$COMBINED_OUTPUT"
                COVERAGE_UNDER_100=true
            fi
        else
            echo "No specific coverage data found for $TARGET_BASENAME. Possibly 0% coverage." >> "$COMBINED_OUTPUT"
            COVERAGE_UNDER_100=true
        fi
    fi
fi

# If coverage is incomplete, we can auto-switch to test mode if not already set
if [ "$COVERAGE_UNDER_100" = true ] && [ "$TEST_MODE" = false ]; then
    echo "Coverage < 100% for $FILE_PATH. Auto-activating TEST_MODE..." | tee -a "$COMBINED_OUTPUT"
    TEST_MODE=true
fi

###############################################################################
# 4.6 Append Full Test File (if found) to the Combined Output
###############################################################################
if [ -n "$TEST_FILE" ]; then
    echo -e "\\n=== FULL TEST FILE CONTENT ===\\n" >> "$COMBINED_OUTPUT"
    echo "Test File Path: $TEST_FILE" >> "$COMBINED_OUTPUT"
    if [ -f "$TEST_FILE" ]; then
        cat "$TEST_FILE" >> "$COMBINED_OUTPUT"
    else
        echo "Error: Could not read test file: $TEST_FILE" >> "$COMBINED_OUTPUT"
    fi
fi

###############################################################################
# 5. Add Prompt (Refactoring or Test Prompt) Depending on Flag
###############################################################################
echo -e "\n=======\nPROMPT:\n**=======**" >> "$COMBINED_OUTPUT"

if [ "$TEST_MODE" = true ]; then
    # Here Document for your test prompt
    cat <<'EOF' >> "$COMBINED_OUTPUT"
# SYSTEM

You are a Python testing expert specializing in writing pytest test cases. You will receive Python function information and create comprehensive test cases following pytest best practices.

## GOALS

1. Create thorough pytest test cases for the given Python function
2. Cover normal operations, edge cases, and error conditions
3. Use pytest fixtures when appropriate
4. Include proper type hints and docstrings
5. Follow pytest naming conventions and best practices

## RULES

1. Always include docstrings explaining test purpose
2. Use descriptive variable names
3. Include type hints for all parameters
4. Create separate test functions for different test cases
5. Use pytest.mark.parametrize for multiple test cases when appropriate
6. Include error case testing with pytest.raises when relevant
7. Add comments explaining complex test logic
8. Follow the standard test_function_name pattern for test names

## CONSTRAINTS

1. Only write valid pytest code
2. Only use standard pytest features and commonly available packages
3. Keep test functions focused and avoid unnecessary complexity
4. Don't test implementation details, only public behavior
5. Don't create redundant tests

## WORKFLOW

1. Analyze the provided function
2. Identify key test scenarios
3. Create appropriate fixtures if needed
4. Write test functions with clear names and docstrings
5. Include multiple test cases and edge cases
6. Add error condition testing
7. Verify all function parameters are tested
8. Add type hints and documentation

## FORMAT

```python
# Test code here
```

# USER

I will provide you with Python function information. Please generate pytest test cases following the above guidelines.

# ASSISTANT

I'll analyze the provided function and create comprehensive pytest test cases following best practices for testing normal behavior, edge cases, and error conditions.

The test code will be properly structured with:
- Clear docstrings explaining test purpose
- Type hints for all parameters
- Appropriate fixtures where needed
- Parametrized tests for multiple cases
- Error case handling
- Meaningful variable names and comments

Let me know if you need any adjustments to the generated test cases.
===
Follow the Pre-test analysis first then write the tests
# Pre-Test Analysis
1. Identify the exact function/code to be tested
   - Copy the target code and read it line by line
   - Note all parameters, return types, and dependencies
   - Mark any async/await patterns
   - List all possible code paths
2. Analyze Infrastructure Requirements
   - Check if async testing is needed
   - Identify required mocks/fixtures
   - Note any special imports or setup needed
   - Check for immutable objects that need special handling
3. Create Test Foundation
   - Write basic fixture setup
   - Test the fixture with a simple case
   - Verify imports work
   - Run once to ensure test infrastructure works
4. Plan Test Cases
   - List happy path scenarios
   - List error cases from function's try/except blocks
   - Map each test to specific lines of code
   - Verify each case tests something unique
5. Write and Verify Incrementally
   - Write one test case
   - Run coverage to verify it hits expected lines
   - Fix any setup issues before continuing
   - Only proceed when each test works
6. Cross-Check Coverage
   - Run coverage report
   - Map uncovered lines to missing test cases
   - Verify edge cases are covered
   - Confirm error handling is tested
7. Final Verification
   - Run full test suite
   - Compare before/after coverage
   - Verify each test targets the intended function
   - Check for test isolation/independence
# Red Flags to Watch For
- Tests that don't increase coverage
- Overly complex test setups
- Tests targeting multiple functions
- Untested fixture setups
- Missing error cases
- Incomplete mock configurations
# Questions to Ask
- Am I actually testing the target function?
- Does each test serve a clear purpose?
- Are the mocks properly configured?
- Have I verified the test infrastructure works?
- Does the coverage report show improvement?
-------
Write pytest code for this code snippet:

EOF

else
    # Here Document for your normal refactoring prompt
    cat <<'EOF' >> "$COMBINED_OUTPUT"
REFACTOR:
=======
The major types of code refactoring mentioned include:

1. **Extract Function**: Extracting code into a function or method (also referred to as Extract Method).
2. **Extract Variable**: Extracting code into a variable.
3. **Inline Function**: The inverse of Extract Function, where a function is inlined back into its calling code.
4. **Inline Variable**: The inverse of Extract Variable, where a variable is inlined back into its usage.
5. **Change Function Declaration**: Changing the names or arguments of functions.
6. **Rename Variable**: Renaming variables for clarity.
7. **Encapsulate Variable**: Encapsulating a variable to manage its visibility.
8. **Introduce Parameter Object**: Combining common arguments into a single object.
9. **Combine Functions into Class**: Grouping functions with the data they operate on into a class.
10. **Combine Functions into Transform**: Merging functions particularly useful with read-only data.
11. **Split Phase**: Organizing modules into distinct processing phases.

These refactorings focus on improving code clarity and maintainability without altering its observable behavior.

For more detailed information, you might consider using tools that could provide further insights or examples related to these refactoring types.

EOF
fi

# Add a final line that indicates the request to show the full code
echo -e "\n====\nFULL CODE:\n====\nshow the full file dont drop comments or existing functionality" >> "$COMBINED_OUTPUT"

###############################################################################
# 6. Add Full Code
###############################################################################
echo -e "\n**====**\nFULL CODE:\n**====**" >> "$COMBINED_OUTPUT"
if [ -f "$FILE_PATH" ]; then
    cat "$FILE_PATH" >> "$COMBINED_OUTPUT"
else
    echo "Error: Could not read analyzed file: $FILE_PATH"
fi

###############################################################################
# 7. Write to local file (always)
# 7. Write to local or custom directory
OUTPUT_FILE="analysis_results_$(basename "$FILE_PATH").txt"

if [ -n "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    OUTPUT_FILE="$OUTPUT_DIR/$OUTPUT_FILE"
fi

cat "$COMBINED_OUTPUT" > "$OUTPUT_FILE"
echo "Results also saved to $OUTPUT_FILE"

###############################################################################
# 8. Copy to clipboard (no script exit on failure)
###############################################################################
if [[ "$OSTYPE" == "darwin"* ]]; then
    if command -v pbcopy >/dev/null 2>&1; then
        if cat "$COMBINED_OUTPUT" | pbcopy 2>/dev/null; then
            echo "Analysis results copied to clipboard (macOS)."
        else
            echo "Clipboard copy failed with pbcopy."
        fi
    else
        echo "pbcopy not found. Could not copy to clipboard."
    fi
elif command -v xclip >/dev/null 2>&1; then
    if cat "$COMBINED_OUTPUT" | xclip -selection clipboard 2>/dev/null; then
        echo "Analysis results copied to clipboard (Linux - xclip)."
    else
        echo "Clipboard copy failed with xclip."
    fi
elif command -v xsel >/dev/null 2>&1; then
    if cat "$COMBINED_OUTPUT" | xsel --clipboard 2>/dev/null; then
        echo "Analysis results copied to clipboard (Linux - xsel)."
    else
        echo "Clipboard copy failed with xsel."
    fi
else
    echo "Could not copy to clipboard. Please install xclip or xsel on Linux, or run on macOS."
fi

echo "All analyses are complete."
exit 0
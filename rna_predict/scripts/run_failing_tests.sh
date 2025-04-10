#!/bin/bash

# =========================================================================
# RNA_PREDICT Test Runner with Progressive Coverage Goals
# Aligned with Kaggle Competition Timeline
# =========================================================================

# Get the number of CPU cores
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    NUM_CORES=$(sysctl -n hw.ncpu)
else
    # Linux
    NUM_CORES=$(nproc)
fi

# Calculate 75% of available cores for parallel execution
NUM_WORKERS=$(( NUM_CORES * 75 / 100 ))

# Ensure at least 1 worker
if [ $NUM_WORKERS -lt 1 ]; then
    NUM_WORKERS=1
fi

# Create coverage directory if it doesn't exist
mkdir -p coverage

# =========================================================================
# Kaggle Competition Timeline Configuration
# =========================================================================

# Kaggle competition dates
# These dates are used for reference but actual dates are loaded from config
LEADERBOARD_REFRESH="2025-04-23"
FINAL_SUBMISSION="2025-05-29"
FUTURE_DATA_PHASE_START="2025-06-01"

# Current date
CURRENT_DATE=$(date +%Y-%m-%d)

# Configuration file for coverage goals
CONFIG_FILE=".coverage_config.json"

# Create a backup of the config file if it exists
if [ -f "$CONFIG_FILE" ]; then
    cp "$CONFIG_FILE" "${CONFIG_FILE}.bak"

    # Check if the file is valid JSON
    if ! jq . "$CONFIG_FILE" > /dev/null 2>&1; then
        echo "Config file is corrupted. Restoring from backup or creating new."
        if [ -f "${CONFIG_FILE}.bak" ] && jq . "${CONFIG_FILE}.bak" > /dev/null 2>&1; then
            cp "${CONFIG_FILE}.bak" "$CONFIG_FILE"
            echo "Restored from backup."
        else
            echo "Creating new config file."
            rm -f "$CONFIG_FILE"
        fi
    fi
fi

# Create default config if it doesn't exist
if [ ! -f "$CONFIG_FILE" ]; then
    cat > "$CONFIG_FILE" << EOF
{
    "last_updated": "$(date +%Y-%m-%d)",
    "base_coverage": 80,
    "current_coverage": 80,
    "max_coverage": 95,
    "phase_coverage": {
        "exploration": {
            "start_date": "2025-02-27",
            "end_date": "2025-03-27",
            "overall": 80,
            "critical_modules": 85,
            "standard_modules": 75,
            "utility_modules": 70
        },
        "development": {
            "start_date": "2025-03-28",
            "end_date": "2025-04-22",
            "overall": 85,
            "critical_modules": 90,
            "standard_modules": 85,
            "utility_modules": 75
        },
        "optimization": {
            "start_date": "2025-04-23",
            "end_date": "2025-05-15",
            "overall": 90,
            "critical_modules": 95,
            "standard_modules": 90,
            "utility_modules": 80
        },
        "final_submission": {
            "start_date": "2025-05-16",
            "end_date": "2025-05-29",
            "overall": 95,
            "critical_modules": 98,
            "standard_modules": 95,
            "utility_modules": 85
        }
    },
    "module_categories": {
        "critical_modules": [
            "rna_predict.pipeline.stageB",
            "rna_predict.pipeline.stageD.diffusion"
        ],
        "standard_modules": [
            "rna_predict.pipeline.stageA",
            "rna_predict.pipeline.stageC",
            "rna_predict.pipeline.stageD.tensor_fixes"
        ],
        "utility_modules": [
            "rna_predict.utils",
            "rna_predict.scripts",
            "rna_predict.dataset"
        ]
    }
}
EOF
fi

# =========================================================================
# Coverage Goal Determination Functions
# =========================================================================

# Function to determine current phase and coverage goal
get_coverage_goal() {
    # Parse the config file using jq
    if command -v jq >/dev/null 2>&1; then
        # Get current phase based on date
        CURRENT_PHASE=""
        for phase in exploration development optimization final_submission; do
            PHASE_START=$(jq -r ".phase_coverage.$phase.start_date" "$CONFIG_FILE")
            PHASE_END=$(jq -r ".phase_coverage.$phase.end_date" "$CONFIG_FILE")

            if [[ "$CURRENT_DATE" > "$PHASE_START" || "$CURRENT_DATE" == "$PHASE_START" ]] && [[ "$CURRENT_DATE" < "$PHASE_END" || "$CURRENT_DATE" == "$PHASE_END" ]]; then
                CURRENT_PHASE=$phase
                break
            fi
        done

        # If no phase matches (e.g., after competition), use the last phase
        if [ -z "$CURRENT_PHASE" ]; then
            if [[ "$CURRENT_DATE" > "$FINAL_SUBMISSION" ]]; then
                CURRENT_PHASE="final_submission"
            else
                CURRENT_PHASE="exploration"  # Default to first phase
            fi
        fi

        # Get current and target coverage for current phase
        CURRENT_COVERAGE=$(jq -r ".current_coverage" "$CONFIG_FILE")
        TARGET_COVERAGE=$(jq -r ".phase_coverage.$CURRENT_PHASE.overall" "$CONFIG_FILE")

        # Calculate days until next phase or deadline
        if [ "$CURRENT_PHASE" == "exploration" ]; then
            NEXT_MILESTONE="development phase"
            NEXT_DATE=$(jq -r ".phase_coverage.development.start_date" "$CONFIG_FILE")
            PHASE_START_DATE=$(jq -r ".phase_coverage.exploration.start_date" "$CONFIG_FILE")
            PHASE_END_DATE=$(jq -r ".phase_coverage.exploration.end_date" "$CONFIG_FILE")
            PREV_PHASE=""
        elif [ "$CURRENT_PHASE" == "development" ]; then
            NEXT_MILESTONE="leaderboard refresh"
            NEXT_DATE="$LEADERBOARD_REFRESH"
            PHASE_START_DATE=$(jq -r ".phase_coverage.development.start_date" "$CONFIG_FILE")
            PHASE_END_DATE=$(jq -r ".phase_coverage.development.end_date" "$CONFIG_FILE")
            PREV_PHASE="exploration"
        elif [ "$CURRENT_PHASE" == "optimization" ]; then
            NEXT_MILESTONE="final submission"
            NEXT_DATE="$FINAL_SUBMISSION"
            PHASE_START_DATE=$(jq -r ".phase_coverage.optimization.start_date" "$CONFIG_FILE")
            PHASE_END_DATE=$(jq -r ".phase_coverage.optimization.end_date" "$CONFIG_FILE")
            PREV_PHASE="development"
        else
            NEXT_MILESTONE="future data phase"
            NEXT_DATE="$FUTURE_DATA_PHASE_START"
            PHASE_START_DATE=$(jq -r ".phase_coverage.final_submission.start_date" "$CONFIG_FILE")
            PHASE_END_DATE=$(jq -r ".phase_coverage.final_submission.end_date" "$CONFIG_FILE")
            PREV_PHASE="optimization"
        fi

        # Calculate days until next milestone and total phase duration
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # Check if GNU date (gdate) is available
            if command -v gdate >/dev/null 2>&1; then
                # Use GNU date for consistent calculations
                DAYS_UNTIL_NEXT=$(( ($(gdate -d "$NEXT_DATE" +%s) - $(gdate -d "$CURRENT_DATE" +%s)) / 86400 ))
                DAYS_SINCE_PHASE_START=$(( ($(gdate -d "$CURRENT_DATE" +%s) - $(gdate -d "$PHASE_START_DATE" +%s)) / 86400 ))
                TOTAL_PHASE_DAYS=$(( ($(gdate -d "$PHASE_END_DATE" +%s) - $(gdate -d "$PHASE_START_DATE" +%s)) / 86400 ))
            else
                # macOS date command - simplified to avoid format issues
                # Just calculate a rough estimate based on month differences
                NEXT_MONTH=$(echo $NEXT_DATE | cut -d'-' -f2)
                NEXT_DAY=$(echo $NEXT_DATE | cut -d'-' -f3)
                CURRENT_MONTH=$(echo $CURRENT_DATE | cut -d'-' -f2)
                CURRENT_DAY=$(echo $CURRENT_DATE | cut -d'-' -f3)

                # Simple calculation: (next_month - current_month) * 30 + (next_day - current_day)
                DAYS_UNTIL_NEXT=$(( (10#$NEXT_MONTH - 10#$CURRENT_MONTH) * 30 + (10#$NEXT_DAY - 10#$CURRENT_DAY) ))

                # Calculate phase start and current date difference
                PHASE_START_MONTH=$(echo $PHASE_START_DATE | cut -d'-' -f2)
                PHASE_START_DAY=$(echo $PHASE_START_DATE | cut -d'-' -f3)
                DAYS_SINCE_PHASE_START=$(( (10#$CURRENT_MONTH - 10#$PHASE_START_MONTH) * 30 + (10#$CURRENT_DAY - 10#$PHASE_START_DAY) ))

                # Calculate total phase duration
                PHASE_END_MONTH=$(echo $PHASE_END_DATE | cut -d'-' -f2)
                PHASE_END_DAY=$(echo $PHASE_END_DATE | cut -d'-' -f3)
                TOTAL_PHASE_DAYS=$(( (10#$PHASE_END_MONTH - 10#$PHASE_START_MONTH) * 30 + (10#$PHASE_END_DAY - 10#$PHASE_START_DAY) ))

                echo "Note: For more accurate date calculations on macOS, consider installing GNU coreutils:"
                echo "  brew install coreutils"
            fi

            # Ensure values aren't negative
            if [ $DAYS_UNTIL_NEXT -lt 0 ]; then
                DAYS_UNTIL_NEXT=0
            fi
            if [ $DAYS_SINCE_PHASE_START -lt 0 ]; then
                DAYS_SINCE_PHASE_START=0
            fi
            if [ $TOTAL_PHASE_DAYS -lt 1 ]; then
                TOTAL_PHASE_DAYS=1
            fi
        else
            # Linux date command
            DAYS_UNTIL_NEXT=$(( ($(date -d "$NEXT_DATE" +%s) - $(date -d "$CURRENT_DATE" +%s)) / 86400 ))
            DAYS_SINCE_PHASE_START=$(( ($(date -d "$CURRENT_DATE" +%s) - $(date -d "$PHASE_START_DATE" +%s)) / 86400 ))
            TOTAL_PHASE_DAYS=$(( ($(date -d "$PHASE_END_DATE" +%s) - $(date -d "$PHASE_START_DATE" +%s)) / 86400 ))
        fi

        # Phase transition smoothing: Check if we just entered a new phase (within the last 3 days)
        if [ $DAYS_SINCE_PHASE_START -lt 3 ] && [ ! -z "$PREV_PHASE" ]; then
            # Get the target coverage of the previous phase
            PREV_TARGET=$(jq -r ".phase_coverage.$PREV_PHASE.overall" "$CONFIG_FILE")

            # If current coverage is significantly below the previous target,
            # adjust the starting point to be closer to the actual coverage
            if (( $(echo "$CURRENT_COVERAGE < $PREV_TARGET - 2" | bc -l) )); then
                # Use the actual coverage plus a small buffer as the new starting point
                CURRENT_COVERAGE=$(echo "$CURRENT_COVERAGE + 1" | bc)
                jq ".current_coverage = $CURRENT_COVERAGE" "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" && mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"
                echo "Phase transition detected. Adjusted starting coverage to $CURRENT_COVERAGE%"
            fi
        fi

        # Calculate days since last run for maximum daily increase cap
        LAST_UPDATED=$(jq -r ".last_updated" "$CONFIG_FILE")
        DAYS_SINCE_LAST_RUN=1  # Default to 1 day

        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS date handling - simplified
            LAST_MONTH=$(echo $LAST_UPDATED | cut -d'-' -f2)
            LAST_DAY=$(echo $LAST_UPDATED | cut -d'-' -f3)
            DAYS_SINCE_LAST_RUN=$(( (10#$CURRENT_MONTH - 10#$LAST_MONTH) * 30 + (10#$CURRENT_DAY - 10#$LAST_DAY) ))
        else
            # Linux date handling
            DAYS_SINCE_LAST_RUN=$(( ($(date -d "$CURRENT_DATE" +%s) - $(date -d "$LAST_UPDATED" +%s)) / 86400 ))
        fi

        # Ensure it's at least 1
        if [ $DAYS_SINCE_LAST_RUN -lt 1 ]; then
            DAYS_SINCE_LAST_RUN=1
        fi

        # Maximum daily increase cap
        MAX_DAILY_INCREASE=0.5  # Maximum 0.5% increase per day
        MAX_ALLOWED_INCREASE=$(echo "$MAX_DAILY_INCREASE * $DAYS_SINCE_LAST_RUN" | bc)

        # Calculate the gradual coverage goal based on days into the phase
        # Formula: base_coverage + (days_since_phase_start / total_phase_days) * (target_coverage - base_coverage)
        COVERAGE_INCREASE=$(echo "scale=2; ($DAYS_SINCE_PHASE_START / $TOTAL_PHASE_DAYS) * ($TARGET_COVERAGE - $CURRENT_COVERAGE)" | bc)

        # Cap the coverage increase
        if (( $(echo "$COVERAGE_INCREASE > $MAX_ALLOWED_INCREASE" | bc -l) )); then
            COVERAGE_INCREASE=$MAX_ALLOWED_INCREASE
            echo "Coverage increase capped at $MAX_ALLOWED_INCREASE% per day"
        fi

        COVERAGE_GOAL=$(echo "scale=2; $CURRENT_COVERAGE + $COVERAGE_INCREASE" | bc)

        # Ensure coverage goal doesn't exceed the target for the phase
        if (( $(echo "$COVERAGE_GOAL > $TARGET_COVERAGE" | bc -l) )); then
            COVERAGE_GOAL=$TARGET_COVERAGE
        fi

        # Update the current coverage and last_updated in the config file
        # Only update if the calculated goal is higher than the stored current coverage
        STORED_CURRENT_COVERAGE=$(jq -r ".current_coverage" "$CONFIG_FILE")
        if (( $(echo "$COVERAGE_GOAL > $STORED_CURRENT_COVERAGE" | bc -l) )); then
            # Create a temporary file with the updated current_coverage and last_updated
            jq ".current_coverage = $COVERAGE_GOAL | .last_updated = \"$CURRENT_DATE\"" "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" && mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"
        fi

        echo "Current Kaggle phase: $CURRENT_PHASE"
        echo "Base coverage: $CURRENT_COVERAGE%"
        echo "Target coverage for this phase: $TARGET_COVERAGE%"
        echo "Current coverage goal (day $DAYS_SINCE_PHASE_START of $TOTAL_PHASE_DAYS): $COVERAGE_GOAL%"
        echo "Days until $NEXT_MILESTONE: $DAYS_UNTIL_NEXT"
        echo "Days since last run: $DAYS_SINCE_LAST_RUN"

        # Return the coverage goal
        echo $COVERAGE_GOAL
    else
        # Fallback if jq is not available
        echo "Warning: jq not found, using default coverage target of 80%"
        echo 80
    fi
}

# =========================================================================
# Main Test Execution
# =========================================================================

# Get the coverage goal
COVERAGE_GOAL=$(get_coverage_goal)
# Extract just the number from the output
COVERAGE_GOAL=$(echo "$COVERAGE_GOAL" | grep -o '[0-9]\+' | tail -1)

# Check if we should run module-specific coverage
RUN_MODULE_SPECIFIC=false
if [ "$1" == "--module-specific" ]; then
    RUN_MODULE_SPECIFIC=true
fi

# Automatically find all Python test files in the tests directory
test_files=$(find tests -type f -name "test_*.py")

if [ "$RUN_MODULE_SPECIFIC" = true ]; then
    # Run tests for each module with its specific goal
    echo "Running module-specific coverage tests..."

    # Parse module categories and their coverage goals
    if command -v jq >/dev/null 2>&1; then
        # Use the same phase determination as in get_coverage_goal
        CURRENT_PHASE=""
        for phase in exploration development optimization final_submission; do
            PHASE_START=$(jq -r ".phase_coverage.$phase.start_date" "$CONFIG_FILE")
            PHASE_END=$(jq -r ".phase_coverage.$phase.end_date" "$CONFIG_FILE")

            if [[ "$CURRENT_DATE" > "$PHASE_START" || "$CURRENT_DATE" == "$PHASE_START" ]] && [[ "$CURRENT_DATE" < "$PHASE_END" || "$CURRENT_DATE" == "$PHASE_END" ]]; then
                CURRENT_PHASE=$phase
                break
            fi
        done

        # If no phase matches (e.g., after competition), use the last phase
        if [ -z "$CURRENT_PHASE" ]; then
            if [[ "$CURRENT_DATE" > "$FINAL_SUBMISSION" ]]; then
                CURRENT_PHASE="final_submission"
            else
                CURRENT_PHASE="exploration"  # Default to first phase
            fi
        fi

        # For each module category
        for category in critical_modules standard_modules utility_modules; do
            CATEGORY_TARGET=$(jq -r ".phase_coverage.$CURRENT_PHASE.$category" "$CONFIG_FILE")
            # Ensure it's just a number
            CATEGORY_TARGET=$(echo "$CATEGORY_TARGET" | grep -o '[0-9]\+')

            echo "Testing $category with coverage goal: $CATEGORY_TARGET%"

            # Get modules in this category
            modules=$(jq -r ".module_categories.${category}[]" "$CONFIG_FILE")

            for module in $modules; do
                echo "Testing module: $module"
                module_path=$(echo $module | sed 's/\./\//g')

                # Find tests for this module
                # This pattern matches tests that might be in a directory structure mirroring the module
                module_tests=$(find tests -type f -name "test_*.py" | grep -i "$(basename $module_path)" || echo "")

                if [ -z "$module_tests" ]; then
                    echo "No tests found for module $module, skipping..."
                    continue
                fi

                # Run tests for this module
                pytest $module_tests -v -n $NUM_WORKERS --dist=loadfile --timeout=300 \
                    --memray --most-allocations=10 --stacks=5 \
                    --cov=$module \
                    --cov-report=html:"coverage/$(basename $module_path)" \
                    --cov-report=term-missing \
                    --cov-fail-under=$CATEGORY_TARGET

                if [ $? -ne 0 ]; then
                    echo "Tests for $module failed or coverage is below threshold!"
                    exit 1
                fi
            done
        done
    else
        echo "jq not found, cannot run module-specific tests"
        exit 1
    fi
else
    # Run tests in parallel using pytest-xdist with increased timeout, memory profiling, and coverage
    echo "Running tests in parallel using $NUM_WORKERS workers (75% of $NUM_CORES cores) with memory profiling and coverage..."
    echo "Current coverage goal: $COVERAGE_GOAL%"

    pytest $test_files -v -n $NUM_WORKERS --dist=loadfile --timeout=300 \
        --memray --most-allocations=10 --stacks=5 \
        --cov=rna_predict \
        --cov-report=html \
        --cov-report=term-missing \
        --cov-fail-under=$COVERAGE_GOAL

    exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo "Some tests failed or coverage is below threshold!"
        exit 1
    fi

    echo "All tests passed successfully!"
    echo "Coverage report generated in coverage/html/index.html"
fi
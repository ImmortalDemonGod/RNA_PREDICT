#!/bin/bash

# Create necessary directories
mkdir -p scripts/test_utils
mkdir -p scripts/coverage
mkdir -p scripts/automation
mkdir -p scripts/analysis

# Move test-related scripts
mv rna_predict/scripts/batch_test_generator.py scripts/test_utils/
mv rna_predict/scripts/mark_slow_tests.py scripts/test_utils/
mv rna_predict/scripts/run_failing_tests.sh scripts/test_utils/
mv rna_predict/scripts/hypot_test_gen.py scripts/test_utils/

# Move coverage-related files
mv rna_predict/scripts/.coverage_config.json scripts/coverage/
mv rna_predict/scripts/update_coverage_config.py scripts/coverage/
mv rna_predict/scripts/show_coverage.py scripts/coverage/
cp -R rna_predict/scripts/coverage/* scripts/coverage/ 2>/dev/null || true
rm -rf rna_predict/scripts/coverage

# Move automation scripts
mv rna_predict/scripts/github_automation.sh scripts/automation/
mv rna_predict/scripts/commit_individual_files.sh scripts/automation/
mv rna_predict/scripts/create_github_issues.py scripts/automation/

# Move analysis scripts
mv rna_predict/scripts/analyze_code.sh scripts/analysis/
mv rna_predict/scripts/batch_analyze.sh scripts/analysis/
mv rna_predict/scripts/count_python_lines.py scripts/analysis/

# Move pipeline scripts to root scripts directory
mv rna_predict/scripts/run_all_pipeline.py scripts/
mv rna_predict/scripts/run_mutation_tests.sh scripts/

# Move screen finder app
mv rna_predict/scripts/screen_finder_app scripts/

# Clean up
rm -f rna_predict/scripts/.DS_Store
rm -f rna_predict/scripts/run_mutation_testing.sh  # Duplicate file
rm -f rna_predict/scripts/reorg_tests.sh  # One-time use script

# Create README files
echo "# Test Utilities
This directory contains scripts for managing and running tests." > scripts/test_utils/README.md

echo "# Coverage Tools
This directory contains tools for managing code coverage analysis." > scripts/coverage/README.md

echo "# Automation Scripts
This directory contains scripts for automating various development tasks." > scripts/automation/README.md

echo "# Analysis Tools
This directory contains scripts for code analysis and metrics." > scripts/analysis/README.md

echo "# RNA Prediction Scripts
This directory contains the main scripts for running the RNA prediction pipeline and related tools.

## Directory Structure
- test_utils/: Test-related utilities
- coverage/: Coverage analysis tools
- automation/: Development automation scripts
- analysis/: Code analysis tools
- screen_finder_app/: Screen finding application" > scripts/README.md 
# Code Quality Improvement: Best Practices and Lessons Learned

This document outlines key lessons and best practices for improving code quality in complex projects, based on our team's experiences with refactoring and code analysis.

## Quick Start: Code Quality Improvement Workflow

This guide outlines a step-by-step process for improving code quality, based on real-world experiences and lessons learned from previous failures.

### Step 1: Baseline Assessment

1. **Clean Slate First**:
   ```bash
   rm analysis_results_*.txt
   ```
   **Why:** Old analysis files have misled developers into thinking they were looking at current results.

2. **Full Analysis**:
   ```bash
   bash rna_predict/scripts/analyze_code.sh path/to/file.py
   ```
   **Expected output:** Creates `analysis_results_file.py.txt` with comprehensive analysis

3. **Review Complete Report**:
   ```bash
   cat analysis_results_file.py.txt
   ```
   **Why:** Skipping this step and using only filtered views has caused teams to miss critical context.

4. **Check Current Score**:
   ```bash
   cat analysis_results_file.py.txt | grep "score"
   ```
   **Rule of thumb:** 
   - Scores below 5.0: Critical improvement needed
   - Scores 5.0-7.0: Significant refactoring required
   - Scores 7.0-8.5: Targeted improvements needed
   - Scores above 8.5: Fine-tuning only

### Step 2: Identify Root Issues (Not Just Symptoms)

1. **Analyze File Size**:
   ```bash
   wc -l path/to/file.py
   ```
   **Decision point:** If file exceeds 500 lines, prioritize splitting it before detailed refactoring.

2. **Find Complex Methods**:
   ```bash
   cat analysis_results_file.py.txt | grep "Complex Method\|Large Method\|Bumpy Road" -A 10
   ```
   **Focus on:** Methods with cyclomatic complexity > 15 or > 50 lines

3. **Identify Type Issues**:
   ```bash
   mypy --strict path/to/file.py
   ```
   **Common mistake:** Treating type errors as separate from design problems. They usually reveal deeper issues.

4. **Check Test Coverage**:
   ```bash
   python -m pytest --cov=module.path tests/path/to/test_file.py -v
   ```
   **Rule of thumb:** Don't refactor code with < 60% test coverage without adding tests first

### Step 3: Create Tactical Plan

Based on the assessment, create a plan following this priority order (learned from past failures):

1. **If file > 500 lines**: Split into logical modules first
   ```
   Example problem:
   - Original: ml_utils.py (900 lines)
   - Better: atom_operations.py, tensor_utils.py, sequence_processing.py
   ```

2. **If high cyclomatic complexity**: Extract helper functions
   ```
   Example problem:
   - Original: process_data() function with complexity of 25
   - Better: Five helper functions with complexity of 5 each
   ```

3. **If deep nesting**: Invert conditions or extract blocks
   ```
   Example problem:
   - Original: 5 levels of nested if-statements
   - Better: Early returns or extracted conditional blocks
   ```

4. **If duplicate logic**: Create shared utilities
   ```
   Example problem:
   - Original: Same tensor manipulation in 6 different functions
   - Better: Shared utility function with clear documentation
   ```

### Step 4: Implementation (With Verification)

For each change:

1. **Always run component tests before making changes**:
   ```bash
   python -m pytest tests/stageX/component_tests/test_specific_component.py -v
   ```
   **Common failure:** Not having a baseline of working tests

2. **Make targeted changes (one issue at a time)**:
   ```
   Example:
   - Extract ONE complex function
   - Fix ONE type error pattern
   - NOT overhauling multiple systems at once
   ```

3. **Run component tests after each change**:
   ```bash
   python -m pytest tests/stageX/component_tests/test_specific_component.py -v
   ```
   **Immediate feedback:** Catch regressions before proceeding

4. **Check for new linter/type issues**:
   ```bash
   mypy --strict path/to/file.py
   ```
   **Warning:** Changes often introduce new type errors

5. **After multiple related changes, run stage pipeline tests**:
   ```bash
   python -m pytest tests/stageX/test_stage_x_*.py -v
   ```
   **Critical mistake:** Skipping integration tests until the end of refactoring

### Step 5: Verify Improvement

1. **Re-run full analysis**:
   ```bash
   rm analysis_results_*.txt  # Start fresh
   bash rna_predict/scripts/analyze_code.sh path/to/file.py
   ```

2. **Compare score improvement**:
   ```bash
   cat analysis_results_file.py.txt | grep "score"
   ```
   **Expected improvements:**
   - File splitting: +1.0-2.0 points
   - Complexity reduction: +0.5-1.0 points per function
   - Type fixes: +0.3-0.7 points

3. **If score improved < 0.3 points**:
   - You likely addressed symptoms, not root causes
   - The most common reason is keeping complex code in the same file
   - Consider more aggressive module separation

### Step 6: Document Lessons

After completing refactoring, add a brief comment to your commit message:

```
Refactored X to improve code quality (score: 5.2 → 7.8)
- Split large file into 3 modules
- Reduced complexity of function Y from 25 to 8
- Fixed 12 type errors
```

### Real-World Example: Transforming Poor Quality Code

**Initial state**:
- `ml_utils.py` (900 lines)
- Code smell score: 5.04
- Cyclomatic complexity of `rename_symmetric_atoms`: 25
- No type hints

**Analysis revealed**:
- Deep nesting in `rename_symmetric_atoms`
- Multiple responsibilities in one file
- Type errors when handling tensor operations

**Successful approach**:
1. Added type hints to understand data flow
2. Extracted helper function `_validate_seq_list` from `scn_atom_embedd`
3. Split processing logic in `rename_symmetric_atoms`
4. Added proper error handling and validation
5. Result: Score improved to 6.5

**Failed approach** (attempted first):
1. Fixed individual type errors without understanding root causes
2. Added small helper functions but kept in same file
3. Added documentation without simplifying logic
4. Result: Score barely improved to 5.2

### Practical Troubleshooting: Common Pitfalls

Here are solutions to common problems faced during code quality improvement:

#### 1. "I made changes but the score barely improved"

**Common causes:**
- Only superficial changes (formatting, variable names)
- Fixed type errors without addressing underlying design issues
- Added documentation without simplifying logic

**Solutions:**
- Use the "Direct CodeScene Review" for immediate feedback:
  ```bash
  cs review path/to/file.py --output-format json
  ```
- Focus on structural changes (file splitting, function extraction)
- Prioritize reducing cyclomatic complexity over other metrics

#### 2. "Tests are failing after my refactoring"

**Common causes:**
- Changed function signatures without updating all call sites
- Modified return types or value ranges
- Changed error handling behavior

**Solutions:**
- Refactor in smaller steps with test verification after each
- Start by adding type hints without changing behavior
- Document function contracts before changing implementation
- Use the debugger to compare before/after execution paths:
  ```bash
  python -m pdb -c "break function_name" -c continue tests/path_to_test.py
  ```

#### 3. "I've reached analysis paralysis - too many issues to fix"

**Common causes:**
- Trying to fix everything at once
- No clear prioritization strategy
- Unclear what "good" looks like

**Solutions:**
- Use this prioritization formula:
  ```
  Priority = (Complexity × 0.4) + (Size × 0.3) + (Dependencies × 0.3)
  ```
- Create a spreadsheet listing issues with their priority scores
- Set a concrete target (e.g., "Improve score from 5.2 to 6.0 this sprint")
- Timebox refactoring sessions (2 hours max per function)

#### 4. "Splitting the file breaks too many dependencies"

**Common causes:**
- Circular dependencies
- Poor separation of concerns in original design
- Functions sharing too many internal variables

**Solutions:**
- Create an intermediate "utils" module for shared functionality
- Use dependency inversion (inject dependencies rather than importing)
- Create proper interfaces before implementation
- Map dependencies visually before starting:
  ```bash
  pydeps path/to/file.py --max-bacon=2 --cluster
  ```

#### 5. "Type hints are causing more problems than they solve"

**Common causes:**
- Adding complex Union/Optional types without refactoring
- Forcing types onto poorly designed functions
- Inconsistent return types

**Solutions:**
- Use type hints to identify design problems, not mask them
- Start with simple types, then refine
- Create custom type aliases for complex structures:
  ```python
  AtomCoordinates = Dict[str, np.ndarray]
  ```
- Use consistent return types (don't return None sometimes and int others)

## Initial Assessment

1. **Holistic Analysis First**: Read and understand the entire analysis report before making any changes. Identify root causes rather than just symptoms.

2. **File Size Matters**: Large files are inherently problematic regardless of internal code quality. Breaking a large file into multiple logical modules should be considered before detailed function refactoring.

3. **Identify Architectural Issues**: Look for structural problems that affect multiple parts of the codebase, not just local code smells.

4. **Understand Quality Metrics**: Know how code quality scores are calculated to effectively target improvements that will have the greatest impact.

## Planning and Prioritization

1. **Create a Refactoring Roadmap**: Develop a structured plan that addresses issues in dependency order.

2. **Set Realistic Targets**: Understand what score improvements are realistically achievable based on structural constraints.

3. **Analyze Test Coverage**: Identify areas with poor test coverage before refactoring to avoid breaking functionality.

4. **Identify High-Value Targets**: Focus on the 20% of issues that cause 80% of the problems.

## Implementation Approach

1. **Module Separation**: Split large files into smaller, focused modules with clear responsibilities before deep refactoring.

2. **Preserve Interfaces**: When extracting functionality, maintain the same public interfaces to minimize ripple effects.

3. **Incremental Verification**: Run tests after each significant change to catch regressions early.

4. **Address TypeScript/Type Errors**: Fix typing issues as they often reveal deeper design problems.

## Measuring Progress

1. **Track Multiple Metrics**: Monitor all relevant quality indicators, not just a single score.

2. **Understand Score Components**: Know how different issues contribute to the overall score.

3. **Verify Real Improvements**: Ensure changes improve maintainability, not just metrics.

4. **Watch for Negative Side Effects**: Monitor if improvements in one area cause degradation in others.

## Common Pitfalls to Avoid

1. **Function Extraction Without Reorganization**: Creating helper functions in the same file can increase file size and complexity.

2. **Over-focusing on Individual Functions**: Missing forest-level issues while fixing tree-level problems.

3. **Losing Context in Analysis**: Using narrow grep filters that miss important contextual information.

4. **Missing Low-Hanging Fruit**: Overlooking simple fixes with high impact (like file splitting).

5. **Ignoring Global Patterns**: Missing repeated patterns that could be addressed with a single systemic change.

## Testing Strategy

1. **Leverage Modular Testing**: In a modular architecture, run tests specific to the component or pipeline stage you're refactoring. This is more efficient than running the entire test suite for every change and provides faster feedback.

2. **Stage Integration Testing**: Run integration tests only after component tests pass to verify that changes haven't broken cross-module interactions.

3. **Create Tests for Uncovered Code**: Add tests for previously uncovered code paths before refactoring.

4. **Validate Edge Cases**: Ensure that refactored code handles edge cases correctly.

5. **Test Performance Impact**: Check that refactoring doesn't introduce performance regressions.

## Useful Commands

To help you with code quality assessment and analysis, here are some useful commands:

* To analyze all python files in the directory:
```bash
bash rna_predict/scripts/analyze_code.sh
```

* To analyze a specific file:
```bash
bash rna_predict/scripts/analyze_code.sh path/to/file.py
```

* To clean up old analysis files before running new analysis:
```bash
rm analysis_results_*.txt
```

* To run staged pipeline tests, which are important after modifying components:
```bash
python -m pytest tests/stageX/test_stage_x_*.py -v
```

* To run the CodeScene CLI tool directly for immediate code quality feedback:
```bash
cs review path/to/file.py --output-format json
```

Remember to review command output thoroughly rather than relying on filtered views for critical decisions.

## Quick Reference: Common Commands

### Assessment Commands

| Purpose | Command | When to Use |
|---------|---------|------------|
| Full code analysis | `bash rna_predict/scripts/analyze_code.sh path/to/file.py` | Start of refactoring, after major changes |
| Clean old results | `rm analysis_results_*.txt` | Before running new analysis |
| View code smell score | `cat analysis_results_file.py.txt \| grep "score"` | Quick check of quality level |
| Count lines of code | `wc -l path/to/file.py` | Determine if file should be split |
| Find complex methods | `cat analysis_results_file.py.txt \| grep "Complex Method" -A 10` | Identify highest-priority refactoring targets |
| Direct quality review | `cs review path/to/file.py --output-format json` | Immediate feedback during refactoring |
| Type checking | `mypy --strict path/to/file.py` | Identify type-related issues |
| Lint checking | `ruff check path/to/file.py` | Identify style and potential bugs |

### Testing Commands

| Purpose | Command | When to Use |
|---------|---------|------------|
| Run component tests | `python -m pytest tests/stageX/component_tests/test_specific.py -v` | Before/after each change |
| Run with coverage | `python -m pytest --cov=module.path tests/path/to/test.py -v` | Identify untested code |
| Run stage pipeline tests | `python -m pytest tests/stageX/test_stage_x_*.py -v` | After completing related changes |
| Debug test failures | `python -m pdb -c "break function_name" -c continue tests/path_to_test.py` | When tests fail after refactoring |

### Dependency Analysis

| Purpose | Command | When to Use |
|---------|---------|------------|
| Visualize dependencies | `pydeps path/to/file.py --max-bacon=2 --cluster` | Before splitting large files |
| Find import usages | `grep -r "from module import" --include="*.py" .` | When restructuring modules |
| Check circular dependencies | `import-linter --config-file .importlinter` | After file restructuring |

### Implementation Workflow

1. **Baseline**: Run tests and analysis
2. **Plan**: Identify highest-priority issues using score breakdown
3. **Change**: Make one targeted improvement
4. **Verify**: Run tests and quick analysis
5. **Repeat**: Until target score is reached
6. **Validate**: Run comprehensive analysis and full test suite

Remember to review complete analysis output rather than relying solely on filtered views for critical decisions.

## Conclusion

Improving code quality is an iterative process that requires both technical skill and strategic thinking. By following the guidelines outlined in this document and leveraging the provided commands, you can systematically transform complex, difficult-to-maintain code into cleaner, more robust implementations. 
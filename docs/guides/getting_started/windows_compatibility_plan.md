# Technical Implementation Plan: Windows Compatibility for RNA_PREDICT Scripts

**Objective:** Modify the `rna_predict` project codebase to allow `rna_predict/scripts/run_all_pipeline.py` and `rna_predict/scripts/run_failing_tests.py` to execute successfully on a Windows operating system.

**Overall Approach:** Identify and refactor platform-specific code, primarily focusing on file path manipulation, while verifying external dependencies. The goal is cross-platform compatibility rather than Windows-only scripts.

## 1. Task Breakdown

1.  Modify path string definitions in `rna_predict/pipeline/stageA/run_stageA.py` to use `os.path.join`.
2.  Modify path string construction in `rna_predict/scripts/run_failing_tests.py` to use `os.path.join`.
3.  Document external dependency requirements for Windows (Java).
4.  Verify core library installation instructions (PyTorch, Pandas, NumPy, potentially Transformers).
5.  Test execution of both target scripts (`run_all_pipeline.py`, `run_failing_tests.py`) on Windows.

## 2. File Modifications

* `rna_predict/pipeline/stageA/run_stageA.py`
* `rna_predict/scripts/run_failing_tests.py`

## 3. Code Section Adjustments & Logic

### File: `rna_predict/pipeline/stageA/run_stageA.py`

* **Location:** Lines 120, 122, 124, 144, 145 (and potentially others defining relative paths).
* **Change:** Replace hardcoded forward slashes (`/`) in path strings with `os.path.join()`.
* **Rationale:** Ensures paths are constructed using the correct OS-specific separator (`\` on Windows, `/` on Unix-like systems), improving robustness, especially when paths are passed to external processes or libraries.
* **Example (Line 120):**
    ```python
    # Before
    checkpoint_zip = "RFold/checkpoints.zip"
    # After
    checkpoint_zip = os.path.join("RFold", "checkpoints.zip")
    ```
* **Example (Line 144):**
    ```python
    # Before
    varna_jar_path = "RFold/VARNAv3-93.jar"
    # After
    varna_jar_path = os.path.join("RFold", "VARNAv3-93.jar")
    ```
* **Impact:** Minimal impact expected. Ensures correct path resolution on Windows. Requires `import os`.

### File: `rna_predict/scripts/run_failing_tests.py`

* **Location:** Line 397 (`module_path = module.replace(".", "/")`)
* **Change:** Replace the string replacement with `os.path.join`.
* **Rationale:** Correctly converts Python module dot notation (e.g., `rna_predict.utils`) into an OS-specific file path (e.g., `rna_predict\utils` on Windows).
* **Example:**
    ```python
    # Before
    module_path = module.replace(".", "/")
    # After (assuming 'os' is imported)
    module_path = os.path.join(*module.split('.'))
    ```
* **Impact:** Corrects potential path issues when specifying coverage sources (`--cov={module}`) or finding related test files on Windows.

* **Location:** Line 421 (f-string for `--cov-report` argument)
* **Change:** Construct the HTML report path using `os.path.join`.
* **Rationale:** Ensures the path provided to pytest for the HTML coverage report uses the correct OS separator.
* **Example:**
    ```python
    # Before
    f"--cov-report=html:coverage/{os.path.basename(module_path)}",
    # After (assuming 'os' is imported)
    f"--cov-report=html:{os.path.join('coverage', os.path.basename(module_path))}",
    ```
* **Impact:** Ensures pytest-cov can correctly create the HTML report directory structure on Windows.

## 4. New Functions/Classes

* No new functions or classes are anticipated for these specific changes.

## 5. Dependencies & Configuration Updates

* **External Software:**
    * **Java:** Required *only* for the optional VARNA visualization feature in `run_stageA.py`. If visualization is needed on Windows, a JRE must be installed and the `java` executable must be in the system's PATH environment variable.
* **Python Libraries:**
    * Ensure `requirements.txt` (and potentially `requirements-test.txt`) list versions of core libraries (PyTorch, NumPy, Pandas, pytest, pytest-cov, potentially transformers) that are known to be compatible with Windows and the target Python version. No changes to the requirements files are planned *unless* testing reveals an incompatible library version. Installation should use standard `pip install -r requirements.txt`.

## 6. Data Structure / Interface Changes

* No changes to data structures or function/class interfaces are anticipated.

## 7. Potential Side Effects

* The changes are minor refactorings aimed at improving cross-platform compatibility. The primary risk is introducing a typo when modifying path construction; careful checking is required.
* The dependency on Java for VARNA remains; if Java is not configured correctly on Windows, the visualization step in `run_stageA.py` will fail (gracefully, based on the code's checks), but the rest of the script should proceed.

## 8. Success Criterion

* Successful, error-free execution of `python rna_predict/scripts/run_all_pipeline.py` on Windows.
* Successful, error-free execution of `python rna_predict/scripts/run_failing_tests.py` (both with and without `--module-specific`) on Windows, including generation of coverage reports.
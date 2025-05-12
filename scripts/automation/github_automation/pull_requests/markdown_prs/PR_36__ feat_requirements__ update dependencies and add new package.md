# Pull Request #36: ✨ feat(requirements): update dependencies and add new package

## Status
- State: MERGED
- Created: 2025-04-08
- Updated: 2025-04-08
- Closed: 2025-04-08
- Merged: 2025-04-08

## Changes
- Additions: 196
- Deletions: 28
- Changed Files: 5

## Author
- Name: ImmortalDemonGod
- Login: ImmortalDemonGod
- Bot: No

## Assignees
- None

## Description
### 🔨 Relax Torch Version & Dependency Update, Enhance File Handling, and Improve AdaptiveLayerNorm Tests

### Summary :memo:
This pull request resolves dependency constraints, improves the robustness of pipeline file path handling, and significantly enhances testing and documentation for the `AdaptiveLayerNorm` module.

---

### Details

#### 🚀 **Dependency Updates**
1. **Relaxed upper bound** for PyTorch (`torch`) to support newer versions.
2. **Added missing dependency**: `protenix`, resolving module import issues.

#### ♻️ **Robust File Path Handling**
- Refactored `run_all_pipeline` script to set `PROJECT_ROOT` dynamically based on the script's location.
- Updated file handling to use relative paths, improving robustness and portability across different environments.

#### ✅ **AdaptiveLayerNorm Test Enhancements**
- Implemented comprehensive tests for `AdaptiveLayerNorm`:
  - **Matched Shapes**: Verifies correct conditioning without unnecessary warnings.
  - **Broadcastable Edge Cases**: Ensures proper handling of inputs with additional broadcastable dimensions.
  - **Incompatible Shapes**: Validates that runtime errors are raised appropriately when dimensions cannot align.

- Updated documentation and examples for clearer guidance on input shapes and expected behaviors.

---

### 🐞 Bugfixes
- Corrected handling of shape mismatches in `AdaptiveLayerNorm` to ensure accurate conditioning rather than masking with warnings.

---

### Impact
These updates:
- Allow usage of modern PyTorch versions.
- Improve pipeline execution reliability.
- Enhance correctness and transparency in layer normalization behavior, backed by robust tests and clear documentation.

---

### Checks
- [x] Closed #29
- [x] Tested Changes
- [x] Stakeholder Approval

## Comments

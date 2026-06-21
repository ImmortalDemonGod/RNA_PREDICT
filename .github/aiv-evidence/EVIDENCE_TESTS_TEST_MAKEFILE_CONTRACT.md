# AIV Evidence File (v1.0)

**File:** `tests/test_makefile_contract.py`
**Commit:** `17be8e3`
**Generated:** 2026-06-20T23:54:26Z
**Protocol:** AIV v2.0 + Addendum 2.7 (Zero-Touch Mandate)

---

## Classification (required)

```yaml
classification:
  risk_tier: R1
  sod_mode: S0
  critical_surfaces: []
  blast_radius: "tests/test_makefile_contract.py"
  classification_rationale: "New test file only; tests are intentionally RED to characterize the existing defect before fixing"
  classified_by: "Claude"
  classified_at: "2026-06-20T23:54:26Z"
```

## Claim(s)

1. test_test_target_does_not_depend_on_lint fails because Makefile:47 lists 'lint' as prerequisite of 'test'
2. test_lint_recipe_does_not_use_unsafe_fixes fails because Makefile:34 uses --unsafe-fixes which mutates source
3. test_test_target_prerequisites_contain_only_test_tooling fails because 'lint' is in test prerequisites
4. No existing tests were modified or deleted during this change.

---

## Evidence

### Class E (Intent Alignment)

- **Link:** [https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/audit/02-static-audit.md#L474](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/audit/02-static-audit.md#L474)
- **Requirements Verified:** Finding s2c0l0-014: test:lint coupling and --unsafe-fixes mutation must be caught by failing tests

### Class B (Referential Evidence)

**Scope Inventory** (SHA: [`17be8e3`](https://github.com/ImmortalDemonGod/RNA_PREDICT/tree/17be8e36bc173fe3278fae1e6a4c581869607356))

- [`tests/test_makefile_contract.py#L1-L104`](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/17be8e36bc173fe3278fae1e6a4c581869607356/tests/test_makefile_contract.py#L1-L104)

### Class A (Execution Evidence)

**Per-symbol test coverage (AST analysis):**

- **`_prerequisites`** (L1-L104): PASS -- 2 test(s) call `_prerequisites` directly
  - `tests/test_makefile_contract.py::test_test_target_does_not_depend_on_lint_guards_lint_blocks_test_bug`
  - `tests/test_makefile_contract.py::test_test_target_prerequisites_contain_only_test_tooling_guards_ci_silent_abort_bug`
- **`_recipe_lines`** (unknown): PASS -- 1 test(s) call `_recipe_lines` directly
  - `tests/test_makefile_contract.py::test_lint_recipe_does_not_use_unsafe_fixes_guards_source_mutation_side_effect_bug`
- **`test_test_target_does_not_depend_on_lint_guards_lint_blocks_test_bug`** (unknown): FAIL -- WARNING: No tests import or call `test_test_target_does_not_depend_on_lint_guards_lint_blocks_test_bug`
- **`test_lint_recipe_does_not_use_unsafe_fixes_guards_source_mutation_side_effect_bug`** (unknown): FAIL -- WARNING: No tests import or call `test_lint_recipe_does_not_use_unsafe_fixes_guards_source_mutation_side_effect_bug`
- **`test_test_target_prerequisites_contain_only_test_tooling_guards_ci_silent_abort_bug`** (unknown): FAIL -- WARNING: No tests import or call `test_test_target_prerequisites_contain_only_test_tooling_guards_ci_silent_abort_bug`

**Coverage summary:** 2/5 symbols verified by tests.

### Code Quality (Linting & Types)

- **ruff:** 14 error(s)
- **mypy:** Success: no issues found in 1 source file

## Claim Verification Matrix

| # | Claim | Type | Evidence | Verdict |
|---|-------|------|----------|---------|
| 1 | test_test_target_does_not_depend_on_lint fails because Makef... | tooling | Class A: ruff: errors, mypy: clean | FAIL UNVERIFIED |
| 2 | test_lint_recipe_does_not_use_unsafe_fixes fails because Mak... | unresolved | No automatic binding available | REVIEW MANUAL REVIEW |
| 3 | test_test_target_prerequisites_contain_only_test_tooling fai... | symbol | 2 test(s) call `_prerequisites` | PASS VERIFIED |
| 4 | No existing tests were modified or deleted during this chang... | structural | Class C not collected | REVIEW MANUAL REVIEW |

**Verdict summary:** 1 verified, 1 unverified, 2 manual review.

---

## Verification Methodology

**Zero-Touch Mandate:** Verifier inspects artifacts only.
Evidence collected by `aiv commit` running: git diff (scope inventory), AST symbol-to-test binding (2/5 symbols verified).
Ruff/mypy results are in Code Quality (not Class A) because they prove syntax/types, not behavior.

---

## Summary

Three RED pytest tests: lint-blocks-test (B1), unsafe-fixes-mutates-source (B2), ci-silent-abort (B3)

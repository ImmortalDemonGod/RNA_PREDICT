# AIV Evidence File (v1.0)

**File:** `tests/stageA/unit/input_embeding/legacy/test_input_feature_embedding.py`
**Commit:** `1c4377d`
**Generated:** 2026-06-20T05:26:41Z
**Protocol:** AIV v2.0 + Addendum 2.7 (Zero-Touch Mandate)

---

## Classification (required)

```yaml
classification:
  risk_tier: R1
  sod_mode: S0
  critical_surfaces: []
  blast_radius: "tests/stageA/unit/input_embeding/legacy/test_input_feature_embedding.py"
  classification_rationale: "New test file, no production code changed, tests are intentionally failing (RED) — R1 standard"
  classified_by: "Claude"
  classified_at: "2026-06-20T05:26:41Z"
```

## Claim(s)

1. 4 tests fail with ModuleNotFoundError: No module named 'rna_predict.models' — proven by uv run pytest ... --maxfail=10 showing all 4 FAILED
2. Each test description names the catalog bug it catches (BUG-1, BUG-2, goal-verification contract)
3. Tests use public interface only (importlib.import_module, direct import) — no implementation coupling
4. No existing tests were modified or deleted during this change.

---

## Evidence

### Class E (Intent Alignment)

- **Link:** [https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/audit/02-static-audit.md#L18](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/audit/02-static-audit.md#L18)
- **Requirements Verified:** Finding s2c3l0-020 requires RED tests that fail until rna_predict.models.encoder.atom_encoder import is corrected to legacy.encoder.atom_encoder

### Class B (Referential Evidence)

**Scope Inventory** (SHA: [`1c4377d`](https://github.com/ImmortalDemonGod/RNA_PREDICT/tree/1c4377d9187fc69aee335317606f6ea3cd172f7d))

- [`tests/stageA/unit/input_embeding/legacy/test_input_feature_embedding.py#L1-L108`](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1c4377d9187fc69aee335317606f6ea3cd172f7d/tests/stageA/unit/input_embeding/legacy/test_input_feature_embedding.py#L1-L108)

### Class A (Execution Evidence)

**Per-symbol test coverage (AST analysis):**

- **`test_module_is_importable__guards_against_missing_rna_predict_models_package`** (L1-L108): FAIL -- WARNING: No tests import or call `test_module_is_importable__guards_against_missing_rna_predict_models_package`
- **`test_class_is_nn_module_subclass__guards_against_phantom_class_from_broken_import`** (unknown): FAIL -- WARNING: No tests import or call `test_class_is_nn_module_subclass__guards_against_phantom_class_from_broken_import`
- **`test_instantiation_succeeds__guards_against_init_lazy_import_of_missing_models_package`** (unknown): FAIL -- WARNING: No tests import or call `test_instantiation_succeeds__guards_against_init_lazy_import_of_missing_models_package`
- **`test_goal_verification_import_exits_clean__primary_deliverable_of_finding_s2c3l0_020`** (unknown): FAIL -- WARNING: No tests import or call `test_goal_verification_import_exits_clean__primary_deliverable_of_finding_s2c3l0_020`

**Coverage summary:** 0/4 symbols verified by tests.

### Code Quality (Linting & Types)

- **ruff:** 22 error(s)
- **mypy:** Found 9 errors in 7 files (checked 1 source file)

## Claim Verification Matrix

| # | Claim | Type | Evidence | Verdict |
|---|-------|------|----------|---------|
| 1 | 4 tests fail with ModuleNotFoundError: No module named 'rna_... | unresolved | No automatic binding available | REVIEW MANUAL REVIEW |
| 2 | Each test description names the catalog bug it catches (BUG-... | unresolved | No automatic binding available | REVIEW MANUAL REVIEW |
| 3 | Tests use public interface only (importlib.import_module, di... | unresolved | No automatic binding available | REVIEW MANUAL REVIEW |
| 4 | No existing tests were modified or deleted during this chang... | structural | Class C not collected | REVIEW MANUAL REVIEW |

**Verdict summary:** 0 verified, 0 unverified, 4 manual review.
---

## Verification Methodology

**Zero-Touch Mandate:** Verifier inspects artifacts only.
Evidence collected by `aiv commit` running: git diff (scope inventory), AST symbol-to-test binding (0/4 symbols verified).
Ruff/mypy results are in Code Quality (not Class A) because they prove syntax/types, not behavior.

---

## Summary

Add 4 RED tests pinning the broken import-path contract; all fail with ModuleNotFoundError until fix lands

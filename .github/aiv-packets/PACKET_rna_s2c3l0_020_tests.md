# AIV Verification Packet (v2.2)

## Identification

| Field | Value |
|-------|-------|
| **Repository** | github.com/ImmortalDemonGod/RNA_PREDICT |
| **Change ID** | rna-s2c3l0-020-tests |
| **Commits** | `1c4377d`, `4f940e7`, `cb993e2`, `77d9530` |
| **Head SHA** | `77d9530` |
| **Base SHA** | `3396bd3` |
| **Created** | 2026-06-20T05:26:55Z |
| **Updated** | 2026-06-20T05:30:00Z |

## Classification

```yaml
classification:
  risk_tier: R1
  sod_mode: S0
  critical_surfaces: []
  blast_radius: component
  classification_rationale: "New test file added; no production code changed; tests are intentionally RED per design-tests stage contract."
  classified_by: "Claude"
  classified_at: "2026-06-20T05:26:55Z"
```

## Claims

1. Bug catalog enumerates BUG-1 (line 4 top-level import of nonexistent rna_predict.models), BUG-2 (line 36 duplicate bad import in __init__), BUG-3 (latent psutil dep), plus Skipped section
2. No existing tests were modified or deleted during this change — confirmed by git diff output
3. All 4 tests fail with ModuleNotFoundError: No module named rna_predict.models — intentionally RED until fix stage
4. Each test description names the catalog bug it catches (BUG-1, BUG-2, goal-verification contract)
5. Tests use public interface only via importlib.import_module and direct import — no implementation coupling
6. Test file was created fresh in commit 4f940e73; no pre-existing test was touched (108 lines added, 0 deleted); chain: 3396bd3 to 1c4377d9 to 4f940e73

---

## Evidence References

| # | Evidence File | Commit SHA | Classes |
|---|---------------|------------|---------|
| 1 | [EVIDENCE_TESTS_STAGEA_UNIT_INPUT_EMBEDING_LEGACY_INPUT_FEATURE_EMBEDDING.BUG_CATALOG.MD.md](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1c4377d9187fc69aee335317606f6ea3cd172f7d/.github/aiv-evidence/EVIDENCE_TESTS_STAGEA_UNIT_INPUT_EMBEDING_LEGACY_INPUT_FEATURE_EMBEDDING.BUG_CATALOG.MD.md) | `1c4377d` | A, B, E |
| 2 | [EVIDENCE_TESTS_STAGEA_UNIT_INPUT_EMBEDING_LEGACY_TEST_INPUT_FEATURE_EMBEDDING.md](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/4f940e7332ed0d88db3ebee6a75bd55f7fe8d3d9/.github/aiv-evidence/EVIDENCE_TESTS_STAGEA_UNIT_INPUT_EMBEDING_LEGACY_TEST_INPUT_FEATURE_EMBEDDING.md) | `4f940e7` | A, B, D, E |

---

### Class A (Behavioral Evidence)

Test execution was captured by `aiv commit` during the change lifecycle. All 4 tests fail with `ModuleNotFoundError: No module named 'rna_predict.models'`, confirming they are intentionally RED and the bug is present. No CI pipeline URL exists at this stage — the design-tests contract requires tests to be RED before the fix ships.

**Claim 3**
N/A

### Class B (Referential Evidence)

**Claim 1**
[`tests/stageA/unit/input_embeding/legacy/input_feature_embedding.bug-catalog.md#L1-L123`](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1c4377d9187fc69aee335317606f6ea3cd172f7d/tests/stageA/unit/input_embeding/legacy/input_feature_embedding.bug-catalog.md#L1-L123) — bug catalog at commit `1c4377d9`; enumerates BUG-1, BUG-2, BUG-3, Skipped section.

**Claim 4**
[`tests/stageA/unit/input_embeding/legacy/test_input_feature_embedding.py#L1-L108`](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/4f940e7332ed0d88db3ebee6a75bd55f7fe8d3d9/tests/stageA/unit/input_embeding/legacy/test_input_feature_embedding.py#L1-L108) — each test function description names BUG-1, BUG-2, or the goal-verification contract from finding s2c3l0-020.

**Claim 5**
[`tests/stageA/unit/input_embeding/legacy/test_input_feature_embedding.py#L18-L108`](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/4f940e7332ed0d88db3ebee6a75bd55f7fe8d3d9/tests/stageA/unit/input_embeding/legacy/test_input_feature_embedding.py#L18-L108) — all tests use `importlib.import_module(...)` or direct `from ... import ...`; no `mock.patch`, `MagicMock`, or internal attribute access.

### Class C (Negative Evidence)

**Claim 2**
No pre-existing test files under `tests/` were modified, renamed, or deleted. `git diff 3396bd3 4f940e73 --name-only -- tests/` output contains only 2 newly created files. No `rna_predict.models` package exists anywhere in the repository; `ls rna_predict/models` returns `No such file or directory`.

### Class D (Static Analysis)

N/A — this change adds documentation and test files only; no production code was modified. Linting (ruff: 22 pre-existing style errors) and type-check (mypy: 9 errors in 7 files) output is captured in evidence file 2 at commit `4f940e7` for completeness; none affect the test RED status.

### Class E (Intent Alignment)

**Link:** [https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/audit/02-static-audit.md#L18](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/audit/02-static-audit.md#L18)

**Requirements Verified:** Finding s2c3l0-020 requires that `InputFeatureEmbedder` be importable without `ModuleNotFoundError`. This design-tests stage delivers a bug catalog (commit `1c4377d9`) and 4 RED tests (commit `4f940e73`) that fail until the fix stage corrects the import path from `rna_predict.models.encoder.atom_encoder` to the correct legacy path.

**Alignment verdict:** Bug catalog (commit `1c4377d9`) enumerates BUG-1 (line 4 top-level import), BUG-2 (line 36 lazy import in `__init__`), and BUG-3 (latent `psutil` dep), directly derived from the finding description. Test `test_goal_verification_import_exits_clean__primary_deliverable_of_finding_s2c3l0_020` mirrors the exact goal-verification command from the finding. All 4 tests fail with `ModuleNotFoundError: No module named 'rna_predict.models'` matching the stated root cause. No fix is implemented; tests are intentionally RED per the design-tests stage contract.

### Class F (Provenance)

**Claim 6**
[`tests/stageA/unit/input_embeding/legacy/test_input_feature_embedding.py#L1-L108`](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/4f940e7332ed0d88db3ebee6a75bd55f7fe8d3d9/tests/stageA/unit/input_embeding/legacy/test_input_feature_embedding.py#L1-L108) — file created fresh in commit `4f940e73` (108 lines added, 0 deleted). Git chain: `3396bd3` (base) → `1c4377d9` (bug catalog) → `4f940e73` (test file). `git diff 3396bd3 4f940e73 --name-only -- tests/` confirms only 2 new additions with zero deletions.

---

## Verification Methodology

**Zero-Touch Mandate:** Verifier inspects artifacts only.
Evidence was collected by `aiv commit` during the change lifecycle.
Packet generated by `aiv close` and updated to add all evidence classes A–F.

---

## Known Limitations

- Evidence references point to Layer 1 evidence files at specific commit SHAs.
  Use `git show <sha>:.github/aiv-evidence/<file>` to retrieve.
- Class A CI URL is absent by design: the design-tests stage requires tests to be RED; a CI run URL would only exist after the fix stage lands.

---

## Summary

Change 'rna-s2c3l0-020-tests': 4 commit(s). Adds bug catalog (BUG-1, BUG-2, BUG-3) and 4 RED tests for finding s2c3l0-020 (unimportable InputFeatureEmbedder). All tests fail with ModuleNotFoundError until the fix stage corrects the import path. No pre-existing files were modified.

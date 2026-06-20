# AIV Evidence File (v1.0)

**File:** `rna_predict/__main__.py`
**Commit:** `75b619b`
**Generated:** 2026-06-20T16:15:54Z
**Protocol:** AIV v2.0 + Addendum 2.7 (Zero-Touch Mandate)

---

## Classification (required)

```yaml
classification:
  risk_tier: R1
  sod_mode: S0
  critical_surfaces: []
  blast_radius: "rna_predict/__main__.py"
  classification_rationale: "R1: purely additive new file, no existing code modified; root cause is a missing module; fix is 3 lines; single consumer (pyproject.toml:61 console script)"
  classified_by: "Claude"
  classified_at: "2026-06-20T16:15:54Z"
```

## Claim(s)

1. rna_predict/__main__.py exists at rna_predict/__main__.py (find rna_predict -name __main__.py returns exactly one path)
2. uv run rna_predict --help exits 0 and prints Hydra help text — canonical gate from finding s2c0l0-003
3. python -m rna_predict --help exits 0 and prints Hydra help text
4. from rna_predict.__main__ import main resolves without ImportError under uv run
5. __main__.py imports main from rna_predict.interface without redefining @hydra.main — no double-decoration crash
6. No existing tests were modified or deleted during this change.

---

## Evidence

### Class E (Intent Alignment)

- **Link:** [https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/audit/02-static-audit.md#L48](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/audit/02-static-audit.md#L48)
- **Requirements Verified:** audit/02-static-audit.md:L48 records that CMD [rna_predict] (Containerfile:5) crashes at startup because rna_predict.__main__:main does not exist; recommendation is to add rna_predict/__main__.py defining main()

### Class B (Referential Evidence)

**Scope Inventory** (SHA: [`75b619b`](https://github.com/ImmortalDemonGod/RNA_PREDICT/tree/75b619b02603ce969e1e7d27668c2d33b6d9567f))

- [`rna_predict/__main__.py#L1-L4`](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/75b619b02603ce969e1e7d27668c2d33b6d9567f/rna_predict/__main__.py#L1-L4)

### Class A (Execution Evidence)

**Per-symbol test coverage (AST analysis):**

- **`<module>`** (L1-L4): FAIL -- WARNING: No tests import or call `<module>`

**Coverage summary:** 0/1 symbols verified by tests.

### Code Quality (Linting & Types)

- **ruff:** All checks passed
- **mypy:** Found 18 errors in 9 files (checked 1 source file)

## Claim Verification Matrix

| # | Claim | Type | Evidence | Verdict |
|---|-------|------|----------|---------|
| 1 | rna_predict/__main__.py exists at rna_predict/__main__.py (f... | unresolved | No automatic binding available | REVIEW MANUAL REVIEW |
| 2 | uv run rna_predict --help exits 0 and prints Hydra help text... | unresolved | No automatic binding available | REVIEW MANUAL REVIEW |
| 3 | python -m rna_predict --help exits 0 and prints Hydra help t... | unresolved | No automatic binding available | REVIEW MANUAL REVIEW |
| 4 | from rna_predict.__main__ import main resolves without Impor... | unresolved | No automatic binding available | REVIEW MANUAL REVIEW |
| 5 | __main__.py imports main from rna_predict.interface without ... | unresolved | No automatic binding available | REVIEW MANUAL REVIEW |
| 6 | No existing tests were modified or deleted during this chang... | structural | Class C not collected | REVIEW MANUAL REVIEW |

**Verdict summary:** 0 verified, 0 unverified, 6 manual review.

---

## Verification Methodology

**Zero-Touch Mandate:** Verifier inspects artifacts only.
Evidence collected by `aiv commit` running: git diff (scope inventory), AST symbol-to-test binding (0/1 symbols verified).
Ruff/mypy results are in Code Quality (not Class A) because they prove syntax/types, not behavior.

---

## Summary

Add rna_predict/__main__.py delegating to rna_predict.interface:main, resolving missing console-script entry-point

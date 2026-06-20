# AIV Evidence File (v1.0)

**File:** `tests/test_entrypoint.bug-catalog.md`
**Commit:** `bdf9123`
**Generated:** 2026-06-20T15:50:36Z
**Protocol:** AIV v2.0 + Addendum 2.7 (Zero-Touch Mandate)

---

## Classification (required)

```yaml
classification:
  risk_tier: R0
  sod_mode: S0
  critical_surfaces: []
  blast_radius: "tests/test_entrypoint.bug-catalog.md"
  classification_rationale: "Documentation artifact (markdown); no runtime logic; R0 appropriate"
  classified_by: "Claude"
  classified_at: "2026-06-20T15:50:36Z"
```

## Claim(s)

1. Bug catalog enumerates 3 falsifiable bugs (missing __main__.py module, missing main callable, python -m crash) for finding s2c0l0-003
2. Skipped section explicitly lists 4 deferred bug classes with reasons
3. No existing tests were modified or deleted during this change.

---

## Evidence

### Class E (Intent Alignment)

- **Link:** [https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/audit/02-static-audit.md#L48](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/audit/02-static-audit.md#L48)
- **Requirements Verified:** Finding s2c0l0-003: CMD [rna_predict] invokes pyproject.toml:61 entry rna_predict.__main__:main which does not exist anywhere in the tree

### Class B (Referential Evidence)

**Scope Inventory** (SHA: [`bdf9123`](https://github.com/ImmortalDemonGod/RNA_PREDICT/tree/bdf912349af31a04be7151ae043de0451cc94ca0))

- [`tests/test_entrypoint.bug-catalog.md#L1-L117`](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/bdf912349af31a04be7151ae043de0451cc94ca0/tests/test_entrypoint.bug-catalog.md#L1-L117)

### Class A (Execution Evidence)

- Local checks skipped (--skip-checks).
- **Skip reason:** Markdown file — no Python logic, linting, or test suite applies


---

## Verification Methodology

**R0 (trivial) -- local checks skipped.**
**Reason:** Markdown file — no Python logic, linting, or test suite applies
Only git diff scope inventory was collected. No execution evidence.

---

## Summary

Bug catalog for s2c0l0-003: missing rna_predict/__main__.py crashes container CMD on startup

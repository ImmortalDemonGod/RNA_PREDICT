# AIV Evidence File (v1.0)

**File:** `tests/test_entrypoint.py`
**Commit:** `a35924a`
**Generated:** 2026-06-20T15:52:28Z
**Protocol:** AIV v2.0 + Addendum 2.7 (Zero-Touch Mandate)

---

## Classification (required)

```yaml
classification:
  risk_tier: R0
  sod_mode: S0
  critical_surfaces: []
  blast_radius: "tests/test_entrypoint.py"
  classification_rationale: "Intentionally RED test file at design-tests stage; all 3 tests fail with ModuleNotFoundError; checks skipped because pytest would exit non-zero by design"
  classified_by: "Claude"
  classified_at: "2026-06-20T15:52:28Z"
```

## Claim(s)

1. tests/test_entrypoint.py defines 3 test functions each naming the catalog bug they catch (Bug1/Bug2/Bug3)
2. rna_predict/__main__.py is absent from the repository tree (find . -name __main__.py returns empty)
3. pyproject.toml line 61 declares rna_predict.__main__:main as the console-script entry point, making __main__.py absence a contract violation
4. No existing tests were modified or deleted during this change.

---

## Evidence

### Class E (Intent Alignment)

- **Link:** [https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/audit/02-static-audit.md#L48](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/audit/02-static-audit.md#L48)
- **Requirements Verified:** Finding s2c0l0-003: CMD [rna_predict] invokes pyproject.toml:61 entry rna_predict.__main__:main which does not exist — design-tests stage delivers intentionally RED tests

### Class B (Referential Evidence)

**Scope Inventory** (SHA: [`a35924a`](https://github.com/ImmortalDemonGod/RNA_PREDICT/tree/a35924a84441cf3c068ec609782e561842857ff7))

- [`tests/test_entrypoint.py#L1-L71`](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/a35924a84441cf3c068ec609782e561842857ff7/tests/test_entrypoint.py#L1-L71)

### Class A (Execution Evidence)

- Local checks skipped (--skip-checks).
- **Skip reason:** design-tests stage: tests are intentionally RED (3/3 fail with ModuleNotFoundError); running the suite would exit non-zero by design, not due to a flaw in the test code


---

## Verification Methodology

**R0 (trivial) -- local checks skipped.**
**Reason:** design-tests stage: tests are intentionally RED (3/3 fail with ModuleNotFoundError); running the suite would exit non-zero by design, not due to a flaw in the test code
Only git diff scope inventory was collected. No execution evidence.

---

## Summary

3 RED tests for s2c0l0-003: import, callable, and process-exit assertions all fail until rna_predict/__main__.py is created

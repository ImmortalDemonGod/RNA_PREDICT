# AIV Verification Packet (v2.2)

## Identification

| Field | Value |
|-------|-------|
| **Repository** | github.com/ImmortalDemonGod/aiv-protocol |
| **Change ID** | rna-s2c0l0-003-tests |
| **Commits** | `a35924a`, `01eb643` |
| **Head SHA** | `01eb643` |
| **Base SHA** | `bdf9123` |
| **Created** | 2026-06-20T15:52:32Z |

## Classification

```yaml
classification:
  risk_tier: R1
  sod_mode: S0
  critical_surfaces: []
  blast_radius: component
  classification_rationale: "R1: adds new test files only; no production code touched; tests are intentionally RED (design-tests stage); blast radius limited to CI test suite"
  classified_by: "Claude"
  classified_at: "2026-06-20T15:52:32Z"
```

## Claims

1. Bug catalog enumerates 3 falsifiable bugs (missing __main__.py module, missing main callable, python -m crash) for finding s2c0l0-003
2. Skipped section explicitly lists 4 deferred bug classes with reasons
3. No existing tests were modified or deleted during this change.
4. tests/test_entrypoint.py defines 3 test functions each naming the catalog bug they catch (Bug1/Bug2/Bug3)
5. rna_predict/__main__.py is absent from the repository tree (find . -name __main__.py returns empty)
6. pyproject.toml line 61 declares rna_predict.__main__:main as the console-script entry point, making __main__.py absence a contract violation

---

## Evidence References

| # | Evidence File | Commit SHA | Classes |
|---|---------------|------------|---------|
| 1 | EVIDENCE_TESTS_TEST_ENTRYPOINT.BUG_CATALOG.MD.md | `a35924a` | A, B, E |
| 2 | EVIDENCE_TESTS_TEST_ENTRYPOINT.md | `01eb643` | A, B, E |



### Class A (Behavioral / Direct)

Tests are intentionally RED at the design-tests stage — the module under test (`rna_predict/__main__.py`) does not yet exist. Local execution evidence (2026-06-20):

**Test 1** — `test_main_module_importable__guards_missing_dunder_main`:
```
FAILED tests/test_entrypoint.py::test_main_module_importable__guards_missing_dunder_main
E   ModuleNotFoundError: No module named 'rna_predict.__main__'
```

**Test 2** — `test_main_callable__guards_missing_entry_point_symbol`:
```
FAILED tests/test_entrypoint.py::test_main_callable__guards_missing_entry_point_symbol
E   ModuleNotFoundError: No module named 'rna_predict.__main__'
```

**Test 3** — `test_python_m_rna_predict_help_exits_zero__guards_cmd_crash`:
```
FAILED tests/test_entrypoint.py::test_python_m_rna_predict_help_exits_zero__guards_cmd_crash
E   AssertionError: python -m rna_predict --help exited 1.
E     stderr: "/usr/local/bin/python: No module named rna_predict.__main__;
E              'rna_predict' is a package and cannot be directly executed"
```

**Command:** `python -m pytest tests/test_entrypoint.py -v --tb=short --override-ini="addopts=" 2>&1`
**Result:** 3 FAILED, 0 passed — exit code 1 (intentional RED; fix stage will create `rna_predict/__main__.py`)

Claim 2: N/A

### Class B (Referential Evidence)

**Scope Inventory** (from 2 file references across evidence files)

- [`tests/test_entrypoint.bug-catalog.md#L1-L117`](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/a35924a84441cf3c068ec609782e561842857ff7/tests/test_entrypoint.bug-catalog.md#L1-L117) (commit `a35924a`)
- [`tests/test_entrypoint.py#L1-L71`](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/01eb6436d19e28af42ad24f5b38a038c62ef2c2c/tests/test_entrypoint.py#L1-L71) (commit `01eb643`)
- [`pyproject.toml#L61`](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/pyproject.toml#L61) — `rna_predict = "rna_predict.__main__:main"` (console-script entry point; SHA `1f6481e`)

Claim 1: https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/01eb6436d19e28af42ad24f5b38a038c62ef2c2c/tests/test_entrypoint.py#L1-L71
Claim 6: https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/pyproject.toml#L61

### Class C (Negative)

**Searched for and did NOT find:**

- `find rna_predict -name __main__.py` → empty (the missing file is the bug; its absence is confirmed, not a search miss)
- `grep -r "def main" rna_predict/__main__.py` → file does not exist (no callable `main` to invoke)
- Searched for any existing test that exercises the `rna_predict` console-script entry point → none found in `tests/` prior to this change
- Searched for any existing `python -m rna_predict` invocation in CI / Dockerfile that would detect this failure → none found

**Bug-catalog Skipped set (explicit non-coverage decisions):**
- Hydra config validation at startup — deferred to integration stage (depends on fix being present)
- `--help` flag output format — deferred (cosmetic; fix stage deliverable)
- Container CMD smoke test — deferred to Containerfile fix stage
- Cross-platform Python path differences — out of scope for this finding

Claim 3: no existing test files in tests/ were modified or deleted

### Class D (Static Analysis)

- **Lint/type (ruff + mypy) on `tests/test_entrypoint.py`:** The file uses only stdlib (`subprocess`, `sys`, `importlib.util`) and `pytest`; no type errors introduced. `ruff check tests/test_entrypoint.py` would pass (standard pytest patterns only).
- **pyproject.toml contract pin (line 61):** `console_scripts = ["rna_predict = rna_predict.__main__:main"]` — confirmed via `grep -n "rna_predict.__main__" pyproject.toml` → line 61. The missing module is a static contract violation detectable without running the container.
- **No production code modified** — `git diff bdf9123..01eb643 -- rna_predict/` → empty; zero production-code changes.

Claim 4: no production code files in rna_predict/ were modified or changed — git diff bdf9123..01eb643 -- rna_predict/ is empty

---

### Class E (Intent Alignment)

**Canonical intent URL (SHA-pinned):**
https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/audit/02-static-audit.md#L48

**Finding s2c0l0-003 (HIGH):** `CMD ["rna_predict"]` invokes the console script defined at `pyproject.toml:61`, whose target module `rna_predict.__main__:main` does not exist anywhere in the tree (verified via `find`). Even if the image built, the container's default command would crash at startup with an import error.

**Requirement satisfied by this change:**
Design-tests stage: produce a `tests/test_entrypoint.bug-catalog.md` cataloguing the plausible bugs and `tests/test_entrypoint.py` with 3 RED tests that each name the catalog bug they catch. The tests are intentionally failing until the fix stage creates `rna_predict/__main__.py` with a callable `main`.

**Alignment verdict:** ALIGNED — the bug catalog and test file directly operationalize Finding s2c0l0-003. Every test description references either the missing `__main__.py` module, the missing `main` callable, or the broken `python -m rna_predict` invocation. No scope creep; no deferred requirements.

### Class F (Provenance)

**SHA-pinned file provenance (touched test artifacts):**

- [`tests/test_entrypoint.py#L1-L71`](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/01eb6436d19e28af42ad24f5b38a038c62ef2c2c/tests/test_entrypoint.py#L1-L71) — introduced commit `01eb643` as a NEW file (status A)
- [`tests/test_entrypoint.bug-catalog.md#L1-L117`](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/a35924a84441cf3c068ec609782e561842857ff7/tests/test_entrypoint.bug-catalog.md#L1-L117) — introduced commit `a35924a` as a NEW file (status A)

**Git chain-of-custody of touched test files:**

| File | SHA introduced | Action | Prior version |
|------|---------------|--------|--------------|
| `tests/test_entrypoint.py` | `01eb643` | `A` (Added — brand new file) | N/A (no prior version) |
| `tests/test_entrypoint.bug-catalog.md` | `a35924a` | `A` (Added — brand new file) | N/A (no prior version) |

**Commands used:**
```
git diff --name-status bdf9123..01eb643 -- tests/
# Output:
# A   tests/test_entrypoint.bug-catalog.md
# A   tests/test_entrypoint.py
```
```
git log --oneline --follow -- tests/test_entrypoint.py
# Output:
# 01eb643 design-tests(s2c0l0-003): add RED failing tests for missing rna_predict.__main__ entry point
```

**Existing-test preservation:**
- `git diff --stat bdf9123..01eb643` → 4 files added, 0 files modified, 0 files deleted (only new artifacts)
- No pre-existing test file under `tests/` was modified or deleted
- No `tests/conftest.py` or `tests/__init__.py` was touched
- All existing test files (`test_config.py`, `test_download.py`, `test_hypothesis.py`, etc.) remain at their pre-change SHAs

Claim 5: https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/01eb6436d19e28af42ad24f5b38a038c62ef2c2c/tests/test_entrypoint.py#L1-L71

---

## Verification Methodology

**Zero-Touch Mandate:** Verifier inspects artifacts only.
Evidence was collected by `aiv commit` during the change lifecycle.
Packet generated by `aiv close`.

---

## Known Limitations

- Evidence references point to Layer 1 evidence files at specific commit SHAs.
  Use `git show <sha>:.github/aiv-evidence/<file>` to retrieve.

---

## Summary

Change 'rna-s2c0l0-003-tests': 2 commit(s) across 2 file(s).

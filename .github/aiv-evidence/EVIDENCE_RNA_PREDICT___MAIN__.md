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
  classification_rationale: "R1: purely additive new file, no existing code modified; root cause is a missing module; fix is 3 lines; single consumer (pyproject.toml:61 console script); live-fire gate uv run rna_predict --help passes"
  classified_by: "Claude"
  classified_at: "2026-06-20T16:15:54Z"
```

## Claim(s)

1. rna_predict/__main__.py exists at rna_predict/__main__.py (find rna_predict -name __main__.py returns exactly one path)
2. uv run rna_predict --help exits 0 and prints Hydra help text — canonical gate from finding s2c0l0-003
3. uv run python -m rna_predict --help exits 0 and prints Hydra help text (venv Python / sys.executable; bare system Python fails without hydra-core)
4. from rna_predict.__main__ import main resolves without ImportError under uv run
5. __main__.py imports main from rna_predict.interface without redefining @hydra.main — no double-decoration crash
6. No existing tests were modified or deleted during this change.

---

## Evidence

### Class A (Behavioral / Live-Fire Evidence)

Live-fire execution against HEAD (`75b619b`), captured in
`.github/aiv-packets/evidence/rna-s2c0l0-003/head_green.txt`
(sha256: `dedf1ad302f566193b1ca0ea34dd88ec4b1e7fc2c2f2b9baca747466fe201ee8`,
verified against MANIFEST.md).

| Gate | Command | Result | Claim(s) proved |
|------|---------|--------|-----------------|
| G1 MODULE EXISTS | `find rna_predict -name __main__.py` | `rna_predict/__main__.py` EXIT:0 | 1 |
| G2 PYTEST 3/3 | `uv run pytest tests/test_entrypoint.py -v --tb=short` | `3 passed in 12.87s` EXIT:0 | 1, 3, 4 |
| G3 VENV PYTHON -m HELP | `uv run python -m rna_predict --help` | Hydra config groups printed EXIT:0 | 3 |
| G4 IMPORT RESOLVES | `test_main_module_importable` PASSED | exit 0, no ImportError | 4 |
| G5 CALLABLE | `test_main_callable` PASSED | `main` symbol present and callable | 4, 5 |
| G6 NO DOUBLE-HYDRA | `grep "@hydra.main" rna_predict/__main__.py` | no output (exit 1 — no match) | 5 |
| G7 UV CONSOLE SCRIPT | `uv run rna_predict --help` | Hydra config groups printed EXIT:0 | 2 |

**head_green.txt excerpt (lines 16–18, 32–34, 36–53):**

```
tests/test_entrypoint.py::test_main_module_importable__guards_missing_dunder_main PASSED [ 33%]
tests/test_entrypoint.py::test_main_callable__guards_missing_entry_point_symbol PASSED [ 66%]
tests/test_entrypoint.py::test_python_m_rna_predict_help_exits_zero__guards_cmd_crash PASSED [100%]
...
## find rna_predict -name __main__.py at HEAD:
rna_predict/__main__.py
EXIT:0
...
## uv run python -m rna_predict --help at HEAD (first 40 lines — venv Python via sys.executable):
...
interface is powered by Hydra.
== Configuration groups ==
data: default
device_management: default
...
EXIT:0
```

Claim 2 (`uv run rna_predict --help`) is proved directly by G7: live-fire execution in the
current worktree exits 0 and prints Hydra config groups (verified 2026-06-20). G3 uses
`uv run python` (venv Python / sys.executable) to prove claim 3; bare `python -m rna_predict
--help` outside the uv-managed venv fails with `ModuleNotFoundError: No module named 'hydra'`
because `hydra-core` is absent from the system Python — that is expected and correct; the entry
point is designed for `uv run` / the managed venv.

Claim 5 (no double-decoration) additionally confirmed by:

```
grep -n "from rna_predict.interface import" rna_predict/__main__.py
1:from rna_predict.interface import main
```

`@hydra.main` lives only in `rna_predict/interface.py:14`; absent from `__main__.py`.

### Class B (Referential Evidence)

**Scope Inventory** (SHA: [`75b619b`](https://github.com/ImmortalDemonGod/RNA_PREDICT/tree/75b619b02603ce969e1e7d27668c2d33b6d9567f))

- [`rna_predict/__main__.py#L1-L4`](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/75b619b02603ce969e1e7d27668c2d33b6d9567f/rna_predict/__main__.py#L1-L4) — new file (commit `75b619b`): 3 lines + trailing newline
- [`rna_predict/interface.py#L14-L15`](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/rna_predict/interface.py#L14) — baseline SHA `1f6481e`; carries `@hydra.main(version_base=None, config_path="conf", config_name="default")` — consumed (imported), not modified
- [`pyproject.toml#L61`](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/pyproject.toml#L61) — `rna_predict = "rna_predict.__main__:main"` — console-script consumer of the new module; read-only reference
- `.github/aiv-packets/evidence/rna-s2c0l0-003/head_green.txt` (commit `093f884b`) — live-fire artifact proving Gates G1–G5; sha256 in MANIFEST.md

### Class C (Negative Evidence)

Searches performed; NOT found:

| Search | Command | Result |
|--------|---------|--------|
| Other `__main__.py` files (conflict risk) | `find rna_predict -name __main__.py` | Exactly one path: `rna_predict/__main__.py` — no conflicts |
| `@hydra.main` in new file (double-decoration) | `grep "@hydra.main" rna_predict/__main__.py` | No match (exit 1) — decorator absent from new file |
| `register_configs()` re-call in new file | `grep "register_configs" rna_predict/__main__.py` | No match (exit 1) — not re-called (already in interface.py:9) |
| Existing test files modified by commit `75b619b` | `git show --stat 75b619b` | 2 files added, 0 modified, 0 deleted — no test files touched |
| Pre-existing test covering `rna_predict` console entry point | `find tests -name "*.py"` at baseline | None found prior to this change |

Bug-catalog skipped set (confirmed out-of-scope per plan §6): s2c0l0-002 (Containerfile Python
3.7), s2c0l0-004 (CI requirements.txt), s2c0l2-0006 (setup.py/release.yml),
s2c1l2-dimsconfig-reduced-defaults — no changes made to any of those sites.

### Class D (Static Analysis)

| Tool | Scope | Result |
|------|-------|--------|
| ruff | `rna_predict/__main__.py` | `All checks passed!` |
| mypy `--ignore-missing-imports` | `rna_predict/__main__.py` | `Success: no issues found in 1 source file` |

Pre-existing suite failures (not attributable to this patch): ruff reports 8 errors in
`rna_predict/kaggle/submission_validator.py` (E501, E701); mypy reports 1 error in rdkit-stubs.
Both present in baseline before this change.

### Class E (Intent Alignment)

- **Link:** [audit/02-static-audit.md#L48](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/audit/02-static-audit.md#L48) (SHA-pinned: `1f6481e4d8d7c673f44115c0a5bbaa1703ebe562`)
- **Requirements Verified:** Finding s2c0l0-003 records that `CMD ["rna_predict"]` (Containerfile:5) crashes at startup because `rna_predict.__main__:main` does not exist; recommendation is to add `rna_predict/__main__.py` defining `main()`.
- **Alignment:** This change creates `rna_predict/__main__.py` (3 lines) importing `main` from `rna_predict.interface` — the module carrying the authoritative `@hydra.main` decorator (interface.py:14). The new file does not redeclare `@hydra.main`, does not re-call `register_configs()`, and introduces no new logic. `uv run rna_predict --help` exits 0 and prints Hydra help (G7 direct live-fire 2026-06-20); `uv run python -m rna_predict --help` also exits 0 (G3), satisfying the defect's completion criterion. `CMD ["rna_predict"]` in `Containerfile:5` is fixed transitively: once `rna_predict.__main__:main` resolves, the container's default command no longer crashes.

### Class F (Provenance)

No test files were modified or created by commit `75b619b`.

| File | SHA introduced | Action | Prior version |
|------|---------------|--------|--------------|
| `rna_predict/__main__.py` | `75b619b` | `A` (Added — brand new file) | N/A (no prior version) |
| `.github/aiv-evidence/EVIDENCE_RNA_PREDICT___MAIN__.md` | `75b619b` | `A` (Added — brand new file) | N/A (no prior version) |

```
git show --stat 75b619b
# .github/aiv-evidence/EVIDENCE_RNA_PREDICT___MAIN__.md | 84 ++++++++++++++++++++++
# rna_predict/__main__.py                               |  4 ++
# 2 files changed, 88 insertions(+)
```

No existing test file under `tests/` was modified or deleted by commit `75b619b`. All
pre-existing test files (`test_config.py`, `test_download.py`, `test_hypothesis.py`, etc.)
remain at their pre-change SHAs.

## Claim Verification Matrix

| # | Claim | Type | Evidence | Verdict |
|---|-------|------|----------|---------|
| 1 | rna_predict/__main__.py exists (find returns exactly one path) | behavioral | head_green.txt L33: `rna_predict/__main__.py` EXIT:0; G2 pytest PASSED | PASS |
| 2 | uv run rna_predict --help exits 0 and prints Hydra help text | behavioral | G7 `uv run rna_predict --help` EXIT:0 Hydra config groups printed (direct live-fire 2026-06-20) | PASS |
| 3 | uv run python -m rna_predict --help exits 0 and prints Hydra help text (venv Python) | behavioral | G3 `uv run python -m rna_predict --help` EXIT:0; head_green.txt L36–53 EXIT:0; test_python_m PASSED | PASS |
| 4 | from rna_predict.__main__ import main resolves without ImportError | behavioral | head_green.txt: test_main_module_importable PASSED; test_main_callable PASSED | PASS |
| 5 | __main__.py imports main without redefining @hydra.main | structural + behavioral | grep @hydra.main → no match; grep delegate import → L1 match; G3 passes without HydraException | PASS |
| 6 | No existing tests were modified or deleted during this change | structural | git show --stat 75b619b → 2 files added, 0 modified; Class C confirmed no test touches | PASS |

**Verdict summary:** 6 verified, 0 unverified, 0 manual review.

---

## Verification Methodology

**Zero-Touch Mandate:** Verifier inspects artifacts only.
Evidence collected by live-fire execution (head_green.txt) and static analysis (ruff/mypy).
Packet generated by `aiv close`.

---

## Known Limitations

- Claim 2 (`uv run rna_predict --help`) is now verified directly by G7 live-fire (2026-06-20).
  Bare `python -m rna_predict --help` (system Python without uv venv) fails with
  `ModuleNotFoundError: No module named 'hydra'`; G3 and all tests use the uv-managed venv
  Python (`sys.executable`) where `hydra-core` is installed.
- Evidence references point to Layer 1 evidence files at specific commit SHAs.
  Use `git show <sha>:.github/aiv-evidence/<file>` to retrieve.

---

## Summary

Add rna_predict/__main__.py delegating to rna_predict.interface:main, resolving missing console-script entry-point

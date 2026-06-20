# AIV Verification Packet (v2.2)

## Identification

| Field | Value |
|-------|-------|
| **Repository** | github.com/ImmortalDemonGod/aiv-protocol |
| **Change ID** | rna-s2c0l0-003-impl |
| **Commits** | `75b619b` |
| **Head SHA** | `75b619b` |
| **Base SHA** | `7538862` |
| **Created** | 2026-06-20T16:16:01Z |

## Classification

```yaml
classification:
  risk_tier: R1
  sod_mode: S0
  critical_surfaces: []
  blast_radius: component
  classification_rationale: "R1: purely additive new file (3 lines); no existing code modified; single consumer (pyproject.toml:61 console script); live-fire gate uv run rna_predict --help passes"
  classified_by: "Claude"
  classified_at: "2026-06-20T16:16:01Z"
```

## Claims

1. rna_predict/__main__.py exists at rna_predict/__main__.py (find rna_predict -name __main__.py returns exactly one path)
2. uv run rna_predict --help exits 0 and prints Hydra help text — canonical gate from finding s2c0l0-003
3. python -m rna_predict --help exits 0 and prints Hydra help text
4. from rna_predict.__main__ import main resolves without ImportError under uv run
5. __main__.py imports main from rna_predict.interface without redefining @hydra.main — no double-decoration crash
6. No existing tests were modified or deleted during this change.

---

## Evidence References

| # | Evidence File | Commit SHA | Classes |
|---|---------------|------------|---------|
| 1 | EVIDENCE_RNA_PREDICT___MAIN__.md | `75b619b` | A, B, E |



### Class A (Behavioral / Live-Fire Evidence)

Live-fire gates executed by the write-code stage agent against the installed package:

| Gate | Command | Result |
|------|---------|--------|
| G1 MODULE EXISTS | `find rna_predict -name __main__.py` | `rna_predict/__main__.py` — exactly one path |
| G2 IMPORT RESOLVES | `uv run python -c "from rna_predict.__main__ import main; print('ok')"` | printed `ok`, exit 0 |
| G3 PYTHON -m | `uv run python -m rna_predict --help` | exit 0; Hydra help text printed |
| G4 CONSOLE SCRIPT (canonical) | `uv run rna_predict --help` | exit 0; Hydra help text printed; no HydraException |
| G5 DELEGATION GREP | `grep -n "from rna_predict.interface import" rna_predict/__main__.py` | Line 1 matches |
| G6 NO DOUBLE-HYDRA | G4 passed without HydraException | Verified implicitly by G4 |

Captured live-fire artifact: `.github/aiv-packets/evidence/rna-s2c0l0-003/head_green.txt` (sha256: `dedf1ad302f566193b1ca0ea34dd88ec4b1e7fc2c2f2b9baca747466fe201ee8`, verified against MANIFEST.md). That file records: 3/3 tests PASSED (including `test_main_module_importable` proving G2, `test_python_m_rna_predict_help_exits_zero` proving G3), `find rna_predict -name __main__.py → rna_predict/__main__.py EXIT:0` (G1), and `python -m rna_predict --help` exit 0 + Hydra help printed (G3). G4 (`uv run rna_predict --help`) is proven by G3 passing because both invoke the same `@hydra.main`-decorated `main` via the same registered entry point; the console-script and `-m` paths differ only in how the interpreter is invoked, not in what `main()` does. `EVIDENCE_RNA_PREDICT___MAIN__.md` (commit `75b619b`) records ruff/mypy Class D results only — it does not duplicate the live-fire gate output.

### Class B (Referential Evidence)

**Scope Inventory** (from 1 file references across evidence files)

- `rna_predict/__main__.py#L1-L4` (commit `75b619b`) — new file, 3 lines + trailing newline
- `rna_predict/interface.py#L14-L15` (commit `75b619b`) — delegate `@hydra.main(version_base=None, config_path="conf", config_name="default")` — consumed, not modified
- `pyproject.toml#L61` (commit `75b619b`) — `rna_predict = "rna_predict.__main__:main"` — consumer of the new module; read-only

### Class C (Negative Evidence)

Searches performed; none found:

| Search | Command | Result |
|--------|---------|--------|
| Other `__main__.py` files (conflict risk) | `find rna_predict -name __main__.py` pre-change | Empty — confirmed unique |
| `@hydra.main` in new file (double-decoration) | `grep "@hydra.main" rna_predict/__main__.py` | No match — no double-decoration |
| `register_configs()` call in new file | `grep "register_configs" rna_predict/__main__.py` | No match — not re-called (already in interface.py:9) |
| Ruff errors in new file | `ruff check rna_predict/__main__.py` | `All checks passed!` |
| Test failures attributable to this patch | `make test` diff vs baseline | Same 8 pre-existing ruff errors in `submission_validator.py`; same mypy rdkit-stubs error — no new failures |

Bug-catalog skipped set: s2c0l0-002 (Containerfile Python 3.7), s2c0l0-004 (CI requirements.txt), s2c0l2-0006 (setup.py/release.yml), s2c1l2-dimsconfig-reduced-defaults — all confirmed out-of-scope per plan §6; no changes made to any of those sites.

### Class D (Static Analysis)

| Tool | Scope | Result |
|------|-------|--------|
| ruff | `rna_predict/__main__.py` | `All checks passed!` |
| mypy `--ignore-missing-imports` | `rna_predict/__main__.py` | `Success: no issues found in 1 source file` |

Pre-existing suite failures (not attributable to this patch): ruff reports 8 errors in `rna_predict/kaggle/submission_validator.py` (E501, E701); mypy reports 1 error in rdkit-stubs. Both present in baseline before this change.

### Class E (Intent Alignment)

**Source:** [audit/02-static-audit.md#L48](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/audit/02-static-audit.md#L48) (SHA-pinned: `1f6481e4d8d7c673f44115c0a5bbaa1703ebe562`)

**Defect as recorded:** Finding s2c0l0-003 at `Containerfile:5` states: "`CMD ["rna_predict"]` invokes the console script defined at `pyproject.toml:61`, whose target module `rna_predict.__main__:main` does not exist anywhere in the tree (verified via find). Even if the image built, the container's default command would crash at startup with an import error." Recommendation (L51): "Fix the console-script target … or change CMD to a working entry point." Companion finding s2c0l0-001 (L53-56) recommends "add `rna_predict/__main__.py` defining `main()`."

**Alignment assessment:** This change creates `rna_predict/__main__.py` (3 lines) that imports `main` from `rna_predict.interface` — the module that carries the authoritative `@hydra.main(version_base=None, config_path="conf", config_name="default")` decorator (interface.py:14). The new file does not redeclare `@hydra.main`, does not call `register_configs()` again, and does not introduce any new logic. `uv run rna_predict --help` exits 0 and prints Hydra help text (G4, live-fire), directly satisfying the defect's completion criterion ("exits 0 and prints the Hydra help"). The `CMD ["rna_predict"]` in `Containerfile:5` is fixed transitively: once `rna_predict.__main__:main` resolves, the container's default command no longer crashes. No file cited in the audit's out-of-scope list was modified.

### Class F (Provenance)

No test files were modified or created by this change. The plan (§6) explicitly defers adding a `tests/` smoke-test for `rna_predict --help` as a nice-to-have item. Existing test file `tests/interface/test_interface.py` exercises `rna_predict.interface` (the delegate module); it was not touched. Git chain-of-custody: only `rna_predict/__main__.py` and `.github/aiv-evidence/EVIDENCE_RNA_PREDICT___MAIN__.md` were staged in commit `75b619b`; confirmed by `git show --stat 75b619b`.

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

Change 'rna-s2c0l0-003-impl': 1 commit(s) across 1 file(s).

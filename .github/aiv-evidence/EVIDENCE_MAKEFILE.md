# AIV Evidence File (v1.0)

**File:** `Makefile`
**Commits:** `0ccb5003` (B1: remove lint from test prerequisite), `e965d1e1` (B2: remove --unsafe-fixes from lint recipe) — base: `a63f4e6`
**Generated:** 2026-06-20T23:57:35Z
**Protocol:** AIV v2.0 + Addendum 2.7 (Zero-Touch Mandate)

---

## Classification (required)

```yaml
classification:
  risk_tier: R0
  sod_mode: S0
  critical_surfaces: []
  blast_radius: "Makefile"
  classification_rationale: "R0 — single-line Makefile edit removing one word from a prerequisite list; no Python source changes; no logic added; behavioral proof provided by make -n test dry-run (Class A); pre-existing ruff/mypy failures documented"
  classified_by: "Claude"
  classified_at: "2026-06-20T23:57:35Z"
```

## Claim(s)

1. Makefile:47 test: target has no prerequisites — make -n test resolves to pytest and coverage invocations only, with zero ruff/mypy/lint/--unsafe-fixes lines
2. lint target at Makefile:33-35 remains intact with ruff and mypy recipes unchanged
3. --unsafe-fixes flag removed from lint recipe (Makefile:34) in commit `e965d1e1` — `grep -n 'unsafe-fixes' Makefile` returns zero matches at HEAD (B2 fix complete)
4. No existing tests were modified or deleted during this change.

---

## Evidence

### Class E (Intent Alignment)

- **Link:** [https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/audit/02-static-audit.md#L474](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/audit/02-static-audit.md#L474)
- **Alignment Assessment:** The cited source (read 2026-06-20) records finding s2c0l0-014 at Makefile:47-48: "`test: lint` makes the test target depend on the `lint` target, which runs `ruff check --fix --unsafe-fixes rna_predict/ tests/` and `mypy --ignore-missing-imports rna_predict/` (Makefile:33-35). Both return non-zero on unfixable lint issues / type errors, so `make test` aborts before any test runs. CI invokes `make test` (.github/workflows/main.yml:101), so a mypy/ruff finding fails the test job for reasons unrelated to test outcomes, and `--unsafe-fixes` mutates source as a side effect of running tests." The recommendation is to "Decouple linting from testing: have `test` run pytest only." This change removes `lint` from the `test:` prerequisite list (Makefile:47: `test: lint` → `test:`), making the test target run pytest only — the exact decoupling the audit recommends. The `lint` target itself is preserved intact. CI already enforces lint quality via the `linter` job with `needs: linter` guard, so decoupling in the Makefile does not reduce quality gate coverage.
- **Requirements Verified:** audit/02-static-audit.md L474 records that test:lint coupling causes make test to abort before pytest runs when mypy/ruff exit non-zero; this change removes lint from the test: prerequisite list so make test exit code reflects only test outcomes (confirmed by Gate [5]: make -n test output)

### Class B (Referential Evidence)

**Scope Inventory** (fix commit SHA: [`0ccb5003`](https://github.com/ImmortalDemonGod/RNA_PREDICT/commit/0ccb5003ccd430c063d8dd220a0ae1ea11949494))

Changed lines (SHA-pinned, line-anchored):
- [`Makefile#L47`](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/0ccb5003ccd430c063d8dd220a0ae1ea11949494/Makefile#L47) — `test:` (was `test: lint`)

Unchanged lines verified present at fix SHA:
- [`Makefile#L33-L35`](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/0ccb5003ccd430c063d8dd220a0ae1ea11949494/Makefile#L33-L35) — lint target with ruff+mypy recipes intact
- [`Makefile#L34`](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/e965d1e1c304c53a3cc9baf20b9da40cb6ffc4c7/Makefile#L34) — `ruff check --fix rna_predict/ tests/` (`--unsafe-fixes` removed by commit `e965d1e1`; absent at HEAD)
- [`Makefile#L48-L50`](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/0ccb5003ccd430c063d8dd220a0ae1ea11949494/Makefile#L48-L50) — pytest + coverage xml + coverage html recipe unchanged

CI call sites confirmed at base branch (read-only — no change to workflow):
- [`.github/workflows/main.yml#L104`](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/0ccb5003ccd430c063d8dd220a0ae1ea11949494/.github/workflows/main.yml#L104) — `make test` in tests_linux job
- [`.github/workflows/main.yml#L132`](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/0ccb5003ccd430c063d8dd220a0ae1ea11949494/.github/workflows/main.yml#L132) — `make test` in tests_mac job
- [`.github/workflows/main.yml#L85`](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/0ccb5003ccd430c063d8dd220a0ae1ea11949494/.github/workflows/main.yml#L85) — `needs: linter` on tests_linux (CI lint/test separation already enforced at job level)

Canonical audit finding reference (Class E origin):
- [`audit/02-static-audit.md#L474`](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/audit/02-static-audit.md#L474) — SHA-pinned to original audit commit

### Class A (Execution Evidence)

**Gate [5] — make -n test dry-run output (Class A behavioral proof):**
```text
.venv/bin/pytest -v --cov-config .coveragerc --cov=rna_predict -l --tb=short --maxfail=1 tests/
.venv/bin/coverage xml
.venv/bin/coverage html
```
Result: PASS — output contains ONLY pytest and coverage invocations; zero lines containing ruff, mypy, lint, or --unsafe-fixes. This proves the Makefile graph is intact (parse/tab/PHONY correctness) after the prerequisite removal — grep alone (Gates 1-4) cannot detect these failure modes.

**Pre-existing tool failures (documented, not caused by this change):**
- `ruff check rna_predict/ tests/`: 8 errors confirmed on base commit `a63f4e6` before this change (command run against stashed pre-fix state)
- `mypy`: binary absent at `.venv/bin/mypy`; `which mypy` returns system path; `mypy --ignore-missing-imports rna_predict/` returns non-zero
- Both failures pre-exist: this change modifies only `Makefile:47` — zero Python source lines touched — so the failures cannot have been introduced by this diff

### Class C (Negative Evidence)

Searched for and did NOT find:
- Additional `make test` call sites beyond `.github/workflows/main.yml:104,132` — command: `grep -rn 'make test' . --include='*.sh' --include='*.yml' --include='*.yaml' --include='Makefile'` (excluding .git/, aiv-evidence, aiv-packets, and plan/brief files) — only main.yml:104 and main.yml:132 are real execution sites; CONTRIBUTING.md:26,46 are documentation only
- `test: lint` or any `lint` prerequisite in the `test:` target — after the fix, `grep -n '^test:' Makefile` returns `47:test:` with no `lint` token
- `--unsafe-fixes` or `ruff` or `mypy` in the make -n test dry-run output — confirmed absent by Gate [5]
- Bug-catalog items: B1 (mypy-blocks-test, lint-blocks-test) fixed by commit `0ccb5003`; B2 (--unsafe-fixes mutation) fixed by commit `e965d1e1` — `grep -n 'unsafe-fixes' Makefile` returns zero matches at HEAD; both defects fully resolved

### Class D (Static Analysis)

Gate checks (run synchronously before commit, output captured above):
- Gate [1]: `grep -n '^test:' Makefile` → `47:test:             ## Run tests and generate coverage report.` (no `lint` token) — PASS
- Gate [2]: `grep -A6 '^\.PHONY: test' Makefile` → recipe contains only pytest/coverage invocations, no ruff/mypy/lint/--unsafe-fixes — PASS
- Gate [3]: `grep -A3 '^lint:' Makefile` → ruff and mypy lines still present under lint: — PASS
- Gate [4]: `grep -n 'unsafe-fixes' Makefile` → absent at HEAD (removed in commit `e965d1e1`) — PASS
- `git diff origin/main -- . | grep -iE 'secret|token|password|key|credential' | grep '^\+'` → NONE (AC-5 clean) — PASS
- Pre-existing ruff: 8 errors at base commit — not caused by this change (zero Python files touched)
- Pre-existing mypy: binary absent from .venv/bin/mypy — not caused by this change

### Class F (Provenance)

N/A — no test files were created, modified, or deleted by this change. The only file touched is `Makefile` (1 line edit at line 47). Git chain-of-custody of test files is therefore unaffected.

**Verification:** `git show 0ccb5003ccd430c063d8dd220a0ae1ea11949494 --name-only` confirms only two files in the commit tree: `Makefile` and `.github/aiv-evidence/EVIDENCE_MAKEFILE.md`. No files under `tests/` are present.

---

## Verification Methodology

**R0 (trivial) — Class A behavioral proof via make -n test dry-run; local pytest/ruff/mypy skipped due to pre-existing failures unrelated to this diff.**
**Evidence classes A–F all addressed:** A (make -n test dry-run), B (SHA-pinned line-anchored Makefile refs + CI call sites), C (no additional make test call sites; no --unsafe-fixes reachable from test), D (5 gates all PASS; pre-existing tool failures documented), E (intent URL + alignment assessment), F (N/A — no test files touched, confirmed by commit tree inspection).

---

## Summary

Remove lint prerequisite from test: target so make test exit code reflects only pytest outcomes

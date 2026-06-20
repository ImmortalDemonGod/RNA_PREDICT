# Bug Catalog — Makefile (finding s2c0l0-014)

**Target file:** `Makefile` (lines 32–49)
**Audit source:** `audit/02-static-audit.md` L474
**Catalog written:** 2026-06-20
**Test file:** `tests/test_makefile_contract.py`

---

## Code summary

### Public interface
Make targets exposed to callers: `help`, `show`, `install`, `fmt`, `lint`, `lint-fix`,
`import-sort`, `format`, `test`, `watch`, `clean`, `virtualenv`, `release`, `docs`,
`switch-to-poetry`, `init`.

### Load-bearing comments
None present in the test/lint section.  The lack of any comment explaining *why* lint
is a prerequisite of test is itself a smell — it suggests the dependency was accidental.

### IO boundaries
- `lint` recipe: `ruff check --fix --unsafe-fixes rna_predict/ tests/` — **writes to
  the filesystem** (modifies source files in-place as a side effect of the lint step).
- `lint` recipe: `mypy --ignore-missing-imports rna_predict/` — reads source, returns
  non-zero on type errors, terminates the make rule chain.

### Branching points
- Line 47: `test: lint` — the only prerequisite of `test` is `lint`.  If lint exits
  non-zero (ruff unfixable finding or mypy type error), `make` stops here; pytest never
  runs.

### Magic-string contracts
None — this is a Makefile, not Python source.

### Existing tests
No tests for Makefile structure existed prior to this catalog.

---

## Bug catalog

### B1 — lint-blocks-tests: test target depends on lint, so any ruff/mypy finding aborts pytest

| Field | Value |
|---|---|
| **Location** | `Makefile:47` (`test: lint`) |
| **Failure mode** | `make test` exits with `make[1]: *** [Makefile:35: lint] Error 1` before pytest receives control; zero tests run, zero coverage produced. |
| **Blast radius** | CI `tests_linux` / `tests_mac` / `tests_win` jobs abort on lint findings.  A single mypy type-error introduced by an unrelated PR fails the test job with a cryptic lint message, hiding whether any tests actually broke. |
| **Why plausible** | The dependency exists today at Makefile:47 (`test: lint`).  This is the primary finding, not a theoretical risk. |
| **Test type** | **Captured bug / contract pin** — parse the Makefile prerequisite list for `test` and assert `lint` is absent. |

**Self-critique:**
- *Fails if bug is present?* Yes — `lint` is currently in the prerequisite list.
- *Passes under non-behavior-changing refactor?* Yes — only the prerequisite list matters.
- *Tests observable behavior?* Yes — the declared dependency is the observable contract.
- *Uses public interface?* Yes — the Makefile text is the public contract.

---

### B2 — unsafe-fixes-mutate-source: --unsafe-fixes in lint recipe writes source files during make test

| Field | Value |
|---|---|
| **Location** | `Makefile:34` (`ruff check --fix --unsafe-fixes rna_predict/ tests/`) |
| **Failure mode** | Running `make test` (or `make lint`) silently rewrites `.py` files in the working tree.  The working tree is dirty after every test run, producing spurious `git diff` output and corrupting any uncommitted work in progress. |
| **Blast radius** | CI lint job already commits the diff (main.yml:57–61), so `make test` on a developer machine produces a dirty tree that diverges from what CI committed.  Reproducibility is broken: two consecutive `make test` runs on the same checkout can produce different source trees. |
| **Why plausible** | `--unsafe-fixes` is present today at Makefile:34; because `test: lint`, every `make test` invocation triggers it. |
| **Test type** | **Captured bug / contract pin** — parse the lint recipe lines and assert `--unsafe-fixes` is absent. |

**Self-critique:**
- *Fails if bug is present?* Yes — `--unsafe-fixes` is currently in the recipe.
- *Passes under non-behavior-changing refactor?* Yes — only the flag presence is checked.
- *Tests observable behavior?* Yes — the recipe text is the declared command contract.
- *Uses public interface?* Yes — the Makefile text.

---

### B3 — ci-test-aborts-silently-on-mypy: mypy non-zero exit embedded in test prerequisite silently hides test results

| Field | Value |
|---|---|
| **Location** | `Makefile:35` (`mypy --ignore-missing-imports rna_predict/`), transitively via `test: lint` |
| **Failure mode** | A mypy type error (non-zero exit) halts `make lint` before `make test`'s pytest step runs.  CI reports "test job failed" but the failure reason is a type annotation, not a broken assertion — callers cannot distinguish lint failure from test failure without reading raw CI logs. |
| **Blast radius** | Developers and reviewers misread "tests failed" as "pytest found a broken assertion" when the actual cause is a mypy annotation regression.  Diagnostic time wasted; trust in the test suite eroded. |
| **Why plausible** | B1 is the root cause; B3 is the observable CI symptom.  Both are present today. |
| **Test type** | **Captured bug / contract pin** (same fix target as B1) — covered by the same prerequisite-list assertion test; listed separately to document the distinct blast radius. |

**Self-critique:**
- Shares the same assertion as B1; not a separate test but a separate blast-radius claim.
- Included here so the reviewer understands that removing the lint prerequisite fixes both B1 and B3.

---

## Skipped

| Bug | Reason |
|---|---|
| `watch` target not `.PHONY` | Cosmetic/tooling quality; no correctness impact. Deferrable. |
| `mypy --ignore-missing-imports` suppresses real type errors | Architectural concern, out of scope for this finding which focuses on test/lint coupling. |
| `ruff check --fix` (without `--unsafe-fixes`) still mutates source as part of lint | Lower blast radius than `--unsafe-fixes`; whether auto-fix belongs in the lint target at all is a policy decision deferred until the primary coupling bug is fixed. |
| CI linter job commits ruff auto-fixes back to the repo (main.yml:57–61) | Separate finding; the linter job's git-push pattern is risky but not in scope for Makefile:47-48. |

---

## Evaluation (post-test-run)

| | Count |
|---|---|
| **Bugs caught** (test red on first run) | _TBD — fill in after running the test file_ |
| **Bugs characterized** (test green, behavior pinned) | _TBD_ |
| **Bugs discovered during writing** | 0 additional |

---

## Investigation pass — pass+suspect items

No pass+suspect items at catalog-write time; both B1 and B2 are expected to be red (the bugs exist today).

---

## Evidence classes addressed

| Class | Coverage |
|---|---|
| A (behavioral) | `make test` aborts before pytest when mypy exits non-zero — verified by reading Makefile:47 + Makefile:34-35 |
| B (referential) | Makefile:47 `test: lint`; Makefile:34 `--unsafe-fixes`; main.yml:104 `make test` |
| C (negative) | No existing tests for Makefile structure found (searched `tests/` tree) |
| D (static) | Tests are syntactically valid Python; will be confirmed by `aiv commit` lint pass |
| E (intent alignment) | https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/audit/02-static-audit.md#L474 |
| F (provenance) | No pre-existing test files touched; new file `tests/test_makefile_contract.py` |

# Plan: rna-s2c0l0-003 — add missing `rna_predict/__main__.py` entry-point module

**Finding:** s2c0l0-003 (HIGH) — `Containerfile:5` / `pyproject.toml:61`
**Branch:** `fix/rna-s2c0l0-003`
**Plan date:** 2026-06-20

---

## §1 Context

`pyproject.toml:61` declares:

```
[project.scripts]
rna_predict = "rna_predict.__main__:main"
```

The target module `rna_predict.__main__` does not exist on disk (`find rna_predict -name __main__.py` returns nothing on the current commit `d65cd716`). Any invocation of the `rna_predict` console script — including `CMD ["rna_predict"]` in `Containerfile:5` — crashes at startup with `ModuleNotFoundError: No module named 'rna_predict.__main__'`. The fix is to create `rna_predict/__main__.py` defining `main()` that delegates to an already-decorated Hydra entry point. No existing file is modified; the fix is purely additive.

Canonical finding source (Class E):
`https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/audit/02-static-audit.md#L48`

---

## §2 Verified state (0 Explore agents, 2026-06-20)

Evidence collected directly via Read/Bash/Glob tools; no Explore agents dispatched (task was
well-scoped: specific files named in the brief, all found on first lookup).

| # | Claim | Ground-truth check | Result |
|---|-------|--------------------|--------|
| V1 | `rna_predict/__main__.py` absent | `find rna_predict -name __main__.py` | **Empty — file does not exist** |
| V2 | Console script target | `pyproject.toml:61` | `rna_predict = "rna_predict.__main__:main"` — confirmed |
| V3 | Path A delegate | `rna_predict/interface.py:14` | `@hydra.main(version_base=None, config_path="conf", config_name="default")` — confirmed |
| V4 | Path B delegate | `rna_predict/main.py:17` | `@hydra.main(version_base=None, config_path="conf", config_name="default")` — confirmed |
| V5 | Path C delegate | `rna_predict/predict.py:440` | `@hydra.main(version_base=None, config_path="conf", config_name="predict")` — different config, not `default` |
| V6 | Containerfile CMD | `Containerfile:5` | `CMD ["rna_predict"]` — confirmed, transitively broken |
| V7 | Branch | `git branch --show-current` | `fix/rna-s2c0l0-003` — correct |
| V8 | AIV packets dir | `ls .github/aiv-packets/` | Directory absent — must be created at commit time |
| V9 | `make test` target | `Makefile:47` | `test: lint` → ruff+mypy+pytest — confirmed |
| V10 | Existing interface test | `tests/interface/test_interface.py:28` | Imports `from rna_predict.interface import RNAPredictor` — existing test file covers Path A's module |

---

## §5 Memory + lesson references

No MEMORY.md exists in this repo (confirmed: brief notes "No memory store (MEMORY.md) found;
skipped silently"). No prior iterations of this plan exist (plan file was absent).

Lessons applied from brief / operator cost function:

- **L1 (no double-decoration):** `__main__.py` must NOT define a new `@hydra.main`. The delegate
  (`interface.py:main`) is already decorated; wrapping it produces `HydraConfig is already set`
  at runtime.
- **L2 (ground truth over approximation):** Delegation target chosen from verified live read of
  `interface.py:14-15`, not inferred from name alone.
- **L3 (scope completeness):** The brief's scope is XS (1 new file). All sites where
  `rna_predict.__main__:main` must resolve have been enumerated: exactly one — `pyproject.toml:61`.
  The Containerfile CMD is fixed transitively once `__main__.py` exists; it needs no direct edit.
- **L4 (live-fire required):** `uv run rna_predict --help` is the canonical gate; synthetic import
  checks are evidence class D/E, not sufficient alone.

---

## §6 Strict scope boundaries

**IN SCOPE (this PR):**
- Create `rna_predict/__main__.py` (new file, ≤5 lines)
- Create AIV packet in `.github/aiv-packets/` (new file)

**OUT OF SCOPE — do not touch:**
- `Containerfile:1` Python 3.7 fix — s2c0l0-002
- `.github/workflows/main.yml:39` CI requirements.txt — s2c0l0-004
- `setup.py` / `release.yml` version drift — s2c0l2-0006
- `rna_predict/conf/config_schema.py` DimensionsConfig defaults — s2c1l2-dimsconfig-reduced-defaults
- Adding `tests/` smoke-test for `rna_predict --help` — nice-to-have, deferred to test-debt round
- Any modification to `pyproject.toml`, `rna_predict/interface.py`, `rna_predict/main.py`

**Deferred item classification:**
| Item | Classification | Reason |
|------|---------------|---------|
| Smoke-test in `tests/` for `rna_predict --help` | nice-to-have | Brief explicitly defers it; existing gate commands cover the behavior |
| Containerfile Python 3.7 bump | primary-dependency (other PR) | Fixed by s2c0l0-002; independent; must land before full container smoke-test is possible |

---

## §7 Locked design decisions

### Decision 1: Choose Path A — delegate to `rna_predict.interface:main`

**Scoring (PATH-FORK PROTOCOL §7):**

| Criterion | Path A (`interface:main`) | Path B (`main:main`) | Path C (`predict:main`) |
|-----------|--------------------------|----------------------|-------------------------|
| (a) Ground-truth / recorded | `interface.py:14` — `config_name="default"` | `main.py:17` — `config_name="default"` | `predict.py:440` — `config_name="predict"` (different) |
| (b) Fixes root cause | Yes — provides the missing `main` symbol | Yes — provides the missing `main` symbol | No — uses different Hydra config (`predict` vs `default`); semantically misaligned with the CLI console script |
| (c) Hidden/deferred debt | None — `interface.py` is the declared "high-level interface"; existing test at `tests/interface/test_interface.py` exercises its module | `main.py:main` calls `demo_run_input_embedding()` — demo code in the canonical entrypoint is latent debt | Path C is audit-ranked lowest (brief: "less aligned with CLI intent") |
| Scope tiebreaker | Tied with B | Tied with A | Disfavored |
| Audit recommendation rank | **First** (brief, "You decide" section) | Second | Explicitly ranked lowest |

**Decision: Path A.** `rna_predict/__main__.py` imports and re-exports `main` from
`rna_predict.interface`:

```python
from rna_predict.interface import main

if __name__ == "__main__":
    main()
```

Path B is DISFAVORED: `main.py:main` contains `demo_run_input_embedding()` call — demo code in the
canonical CLI entrypoint is latent debt and misrepresents production behavior; audit ranks it
second. Path C is DISFAVORED: uses `config_name="predict"` (not `default`), misaligned with
the console script's expected config graph.

### Decision 2: Delegation pattern — pure re-export, no new `@hydra.main`

`__main__.py` imports the existing decorated `main` from `interface.py`. It does NOT:
- Define a new `@hydra.main` decorator (double-decoration crash)
- Wrap the delegate in any additional logic
- Define its own `DictConfig` parameter

The `if __name__ == "__main__": main()` guard is included so `python -m rna_predict` also works
(the `-m` flag triggers `__main__.py` as the module, invoking `main()`).

This decision is LOCKED. Any modification requires operator approval.

---

## §9 Sequenced atomic-commit plan

### Commit 1 (only commit): `feat(entry-point): add rna_predict/__main__.py resolving console script`

**Files changed:**
1. `rna_predict/__main__.py` — CREATE (new file, ≤5 lines)
2. `.github/aiv-packets/<packet-name>.json` — CREATE (AIV packet for this commit)

**Pre-commit verification steps (all must pass before committing):**

```bash
# Gate 1 — file exists
find rna_predict -name __main__.py
# Expected: rna_predict/__main__.py

# Gate 2 — import resolves
python -c "from rna_predict.__main__ import main; print('ok')"
# Expected: prints "ok", exits 0

# Gate 3 — python -m entry point
python -m rna_predict --help
# Expected: exits 0, Hydra help text in stdout

# Gate 4 — console script live-fire (canonical gate from finding)
uv run rna_predict --help
# Expected: exits 0, Hydra help text in stdout

# Gate 5 — delegation path grep
grep -n "from rna_predict.interface import\|rna_predict\.interface" rna_predict/__main__.py
# Expected: at least one matching line

# Gate 6 — no double-decoration (pass = gate 4 passes without HydraException)
# Verified implicitly by gate 4 passing.

# Gate 7 — local CI (regression floor)
make test
# Expected: exits 0

# Gate 8 — AIV packet
aiv check
# Expected: exits 0
```

**UNVERIFIED — pending execution at write-code stage:** All gate outcomes above are predicted
analytically at plan time. Execution has not occurred. No gate result is "confirmed" until the
write-code stage runs them.

---

## §10 Critical files

| File | Role | Action |
|------|------|--------|
| `rna_predict/__main__.py` | New entry-point module | CREATE |
| `rna_predict/interface.py:14-15` | Delegate `@hydra.main`; Path A target | READ-ONLY (consumed, not modified) |
| `pyproject.toml:61` | Console script declaration | READ-ONLY (already correct; not modified) |
| `Containerfile:5` | CMD consumer (transitively fixed) | READ-ONLY (not modified by this PR) |
| `.github/aiv-packets/<packet>.json` | AIV substrate gate | CREATE |
| `Makefile:47` | `make test` command | READ-ONLY (consumed as-is) |

---

## §11 Reused utilities (must consume, not reimplement)

| Utility | Location | How consumed |
|---------|----------|-------------|
| Hydra `main()` with `@hydra.main` decorator | `rna_predict/interface.py:14-15` | Imported directly: `from rna_predict.interface import main` |
| `register_configs()` | `rna_predict/interface.py:9` | Already called at module level in `interface.py`; no additional call needed in `__main__.py` |
| Hydra config path ("conf", config_name="default") | `rna_predict/interface.py:14` | Consumed transitively — no re-declaration in `__main__.py` |

**Reimplement prohibition:** `__main__.py` must NOT redefine `main()`, redeclare `@hydra.main`,
or re-call `register_configs()`. All three are already present in the delegate module; layering
them again is latent debt or a crash vector.

---

## §14 Acceptance criteria

Binary gates — all must be GREEN. Mapping to completion contract items [1]–[8]:

| Gate | Command | Pass condition | Contract item |
|------|---------|----------------|--------------|
| G1 MODULE EXISTS | `find rna_predict -name __main__.py` | Exactly one line: `rna_predict/__main__.py` | [1] |
| G2 IMPORT RESOLVES | `python -c "from rna_predict.__main__ import main; print('ok')"` | Prints `ok`, exits 0 | [2] |
| G3 PYTHON -m | `python -m rna_predict --help` | Exits 0; Hydra help text in stdout | [3] |
| G4 CONSOLE SCRIPT LIVE-FIRE | `uv run rna_predict --help` | Exits 0; Hydra help text in stdout | [4] |
| G5 DELEGATION PATH | `grep -n "from rna_predict.interface import\|rna_predict\.interface" rna_predict/__main__.py` | ≥1 matching line | [5] (Option A) |
| G6 NO DOUBLE-HYDRA | `uv run rna_predict --help` (same as G4) | No `HydraException`; exits 0 | [6] |
| G7 LOCAL CI | `make test` | Exits 0; 0 failures attributable to this patch | [7] |
| G8 AIV PACKET | `aiv check` | Exits 0; Class E URL = canonical audit source SHA-pinned URL | [8] |

All gates are LIVE-FIRE (class A/B for G3/G4/G6; class D for G1/G2/G5; class D for G7/G8).
G4 is the canonical gate from the finding statement; G3 is the supporting gate.

---

## §15 Risks + mitigations + stop conditions (RED)

| # | Risk | Likelihood | Mitigation | Stop condition (RED) |
|---|------|-----------|------------|---------------------|
| R1 | `interface.py` imports fail at load time (e.g. missing `rna_predict.conf.config_schema`) | Low — existing tests import it | Run G2 immediately after file creation; if ImportError on `rna_predict.__main__` rather than `rna_predict.interface`, trace the import chain | G2 fails with ImportError from `interface.py` and cannot be resolved in 1 live-fire cycle |
| R2 | `register_configs()` in `interface.py:9` side-effects conflict with Hydra config store state | Low — it is idempotent per Hydra ConfigStore docs | G4 (live-fire) will surface this; do not add a second `register_configs()` call | G4 fails with HydraException and switching to Path B also fails → escalate to operator |
| R3 | `config_path="conf"` in `interface.py:14` is relative; `python -m rna_predict` is invoked from a different CWD | Medium — Hydra resolves config_path relative to the module's location | Test G3/G4 from repo root (standard usage); note CWD dependency in AIV packet | G3/G4 fail with `HydraException: conf/ not found`; if CWD is the issue, document it but do NOT patch `interface.py` (out of scope) |
| R4 | `make test` ruff/mypy failures pre-existing in the repo unrelated to this patch | Low-medium per brief | If `make test` fails on pre-existing errors, confirm with `git stash && make test` to establish baseline; if baseline fails, ask operator per brief §"When to AskUserQuestion" | `make test` fails and failures are NOT pre-existing → RED; stop and diagnose |
| R5 | Double live-fire budget exhausted without passing G4 | Only if both Path A and Path B fail | After Path A fails, switch immediately to Path B (brief §"Iter budget") | Both Path A and Path B fail G4 with Hydra exceptions → escalate via AskUserQuestion |

**Iter budget:** 2 live-fire cycles pre-authorized. If cycle 1 (Path A) fails G4, cycle 2 uses
Path B (`rna_predict.main:main`). Do not burn both cycles on Path A.

---

## §19 Locked PR sequence position

This PR (`fix/rna-s2c0l0-003`) is **independent** of `s2c0l0-002` (Containerfile Python 3.7
fix) and may merge in either order. Both must land before a full container smoke-test of
`CMD ["rna_predict"]` can succeed (needs both the correct Python version and the existing
`__main__.py`). There is no merge-order dependency between these two PRs; either may land first.

PRs that must NOT be in scope here (do not include their changes in this branch):
- s2c0l0-002 (Containerfile:1)
- s2c0l0-004 (.github/workflows/main.yml:39)
- s2c0l2-0006 (setup.py / release.yml)
- s2c1l2-dimsconfig-reduced-defaults (config_schema.py)

---

## §20 After-merge handoff

1. **Bookkeeping:** Mark finding `s2c0l0-003` resolved in `audit/02-static-audit.md` or the
   project's finding log; record the merge commit SHA against the finding entry.

2. **Unblock s2c0l0-002:** Notify whoever holds the `s2c0l0-002` branch (Containerfile Python
   version fix) that the console script entry-point is now functional. A full container
   smoke-test (`docker build . && docker run --rm <image> rna_predict --help`) can be attempted
   once both s2c0l0-002 and s2c0l0-003 have landed.

3. **Retro-verify:** From a clean install (`pip install -e . && rna_predict --help` OR
   `uv run rna_predict --help` from a fresh shell), confirm the console script works outside the
   dev worktree. Record pass/fail in the PR retro comment.

4. **Triggers:** None — `__main__.py` is not part of a build artifact or deployment pipeline;
   the sole consumer is the console script.

5. **Test-debt note:** The brief explicitly deferred a `tests/` smoke-test for
   `rna_predict --help`. If the project's test-debt round is scheduled, file it as a
   nice-to-have item referencing this PR.

---

## Revision log

*(No prior version — initial creation.)*

# Plan: rna-s2c3l0-020 — Fix unimportable legacy InputFeatureEmbedder

Finding ID: s2c3l0-020  
Branch: fix/rna-s2c3l0-020  
Base: origin/main  
Risk tier: R1 (legacy-only, 0 production callers, XS scope)

---

## §1 Context

`rna_predict/pipeline/stageA/input_embedding/legacy/encoder/input_feature_embedding.py`
raises `ModuleNotFoundError` on import. Two broken import sites reference
`rna_predict.models.encoder.atom_encoder`, a package that has never existed in this
repository. The real home of `AtomAttentionEncoder` and `AtomEncoderConfig` is the sibling
file `rna_predict/pipeline/stageA/input_embedding/legacy/encoder/atom_encoder.py`, whose
own line 6 self-documents the correct path with the comment
`# Corrected import path from models.attention to legacy.attention`.

Canonical audit source (Class E intent URL):
https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/audit/02-static-audit.md#L18

Two resolution paths exist:
- **Path A**: Fix the two broken import lines (4 and 36) to point to the sibling; verify the
  `PairformerWrapper` import at line 39 is also resolvable. Module becomes importable.
- **Path B**: Delete the file entirely. Requires 0-caller confirmation. Dead code eliminated;
  no future maintenance burden.

The operator's "When to AskUserQuestion" rule in the brief requires explicit Path A / Path B
confirmation at the write-code stage before any file is changed or deleted.

---

## §2 Verified state (direct tool queries, 2026-06-20)

All facts below were obtained by direct file reads and shell commands — no approximation.

| Claim | Evidence | Result |
|---|---|---|
| `rna_predict/models/` does not exist | `ls rna_predict/models` → "No such file or directory" | CONFIRMED |
| Line 4 broken import | Read file:4 — `from rna_predict.models.encoder.atom_encoder import AtomAttentionEncoder` | CONFIRMED |
| Line 36 broken import | Read file:36-38 — `from rna_predict.models.encoder.atom_encoder import AtomEncoderConfig` | CONFIRMED |
| `AtomAttentionEncoder` + `AtomEncoderConfig` at sibling | Read `atom_encoder.py` lines 13 and 41 | CONFIRMED |
| Sibling self-documents correct path | `atom_encoder.py:6` comment `# Corrected import path from models.attention to legacy.attention` | CONFIRMED |
| `PairformerWrapper` file exists | `ls rna_predict/pipeline/stageB/pairwise/pairformer_wrapper.py` → file found | CONFIRMED |
| `legacy/encoder/__init__.py` is empty (no re-exports) | Read file: system reports "1 line" with no content; brief confirms empty | CONFIRMED |
| Zero callers of the broken module | `grep -rn "from rna_predict.pipeline.stageA.input_embedding.legacy.encoder.input_feature_embedding" rna_predict/ tests/` → EXIT:1 (0 matches) | ZERO-CALLERS-CONFIRMED |
| Zero references to `input_feature_embedding` anywhere | `grep -rn "input_feature_embedding" rna_predict/ tests/` → EXIT:1 (0 matches) | CONFIRMED |
| mypy baseline (pre-patch, --ignore-missing-imports) | `mypy rna_predict/pipeline/stageA/input_embedding/legacy/encoder/ --ignore-missing-imports 2>&1 \| tail -5` → `Success: no issues found in 3 source files` EXIT:0 | BASELINE: 0 errors |
| Active `InputFeatureEmbedder` at `current/embedders.py:35` | Read `current/embedders.py` (per brief) — production callers: `benchmarks/benchmark.py:10`, `stageB/protenix_integration.py:28` | CONFIRMED INDEPENDENT |

Note: `grep` returns exit code 1 when it finds 0 matches (no error); exit code 0 when matches
are found. EXIT:1 here means "no matches" — ZERO-CALLERS-CONFIRMED passes.

---

## §5 Memory + lesson references

No project `MEMORY.md` found (brief confirms absent). Universal principles apply:

- Never merge autonomously; the human is the merge gate (H2).
- Run `pytest` before every push; do not push knowing CI will fail.
- Read the code-review body, not just its status.
- Never edit a test to make it pass without first establishing which side is wrong.
- Every deferred item must point to a follow-up (PR ID, stage, or issue #).
- AIV packets pass `aiv check`; do not restate spec rules in packet prose.
- AskUserQuestion before implementing: confirm Path A or Path B with the operator.

Lesson from sibling `atom_encoder.py:6`: when a legacy file already self-documents the
corrected import path in a comment, that comment is ground truth for Path A.

---

## §6 Strict scope boundaries

**In scope:**
- `rna_predict/pipeline/stageA/input_embedding/legacy/encoder/input_feature_embedding.py`
  (edit lines 4 and 36 under Path A, or delete under Path B)
- `rna_predict/pipeline/stageA/input_embedding/legacy/encoder/__init__.py`
  (inspect only; confirmed empty — no action needed under either path)
- `.github/aiv-packets/rna-s2c3l0-020.json` (new AIV packet, exactly one per commit)

**Explicitly out of scope (do not touch):**
- `rna_predict/pipeline/stageA/input_embedding/current/embedders.py` — functional, independent
- `rna_predict/pipeline/stageA/input_embedding/legacy/encoder/atom_encoder.py` — correct already
- `rna_predict/pipeline/stageA/input_embedding/legacy/attention/` — unrelated to this finding
- `rna_predict/pipeline/stageB/pairwise/protenix_integration.py` — imports from `current/`
- `rna_predict/pipeline/stageD/diffusion/` — imports from `current/`
- `rna_predict/benchmarks/benchmark.py` — imports from `current/`
- All other `audit/02-static-audit.md` findings — each has its own PR in the `fix/rna-s2c*` lineage

**Deferred items:**

| Item | Classification | Action |
|---|---|---|
| Other broken legacy imports across the audit | nice-to-have (each finding has its own PR) | Addressed in sibling PRs per audit triage |
| `atom_encoder.py` TODO comments (lines 62, 67, 113) | nice-to-have | Defer; out of scope for this finding |
| `atom_encoder.py` forward `# noqa: C901` suppression | nice-to-have | Defer; out of scope |

---

## §7 Locked design decisions

### Path fork: Path B (delete) is the recommended approach

**CORRECTNESS scoring (Path A vs Path B):**

| Criterion | Path A (fix imports) | Path B (delete file) |
|---|---|---|
| (a) Ground-truth data | Uses recorded sibling path | Uses grep-confirmed 0-caller count |
| (b) Root cause vs symptom | Fixes symptom (2 broken lines); dead code remains | Eliminates root cause; dead file removed entirely |
| (c) Hidden/deferred debt | Preserves ~115 lines of untested, uncalled legacy code | Zero residual debt |

**Operator cost function scoring:**

- **Drive A (scope):** Path B is strictly more complete — 0 callers means the invariant
  "module must be importable" holds for no caller; deleting eliminates all affected sites
  (there are none to fix). Path A fixes 2 import lines but leaves 115 lines of dead code
  as ongoing maintenance debt. **Path B preferred.**
- **Drive B (exemptions):** Neither path requires an exemption.
- **Drive C (ground truth):** The 0-caller count is system-recorded ground truth (grep result),
  not an approximation. Path B consumes it directly. **Both paths equal; Path B acts on it.**
- **Drive D (false completion):** Path A produces a module that is technically importable but
  has zero callers — a stub that makes callers believe behavior exists where no callers use it.
  This is the "looks done but isn't used" anti-pattern. Path B produces a clean deletion with
  no ambiguity. **Path B preferred.**
- **Drive E (live-fire):** Both paths require live-fire subprocess validation. Equal.

**Verdict: Path B (delete) is the operator-preferred approach** and is disfavored for Path A
with justification: Path A approximates "fix" when the correct semantic is "eliminate dead code
confirmed by ground-truth grep."

**HOWEVER**: The brief mandates AskUserQuestion at write-code stage before implementing.
The write-code agent MUST present the caller count (observed: 0) and ask the operator to
confirm Path B (delete) vs Path A (fix imports) before touching any file.

If the operator chooses **Path A**, the locked sub-decisions are:
- Line 4: replace `from rna_predict.models.encoder.atom_encoder import AtomAttentionEncoder`
  with `from rna_predict.pipeline.stageA.input_embedding.legacy.encoder.atom_encoder import AtomAttentionEncoder`
- Lines 36-38 (inside `__init__`): replace `from rna_predict.models.encoder.atom_encoder import (AtomEncoderConfig,)`
  with `from rna_predict.pipeline.stageA.input_embedding.legacy.encoder.atom_encoder import (AtomEncoderConfig,)`
- Do NOT move the top-level import inside `__init__` — fix it at its existing site (line 4).
- `PairformerWrapper` import at line 39 is already at the correct absolute path
  (`rna_predict.pipeline.stageB.pairwise.pairformer_wrapper`) and the file exists —
  no change needed there. UNVERIFIED — pending execution at write-code/design-tests.

If the operator chooses **Path B**, the locked sub-decisions are:
- Delete `input_feature_embedding.py` only.
- `__init__.py` confirmed empty — no re-export removal needed. Confirm again at write-code.
- Post-deletion, re-run `grep -rn "input_feature_embedding" rna_predict/ tests/` to confirm
  0 residual references.

---

## §9 Sequenced atomic-commit plan

Exactly one commit for the functional change + one AIV packet, per the atomic-commit rule.

**Pre-action (before any edit/delete — at write-code stage):**
1. AskUserQuestion: present caller count (0), recommend Path B, await operator confirmation.
2. Re-run zero-caller grep to confirm still 0 matches at write time:
   `grep -rn "input_feature_embedding" rna_predict/ tests/`

**Commit 1 — functional change (Path B) or import fix (Path A):**

Under Path B:
```
git rm rna_predict/pipeline/stageA/input_embedding/legacy/encoder/input_feature_embedding.py
git add .github/aiv-packets/rna-s2c3l0-020.json
git commit -m "fix(s2c3l0-020): delete unimportable dead-code InputFeatureEmbedder (0 callers)"
```

Under Path A:
```
git add rna_predict/pipeline/stageA/input_embedding/legacy/encoder/input_feature_embedding.py
git add .github/aiv-packets/rna-s2c3l0-020.json
git commit -m "fix(s2c3l0-020): repair broken models.encoder import path in legacy InputFeatureEmbedder"
```

No additional commits. This is a single-file change (+ packet).

**Test-layer contract for each gate:**

| Gate | Behavior under test | Input source | Layer | Evidence class |
|---|---|---|---|---|
| ZERO-CALLERS-CONFIRMED | No production file imports the broken module | `rna_predict/` + `tests/` filesystem | Static grep (live-fire against real FS) | Class A/B |
| IMPORT-OK-OR-FILE-GONE | Module importable (A) / no references remain (B) | Python import system / grep | Live-fire subprocess | Class A |
| SIBLING-IMPORT-INTACT | `atom_encoder.py` exports both names unbroken | Python import system | Live-fire subprocess | Class A |
| CURRENT-EMBEDDER-INTACT | `current/embedders.py` unaffected | Python import system | Live-fire subprocess | Class A |
| NO-REGRESSION | `pytest tests/stageA/` exits 0 | Real test suite against real source | Integration live-fire | Class A |
| TYPECHECK-LOCAL | mypy error count ≤ baseline (0) | Source tree (post-patch) | Static analysis | Class D |
| PACKET-VALIDATES | `aiv check` exits 0; Class E URL matches | AIV packet file | Tool invocation | Class D |
| ISSUE-CLOSED | PR body contains "s2c3l0-020" | PR description text | Manual check | Class E |

All "UNVERIFIED — pending execution at write-code/design-tests" until the write-code stage
runs each command and records its output in the AIV packet.

---

## §10 Critical files

| File | Role | Action |
|---|---|---|
| `rna_predict/pipeline/stageA/input_embedding/legacy/encoder/input_feature_embedding.py` | Subject of the finding; broken imports at lines 4, 36 | DELETE (Path B) or EDIT lines 4, 36 (Path A) |
| `rna_predict/pipeline/stageA/input_embedding/legacy/encoder/atom_encoder.py` | Ground truth for correct import path; provides `AtomAttentionEncoder` + `AtomEncoderConfig` | READ ONLY — do not modify |
| `rna_predict/pipeline/stageA/input_embedding/legacy/encoder/__init__.py` | Re-export surface; confirmed empty | READ ONLY — confirm still empty at write time |
| `rna_predict/pipeline/stageB/pairwise/pairformer_wrapper.py` | `PairformerWrapper` source; used at line 39 of subject file | READ ONLY — import path already correct |
| `rna_predict/pipeline/stageA/input_embedding/current/embedders.py` | Active `InputFeatureEmbedder`; must remain untouched | DO NOT TOUCH |
| `.github/aiv-packets/rna-s2c3l0-020.json` | AIV packet for this finding | CREATE |
| `audit/02-static-audit.md` | Canonical finding record; Class E anchor | READ (post-merge: annotate line 18 as resolved) |

---

## §11 Reused utilities (must consume, not reimplement)

- **`aiv check`** — the existing AIV validation toolchain. Do not reimplement packet
  validation; run `aiv check` and cite its exit code.
- **`pytest`** — the project's registered test runner (confirmed via `pytest.ini` and
  `pyproject.toml`). Do not use `npx vitest run` (the config default does not apply to
  this Python project).
- **`mypy --ignore-missing-imports`** — the project's type-checker as specified in the
  completion contract. Consume the recorded baseline (0 errors) rather than re-deriving it.
- **`uv run python -c`** — the project's virtualenv-aware Python runner. All import-check
  commands must use `uv run` rather than bare `python` to ensure the correct environment.
- **`grep -rn`** — for all caller/reference scans. Do not substitute `find` or `ag`.

---

## §14 Acceptance criteria

Binary green/red per the completion contract. The write-code stage must run each command,
capture its output, and record it verbatim in the AIV packet Class A evidence.

| # | Gate | Command | Pass condition |
|---|---|---|---|
| 1 | ZERO-CALLERS-CONFIRMED | `grep -rn "from rna_predict.pipeline.stageA.input_embedding.legacy.encoder.input_feature_embedding" rna_predict/ tests/` | EXIT:1, 0 lines output |
| 2A | IMPORT-OK (Path A only) | `uv run python -c 'from rna_predict.pipeline.stageA.input_embedding.legacy.encoder.input_feature_embedding import InputFeatureEmbedder'` | Exit 0, no output |
| 2B | FILE-GONE (Path B only) | `grep -rn "input_feature_embedding" rna_predict/ tests/` | EXIT:1, 0 lines output |
| 3 | SIBLING-IMPORT-INTACT | `uv run python -c 'from rna_predict.pipeline.stageA.input_embedding.legacy.encoder.atom_encoder import AtomAttentionEncoder, AtomEncoderConfig'` | Exit 0 |
| 4 | CURRENT-EMBEDDER-INTACT | `uv run python -c 'from rna_predict.pipeline.stageA.input_embedding.current.embedders import InputFeatureEmbedder'` | Exit 0 |
| 5 | NO-REGRESSION | `pytest tests/stageA/ -x --tb=short` | Exit 0 |
| 6 | TYPECHECK-LOCAL | `mypy rna_predict/pipeline/stageA/input_embedding/legacy/encoder/ --ignore-missing-imports 2>&1 \| tail -1` | Exit 0, error count ≤ baseline (0) |
| 7 | PACKET-VALIDATES | `aiv check` | Exit 0 |
| 8 | ISSUE-CLOSED | PR body text | Contains "s2c3l0-020" |

**Pre-merge checklist (non-command):**
- [ ] AskUserQuestion answered: operator confirmed Path A or Path B before implementing
- [ ] Diff scope: only `input_feature_embedding.py` (edited or deleted) + AIV packet; no other files
- [ ] AIV packet Class E `intent_url` exactly matches the canonical audit URL:
  `https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/audit/02-static-audit.md#L18`

---

## §15 Risks + mitigations + stop conditions (RED)

| Risk | Likelihood | Mitigation | RED stop condition |
|---|---|---|---|
| New caller appears in a freshly pushed branch between plan and write-code | Very low | Re-run ZERO-CALLERS grep at write-code start (step 2 in §9) | If any caller found: stop, do not delete, do not fix — AskUserQuestion with caller location |
| Path A: `PairformerWrapper` import at line 39 resolves but has a transitive broken import | Low | Run the full `ImportFeatureEmbedder` import test (gate 2A) which exercises all 3 imports | If 2A fails: AskUserQuestion — does the operator want Path B or a broader Path A? |
| Path B: `__init__.py` gained a re-export since plan-time observation | Very low | Re-read `__init__.py` at write-code start | If re-export found: stop — AskUserQuestion on whether to remove re-export too |
| pytest tests/stageA/ suite reveals tests that import the broken module | Very low (grep showed 0 references) | Gate 5 catches this immediately | If pytest exits non-zero due to import of deleted file: re-examine grep scope (tests/ subdirs) |
| AIV packet `aiv check` fails | Medium (toolchain config) | Run `aiv check` before committing; fix packet before push | If `aiv check` fails after packet edits: fix before pushing — never push with failing packet |
| mypy error count increases above baseline post-patch | Path A only; very low | Post-patch mypy run is required gate 6 | If count > 0 (baseline): fix new mypy errors or revert and AskUserQuestion |

**Hard RED stops (stop the PR, AskUserQuestion immediately):**
- Any caller found that was not in the plan-time grep
- Gate 2A (Path A) fails for a reason other than the two known broken imports
- Gate 5 (pytest) fails for a reason related to the change (not pre-existing flakiness)
- Any file outside the §6 scope is modified by the change

---

## §19 Locked PR sequence position

This PR is **standalone** in the `fix/rna-s2c*` lineage. Per the brief:

- Base branch: `origin/main`
- Branch: `fix/rna-s2c3l0-020`
- No upstream dependency: this fix does not require any other finding's PR to merge first.
- No downstream dependency: no other `fix/rna-s2c*` PR is gated on this fix.
- The PR may be opened and merged independently of all sibling audit-finding PRs.
- Iter budget: 2 live-fire cycles pre-authorized (R1 scope, isolated legacy file).

---

## §20 After-merge handoff

1. **Audit annotation**: in `audit/02-static-audit.md`, add a resolved annotation at L18,
   e.g., `<!-- RESOLVED: s2c3l0-020 — PR fix/rna-s2c3l0-020 merged YYYY-MM-DD -->`.
   Reference the merged PR number.
2. **Issue closure**: close any GitHub issue linked to finding s2c3l0-020.
3. **Unblock**: N/A — no downstream PRs are gated on this fix.
4. **Triggers**: N/A — no dependent stages or sibling PRs require this fix.
5. **Retro-verify**: N/A — R1 scope; CI passing post-merge is sufficient.
6. **Dead-code inventory**: if the team maintains a dead-code registry or audit triage
   tracker, mark `input_feature_embedding.py` as eliminated (Path B) or restored (Path A).

---

## Revision log

_(No prior iterations — fresh plan, 2026-06-20)_

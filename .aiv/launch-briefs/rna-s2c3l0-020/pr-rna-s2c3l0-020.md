# PR-rna-s2c3l0-020 - Fix unimportable legacy InputFeatureEmbedder

> **Config fallbacks (no `.aiv-workflow.yml` found):** all values are defaults.
> `launch_brief.out_dir` = `.aiv/launch-briefs/`; `branch.base` = `origin/main`;
> `aiv.check_cmd` = `aiv check`; `aiv.packets_dir` = `.github/aiv-packets`;
> `ci.test_cmd` default is `npx vitest run` — overridden by project discovery to `pytest`
> (confirmed via `pytest.ini` + `pyproject.toml`).
> `review.coord_file` absent — coord-file slot dropped.
> `review.spec_sections.progress_tracker` absent — progress-tracker closure slot dropped.
> `ci.local_replica_cmd` absent — pre-push-replica gate dropped (warning: verify CI locally before push via `pytest`).
> Memory store (`MEMORY.md`) absent — skipped silently; universal principles apply.

---

## Goal

Close finding s2c3l0-020 (critical). The file
`rna_predict/pipeline/stageA/input_embedding/legacy/encoder/input_feature_embedding.py`
raises `ModuleNotFoundError` on import because lines 4 and 36 reference
`rna_predict.models.encoder.atom_encoder`, a package that does not exist. Either repoint
both broken imports to the real location (`legacy.encoder.atom_encoder`) to make the class
importable (Path A), or delete the file entirely if zero production callers are confirmed
(Path B). Canonical audit record:
https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/audit/02-static-audit.md#L18

---

## High-level facts (verify each yourself)

- `rna_predict/models/` does not exist.
  Probe: `ls rna_predict/models` → "No such file or directory"

- Two broken import sites in `input_feature_embedding.py`:
  - Line 4 (module top-level): `from rna_predict.models.encoder.atom_encoder import AtomAttentionEncoder`
  - Line 36 (inside `__init__`): `from rna_predict.models.encoder.atom_encoder import AtomEncoderConfig`

- `AtomAttentionEncoder` and `AtomEncoderConfig` both live at:
  `rna_predict/pipeline/stageA/input_embedding/legacy/encoder/atom_encoder.py`
  The sibling file already self-documents this at line 6: `# Corrected import path from models.attention to legacy.attention`

- No production file imports from the broken module.
  Probe: `grep -rn "from rna_predict.pipeline.stageA.input_embedding.legacy.encoder.input_feature_embedding" rna_predict/ tests/`
  Observed result: 0 matches (file is dead code as of HEAD `1f6481e4`).

- The active `InputFeatureEmbedder` class is at
  `rna_predict/pipeline/stageA/input_embedding/current/embedders.py:35` and is what
  production callers use:
  - `rna_predict/benchmarks/benchmark.py:10`
  - `rna_predict/pipeline/stageB/pairwise/protenix_integration.py:28`

- `rna_predict/pipeline/stageA/input_embedding/legacy/encoder/__init__.py` is empty
  (no re-exports of `InputFeatureEmbedder`). Confirm: `cat` the file.

- The legacy file also imports `PairformerWrapper` from
  `rna_predict.pipeline.stageB.pairwise.pairformer_wrapper` inside `__init__` (line 38).
  If Path A is chosen, verify this import is also resolvable before committing.

---

## You decide

- **Path A vs Path B** — primary fork:
  - **Path A**: Fix lines 4 and 36 to import from
    `rna_predict.pipeline.stageA.input_embedding.legacy.encoder.atom_encoder`. Also verify
    the `PairformerWrapper` import at line 38 is resolvable. Result: module becomes
    importable and the legacy class is preserved.
  - **Path B**: Delete `input_feature_embedding.py`. Requires independently confirming
    zero callers (grep in production + tests), then removing the file and verifying no
    dangling references remain anywhere. Result: dead code eliminated; simpler diff.
  - Path B is the likely correct choice given 0 observed callers, but the implementing agent
    must verify independently and confirm with the operator before deleting.

- **`__init__.py` re-exports** (whichever path): If `legacy/encoder/__init__.py` contains
  anything that re-exports `InputFeatureEmbedder`, that export must be removed under Path B
  or updated under Path A. (Current observation: empty; verify before acting.)

---

## Worktree + branch

The worktree for this finding already exists on branch `fix/rna-s2c3l0-020`
(the start-PR ritual has been run). Base: `origin/main`.

If the harness creates a new branch via `branch.pattern` = `feat/{stage}-pr-{slug}`, the
actual branch name is owned by the harness — do not hard-code it in the AIV packet or commit
message. Reference the finding ID (`rna-s2c3l0-020`) as the work identifier.

---

## Gates (binary)

- **ZERO-CALLERS-CONFIRMED**: `grep -rn "from rna_predict.pipeline.stageA.input_embedding.legacy.encoder.input_feature_embedding" rna_predict/ tests/` exits 0 with 0 matches. Required before Path B; should be verified even for Path A to confirm bounded scope.

- **IMPORT-OK or FILE-GONE** (path-conditional):
  - Path A: `uv run python -c 'from rna_predict.pipeline.stageA.input_embedding.legacy.encoder.input_feature_embedding import InputFeatureEmbedder'` exits 0.
  - Path B: `grep -rn "input_feature_embedding" rna_predict/ tests/` exits 0 with 0 matches.

- **SIBLING-IMPORT-INTACT**: `uv run python -c 'from rna_predict.pipeline.stageA.input_embedding.legacy.encoder.atom_encoder import AtomAttentionEncoder, AtomEncoderConfig'` exits 0 (sibling not accidentally broken).

- **CURRENT-EMBEDDER-INTACT**: `uv run python -c 'from rna_predict.pipeline.stageA.input_embedding.current.embedders import InputFeatureEmbedder'` exits 0 (current/ tree unaffected).

- **NO-REGRESSION**: `pytest tests/stageA/ -x --tb=short` exits 0.

- **TYPECHECK-LOCAL**: `mypy rna_predict/pipeline/stageA/input_embedding/legacy/encoder/ --ignore-missing-imports` exits 0 or error count same-or-lower than pre-patch baseline (record baseline on unpatched tree first).

- **PACKET-VALIDATES**: `aiv check` exits 0; packet Class E `intent_url` is the canonical audit URL above.

- **ISSUE-CLOSED**: PR description contains a reference to `s2c3l0-020`.

---

## Iter budget

2 live-fire cycles pre-authorized (R1 scope, isolated legacy file, no production callers).

Escalation path: if Path A reveals that the `PairformerWrapper` import or another downstream
dependency is also broken, pause and AskUserQuestion before expanding scope.

---

## When to AskUserQuestion

- **Before implementing**: confirm Path A or Path B with the operator. Present the caller
  count (expected: 0) and ask whether to fix the imports or delete the file.
- **Scope expansion**: if `grep` reveals any caller that was missed, stop and ask before
  deciding to fix vs. delete.
- **Unexpected blocker**: if `PairformerWrapper` import (line 38) is also broken and Path A
  is chosen, ask whether to fix that import too or switch to Path B.

---

## Risk tier + scope estimate

**R1** — legacy-only file, no production callers, contained import fix or single-file deletion.
Scope: XS (1-3 lines changed under Path A; 1 file deleted under Path B, plus a 1-line
`__init__.py` check). No public API surface changed.

---

## Out-of-scope

- `rna_predict/pipeline/stageA/input_embedding/current/embedders.py` (`InputFeatureEmbedder`) — fully functional, independent of this finding. By design: not broken.
- `rna_predict/pipeline/stageA/input_embedding/legacy/encoder/atom_encoder.py` — already uses the correct import path (see line 6 comment). By design: not broken.
- `rna_predict/pipeline/stageD/diffusion/` references to `AtomAttentionEncoder` — import from `current/` tree, unaffected. By design: not broken.
- `rna_predict/pipeline/stageB/pairwise/protenix_integration.py` — imports from `current/` tree. By design: not broken.
- Other legacy/ components not directly causing this finding's import failure → defer to the ongoing audit/02-static-audit.md findings triage (each finding gets its own PR in the `fix/rna-s2c*` lineage).

---

## Reading order before start-PR

1. `rna_predict/pipeline/stageA/input_embedding/legacy/encoder/input_feature_embedding.py` (the broken file — read all lines)
2. `rna_predict/pipeline/stageA/input_embedding/legacy/encoder/atom_encoder.py` (correct sibling providing `AtomAttentionEncoder` + `AtomEncoderConfig`)
3. `rna_predict/pipeline/stageA/input_embedding/legacy/encoder/__init__.py` (re-export surface — expected empty)
4. `rna_predict/pipeline/stageA/input_embedding/current/embedders.py` (active `InputFeatureEmbedder` — confirm independence)
5. `audit/02-static-audit.md` line 18 — the canonical finding record (local copy of the SHA-pinned URL)

Universal principles (no memory store found; these always apply):
- Never merge autonomously. The human is the merge gate.
- Run `pytest` before every push; do not push knowing CI will fail.
- Read the code-review body, not just its status. A green review status is not zero findings.
- Never edit a test to make it pass without first establishing which side is wrong.
- Every deferred item must point to a follow-up (PR ID, stage, or issue #).
- AIV packets pass `aiv check`; do not restate spec rules in packet prose.

---

Now run the start-PR ritual.

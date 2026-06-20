===== PR-rna-s2c3l0-020 COMPLETION CONTRACT - Fix unimportable legacy InputFeatureEmbedder =====

GOAL: Close finding s2c3l0-020 (critical). The file
`rna_predict/pipeline/stageA/input_embedding/legacy/encoder/input_feature_embedding.py`
is unimportable: lines 4 and 36 reference `rna_predict.models.encoder.atom_encoder`, a
package that does not exist. Fix both broken imports to the real location
(`rna_predict.pipeline.stageA.input_embedding.legacy.encoder.atom_encoder`) making
`InputFeatureEmbedder` importable (Path A), or delete the file entirely after confirming
zero production callers (Path B). Canonical intent (Class E):
https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/audit/02-static-audit.md#L18

---

VERIFY (binary green/red):

[1] ZERO-CALLERS-CONFIRMED
cmd: grep -rn "from rna_predict.pipeline.stageA.input_embedding.legacy.encoder.input_feature_embedding" rna_predict/ tests/
pass: exits 0 with 0 matches
note: Required prerequisite for Path B; must also be run under Path A to confirm scope is bounded.

[2] IMPORT-OK-OR-FILE-GONE (path-conditional)
Option A — imports fixed:
  cmd: uv run python -c 'from rna_predict.pipeline.stageA.input_embedding.legacy.encoder.input_feature_embedding import InputFeatureEmbedder'
  pass: exits 0 with no output
Option B — file deleted:
  cmd: grep -rn "input_feature_embedding" rna_predict/ tests/
  pass: exits 0 with 0 matches (file gone, no dangling references)

[3] SIBLING-IMPORT-INTACT
cmd: uv run python -c 'from rna_predict.pipeline.stageA.input_embedding.legacy.encoder.atom_encoder import AtomAttentionEncoder, AtomEncoderConfig'
pass: exits 0
note: Confirms the sibling atom_encoder.py was not accidentally disturbed.

[4] CURRENT-EMBEDDER-INTACT
cmd: uv run python -c 'from rna_predict.pipeline.stageA.input_embedding.current.embedders import InputFeatureEmbedder'
pass: exits 0
note: Confirms the current/ production tree is unaffected by the patch.

[5] NO-REGRESSION
cmd: pytest tests/stageA/ -x --tb=short
pass: exits 0
note: ci.test_cmd config default (npx vitest run) does not apply to this Python project;
project test runner resolved as pytest from pytest.ini and pyproject.toml.

[6] TYPECHECK-LOCAL
cmd: mypy rna_predict/pipeline/stageA/input_embedding/legacy/encoder/ --ignore-missing-imports
pass: exits 0, or error count same-or-lower than the pre-patch baseline
drill: Record baseline on the unpatched file before applying the fix:
  mypy rna_predict/pipeline/stageA/input_embedding/legacy/encoder/ --ignore-missing-imports 2>&1 | tail -1
  Note the baseline error count; post-patch count must not exceed it.

[7] PACKET-VALIDATES
cmd: aiv check
pass: exits 0
drill: Packet Class E intent_url MUST be exactly:
  https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/audit/02-static-audit.md#L18
  (the canonical audit source for finding s2c3l0-020 — never a taskmaster task or a local file)
note: ci.local_replica_cmd absent — no pre-push local-CI replica gate available; run
  `pytest tests/stageA/` manually before push.

[8] ISSUE-CLOSED
check: PR description body contains a reference to finding ID s2c3l0-020
pass: text present (e.g. "Closes s2c3l0-020" or "Fixes finding s2c3l0-020")
note: progress-tracker closure slot dropped — review.spec_sections.progress_tracker not configured.

---

PRE-MERGE:

- [ ] All VERIFY items [1]-[8] are green (copy the cmd output for each into the PR body or AIV packet Class A evidence)
- [ ] AskUserQuestion answered: operator confirmed Path A or Path B before implementing
- [ ] Diff scope confirmed: only `input_feature_embedding.py` (edited or deleted) plus the AIV packet under `.github/aiv-packets/`; no unrelated files in the diff
- [ ] No coord-file slot (review.coord_file not configured — note dropped)

---

POST-MERGE:

- Bookkeeping: Mark finding s2c3l0-020 as resolved in the audit triage tracker or in `audit/02-static-audit.md` (add a resolved annotation at L18). Close any linked issue.
- Unblock: N/A — this is a standalone legacy fix; no downstream PRs are gated on it.
- Triggers: N/A — no dependent stages or sibling PRs require this fix.
- Retro-verify: N/A — R1 scope; CI passing post-merge is sufficient; no separate retro-verification step required.

---

OUT-OF-SCOPE REMINDERS:

- The current/ InputFeatureEmbedder (embedders.py:35) is independently functional — do not touch it.
- Other audit findings in audit/02-static-audit.md are each addressed in their own PR in the fix/rna-s2c* lineage.
- The stageD and stageB references to AtomAttentionEncoder import from current/ and are unaffected.

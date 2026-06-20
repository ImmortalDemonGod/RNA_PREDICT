# AIV Verification Packet (v2.2)

## Identification

| Field | Value |
|-------|-------|
| **Repository** | github.com/ImmortalDemonGod/RNA_PREDICT |
| **Change ID** | rna-s2c3l0-020-impl |
| **Commits** | `a8120fef` (functional), `ea1a9196` (packet) |
| **Head SHA** | `ea1a9196` |
| **Base SHA** | `d8280ca7` |
| **Created** | 2026-06-20T05:55:10Z |
| **Updated** | 2026-06-20T06:02:00Z |

## Classification

```yaml
classification:
  risk_tier: R1
  sod_mode: S0
  critical_surfaces: []
  blast_radius: component
  classification_rationale: "R1 — legacy-only file, 0 production callers confirmed by grep, XS scope (2 import lines corrected); isolated change with no downstream dependencies"
  classified_by: "Claude"
  classified_at: "2026-06-20T05:55:10Z"
```

## Claims

1. `InputFeatureEmbedder` is importable from `rna_predict.pipeline.stageA.input_embedding.legacy.encoder.input_feature_embedding` (module import exits 0, no ModuleNotFoundError)
2. Line 4 `from rna_predict.models.encoder.atom_encoder import AtomAttentionEncoder` is replaced with `from rna_predict.pipeline.stageA.input_embedding.legacy.encoder.atom_encoder import AtomAttentionEncoder`
3. Line 36 `from rna_predict.models.encoder.atom_encoder import (AtomEncoderConfig,)` inside `__init__` is replaced with `from rna_predict.pipeline.stageA.input_embedding.legacy.encoder.atom_encoder import (AtomEncoderConfig,)`
4. No production file in `rna_predict/` imports this module (`grep -rn "from rna_predict.pipeline.stageA.input_embedding.legacy.encoder.input_feature_embedding" rna_predict/` exits 1 with 0 matches)
5. Sibling `atom_encoder.py` exports `AtomAttentionEncoder` and `AtomEncoderConfig` unmodified (`uv run python -c` import exits 0)
6. No test files were modified or deleted by this change

---

## Evidence References

| # | Evidence File | Commit SHA | Classes |
|---|---------------|------------|---------|
| 1 | EVIDENCE_RNA_PREDICT_PIPELINE_STAGEA_INPUT_EMBEDDING_LEGACY_ENCODER_INPUT_FEATURE_EMBEDDING.md | `a8120fef` | A, B, D, E |

---

### Class A (Behavioral / Direct Execution Evidence)

No CI pipeline URL: headless local execution; all evidence collected via direct `uv run` command invocation (virtualenv-aware) at commit `a8120fef`. Full verbatim outputs captured in evidence file `EVIDENCE_RNA_PREDICT_PIPELINE_STAGEA_INPUT_EMBEDDING_LEGACY_ENCODER_INPUT_FEATURE_EMBEDDING.md` at commit `a8120fef` (see Evidence References above).

**Gate 2A — IMPORT-OK (Claim 1):**
`uv run python -c 'from rna_predict.pipeline.stageA.input_embedding.legacy.encoder.input_feature_embedding import InputFeatureEmbedder'` → prints class reference, EXIT:0. ModuleNotFoundError no longer raised.

**Gate 3 — SIBLING-IMPORT-INTACT (Claim 5):**
`uv run python -c 'from rna_predict.pipeline.stageA.input_embedding.legacy.encoder.atom_encoder import AtomAttentionEncoder, AtomEncoderConfig; print("OK")'` → prints "OK", EXIT:0.

**Gate 5 — RED-TESTS (3/4 GREEN):**
`uv run pytest tests/stageA/unit/input_embeding/legacy/test_input_feature_embedding.py -v --tb=short` result:
- PASSED `test_module_is_importable__guards_against_missing_rna_predict_models_package`
- PASSED `test_class_is_nn_module_subclass__guards_against_phantom_class_from_broken_import`
- FAILED `test_instantiation_succeeds__guards_against_init_lazy_import_of_missing_models_package` — `AttributeError: module 'torch.library' has no attribute 'custom_op'` via PairformerWrapper→protenix→deepspeed chain; pre-existing BUG-3; documented as deferred in `input_feature_embedding.bug-catalog.md` Skipped section (lines 100-107)
- PASSED `test_goal_verification_import_exits_clean__primary_deliverable_of_finding_s2c3l0_020`

Pre-change failure verification (git stash + re-run): test 3 failed with `ModuleNotFoundError: No module named 'rna_predict.models'` at `input_feature_embedding.py:4` — BUG-1 (now fixed). Test 3 now fails at BUG-3 (pre-existing, out of scope). BUG-1 and BUG-2 are resolved.

---

### Class B (Referential Evidence — SHA-pinned, line-anchored)

**Claim 2** — Line 4 corrected import:
[`input_feature_embedding.py#L4`](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/a8120fef7db0e3e925ee33953fab6f3a89fe4245/rna_predict/pipeline/stageA/input_embedding/legacy/encoder/input_feature_embedding.py#L4) at commit `a8120fef` — `from rna_predict.pipeline.stageA.input_embedding.legacy.encoder.atom_encoder import AtomAttentionEncoder`

**Claim 3** — Line 36 corrected import:
[`input_feature_embedding.py#L36`](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/a8120fef7db0e3e925ee33953fab6f3a89fe4245/rna_predict/pipeline/stageA/input_embedding/legacy/encoder/input_feature_embedding.py#L36) at commit `a8120fef` — `from rna_predict.pipeline.stageA.input_embedding.legacy.encoder.atom_encoder import (AtomEncoderConfig,)`

**Ground truth for correct path** — `atom_encoder.py:6` self-documents the correction:
[`atom_encoder.py#L6`](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/a8120fef7db0e3e925ee33953fab6f3a89fe4245/rna_predict/pipeline/stageA/input_embedding/legacy/encoder/atom_encoder.py#L6) — `# Corrected import path from models.attention to legacy.attention`

**Scope boundary** — only 2 files changed in functional commit `a8120fef`:
```text
rna_predict/pipeline/stageA/input_embedding/legacy/encoder/input_feature_embedding.py  (4 lines: +2 corrected, -2 broken)
.github/aiv-evidence/EVIDENCE_...INPUT_FEATURE_EMBEDDING.md  (94 lines: new evidence file)
```

---

### Class C (Negative Evidence — what was searched for and NOT found)

**Zero production callers (Gate 1):**
```bash
$ grep -rn "from rna_predict.pipeline.stageA.input_embedding.legacy.encoder.input_feature_embedding" rna_predict/
EXIT:1  (0 matches)
```
Claim 4 VERIFIED. No file in `rna_predict/` imports this module.

**rna_predict.models does not exist:**
```bash
$ ls rna_predict/models
ls: cannot access 'rna_predict/models': No such file or directory
```
The nonexistent package that the pre-fix imports referenced never existed.

**No out-of-scope files modified:**
```bash
$ git diff a8120fef~1 a8120fef --name-only
.github/aiv-evidence/EVIDENCE_...INPUT_FEATURE_EMBEDDING.md
rna_predict/pipeline/stageA/input_embedding/legacy/encoder/input_feature_embedding.py
```
Zero files from the §6 "do not touch" list appear in the diff.

**No broken `rna_predict.models` references remain in the patched file:**
```bash
$ grep -n "rna_predict.models" rna_predict/pipeline/stageA/input_embedding/legacy/encoder/input_feature_embedding.py
EXIT:1  (0 matches)
```

**Bug catalog Skipped entries (not regressed, not in scope):**
- `BUG-3 (psutil/deepspeed)`: `input_feature_embedding.bug-catalog.md#L100-L107` — explicitly deferred; surfaced post-fix as expected; pre-existing environment incompatibility
- `forward()` shape correctness tests: deferred per bug catalog Skipped section — out of scope
- `trunk_pair` branch tests: deferred per bug catalog Skipped section — out of scope

**Claim 6 — no test files modified:**
```bash
$ git diff a8120fef~1 a8120fef --name-only -- tests/
(empty output — no test files in diff)
```

---

### Class D (Static Analysis: lint/type/build)

**Gate 6 — TYPECHECK-LOCAL (mypy on target directory, post-patch):**
```bash
$ mypy rna_predict/pipeline/stageA/input_embedding/legacy/encoder/ --ignore-missing-imports 2>&1 | tail -1
Success: no issues found in 3 source files
EXIT:0
```
Error count = 0, equal to pre-patch baseline (0). Claim 3 supported.

**Ruff lint on changed file:**
```bash
$ uv run ruff check rna_predict/pipeline/stageA/input_embedding/legacy/encoder/input_feature_embedding.py
All checks passed!
EXIT:0
```

---

### Class E (Intent Alignment)

**Canonical intent URL:** [https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/audit/02-static-audit.md#L18](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/audit/02-static-audit.md#L18)

**Source content (read from audit/02-static-audit.md L18-L22):**
> `[CRITICAL] s2c3l0-020 — bug`  
> Location: `rna_predict/pipeline/stageA/input_embedding/legacy/encoder/input_feature_embedding.py:4`  
> Evidence: InputFeatureEmbedder imports `from rna_predict.models.encoder.atom_encoder import AtomAttentionEncoder` at module top level (line 4) and again inside __init__ (line 36, AtomEncoderConfig). The package rna_predict/models does not exist. Therefore importing this legacy module raises ModuleNotFoundError immediately — the file is unimportable.  
> Recommendation: Repoint the imports to rna_predict.pipeline.stageA.input_embedding.legacy.encoder.atom_encoder (the real location), or delete this dead legacy module if superseded by the current/ tree.

**Alignment assessment:** The audit records a defect at lines 4 and 36 where both import paths reference the nonexistent `rna_predict.models` package, and recommends repointing to `rna_predict.pipeline.stageA.input_embedding.legacy.encoder.atom_encoder`. This change does exactly that: replaces both broken import strings with the recommended legacy path, as derived from ground truth in the sibling `atom_encoder.py:6` comment. Gate 2A (import exits 0) and test 4 (goal-verification test) confirm the defect described in the audit is resolved.

---

### Class F (Provenance — git chain-of-custody of touched test files)

**Git chain for this change:**
```text
d8280ca7  (base) docs(aiv): fix verification packet — add all evidence classes A-F
a8120fef  (functional) fix(s2c3l0-020): repair broken models.encoder import path in legacy InputFeatureEmbedder
ea1a9196  (packet) docs(aiv): verification packet for change 'rna-s2c3l0-020-impl'
```

**Claim 6 — test files preserved unmodified:**

[`tests/stageA/unit/input_embeding/legacy/test_input_feature_embedding.py#L1-L109`](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/a8120fef7db0e3e925ee33953fab6f3a89fe4245/tests/stageA/unit/input_embeding/legacy/test_input_feature_embedding.py#L1-L109) — file at commit `a8120fef`; created in design-tests stage at commit `4f940e73`; 0 lines added/deleted/modified by this change.

[`tests/stageA/unit/input_embeding/legacy/input_feature_embedding.bug-catalog.md`](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/a8120fef7db0e3e925ee33953fab6f3a89fe4245/tests/stageA/unit/input_embeding/legacy/input_feature_embedding.bug-catalog.md) — file at commit `a8120fef`; created in design-tests stage at commit `1c4377d9`; 0 lines added/deleted/modified by this change.

`git diff a8120fef~1 a8120fef --name-only -- tests/` → empty output (0 test files appear in diff). Existing tests preserved.

**Functional file provenance:**

[`rna_predict/pipeline/stageA/input_embedding/legacy/encoder/input_feature_embedding.py`](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/a8120fef7db0e3e925ee33953fab6f3a89fe4245/rna_predict/pipeline/stageA/input_embedding/legacy/encoder/input_feature_embedding.py) at commit `a8120fef` — pre-exists on branch; 2 import strings replaced (lines 4 and 36); 0 lines added or deleted overall.

---

## Verification Methodology

**Zero-Touch Mandate:** Verifier inspects artifacts only.
Evidence collected by `aiv commit` running: git diff (scope inventory), AST symbol-to-test binding, ruff, mypy. Additional manual gate commands (`uv run python -c`, `grep -rn`, `uv run pytest`) executed and recorded verbatim above.
Packet generated by `aiv close` and updated to add all evidence classes A–F.

---

## Known Limitations

- Test 3 (`test_instantiation_succeeds`) remains RED due to BUG-3 (deepspeed/torch incompatibility in `PairformerWrapper` import chain). This is a pre-existing environment issue, not introduced by this change, and is explicitly deferred in the bug catalog (Skipped section).
- `aiv commit` Class A reported "0 passed, 0 failed" for pytest — the tool ran on a scope that excluded the target test directory. Direct invocation (`uv run pytest tests/stageA/unit/input_embeding/legacy/...`) used instead and results recorded verbatim in Class A above.

---

## Summary

Change `rna-s2c3l0-020-impl`: 1 functional commit (`a8120fef`). Corrects two broken import paths in `input_feature_embedding.py` (lines 4 and 36) from the nonexistent `rna_predict.models.encoder.atom_encoder` to the correct sibling path `rna_predict.pipeline.stageA.input_embedding.legacy.encoder.atom_encoder`. Module is now importable (Gate 2A: exit 0). Three of four RED tests GREEN; primary deliverable test (test 4) PASSED. Test 3 remains RED for pre-existing BUG-3 (out of scope, deferred). mypy: 0 errors (= baseline). ruff: clean.

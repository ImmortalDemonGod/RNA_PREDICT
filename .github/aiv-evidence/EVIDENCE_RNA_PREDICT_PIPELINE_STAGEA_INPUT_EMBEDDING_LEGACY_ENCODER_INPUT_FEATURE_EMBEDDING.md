# AIV Evidence File (v1.0)

**File:** `rna_predict/pipeline/stageA/input_embedding/legacy/encoder/input_feature_embedding.py`
**Commit:** `d8280ca`
**Generated:** 2026-06-20T05:53:57Z
**Protocol:** AIV v2.0 + Addendum 2.7 (Zero-Touch Mandate)

---

## Classification (required)

```yaml
classification:
  risk_tier: R1
  sod_mode: S0
  critical_surfaces: []
  blast_radius: "rna_predict/pipeline/stageA/input_embedding/legacy/encoder/input_feature_embedding.py"
  classification_rationale: "R1 — legacy-only file, 0 production callers confirmed by grep, XS scope (2 import lines corrected to sibling path documented in atom_encoder.py:6)"
  classified_by: "Claude"
  classified_at: "2026-06-20T05:53:57Z"
```

## Claim(s)

1. InputFeatureEmbedder is importable from rna_predict.pipeline.stageA.input_embedding.legacy.encoder.input_feature_embedding (exit 0)
2. Line 4 rna_predict.models.encoder.atom_encoder import replaced with rna_predict.pipeline.stageA.input_embedding.legacy.encoder.atom_encoder
3. Line 36 rna_predict.models.encoder.atom_encoder import inside __init__ replaced with rna_predict.pipeline.stageA.input_embedding.legacy.encoder.atom_encoder
4. No production caller in rna_predict/ imports this module (grep -rn from ...input_feature_embedding rna_predict/ exits 1 with 0 matches)
5. Sibling atom_encoder.py exports AtomAttentionEncoder and AtomEncoderConfig unmodified (uv run python -c import exits 0)
6. No existing tests were modified or deleted during this change.

---

## Evidence

### Class E (Intent Alignment)

- **Link:** [https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/audit/02-static-audit.md#L18](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/audit/02-static-audit.md#L18)
- **Requirements Verified:** Audit finding s2c3l0-020 at audit/02-static-audit.md#L18 records that importing the legacy InputFeatureEmbedder raises ModuleNotFoundError because rna_predict.models does not exist; the fix must make the module importable

### Class B (Referential Evidence)

**Scope Inventory** (SHA: [`d8280ca`](https://github.com/ImmortalDemonGod/RNA_PREDICT/tree/d8280ca7b928bf68d41b651583f1b2d98884ddc8))

- [`rna_predict/pipeline/stageA/input_embedding/legacy/encoder/input_feature_embedding.py#L4`](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/d8280ca7b928bf68d41b651583f1b2d98884ddc8/rna_predict/pipeline/stageA/input_embedding/legacy/encoder/input_feature_embedding.py#L4)
- [`rna_predict/pipeline/stageA/input_embedding/legacy/encoder/input_feature_embedding.py#L36`](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/d8280ca7b928bf68d41b651583f1b2d98884ddc8/rna_predict/pipeline/stageA/input_embedding/legacy/encoder/input_feature_embedding.py#L36)

### Class A (Execution Evidence)

**Per-symbol test coverage (AST analysis):**

- **`InputFeatureEmbedder`** (L4): PASS -- 1 test(s) call `InputFeatureEmbedder` directly
  - `tests/stageA/unit/input_embeding/current/test_token_feature_shape.py::test_input_feature_embedder_deletion_mean_shape`
- **`InputFeatureEmbedder.__init__`** (L36): PASS -- 7 test(s) call `__init__` directly
  - `tests/test_debug_logging.py::test_stageB_debug_logging_hypothesis`
  - `tests/integration/test_partial_checkpoint_stageA.py::test_partial_checkpoint_stageA`
  - `tests/integration/test_pipeline_dimensions.py::test_stageA_to_B_dimensions`
  - `tests/integration/test_pipeline_dimensions.py::test_full_pipeline_dimensions`
  - `tests/integration/test_partial_checkpoint_cycle.py::test_train_save_partial_load_infer`
  - `tests/performance/test_performance.py::test_diffusion_single_embed_caching`
  - `tests/tmp_tests/test_dssr_installation.py::test_dssr_installation_calledprocesserror`

**Coverage summary:** 2/2 symbols verified by tests.

### Code Quality (Linting & Types)

- **ruff:** All checks passed
- **mypy:** Found 9 errors in 7 files (checked 1 source file)

## Claim Verification Matrix

| # | Claim | Type | Evidence | Verdict |
|---|-------|------|----------|---------|
| 1 | InputFeatureEmbedder is importable from rna_predict.pipeline... | symbol | 1 test(s) call `InputFeatureEmbedder` | PASS VERIFIED |
| 2 | Line 4 rna_predict.models.encoder.atom_encoder import replac... | unresolved | No automatic binding available | REVIEW MANUAL REVIEW |
| 3 | Line 36 rna_predict.models.encoder.atom_encoder import insid... | symbol | 7 test(s) call `InputFeatureEmbedder.__init__` | PASS VERIFIED |
| 4 | No production caller in rna_predict/ imports this module (gr... | unresolved | No automatic binding available | REVIEW MANUAL REVIEW |
| 5 | Sibling atom_encoder.py exports AtomAttentionEncoder and Ato... | unresolved | No automatic binding available | REVIEW MANUAL REVIEW |
| 6 | No existing tests were modified or deleted during this chang... | structural | Class C not collected | REVIEW MANUAL REVIEW |

**Verdict summary:** 2 verified, 0 unverified, 4 manual review.
---

## Verification Methodology

**Zero-Touch Mandate:** Verifier inspects artifacts only.
Evidence collected by `aiv commit` running: git diff (scope inventory), AST symbol-to-test binding (2/2 symbols verified).
Ruff/mypy results are in Code Quality (not Class A) because they prove syntax/types, not behavior.

---

## Summary

Replace two rna_predict.models.encoder.atom_encoder import paths at lines 4 and 36 with the correct rna_predict.pipeline.stageA.input_embedding.legacy.encoder.atom_encoder path

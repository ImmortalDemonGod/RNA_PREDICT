# AIV Evidence File (v1.0)

**File:** `tests/stageA/unit/input_embeding/legacy/input_feature_embedding.bug-catalog.md`
**Commit:** `3396bd3`
**Generated:** 2026-06-20T05:25:35Z
**Protocol:** AIV v2.0 + Addendum 2.7 (Zero-Touch Mandate)

---

## Classification (required)

```yaml
classification:
  risk_tier: R0
  sod_mode: S0
  critical_surfaces: []
  blast_radius: "tests/stageA/unit/input_embeding/legacy/input_feature_embedding.bug-catalog.md"
  classification_rationale: "Pure documentation artifact — markdown bug catalog, no executable code, R0 trivial tier appropriate"
  classified_by: "Claude"
  classified_at: "2026-06-20T05:25:35Z"
```

## Claim(s)

1. Bug catalog enumerates BUG-1 (line 4 top-level import of nonexistent rna_predict.models), BUG-2 (line 36 duplicate bad import in __init__), BUG-3 (latent psutil dep), plus Skipped section
2. No existing tests were modified or deleted during this change.

---

## Evidence

### Class E (Intent Alignment)

- **Link:** [https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/audit/02-static-audit.md#L18](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/audit/02-static-audit.md#L18)
- **Requirements Verified:** Finding s2c3l0-020 requires test-design stage to produce bug catalog next to the test file before writing tests

### Class B (Referential Evidence)

**Scope Inventory** (SHA: [`3396bd3`](https://github.com/ImmortalDemonGod/RNA_PREDICT/tree/3396bd3702597e2e05b21610dda763d95b3484ce))

- [`tests/stageA/unit/input_embeding/legacy/input_feature_embedding.bug-catalog.md#L1-L123`](https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/3396bd3702597e2e05b21610dda763d95b3484ce/tests/stageA/unit/input_embeding/legacy/input_feature_embedding.bug-catalog.md#L1-L123)

### Class A (Execution Evidence)

- Local checks skipped (--skip-checks).
- **Skip reason:** Markdown documentation file only — no Python code to lint, type-check, or test


---

## Verification Methodology

**R0 (trivial) -- local checks skipped.**
**Reason:** Markdown documentation file only — no Python code to lint, type-check, or test
Only git diff scope inventory was collected. No execution evidence.

---

## Summary

Add bug catalog cataloging 3 bugs in InputFeatureEmbedder: primary broken import path at module level

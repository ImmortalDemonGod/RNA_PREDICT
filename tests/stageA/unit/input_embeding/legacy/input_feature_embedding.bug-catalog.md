# Bug Catalog: `rna_predict/pipeline/stageA/input_embedding/legacy/encoder/input_feature_embedding.py`

Generated: 2026-06-20  
Finding: s2c3l0-020 (critical)  
Canonical intent: https://github.com/ImmortalDemonGod/RNA_PREDICT/blob/1f6481e4d8d7c673f44115c0a5bbaa1703ebe562/audit/02-static-audit.md#L18

---

## Code Summary

### Public interface
Single export: `InputFeatureEmbedder(nn.Module)`.  
Constructor signature: `(c_token=384, restype_dim=32, profile_dim=32, c_atom=128, c_pair=32, num_heads=4, num_layers=3, use_optimized=False, pairformer_blocks=48)`.  
`forward(f, trunk_sing=None, trunk_pair=None, block_index=None) -> Tensor[N_token, c_token]`.

### Load-bearing comments
- `atom_encoder.py:6` – "Corrected import path from models.attention to legacy.attention" – this comment documents that `rna_predict.models.*` was a stale, wrong path that had already been fixed in one sibling but not here.

### IO boundaries
No filesystem/network/DB. Computation only once instantiated. The only boundary is the Python module import system itself: the file depends on two other modules being resolvable at import time.

### Branching points
- Line 98: `if trunk_pair is not None` – activates `PairformerWrapper` path.
- Line 102: `if block_index is None` – full-attention mask vs. provided mask.
- Line 95: early `single_emb = a_token + extras_emb` before optional pairformer.

### Type definitions of magic-string contracts
No magic strings. All tensor keys are dict string literals (`"restype"`, `"profile"`, `"deletion_mean"`, `"ref_pos"`, `"ref_charge"`, `"ref_element"`, `"ref_atom_name_chars"`, `"atom_to_token"`).

### Existing tests
None. No test file exists for `InputFeatureEmbedder` prior to this session. The `.md` file alongside the source is a test-design prompt document, not a test.

---

## Bug Catalog

### BUG-1 (PRIMARY) — Top-level import of nonexistent `rna_predict.models` package causes `ModuleNotFoundError` at import time

**Failure mode:** `import rna_predict.pipeline.stageA.input_embedding.legacy.encoder.input_feature_embedding` raises `ModuleNotFoundError: No module named 'rna_predict.models'` immediately — no code in the file can execute.

**Location:** `input_feature_embedding.py:4`  
```python
from rna_predict.models.encoder.atom_encoder import AtomAttentionEncoder
```

**Blast radius:** The entire module is unimportable. Any caller that attempts `from ... import InputFeatureEmbedder` crashes at import time. StageA pipeline initialization fails before any forward pass is attempted. The goal-verification command `uv run python -c 'from rna_predict.pipeline.stageA.input_embedding.legacy.encoder.input_feature_embedding import InputFeatureEmbedder'` exits non-zero.

**Why it's plausible:** The file was written when `rna_predict.models.encoder.atom_encoder` existed (or was planned). The actual module was later placed at `rna_predict.pipeline.stageA.input_embedding.legacy.encoder.atom_encoder` (as `atom_encoder.py:6` documents), but `input_feature_embedding.py` was never updated. Grep confirms `rna_predict/models/` does not exist anywhere in the repo.

**Verified:** `uv run python -c 'from rna_predict.pipeline.stageA.input_embedding.legacy.encoder.input_feature_embedding import InputFeatureEmbedder'` → `ModuleNotFoundError: No module named 'rna_predict.models'` (exit 1).

**Test type:** Import-contract / captured-bug pin.  
**Mapped test:** `test_module_is_importable__guards_against_missing_rna_predict_models_package`

---

### BUG-2 (SECONDARY) — Lazy import inside `__init__` also references the nonexistent `rna_predict.models` path

**Failure mode:** Even if line 4 were patched, `InputFeatureEmbedder.__init__` at line 36 re-imports `AtomEncoderConfig` from `rna_predict.models.encoder.atom_encoder`, so instantiation also raises `ModuleNotFoundError`.

**Location:** `input_feature_embedding.py:36`  
```python
from rna_predict.models.encoder.atom_encoder import (
    AtomEncoderConfig,
)
```

**Blast radius:** Any instantiation of `InputFeatureEmbedder` fails even if the top-level import were somehow mocked. This is a belt-and-suspenders confirmation that the fix must update both line 4 and line 36.

**Why it's plausible:** Duplicated copy-paste of the wrong path. Both lines reference the same nonexistent package; a partial fix that patches only one would still leave the class unusable.

**Test type:** Captured bug / contract pin.  
**Mapped test:** `test_class_is_nn_module_subclass__guards_against_phantom_class_from_broken_import` and `test_instantiation_succeeds__guards_against_init_lazy_import_of_missing_models_package`

---

### BUG-3 (LATENT) — `PairformerWrapper` import inside `__init__` (line 39) may also fail if `psutil` is absent

**Failure mode:** `pairformer_wrapper.py:26` does `import psutil`, which is not in `pyproject.toml` dependencies. If `psutil` is not installed, the lazy import of `PairformerWrapper` inside `__init__` also raises `ModuleNotFoundError: No module named 'psutil'`.

**Location:** `input_feature_embedding.py:39` → `pairformer_wrapper.py:26`

**Blast radius:** Instantiation fails if `psutil` is absent, even after the `rna_predict.models` path is fixed. Separate from BUG-1/BUG-2 but composes with them.

**Why it's plausible:** `psutil` is an optional monitoring dependency not listed in the core requirements. In CI environments without it, `InputFeatureEmbedder()` raises even after the primary path bug is fixed.

**Test type:** Negative path / dependency-contract pin.  
**Mapped test:** Deferred — see Skipped section. BUG-1 and BUG-2 must be fixed first; this bug only surfaces after the primary import path is corrected.

---

## Self-Critique

| Test | Catches catalog bug? | Passes for wrong-but-stable output? | Fails under behavior-preserving refactor? |
|---|---|---|---|
| `test_module_is_importable__guards_against_missing_rna_predict_models_package` | Yes — BUG-1 | No — only assertion is that import succeeds | No — tests observable contract (importability) |
| `test_class_is_nn_module_subclass__guards_against_phantom_class_from_broken_import` | Yes — BUG-1 (prerequisite) + confirms class shape | No — asserts inheritance, not internal structure | No — `issubclass(X, nn.Module)` is stable contract |
| `test_instantiation_succeeds__guards_against_init_lazy_import_of_missing_models_package` | Yes — BUG-2 | No — asserts instance exists | No — tests public API |

---

## Skipped

| Bug / scenario | Reason for exclusion |
|---|---|
| BUG-3 (`psutil` absence) | Blocked by BUG-1 and BUG-2; fix those first, then pin this as a separate finding if `psutil` is still missing. Not in scope of this finding (s2c3l0-020). |
| `forward()` shape correctness (restype/profile tensor dims) | Out of scope for this finding; the module is unimportable — testing forward behavior requires the import fix first. Deferred to a follow-up. |
| `trunk_pair` branch in `forward()` | Same reason — deferred until module is importable. Nice-to-have characterization. |
| `block_index=None` default-mask branch | Same as above. |
| `extras_emb` dimension mismatch if `restype_dim` != actual restype tensor width | Architectural-correctness concern but dependent on fix landing first; classify as deferrable until after s2c3l0-020 is merged. |

---

## Evaluation (to be filled after test run)

### Bugs caught (test failed first run, fix needed)
_Pending — tests written in RED state intentionally._

### Bugs characterized (test passed first run, behavior pinned)
_Pending._

### Bugs discovered during writing not in original catalog
- BUG-3 (`psutil` import chain failure inside `PairformerWrapper`) — surfaced while tracing the `__init__` import chain. Cataloged above as latent; deferred pending primary fix.

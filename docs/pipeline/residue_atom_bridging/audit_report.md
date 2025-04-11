# Audit Report: Residue vs. Atom Representation Mismatches (Stages B-D)

**Date:** 2025-04-10
**Auditor:** Roo

## 1. Introduction

This report documents the findings of a code audit focused on identifying mismatches and ad-hoc fixes related to residue-level vs. atom-level tensor representations between Stages B, C, and D of the M2 RNA prediction pipeline. Stage B outputs are typically residue-level (`[N_residue, c_s]`), while Stage D often requires atom-level inputs (`[N_atom, c_s]` or `[N_atom, 3]`). The goal is to locate existing fixes (e.g., tensor repeats, slicing, warnings) to inform the design of a systematic bridging mechanism.

## 2. Audit Findings

The audit focused on `rna_predict/pipeline/stageD/` and its subdirectories.

### 2.1. Explicit Shape Validation and Fixing

*   **Location:** `rna_predict/pipeline/stageD/diffusion/run_stageD_unified.py`
*   **Function:** `validate_and_fix_shapes(partial_coords, trunk_embeddings, input_features, ...)`
*   **Description:** This function explicitly checks the sequence dimensions of input `trunk_embeddings` tensors (specifically keys `s_trunk`, `s_inputs`, `sing`, `pair`) against the number of atoms (`num_atoms`) derived from the `partial_coords` tensor (`partial_coords.shape[1]`).
*   **Fix Mechanism:**
    *   **Expansion:** If an embedding's sequence dimension is *smaller* than `num_atoms`, `torch.Tensor.repeat()` is used to duplicate the tensor along that dimension until it matches or exceeds `num_atoms`. The result is then truncated to exactly `num_atoms`. (Lines 61-68 for single dim, 91-99 for pair dims).
    *   **Truncation:** If an embedding's sequence dimension is *larger* than `num_atoms`, simple slicing (`[:, :num_atoms]`) is used. (Lines 70-71 for single dim, 101-102 for pair dims).
*   **Logging:** Issues `warnings.warn` messages when adjustments are made:
    *   `"Adjusting sequence length for {key} from {shape[1]} to {num_atoms}"`
    *   `"Adjusting sequence lengths for {key} from ({shape[1]}, {shape[2]}) to ({num_atoms}, {num_atoms})"`
*   **Partial Coordinate Usage:** `partial_coords` (assumed `[B, N_atom, 3]`) serves as the reference for the target atom dimension (`num_atoms`).

### 2.2. Implicit Shape Fixing via Monkey-Patching

*   **Location:** `rna_predict/pipeline/stageD/tensor_fixes/__init__.py`
*   **Function:** `apply_tensor_fixes()` (called from `rna_predict/pipeline/stageD/diffusion/run_stageD_unified.py::run_stageD_diffusion`)
*   **Description:** This function applies several monkey-patches to core PyTorch operations to implicitly handle shape mismatches during runtime. Two relevant patches identified:
    1.  **`fix_tensor_add` (Patches `torch.Tensor.__add__`)**:
        *   **Fix Mechanism:** Intercepts `RuntimeError` during tensor addition. Attempts to resolve mismatches by:
            *   Handling different dimension counts (`_handle_dimension_count_mismatch`).
            *   Handling specific attention bias cases (`_handle_attention_bias_mismatch`).
            *   Attempting manual broadcasting (`_try_manual_broadcasting`).
            *   Uses `_expand_tensor_dimension` which employs `torch.Tensor.repeat_interleave()` for expansion or `torch.nn.functional.adaptive_avg_pool1d` for reduction along the mismatched dimension. This *could* affect residue-to-atom dimension mismatches if such tensors are added together.
        *   **Logging:** None directly within the patch; relies on catching the `RuntimeError`.
    2.  **`fix_atom_attention_encoder` (Patches `torch.nn.Module.forward`)**:
        *   **Fix Mechanism:** Specifically targets modules likely named `AtomAttentionEncoder`. If input tensors `r_l` and `s` have different sequence lengths (dimension 1), it truncates both to the minimum length using slicing (`[:, :min_len]`). This directly addresses sequence length mismatches, potentially between residue and atom representations if `r_l` and `s` represent these differently within this module type. (Lines 367-372).
        *   **Logging:** None.

## 3. Partial Coordinate Usage Summary

*   **Explicit:** `partial_coords` are directly used in `validate_and_fix_shapes` to determine the target `num_atoms` dimension for explicit reshaping of `trunk_embeddings`. They are also assigned to `input_features['ref_pos']`.
*   **Implicit:** Partial coordinates might be involved in operations affected by the monkey-patches in `tensor_fixes` (e.g., additions handled by `fix_tensor_add`), but this is indirect and depends on the specific model architecture and data flow within Stage D.

## 4. Conclusion

The audit identified two primary mechanisms for handling residue-atom representation mismatches:
1.  An explicit validation and fix function (`validate_and_fix_shapes`) primarily targeting `trunk_embeddings` based on `partial_coords`.
2.  Implicit fixes via monkey-patching core PyTorch operations (`tensor_fixes`), attempting to resolve errors during runtime, including dimension expansion/reduction and truncation in specific contexts.

These ad-hoc fixes, particularly the tensor repetitions and warnings in `validate_and_fix_shapes`, confirm the need for a more systematic and robust bridging function to map residue-level representations (from Stage B/C) to the atom-level representations required by Stage D. The monkey-patching approach, while potentially functional, obscures the handling of these mismatches and makes debugging difficult.
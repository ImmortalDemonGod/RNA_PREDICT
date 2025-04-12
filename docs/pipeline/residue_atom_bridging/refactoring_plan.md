# Refactoring Plan: Bridging Function Integration (Residue-to-Atom)

**Version:** 1.1
**Date:** 2025-04-10
**Author:** Roo (Architect Mode)

**Objective:** Integrate the `residue_to_atoms` function (and its helper `derive_residue_atom_map`) into the M2 pipeline, replacing ad-hoc residue-to-atom expansion logic identified in `docs/pipeline/audit_residue_atom_mismatch.md`.

**References:**
*   Audit Report: `docs/pipeline/audit_residue_atom_mismatch.md`
*   Bridging Function Design: `docs/pipeline/bridging_function_design.md`
*   User Feedback (Provided 2025-04-10 ~17:11 PM CT)

## 1. Integration Point

*   **Chosen Location:** Within the main execution flow of Stage D, specifically in the script `rna_predict/pipeline/stageD/diffusion/run_stageD_unified.py`.
*   **Timing:** The bridging functions (`derive_residue_atom_map` and `residue_to_atoms`) should be called *after* receiving the data dictionary from the previous stage (containing residue-level embeddings like `trunk_embeddings`, sequence information, and potentially `partial_coords`) but *before* this data is passed into the core Stage D model components (e.g., the diffusion model itself or modules like `AtomAttentionEncoder`). Crucially, this should happen *before* the point where the now-obsolete `validate_and_fix_shapes` function was called.
*   **Rationale:**
    *   This point ensures that all necessary inputs for `derive_residue_atom_map` (sequence, potentially partial coords) and `residue_to_atoms` (residue embeddings) are available from the upstream stages.
    *   It centralizes the conversion logic, ensuring that all downstream components within Stage D receive consistently shaped atom-level embeddings.
    *   It directly replaces the location where the primary ad-hoc fix (`validate_and_fix_shapes`) was previously applied.
    *   Integrating earlier (e.g., `run_full_pipeline.py`) might be premature if Stage C outputs are strictly residue-level. Integrating later (within specific Stage D sub-modules) would lead to redundant calls and complex data passing.

## 2. Code Removal / Modification

The following code sections, identified in the audit, need modification or removal:

1.  **File:** `rna_predict/pipeline/stageD/diffusion/run_stageD_unified.py`
    *   **Action:** **Remove** the entire `validate_and_fix_shapes` function.
        *   *Reasoning:* Its logic (checking shapes and using `torch.Tensor.repeat()` or slicing based on `partial_coords`) is entirely superseded by the explicit `residue_to_atoms` bridging function.
    *   **Action:** **Remove** any calls to `validate_and_fix_shapes`.
    *   **Action:** **Remove** associated `warnings.warn` calls related to shape adjustments within the old function.

2.  **File:** `rna_predict/pipeline/stageD/tensor_fixes/__init__.py`
    *   **Action:** **Remove** the `fix_atom_attention_encoder` monkey-patch function and its application logic within `apply_tensor_fixes`.
        *   *Reasoning:* Redundant due to guaranteed correct input shapes from the bridging function.
    *   **Action:** **Modify** the `fix_tensor_add` monkey-patch.
        *   *Details:* **Remove** any logic within `fix_tensor_add` (and its helpers like `_expand_tensor_dimension`) that specifically attempts to resolve dimension mismatches by repeating/expanding tensors in a way that mimics residue-to-atom conversion (e.g., expanding dimension N_res to N_atom based on size difference).
        *   *Details:* **Retain** the general-purpose shape mismatch handling logic within `fix_tensor_add` if it addresses other confirmed issues in the pipeline (e.g., dimension count differences, attention bias adjustments, general broadcasting attempts unrelated to residue/atom expansion). This requires careful analysis during implementation to isolate and preserve only the necessary non-residue/atom fixes.
        *   *Reasoning:* Avoid redundant residue/atom expansion logic while preserving fixes for other potential shape issues, as confirmed by user feedback.
    *   **Action:** **Retain** the call to `apply_tensor_fixes()` from `run_stageD_unified.py` (assuming `fix_tensor_add` is partially retained). If implementation analysis reveals `fix_tensor_add` is *entirely* unnecessary after removing residue/atom logic, this call can be removed then.

3.  **File:** `rna_predict/utils/tensor_utils.py`
    *   **Action:** **Add** the new functions `derive_residue_atom_map` and `residue_to_atoms` as specified in `docs/pipeline/bridging_function_design.md`.
    *   **Action:** **Add** the `STANDARD_RNA_ATOMS` dictionary (requires verification of atom names/counts against project standards during implementation).
    *   **Action:** **Add** necessary imports (`torch`, `typing`).

## 3. Data Flow / Configuration Changes

1.  **Integration Logic in `run_stageD_unified.py`:**
    *   Retrieve necessary inputs from the data dictionary passed from the previous stage. **Implementation Note:** Verify exact keys during implementation (e.g., `'sequence'` vs `'rna_sequence'`, `'s_trunk'` vs `'s_embeddings'`, identify all relevant single-embedding keys needing conversion like `'s_inputs'`). Assumed keys:
        *   `'sequence'`
        *   `'partial_coords'` (Optional)
        *   `'trunk_embeddings'` (or similar dict/tensor containing residue embeddings)
        *   `'atom_metadata'` (Optional)
    *   **Step 1:** Call `derive_residue_atom_map` once:
        ```python
        # Example call signature (verify keys during implementation)
        residue_atom_map = derive_residue_atom_map(
            sequence=pipeline_data['sequence'],
            partial_coords=pipeline_data.get('partial_coords'),
            atom_metadata=pipeline_data.get('atom_metadata')
            # atom_counts_map will be derived internally from STANDARD_RNA_ATOMS if needed
        )
        ```
    *   **Step 2:** Iterate through the relevant residue-level embedding tensors (e.g., `pipeline_data['trunk_embeddings']['s_trunk']`, `pipeline_data['s_inputs']`, etc.). For each `s_emb_res`:
        ```python
        # Example call signature
        s_emb_atom = residue_to_atoms(
            s_emb=s_emb_res,
            residue_atom_map=residue_atom_map
        )
        # Update the pipeline data dictionary with the atom-level tensor
        # (e.g., replace pipeline_data['trunk_embeddings']['s_trunk'] with s_emb_atom)
        ```
    *   **Step 3:** Ensure the updated data dictionary (now containing atom-level embeddings) is passed to downstream Stage D components.

2.  **Data Structures:**
    *   The primary change is modifying the *values* within the existing data dictionary passed to Stage D (e.g., `pipeline_data` or `input_features`). Residue-level tensors (`[B, N_res, C_s]`) identified as needing conversion will be replaced by their atom-level counterparts (`[B, N_atom, C_s]`).
    *   No new persistent data structures are introduced, but the *state* of the data dictionary changes at the integration point.

3.  **Configuration:**
    *   No changes to configuration files are anticipated based on the audit. The previous ad-hoc logic did not appear to be config-driven.

4.  **Data Loaders:**
    *   No changes anticipated for data loaders. They are assumed to provide the necessary base information (sequence, potentially coordinates, residue-level features) upon which the pipeline stages operate.

## 4. Partial Coordinates Handling

*   `partial_coords` (shape `[B, N_atom, 3]`), if provided by Stage C, will be passed as an *optional* input to `derive_residue_atom_map`.
*   `derive_residue_atom_map` can use the `N_atom` dimension from `partial_coords.shape[-2]` to validate or infer the residue-to-atom mapping (Method 2 in the design doc).
*   The `residue_to_atoms` function *transforms embeddings*, it does *not* modify the `partial_coords` tensor itself.
*   The target `N_atom` dimension for the output embeddings from `residue_to_atoms` will be consistent with the `N_atom` dimension derived by `derive_residue_atom_map` (which may have used `partial_coords` shape as input). This ensures consistency.

## 5. Limitations

*   This plan and the underlying bridging function design **only address single embeddings** (e.g., `s_trunk`, `s_inputs` with shape `[N_res, C_s]`).
*   **Pair embeddings** (e.g., `pair` features with shape `[N_res, N_res, C_z]`) are **out of scope** for this refactoring task. Converting these to atom-level (`[N_atom, N_atom, C_z]`) requires a separate, more complex design and implementation effort if needed by Stage D.

## 6. Implementation Considerations

*   **Non-Canonicals:** Ensure `derive_residue_atom_map` raises appropriate errors if unknown residue types are encountered without explicit metadata.
*   **Device Placement:** Ensure tensors created or manipulated by the bridging functions are placed on the correct device (`.to(device)`), especially in distributed/multi-GPU scenarios.
*   **Testing:** Test thoroughly with real PDB data inputs to confirm shapes are correct and no mismatch warnings/errors occur in Stage D.

## 7. Diagram: Integration Point

```mermaid
graph TD
    subgraph Pipeline Flow
        StageB_C[Stage B/C Output] --> DataDictRes{Data Dictionary (Residue Lvl)};
        DataDictRes -- Sequence, Embeddings, Coords? --> StageD_Entry(run_stageD_unified.py);

        subgraph StageD_Entry [run_stageD_unified.py]
            direction LR
            Start --> Call_DeriveMap[Call derive_residue_atom_map];
            Call_DeriveMap -- residue_atom_map --> Call_Bridge[Loop: Call residue_to_atoms for each s_emb];
            Call_Bridge -- atom_level_embeddings --> UpdateDict[Update Data Dictionary];
            UpdateDict --> DataDictAtom{Data Dictionary (Atom Lvl)};
            DataDictAtom --> StageD_Model[Stage D Core Model];
        end

        StageD_Model --> StageD_Output[Stage D Output];

        subgraph Obsolete/Modified Code
           Removed_Validate[validate_and_fix_shapes (Removed)]
           Modified_MonkeyPatch[tensor_fixes (Modified: fix_tensor_add)]
           Removed_MonkeyPatch2[tensor_fixes (Removed: fix_atom_attention_encoder)]
        end
    end

    style Call_DeriveMap fill:#f9f,stroke:#333,stroke-width:2px
    style Call_Bridge fill:#ccf,stroke:#333,stroke-width:2px
    style Removed_Validate fill:#ddd,stroke:#666,stroke-width:1px,stroke-dasharray: 5 5
    style Modified_MonkeyPatch fill:#ffe0b3,stroke:#666,stroke-width:1px,stroke-dasharray: 2 2 /* Orange dashed for modified */
    style Removed_MonkeyPatch2 fill:#ddd,stroke:#666,stroke-width:1px,stroke-dasharray: 5 5
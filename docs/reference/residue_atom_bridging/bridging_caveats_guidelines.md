# Extended Guidelines for Residue-to-Atom Bridging Caveats

This document provides extended guidelines and discusses potential caveats related to the residue-to-atom bridging mechanism implemented in the pipeline. It addresses handling non-canonical residues, aligning partial coordinates, managing pair embeddings, and ensuring legacy code cleanup.

## 1. Non-Canonical Residue Handling

### Explanation & Pitfalls

*   **Standard vs. Non-Standard:** The current residue-to-atom mapping assumes canonical residues (A, U, G, C) with a fixed number of heavy atoms. Non-canonical residues (e.g., pseudouridine Ψ, 5-methylcytosine m⁵C) present challenges as they may have:
    *   A different atom count than standard residues.
    *   Distinct structures or partial occupancy in experimental data, leading to missing coordinates.
*   **Potential Mismatches:** Encountering an unrecognized residue can cause the bridging function to:
    *   Fail with a `KeyError` if the residue isn't found in the canonical dictionary.
    *   Use an inaccurate default atom count, leading to indexing errors or incorrect zero-padding.
*   **Data Integration:** Partial coordinates from Stage C or external sources might contain the correct atom set for modified residues. Failing to integrate this data can lead to shape mismatches or inaccurate embeddings.

### Recommendations

1.  **Dynamic Mapping:**
    *   Implement logic (e.g., in `derive_residue_atom_map`) to check if a residue is recognized.
    *   If recognized: Use the standard canonical dictionary.
    *   If unrecognized: Consult a "modification dictionary" or parse partial coordinates/metadata specifying the actual atoms.
2.  **Metadata-Based Approach:**
    *   Allow users to provide metadata (e.g., CSV, JSON) listing non-canonical residue names, atom counts, and naming conventions.
    *   Use partial coordinates as a fallback if a residue is not found in any dictionary.
3.  **Error Handling:**
    *   Provide clear errors when a residue is missing from all references (e.g., "Residue X is unrecognized. Provide partial coordinates or a custom atom map in config.").
4.  **Testing:**
    *   Include test cases with non-canonical residues (or simplified mocks) to verify the bridging logic handles them correctly.

## 2. Partial Coordinates Alignment

### Explanation & Pitfalls

*   **Alignment Challenge:** Stage C often produces partial or full 3D coordinates (`[N_atom, 3]`) that need to align with the bridging function's expected atom count and order for the single-sequence embeddings (`[N_atom, c_s]`). Discrepancies can arise if:
    *   Stage C coordinates skip atoms.
    *   Ring-closure logic adds or removes atoms, changing the final `N_atom`.
*   **Missing Coordinates:** Unresolved or omitted atoms in experimental data can reduce the practical `N_atom`, causing mismatches with the bridging function's theoretical count.
*   **Ring-Closure & Edge Cases:** Dynamic modifications like ring closure or sugar pucker corrections can alter atom counts or ordering. The bridging mechanism must remain consistent with the final atom representation.

### Recommendations

1.  **Consistent Atom Ordering:**
    *   Define and enforce a canonical atom order within each residue (e.g., 5' to 3' backbone order, standard base ring order).
    *   Ensure Stage C outputs preserve or record this order for consistent indexing during bridging.
2.  **Fallback for Missing Atoms:**
    *   Establish a clear policy for handling atoms missing in partial coordinates: Are they excluded entirely, or represented with zero-embeddings (`[0, ..., 0]`)?
3.  **Validation Step:**
    *   Implement a consistency check post-Stage C processing (e.g., after ring closure) to verify that the final `N_atom` matches the bridging function's expectation. Fail fast with informative errors if they mismatch.
4.  **Documentation:**
    *   Clearly document the requirement for Stage C coordinate outputs to remain synchronized with the bridging approach, especially regarding potential changes in atom number or order.

## 3. Pair Embeddings

### Explanation & Pitfalls

*   **Higher Dimensionality:** Bridging pair embeddings (`[N, N, c_z]`) to the atom level (`[N_atom, N_atom, c_z]`) is significantly more complex than bridging single embeddings. Each residue pair corresponds to numerous potential atom-atom interactions.
*   **Memory Usage:** A naive expansion from `[N, N, c_z]` to `[N_atom, N_atom, c_z]` can lead to substantial memory increases, especially if `N_atom` is much larger than `N`. (e.g., N=8 residues to N_atom=24 atoms implies a 9x increase in pair embedding size).
*   **Limited Stage D Requirements:** Currently, Stage D might not require atom-level pair embeddings, making the bridging effort potentially unnecessary overhead. The existing plan defers this, suggesting Stage D works with single-atom embeddings or its own adjacency logic.
*   **Future Feature:** If atom-level pair embeddings become necessary for Stage D later, a dedicated bridging strategy will be required, potentially involving mapping residue pairs to sub-blocks of atom pairs.

### Recommendations

1.  **Confirm Stage D Requirements:**
    *   Verify whether Stage D absolutely needs atom-level pair embeddings or if single embeddings plus adjacency information suffice.
2.  **Explicitly Mark as "Future Task":**
    *   Clearly state in documentation that bridging pair embeddings (`[N, N, c_z]` -> `[N_atom, N_atom, c_z]`) is a recognized potential future requirement but is currently out of scope to manage complexity.
3.  **Performance Considerations:**
    *   If implemented later, prioritize memory efficiency. Consider block-sparse representations or on-demand expansion instead of dense `[N_atom, N_atom, c_z]` tensors.
4.  **Potential Data Structures:**
    *   Explore data structures that reference residue-residue pairs and enumerate the corresponding sub-atom pairs, which might be more memory-efficient than a dense tensor.

## 4. Legacy Code Cleanup

### Explanation & Pitfalls

*   **Scattered Fixes:** Previous patches addressing shape mismatches (e.g., in `validate_and_fix_shapes`, `tensor_fixes/`) might be scattered throughout the codebase, potentially undocumented or partially implemented.
*   **Risk of Redundant Behavior:** Leftover expansion logic could conflict with the new bridging mechanism, causing double-application of expansions or incorrect shapes (e.g., logs showing "Adjusting from 24 to 72").
*   **Regression Risk:** Unremoved legacy code could lead to regressions where the pipeline reverts to old expansion behaviors due to minor configuration changes or updates.

### Recommendations

1.  **Global Search:**
    *   Perform a repository-wide search for keywords related to shape expansion and mismatch fixing (e.g., `expand`, `clamp`, `Adjusting sequence length`, `validate_and_fix_shapes`, `res->atom mismatch`).
2.  **Modular Removal:**
    *   Systematically remove or comment out code specifically intended to fix residue-to-atom mismatches. Test the pipeline after each removal to ensure the bridging function correctly handles the scenario.
3.  **Identify Unrelated Fixes:**
    *   Preserve shape or memory patches that address genuinely unrelated issues (e.g., 4D -> 5D tensor reshaping for attention mechanisms) and do not interfere with residue-to-atom bridging.
4.  **Continuous Integration Checks:**
    *   Enhance CI tests to fail if legacy expansion logic is reintroduced or if logs indicating shape mismatches reappear. This helps prevent accidental reintroduction of problematic code.

## Additional Considerations

### Implementation Examples

*   Provide concise code snippets or references demonstrating the core bridging function logic.
*   Include pseudo-code examples for handling partial coordinate mismatches or constructing a "modification dictionary" for non-canonical residues. *(Example to be added based on final implementation)*

### Testing Protocol

*   Outline specific test strategies to verify:
    *   Correct handling of partial coordinates by the bridging logic.
    *   Successful removal of legacy expansion code without introducing regressions.
    *   Graceful handling of non-canonical residues (if supported).
*   *(Specific test cases/files to be referenced here)*

### FAQ / Notes

*   **Q: What if a residue has partial occupancy in PDB data?**
    *   A: The handling depends on the chosen policy for missing atoms (Section 2, Recommendation 2). If partial coordinates are used as input, atoms with zero occupancy might be treated as missing. The bridging function should either exclude them or represent them with zero-embeddings based on the defined policy.
*   *(Add other relevant FAQs as they arise)*
*   Link to relevant design documents (e.g., `docs/pipeline/residue_atom_bridging/design_spec.md`) and automated test results/logs for further reference.



# Design Specification: Residue-to-Atom Bridging Function

**Version:** 1.0
**Date:** 2025-04-10
**Author:** Roo (Architect Mode)

## 1. Introduction & Goal

This document outlines the design for a systematic bridging mechanism to convert residue-level tensor representations (e.g., single embeddings `s_emb` with shape `[N_residue, C_s]`) generated in earlier pipeline stages (like Stage B) to the atom-level representations (shape `[N_atom, C_s]`) often required by later stages (like Stage D).

This replaces previous ad-hoc methods (like simple tensor repeats based on `partial_coords` shape or runtime monkey-patching) identified in `docs/pipeline/audit_residue_atom_mismatch.md` with a robust and explicit approach.

The core components are:
1.  A helper function `derive_residue_atom_map` to determine the mapping between residue indices and their corresponding global atom indices.
2.  The main bridging function `residue_to_atoms` that uses this map to expand the residue embeddings.

## 2. Target File & Location

*   **File:** `rna_predict/utils/tensor_utils.py`
*   **Rationale:** Centralized utility location, accessible across pipeline stages, avoids duplication, and promotes reuse for potential future tensor manipulation utilities.

## 3. Function Signatures & Data Types

```python
# In rna_predict/utils/tensor_utils.py
import torch
from typing import List, Union, Optional, Dict

# Define standard atom names (Example - requires verification/update with actual RNA atom names used in the project)
STANDARD_RNA_ATOMS = {
    'A': ['P', 'O5\'', 'C5\'', 'C4\'', 'O4\'', 'C3\'', 'O3\'', 'C2\'', 'C1\'', 'N9', 'C8', 'N7', 'C5', 'C6', 'N6', 'N1', 'C2', 'N3', 'C4'], # 19 atoms
    'U': ['P', 'O5\'', 'C5\'', 'C4\'', 'O4\'', 'C3\'', 'O3\'', 'C2\'', 'O2\'', 'C1\'', 'N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C6'], # 18 atoms
    'G': ['P', 'O5\'', 'C5\'', 'C4\'', 'O4\'', 'C3\'', 'O3\'', 'C2\'', 'C1\'', 'N9', 'C8', 'N7', 'C5', 'C6', 'O6', 'N1', 'C2', 'N2', 'N3', 'C4'], # 20 atoms
    'C': ['P', 'O5\'', 'C5\'', 'C4\'', 'O4\'', 'C3\'', 'O3\'', 'C2\'', 'C1\'', 'N1', 'C2', 'O2', 'N3', 'C4', 'N4', 'C5', 'C6'], # 17 atoms
    # Add other standard residues if needed
}

# Type alias for clarity
ResidueAtomMap = List[List[int]] # List where index is residue_idx, value is list of global atom indices for that residue

def derive_residue_atom_map(
    sequence: Union[str, List[str]],
    partial_coords: Optional[torch.Tensor] = None, # Shape [B, N_atom, 3] or [N_atom, 3]
    atom_metadata: Optional[Dict] = None, # e.g., {'atom_names': ['P', ...], 'residue_indices': [0, 0, ... 1, ...]}
    atom_counts_map: Optional[Dict[str, int]] = None # Fallback: {'A': 19, 'U': 18, ...} derived from STANDARD_RNA_ATOMS
) -> ResidueAtomMap:
    """
    Derives the mapping from residue index to a list of corresponding global atom indices.

    This helper function determines how atoms are grouped by residue, which is essential
    for the `residue_to_atoms` bridging function.

    Args:
        sequence (Union[str, List[str]]): The RNA sequence (e.g., "AUCG" or ['A', 'U', 'C', 'G']). Length must match N_residue.
        partial_coords (Optional[torch.Tensor]): Atom coordinates, potentially used to infer N_atom. Shape [B, N_atom, 3] or [N_atom, 3].
        atom_metadata (Optional[Dict]): Explicit metadata linking atoms to residues. Expected keys might include 'residue_indices' (list/tensor of length N_atom mapping each atom to its residue index).
        atom_counts_map (Optional[Dict[str, int]]): A fallback map providing the number of atoms for each standard residue type (e.g., derived from STANDARD_RNA_ATOMS). Used when `partial_coords` and `atom_metadata` are insufficient.

    Priority for Derivation:
    1. Use `atom_metadata` if provided (most explicit and reliable).
    2. Use `partial_coords` shape and assume contiguous block ordering based on `sequence` and standard atom counts if `atom_metadata` is missing. Requires validation that total inferred atoms match `partial_coords.shape[-2]`.
    3. Use `atom_counts_map` (e.g., from STANDARD_RNA_ATOMS lengths) and `sequence` if coordinates and metadata are both missing. Assumes contiguous blocks.

    Returns:
        ResidueAtomMap: Map where index `i` contains the list of global atom indices for residue `i`.

    Raises:
        ValueError: If insufficient information is provided, inputs are inconsistent (e.g., sequence length mismatch, atom count mismatch), or assumptions (like contiguous order) are violated based on available data.
        KeyError: If a residue in `sequence` is not found in `atom_counts_map` when operating in fallback mode (Method 3).
    """
    # Implementation details deferred to Code mode.
    # Key logic: Input validation, selecting derivation method based on priority,
    # generating the list of lists, handling errors.
    pass

def residue_to_atoms(
    s_emb: torch.Tensor,
    residue_atom_map: ResidueAtomMap,
) -> torch.Tensor:
    """
    Expands residue-level embeddings to atom-level embeddings using a precomputed map.

    Assigns the embedding of residue `i` to all atoms belonging to residue `i` as defined
    by the `residue_atom_map`.

    Args:
        s_emb (torch.Tensor): Residue embeddings. Shape [N_residue, C_s] or [B, N_residue, C_s].
        residue_atom_map (ResidueAtomMap): A list where index `i` contains the list of
                                           global atom indices corresponding to residue `i`.
                                           The length of this list must equal N_residue.
                                           The union of all inner lists must cover all atoms
                                           from 0 to N_atom-1 exactly once and without overlaps.

    Returns:
        torch.Tensor: Atom-level embeddings. Shape [N_atom, C_s] or [B, N_atom, C_s].

    Raises:
        ValueError: If input shapes are inconsistent (e.g., `len(residue_atom_map)` != `s_emb.shape[-2]`)
                    or `residue_atom_map` is invalid (e.g., gaps or overlaps in atom indices).
    """
    # Implementation details deferred to Code mode.
    # Key logic: Handle batched/unbatched input, calculate N_atom from map,
    # initialize output tensor, use map to efficiently assign s_emb[i] to atom_embs[atom_indices],
    # validate map integrity. Consider using torch.gather with an atom_to_residue_idx tensor for efficiency.
    pass
```

## 4. Residue-to-Atom Mapping (`residue_atom_map`)

*   **Role:** This `List[List[int]]` structure is the critical link provided *to* the `residue_to_atoms` function. It explicitly defines the atom indices associated with each residue index.
*   **Generation:** The `derive_residue_atom_map` helper function is responsible for creating this map based on available inputs (sequence, coords, metadata), prioritizing explicit metadata over inferred mappings.
*   **Standard Atoms:** The `STANDARD_RNA_ATOMS` dictionary provides the default atom names and counts for A, U, G, C, used by `derive_residue_atom_map` primarily in its fallback mode or for validation. **Note:** The specific atom names in `STANDARD_RNA_ATOMS` must be verified against the project's conventions.
*   **Non-Canonical Handling:** Handled within `derive_residue_atom_map`. By default, if operating in fallback mode (using `atom_counts_map`), encountering an unknown residue type in the `sequence` will raise a `KeyError` or `ValueError`, forcing explicit handling rather than silent failure. If derived from explicit `atom_metadata`, non-canonicals are handled if the metadata correctly describes them.

## 5. Core Logic (`residue_to_atoms`)

1.  **Inputs:** Takes `s_emb` (`[N_res, C_s]` or `[B, N_res, C_s]`) and the validated `residue_atom_map`.
2.  **Calculate `N_atom`:** Determine the total number of atoms by summing the lengths of the inner lists in `residue_atom_map` (or checking the max index + 1).
3.  **Initialize Output:** Create `atom_embs` tensor of shape `[N_atom, C_s]` or `[B, N_atom, C_s]` with the same dtype and device as `s_emb`.
4.  **Expand/Assign:** Iterate through `residue_atom_map`. For each residue `i` and its `atom_indices`:
    *   Select the source embedding: `s_emb[i]` or `s_emb[:, i]`.
    *   Assign this embedding vector to the target atoms: `atom_embs[atom_indices]` or `atom_embs[:, atom_indices]`. (Implementation should aim for efficiency, e.g., using advanced indexing or potentially constructing an `atom_to_residue_idx` tensor and using `torch.gather`).
5.  **Return:** Output the populated `atom_embs` tensor.

## 6. Process Flow Diagram

```mermaid
graph TD
    subgraph Input Preparation Phase
        I1[Sequence Info (str/list)] --> D(derive_residue_atom_map);
        I2[Partial Coords [N_atom, 3] (Optional)] --> D;
        I3[Atom Metadata (Optional, e.g., residue_indices)] --> D;
        I4[Standard Atom Counts Map (Fallback)] --> D;
    end

    subgraph Bridging Phase
        E1[s_emb [N_res, C_s]] --> R(residue_to_atoms);
        D -- residue_atom_map [List[List[int]]] --> R;
        R -- atom_embs [N_atom, C_s] --> O[Output Atom Embeddings];
    end

    style D fill:#f9f,stroke:#333,stroke-width:2px
    style R fill:#ccf,stroke:#333,stroke-width:2px
```

## 7. Edge Case Handling

*   **Empty Inputs:** Both functions should handle empty sequences/embeddings gracefully, returning empty maps/tensors of the correct dimensionality.
*   **Inconsistencies:** `derive_residue_atom_map` must rigorously validate consistency between sequence length, `N_residue` inferred from inputs, and `N_atom` inferred from inputs. Raise `ValueError` on mismatch.
*   **Invalid Map:** `residue_to_atoms` must validate the received `residue_atom_map` ensures all atoms 0 to `N_atom-1` are covered exactly once. Raise `ValueError` if gaps or overlaps exist. `len(residue_atom_map)` must match `s_emb.shape[-2]`.
*   **Non-Canonicals:** Explicitly handled by `derive_residue_atom_map` raising an error in fallback mode if an unknown residue type is encountered.

## 8. Logging

*   **`derive_residue_atom_map`:**
    *   INFO: Log the method used for map generation (metadata, coords, fallback).
    *   WARN: Log any assumptions made (e.g., assuming contiguous atom order when using coords without explicit residue indices).
    *   ERROR: Log errors for inconsistent inputs or unknown residues in fallback mode.
*   **`residue_to_atoms`:**
    *   DEBUG: Log input and output tensor shapes.
    *   WARN: Log warnings for empty inputs being processed.
    *   ERROR: Log errors for invalid `residue_atom_map`.

## 9. Pair Embeddings (`c_z`)

*   **Out of Scope:** This design specification *only* covers the bridging of single embeddings (`s_emb`, shape `[N_res, C_s]`).
*   **Future Consideration:** Bridging pair embeddings (`[N_res, N_res, C_z]` -> `[N_atom, N_atom, C_z]`) is significantly more complex. It requires mapping every pair of residues to the corresponding block of atom-atom pairs and likely involves substantial memory overhead (`N_atom^2` vs `N_res^2`). This is deferred for future work if required by Stage D or other downstream modules.

## 10. Implementation Notes

*   Focus on efficiency within `residue_to_atoms`, potentially avoiding Python loops over residues/atoms if possible by using vectorized PyTorch operations like advanced indexing or `torch.gather`.
*   Ensure comprehensive unit tests cover various input scenarios, edge cases, and map generation methods.
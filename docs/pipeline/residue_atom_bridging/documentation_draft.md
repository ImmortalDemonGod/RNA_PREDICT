# Draft Documentation for Residue-Atom Bridging

## Draft Code Comments/Docstrings

```python
import torch
from typing import Dict, List, Optional, Tuple, Union

# Placeholder for actual atom definitions
RNA_ATOM_MAP = {
    'A': ['N9', 'C8', 'N7', 'C5', 'C6', 'N6', 'N1', 'C2', 'N3', 'C4', "C1'", "C2'", "O2'", "C3'", "O3'", "C4'", "O4'", "C5'", "O5'", 'P', 'OP1', 'OP2'],
    'U': ['N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C6', "C1'", "C2'", "O2'", "C3'", "O3'", "C4'", "O4'", "C5'", "O5'", 'P', 'OP1', 'OP2'],
    'G': ['N9', 'C8', 'N7', 'C5', 'C6', 'O6', 'N1', 'C2', 'N2', 'N3', 'C4', "C1'", "C2'", "O2'", "C3'", "O3'", "C4'", "O4'", "C5'", "O5'", 'P', 'OP1', 'OP2'],
    'C': ['N1', 'C2', 'O2', 'N3', 'C4', 'N4', 'C5', 'C6', "C1'", "C2'", "O2'", "C3'", "O3'", "C4'", "O4'", "C5'", "O5'", 'P', 'OP1', 'OP2'],
    # Add other canonical/non-canonical mappings as needed
}

DEFAULT_ATOMS_PER_RESIDUE = max(len(v) for v in RNA_ATOM_MAP.values()) # Example default

def derive_residue_atom_map(
    sequence: List[str],
    partial_coords: Optional[torch.Tensor] = None,
    atom_definitions: Dict[str, List[str]] = RNA_ATOM_MAP
) -> Tuple[Dict[int, List[int]], int, List[str]]:
    """
    Derives a mapping from residue indices to corresponding atom indices.

    This function constructs a map indicating which atoms belong to each residue
    in the sequence. It primarily uses standard definitions but can potentially
    be extended to use partial coordinates if available and necessary for
    handling modifications or non-standard residues.

    Args:
        sequence (List[str]): List of residue identifiers (e.g., ['A', 'U', 'G']).
        partial_coords (Optional[torch.Tensor]): Tensor of partial coordinates
            (shape [N_atom_partial, 3]), potentially used for map derivation
            in complex cases (currently placeholder). Defaults to None.
        atom_definitions (Dict[str, List[str]]): A dictionary mapping residue
            types to their constituent atom names. Defaults to RNA_ATOM_MAP.

    Returns:
        Tuple[Dict[int, List[int]], int, List[str]]:
        - residue_to_atom_indices (Dict[int, List[int]]): Dictionary mapping
          residue index (0 to N-1) to a list of global atom indices
          (0 to N_atom-1) belonging to that residue.
        - total_atoms (int): The total number of atoms inferred for the sequence.
        - atom_names (List[str]): Flat list of atom names corresponding to the
          global atom indices.
    """
    residue_to_atom_indices = {}
    atom_names = []
    current_atom_index = 0
    total_atoms = 0

    # Basic implementation using standard definitions
    # TODO: Enhance logic if partial_coords are needed for complex cases
    if partial_coords is not None:
        # Placeholder: Logic to use partial_coords if needed, e.g.,
        # to infer atom counts or handle non-standard residues.
        # This could involve checking coordinate presence/absence.
        print("Warning: Partial coordinate usage in map derivation not fully implemented.")

    for i, res_type in enumerate(sequence):
        if res_type in atom_definitions:
            atoms = atom_definitions[res_type]
            num_atoms_in_residue = len(atoms)
            indices = list(range(current_atom_index, current_atom_index + num_atoms_in_residue))
            residue_to_atom_indices[i] = indices
            atom_names.extend(atoms)
            current_atom_index += num_atoms_in_residue
        else:
            # Handle unknown residue types (e.g., use default, skip, error)
            # Option: Use a default number of atoms or raise error
            print(f"Warning: Unknown residue type '{res_type}' at index {i}. Using default atom count (if defined) or skipping.")
            # Example fallback (needs refinement based on requirements)
            # num_atoms_in_residue = DEFAULT_ATOMS_PER_RESIDUE
            # indices = list(range(current_atom_index, current_atom_index + num_atoms_in_residue))
            # residue_to_atom_indices[i] = indices
            # atom_names.extend([f"UNK_{j}" for j in range(num_atoms_in_residue)])
            # current_atom_index += num_atoms_in_residue
            residue_to_atom_indices[i] = [] # Or handle differently

    total_atoms = current_atom_index
    print(f"Derived residue-atom map for {len(sequence)} residues -> {total_atoms} atoms.")
    return residue_to_atom_indices, total_atoms, atom_names

def residue_to_atoms(
    s_emb: torch.Tensor,
    sequence: List[str],
    residue_atom_map: Optional[Dict[int, List[int]]] = None,
    total_atoms: Optional[int] = None,
    partial_coords: Optional[torch.Tensor] = None,
    atom_definitions: Dict[str, List[str]] = RNA_ATOM_MAP
) -> torch.Tensor:
    """
    Expands residue-level embeddings to atom-level embeddings.

    Takes a tensor of residue embeddings (e.g., from Stage B) and expands it
    to atom-level embeddings suitable for Stage D, using a residue-to-atom map.

    Args:
        s_emb (torch.Tensor): Residue-level embeddings tensor (shape [N, c_s]).
        sequence (List[str]): List of residue identifiers corresponding to s_emb.
        residue_atom_map (Optional[Dict[int, List[int]]]): Precomputed map from
            residue index to list of global atom indices. If None, it will be
            derived using `derive_residue_atom_map`. Defaults to None.
        total_atoms (Optional[int]): Precomputed total number of atoms. If None,
            it will be derived. Defaults to None.
        partial_coords (Optional[torch.Tensor]): Tensor of partial coordinates
            (shape [N_atom_partial, 3]), passed to `derive_residue_atom_map`
            if the map needs to be derived. Defaults to None.
        atom_definitions (Dict[str, List[str]]): Definitions used if deriving
            the map. Defaults to RNA_ATOM_MAP.

    Returns:
        torch.Tensor: Atom-level embeddings tensor (shape [N_atom, c_s]).

    Raises:
        ValueError: If the residue dimension of s_emb doesn't match the sequence length.
        ValueError: If the derived/provided map is inconsistent.
    """
    N, c_s = s_emb.shape
    if N != len(sequence):
        raise ValueError(f"Residue dimension mismatch: s_emb has {N} residues, sequence has {len(sequence)}.")

    if residue_atom_map is None or total_atoms is None:
        print("Deriving residue-atom map internally...")
        residue_atom_map, total_atoms, _ = derive_residue_atom_map(
            sequence, partial_coords, atom_definitions
        )
        if total_atoms == 0 and N > 0:
             raise ValueError("Failed to derive a valid residue-atom map; total atoms is zero.")
        elif total_atoms == 0 and N == 0:
             print("Input sequence is empty, returning empty atom tensor.")
             return torch.empty((0, c_s), dtype=s_emb.dtype, device=s_emb.device)


    # Create the target atom embedding tensor
    atom_embs = torch.zeros((total_atoms, c_s), dtype=s_emb.dtype, device=s_emb.device)

    # Expand embeddings using the map
    all_mapped_indices = []
    for res_idx, atom_indices in residue_atom_map.items():
        if res_idx < N: # Ensure residue index is valid for s_emb
            if atom_indices: # Check if list is not empty
                # Assign the residue embedding to all its corresponding atoms
                atom_embs[atom_indices, :] = s_emb[res_idx, :]
                all_mapped_indices.extend(atom_indices)
        else:
            print(f"Warning: Residue index {res_idx} in map is out of bounds for s_emb (size {N}). Skipping.")

    # Verification (optional but recommended)
    if len(all_mapped_indices) != total_atoms:
         unmapped_indices = set(range(total_atoms)) - set(all_mapped_indices)
         print(f"Warning: Mismatch in mapped atoms. Expected {total_atoms}, mapped {len(all_mapped_indices)}. Unmapped indices: {unmapped_indices}")
         # Decide handling: error or proceed with potentially zero-filled embeddings for unmapped atoms

    print(f"Expanded residue embeddings [N={N}, c_s={c_s}] to atom embeddings [N_atom={total_atoms}, c_s={c_s}]")
    return atom_embs

```

## Draft README.md Section

### Pipeline: Residue vs. Atom Representation Bridging

**Problem:**
The M2 pipeline involves stages operating at different levels of molecular representation. Specifically, Stage B (Pairformer + TorsionBERT) primarily outputs residue-level information (e.g., sequence embeddings `s_emb` with shape `[N, c_s]`, where `N` is the number of residues), while Stage D (Diffusion) requires atom-level inputs (e.g., atom embeddings `atom_embs` with shape `[N_atom, c_s]` or coordinates `[N_atom, 3]`, where `N_atom` is the total number of atoms). Previously, this mismatch was handled by ad-hoc fixes within Stage D or utility functions, leading to potential inconsistencies and reduced maintainability.

**Solution: Bridging Function**
To address this systematically, a dedicated bridging mechanism has been introduced between the residue-level outputs and atom-level inputs. This is implemented primarily via the `residue_to_atoms` utility function (likely located in `rna_predict/utils/tensor_utils.py` or similar). This function takes residue-level tensors (like `s_emb`) and expands them to the corresponding atom-level representation based on a defined mapping.

**Residue-to-Atom Map Derivation:**
The core of the bridging process relies on a `residue_atom_map`. This map defines which atoms constitute each residue in the sequence. The map is typically derived by the helper function `derive_residue_atom_map` using standard definitions for canonical RNA nucleotides (A, U, G, C) stored in a structure like `RNA_ATOM_MAP`. The derivation logic can potentially be extended to handle non-canonical residues or utilize partial coordinate information (e.g., from Stage C) if available and necessary, although the primary method relies on standard definitions for robustness.

**Data Flow:**
The typical data flow involving the bridge is:
1. Stage B produces residue-level outputs (e.g., `s_emb: [N, c_s]`).
2. Before Stage D, the `residue_to_atoms` function is called.
3. `residue_to_atoms` either uses a provided map or calls `derive_residue_atom_map` using the sequence information (and potentially partial coordinates).
4. The function expands `s_emb` to `atom_embs: [N_atom, c_s]`.
5. Stage D receives the correctly shaped `atom_embs` tensor.

**Interaction with Partial Coordinates (Stage C):**
Stage C might provide partial coordinates (e.g., backbone atoms). While the primary map derivation uses standard definitions, the `derive_residue_atom_map` function is designed to potentially accept these partial coordinates. This allows for future enhancements where partial coordinates could help resolve ambiguities, identify present atoms, or handle non-standard residues more accurately if needed. Currently, the standard definitions are the default mechanism.

**Conceptual Usage Example:**
The bridging function is integrated into the pipeline script (e.g., `run_full_pipeline.py` or the entry point of Stage D) after Stage B/C outputs are generated and before Stage D processing begins:

```python
# --- In run_full_pipeline.py or similar ---

# Assume stage_b_output contains 's_emb' [N, c_s] and 'sequence' list
# Assume stage_c_output might contain 'partial_coords' [N_atom_partial, 3] (optional)

from rna_predict.utils.tensor_utils import residue_to_atoms # Adjust import path

residue_embeddings = stage_b_output['s_emb']
sequence = stage_b_output['sequence']
partial_coords = stage_c_output.get('partial_coords', None) # Get partial coords if available

# Bridge residue embeddings to atom embeddings
atom_embeddings = residue_to_atoms(
    s_emb=residue_embeddings,
    sequence=sequence,
    partial_coords=partial_coords # Pass partial coords if they should influence map derivation
)

# Prepare inputs for Stage D using atom_embeddings
stage_d_input = {
    # ... other inputs ...
    'atom_embs': atom_embeddings,
    'sequence': sequence, # Pass sequence along if needed by Stage D
    # ... potentially pass atom_mask derived from map if needed ...
}

# Run Stage D
stage_d_output = run_stage_d(stage_d_input)

# --- End Example ---
```

This structured approach ensures that Stage D consistently receives atom-level data derived systematically from residue-level precursors, improving pipeline robustness and clarity.
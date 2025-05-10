"""
Residue-to-atom mapping utilities.

This module provides functions for creating and managing mappings between
residue indices and atom indices, which is essential for bridging between
residue-level and atom-level representations in RNA structures.
"""

import torch
from typing import Dict, List, Optional, Tuple, Union

from .types import ResidueAtomMap, STANDARD_RNA_ATOMS, logger


def _prepare_sequence_and_counts(
    sequence: Union[str, List[str], None],
    atom_counts_map: Optional[Dict[str, int]] = None,
) -> Tuple[List[str], Dict[str, int], int]:
    """
    Prepare sequence and atom counts map for residue-atom mapping.

    Args:
        sequence: RNA sequence as string or list of residue types
        atom_counts_map: Optional mapping from residue type to atom count

    Returns:
        Tuple of (sequence_list, atom_counts_map, n_residues)
    """
    # Handle missing sequence: defer to metadata branch later
    if sequence is None:
        sequence_list: List[str] = []
        n_residues = 0
    else:
        sequence_list = list(sequence) if isinstance(sequence, str) else sequence
        n_residues = len(sequence_list)

    # Prepare atom counts map: use default if not provided
    counts_map = atom_counts_map or {res_type: len(atoms) for res_type, atoms in STANDARD_RNA_ATOMS.items()}

    return sequence_list, counts_map, n_residues


class MetadataMapConfig:
    """Configuration for deriving residue-atom map from metadata."""
    def __init__(
        self,
        residue_indices: Union[List[Union[int, str]], torch.Tensor],
        sequence_list: List[str],
        n_residues: int,
    ) -> None:
        self.residue_indices = residue_indices
        self.sequence_list = sequence_list
        self.n_residues = n_residues


def _convert_indices_to_list(
    residue_indices: Union[List[Union[int, str]], torch.Tensor],
) -> List[Union[int, str]]:
    """
    Convert residue indices to a list.

    Args:
        residue_indices: List or tensor of residue indices for each atom, can be integers or strings

    Returns:
        List of residue indices as integers or strings
    """
    if isinstance(residue_indices, torch.Tensor):
        return [int(x) for x in residue_indices.cpu().tolist()]
    return residue_indices


def _convert_to_int_list(indices: List[Union[int, str]]) -> List[int]:
    """
    Convert a list of indices to integers.

    Args:
        indices: List of indices that can be either integers or strings

    Returns:
        List of integer indices

    Raises:
        ValueError: If string indices cannot be converted to integers
    """
    if not indices:
        return []
    
    result = []
    for idx in indices:
        if isinstance(idx, str):
            try:
                result.append(int(idx))
            except ValueError as e:
                raise ValueError(f"Failed to convert string residue index '{idx}' to integer: {e}")
        elif isinstance(idx, int):
            result.append(idx)
        else:
            raise ValueError(f"Residue index must be string or integer, got {type(idx)}")
    
    return result


def _populate_residue_atom_map(
    indices: List[int],
    n_residues: int,
) -> ResidueAtomMap:
    """
    Populate residue-atom map from residue indices.

    Args:
        indices: List of residue indices for each atom
        n_residues: Number of residues in the sequence

    Returns:
        Mapping from residue indices to atom indices

    Raises:
        ValueError: If residue indices are invalid
    """
    # Initialize the map
    residue_atom_map: ResidueAtomMap = [[] for _ in range(n_residues)]

    # Populate the map
    for atom_idx, res_idx in enumerate(indices):
        if 0 <= res_idx < n_residues:
            residue_atom_map[res_idx].append(atom_idx)
        else:
            raise ValueError(f"Residue index {res_idx} in atom_metadata is out of bounds for sequence of length {n_residues}.")

    return residue_atom_map


def _validate_residue_atom_map(
    residue_atom_map: ResidueAtomMap,
    sequence_list: List[str],
) -> None:
    """
    Validate that all residues have at least one atom.

    Args:
        residue_atom_map: Mapping from residue indices to atom indices
        sequence_list: List of residue types
    """
    for res_idx, atom_indices in enumerate(residue_atom_map):
        if not atom_indices:
            logger.warning(f"Residue {res_idx} ({sequence_list[res_idx]}) has no atoms assigned in the metadata.")


def _derive_map_from_metadata(
    residue_indices: Union[List[Union[int, str]], torch.Tensor],
    sequence_list: List[str],
    n_residues: int,
) -> ResidueAtomMap:
    """
    Derive residue-atom map from residue indices in metadata.

    Args:
        residue_indices: List or tensor of residue indices for each atom, can be integers or strings
        sequence_list: List of residue types
        n_residues: Number of residues in the sequence

    Returns:
        Mapping from residue indices to atom indices

    Raises:
        ValueError: If residue indices are invalid or string indices cannot be converted to integers
    """
    # Create configuration
    config = MetadataMapConfig(residue_indices, sequence_list, n_residues)

    # Convert to list if it's a tensor
    indices = _convert_indices_to_list(config.residue_indices)

    # Convert to integer list
    int_indices = _convert_to_int_list(indices)

    # Populate the map
    residue_atom_map = _populate_residue_atom_map(int_indices, config.n_residues)

    # Validate that all residues have at least one atom
    _validate_residue_atom_map(residue_atom_map, config.sequence_list)

    return residue_atom_map


def _get_atom_count_for_residue(
    res_type: str,
    counts_map: Dict[str, int],
) -> int:
    """
    Get atom count for a specific residue type.

    Args:
        res_type: Residue type
        counts_map: Mapping from residue type to atom count

    Returns:
        Number of atoms for the residue type

    Raises:
        KeyError: If residue type is not found in counts_map
    """
    if res_type in counts_map:
        return counts_map[res_type]
    raise KeyError(f"Residue type '{res_type}' not found in atom_counts_map. Cannot infer atom count.")


def _get_counts_for_sequence(
    sequence_list: List[str],
    counts_map: Dict[str, int],
) -> List[int]:
    """
    Get atom counts for each residue in the sequence.

    Args:
        sequence_list: List of residue types
        counts_map: Mapping from residue type to atom count

    Returns:
        List of atom counts for each residue

    Raises:
        KeyError: If a residue type is not found in counts_map
    """
    return [_get_atom_count_for_residue(res_type, counts_map) for res_type in sequence_list]


def _adjust_counts_to_match_total(
    expected_atom_counts: List[int],
    n_atoms: int,
) -> List[int]:
    """
    Adjust atom counts to match the total number of atoms.

    Args:
        expected_atom_counts: List of expected atom counts for each residue
        n_atoms: Target total number of atoms

    Returns:
        Adjusted list of atom counts
    """
    expected_total_atoms = sum(expected_atom_counts)

    if expected_total_atoms <= 0:
        return expected_atom_counts

    # Scale all counts proportionally
    scale_factor = n_atoms / expected_total_atoms
    adjusted_counts = [int(count * scale_factor) for count in expected_atom_counts]

    # Ensure the sum matches n_atoms by adjusting the last residue
    diff = n_atoms - sum(adjusted_counts)
    adjusted_counts[-1] += diff

    return adjusted_counts


def _calculate_expected_atom_counts(
    sequence_list: List[str],
    counts_map: Dict[str, int],
    n_atoms: Optional[int] = None,
) -> List[int]:
    """
    Calculate expected atom counts for each residue with optional adjustment.

    Args:
        sequence_list: List of residue types
        counts_map: Mapping from residue type to atom count
        n_atoms: Optional total number of atoms to adjust counts to

    Returns:
        List[int]: List of expected atom counts for each residue

    Raises:
        KeyError: If a residue type is not found in counts_map
    """
    # Get atom counts for each residue in the sequence
    expected_atom_counts = _get_counts_for_sequence(sequence_list, counts_map)
    expected_total_atoms = sum(expected_atom_counts)

    # Adjust counts if n_atoms is provided and different from expected
    if n_atoms is not None and expected_total_atoms != n_atoms:
        logger.warning(
            f"Mismatch between expected atom count ({expected_total_atoms}) and actual atom count ({n_atoms}) "
            f"from partial_coords. This might indicate non-standard residues or missing atoms."
        )
        return _adjust_counts_to_match_total(expected_atom_counts, n_atoms)

    return expected_atom_counts


def _get_atom_count_from_coords(
    partial_coords: torch.Tensor,
) -> int:
    """
    Get atom count from partial coordinates tensor.

    Args:
        partial_coords: Tensor of partial coordinates

    Returns:
        Number of atoms

    Raises:
        ValueError: If partial_coords has unexpected shape
    """
    if partial_coords.dim() == 3:  # [B, N_atom, 3]
        return partial_coords.shape[1]
    elif partial_coords.dim() == 2:  # [N_atom, 3]
        return partial_coords.shape[0]
    else:
        raise ValueError(f"partial_coords has unexpected shape: {partial_coords.shape}. Expected [B, N_atom, 3] or [N_atom, 3].")


def _create_residue_atom_map_from_counts(
    expected_atom_counts: List[int],
) -> ResidueAtomMap:
    """
    Create residue-atom map from expected atom counts.

    Args:
        expected_atom_counts: List of expected atom counts for each residue

    Returns:
        Mapping from residue indices to atom indices
    """
    residue_atom_map = []
    current_atom_idx = 0

    for atom_count in expected_atom_counts:
        atom_indices = list(range(current_atom_idx, current_atom_idx + atom_count))
        residue_atom_map.append(atom_indices)
        current_atom_idx += atom_count

    return residue_atom_map


def _cast_residue_indices(
    residue_indices: Union[List[str], List[int], torch.Tensor]
) -> List[Union[int, str]]:
    """
    Cast residue indices to the expected type.

    Args:
        residue_indices: List or tensor of residue indices

    Returns:
        List of residue indices as integers or strings
    """
    if isinstance(residue_indices, torch.Tensor):
        return residue_indices.tolist()  # Convert tensor to list
    return [x for x in residue_indices]  # Convert to List[Union[int, str]]


def derive_residue_atom_map(
    sequence: Union[str, List[str], None],
    partial_coords: Optional[torch.Tensor] = None,  # Shape [B, N_atom, 3] or [N_atom, 3]
    atom_metadata: Optional[Dict[str, Union[List[str], List[int], torch.Tensor]]] = None,  # e.g., {'atom_names': ['P', ...], 'residue_indices': [0, 0, ... 1, ...]}
    atom_counts_map: Optional[Dict[str, int]] = None  # Fallback: {'A': 22, 'U': 20, ...} derived from STANDARD_RNA_ATOMS
) -> ResidueAtomMap:
    """
    Derives the mapping from residue index to a list of corresponding global atom indices.

    This helper function determines how atoms are grouped by residue, which is essential
    for the `residue_to_atoms` bridging function.

    Args:
        sequence (Union[str, List[str]]): The RNA sequence (e.g., "AUCG" or ['A', 'U', 'C', 'G']).
                                         Length must match N_residue.
        partial_coords (Optional[torch.Tensor]): Atom coordinates, potentially used to infer N_atom.
                                                Shape [B, N_atom, 3] or [N_atom, 3].
        atom_metadata (Optional[Dict]): Explicit metadata linking atoms to residues.
                                       Expected keys might include 'residue_indices'
                                       (list/tensor of length N_atom mapping each atom to its residue index).
        atom_counts_map (Optional[Dict[str, int]]): A fallback map providing the number of atoms for each
                                                   standard residue type (e.g., derived from STANDARD_RNA_ATOMS).
                                                   Used when `partial_coords` and `atom_metadata` are insufficient.

    Priority for Derivation:
    1. Use `atom_metadata` if provided (most explicit and reliable).
    2. Use `partial_coords` shape and assume contiguous block ordering based on `sequence` and standard atom counts
       if `atom_metadata` is missing. Requires validation that total inferred atoms match `partial_coords.shape[-2]`.
    3. Use `atom_counts_map` (e.g., from STANDARD_RNA_ATOMS lengths) and `sequence` if coordinates and metadata
       are both missing. Assumes contiguous blocks.

    Returns:
        ResidueAtomMap: Map where index `i` contains the list of global atom indices for residue `i`.

    Raises:
        ValueError: If insufficient information is provided, inputs are inconsistent (e.g., sequence length mismatch,
                   atom count mismatch), or assumptions (like contiguous order) are violated based on available data.
        KeyError: If a residue in `sequence` is not found in `atom_counts_map` when operating in fallback mode (Method 3).
    """
    # Prepare sequence and atom counts
    sequence_list, counts_map, n_residues = _prepare_sequence_and_counts(sequence, atom_counts_map)

    # Method 1: Use atom_metadata if provided (most explicit and reliable)
    if atom_metadata is not None and 'residue_indices' in atom_metadata:
        residue_indices = atom_metadata['residue_indices']
        # Derive number of residues from metadata if needed
        if residue_indices:
            if isinstance(residue_indices, torch.Tensor):
                n_residues_meta = int(residue_indices.max().item()) + 1
            else:
                # Convert all indices to integers for max operation
                int_indices = _convert_to_int_list(_cast_residue_indices(residue_indices))
                n_residues_meta = max(int_indices) + 1
        else:
            n_residues_meta = n_residues
        logger.info("Deriving residue-atom map from explicit atom metadata.")
        return _derive_map_from_metadata(_cast_residue_indices(residue_indices), sequence_list or [], n_residues_meta)

    # If no residues, return empty map
    if n_residues == 0:
        logger.info("Empty sequence provided, returning empty residue-atom map.")
        return []

    # Method 2: Use partial_coords shape and assume contiguous block ordering
    if partial_coords is not None:
        logger.info("Deriving residue-atom map from partial coordinates shape and sequence.")
        n_atoms = _get_atom_count_from_coords(partial_coords)
        logger.info(f"[DEBUG][StageD] sequence_list: {sequence_list}")
        logger.info(f"[DEBUG][StageD] counts_map: {counts_map}")
        logger.info(f"[DEBUG][StageD] n_atoms (from partial_coords): {n_atoms}")
        expected_atom_counts = _calculate_expected_atom_counts(sequence_list, counts_map, n_atoms)
        logger.info(f"[DEBUG][StageD] expected_atom_counts: {expected_atom_counts}, sum={sum(expected_atom_counts)}")
        return _create_residue_atom_map_from_counts(expected_atom_counts)
    # Method 3: Use atom_counts_map and sequence if coordinates and metadata are both missing
    logger.info("Deriving residue-atom map from sequence and standard atom counts.")
    logger.info(f"[DEBUG][StageD] sequence_list: {sequence_list}")
    logger.info(f"[DEBUG][StageD] counts_map: {counts_map}")
    expected_atom_counts = _calculate_expected_atom_counts(sequence_list, counts_map)
    logger.info(f"[DEBUG][StageD] expected_atom_counts: {expected_atom_counts}, sum={sum(expected_atom_counts)}")
    return _create_residue_atom_map_from_counts(expected_atom_counts)


def _get_residue_id(residue_id: Union[str, int]) -> str:
    """Convert residue ID to string format."""
    return str(residue_id)


def _format_residue_id(residue_id: Union[str, int]) -> str:
    """Format residue ID for consistent comparison."""
    return str(residue_id).strip()


def get_residue_key(chain_id: str, residue_id: Union[str, int]) -> str:
    """Generate a unique key for a residue based on chain ID and residue ID."""
    formatted_id = _format_residue_id(residue_id)
    return f"{chain_id}_{formatted_id}"

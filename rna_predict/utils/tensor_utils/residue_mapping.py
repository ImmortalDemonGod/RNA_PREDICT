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
    sequence: Union[str, List[str]],
    atom_counts_map: Optional[Dict[str, int]] = None,
) -> Tuple[List[str], Dict[str, int], int]:
    """
    Prepares the RNA sequence list and atom counts mapping.
    
    Converts the input sequence to a list of residues and calculates the total number of residues.
    If no atom counts map is provided, a default mapping based on standard RNA atoms is used.
    
    Args:
        sequence: RNA sequence provided as a string of residue types or as a list.
        atom_counts_map: Optional mapping from residue type to expected atom count.
    
    Returns:
        A tuple containing:
          - The sequence as a list of residue types.
          - The atom counts mapping.
          - The total number of residues.
    """
    # Convert sequence to list if it's a string
    sequence_list = list(sequence) if isinstance(sequence, str) else sequence
    n_residues = len(sequence_list)

    # Prepare atom_counts_map if not provided
    counts_map = atom_counts_map or {res_type: len(atoms) for res_type, atoms in STANDARD_RNA_ATOMS.items()}

    return sequence_list, counts_map, n_residues


class MetadataMapConfig:
    """Configuration for deriving residue-atom map from metadata."""
    def __init__(
        self,
        residue_indices: Union[List[int], torch.Tensor],
        sequence_list: List[str],
        n_residues: int,
    ) -> None:
        """
        Initializes the metadata map configuration.
        
        Args:
            residue_indices (Union[List[int], torch.Tensor]): Residue indices for mapping.
            sequence_list (List[str]): List of residue names in order.
            n_residues (int): The total number of residues.
        """
        self.residue_indices = residue_indices
        self.sequence_list = sequence_list
        self.n_residues = n_residues


def _convert_indices_to_list(
    residue_indices: Union[List[int], torch.Tensor],
) -> List[int]:
    """
    Convert residue indices to a list.

    Args:
        residue_indices: List or tensor of residue indices for each atom

    Returns:
        List of residue indices
    """
    return residue_indices.cpu().tolist() if isinstance(residue_indices, torch.Tensor) else residue_indices


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
    Validates that each residue has at least one atom.
    
    Checks each residue in the residue-atom map and logs a warning if a residue is assigned no atoms.
    The associated residue type from the sequence list is included in the warning message.
    
    Args:
        residue_atom_map: Mapping from residue indices to lists of atom indices.
        sequence_list: List of residue types corresponding to each residue in the map.
    """
    for res_idx, atom_indices in enumerate(residue_atom_map):
        if not atom_indices:
            logger.warning(f"Residue {res_idx} ({sequence_list[res_idx]}) has no atoms assigned in the metadata.")


def _derive_map_from_metadata(
    residue_indices: Union[List[int], torch.Tensor],
    sequence_list: List[str],
    n_residues: int,
) -> ResidueAtomMap:
    """
    Derive residue-atom map from residue indices in metadata.

    Args:
        residue_indices: List or tensor of residue indices for each atom
        sequence_list: List of residue types
        n_residues: Number of residues in the sequence

    Returns:
        Mapping from residue indices to atom indices

    Raises:
        ValueError: If residue indices are invalid
    """
    # Create configuration
    config = MetadataMapConfig(residue_indices, sequence_list, n_residues)

    # Convert to list if it's a tensor
    indices = _convert_indices_to_list(config.residue_indices)

    # Populate the map
    residue_atom_map = _populate_residue_atom_map(indices, config.n_residues)

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
    Proportionally adjust residue atom counts to match a target total.
    
    Scales each residue's expected atom count based on the ratio of the target total to the
    sum of expected counts. The counts are rounded down to integers, and any rounding error is
    compensated by adjusting the last residue's count. If the total of expected counts is zero
    or negative, the original counts are returned.
     
    Args:
        expected_atom_counts: Expected atom counts for each residue.
        n_atoms: Target total number of atoms.
    
    Returns:
        A list of adjusted atom counts that sum exactly to n_atoms.
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
    Calculates the expected number of atoms for each residue in a sequence.
    
    The function computes atom counts for each residue based on the provided counts_map.
    If a total atom count (n_atoms) is provided and differs from the sum of these counts,
    the list is adjusted so that the total matches n_atoms.
    
    Args:
        sequence_list: List of residue types.
        counts_map: Dictionary mapping residue types to their standard atom counts.
        n_atoms: Optional total number of atoms to which the calculated counts are adjusted.
    
    Returns:
        A list of integers representing the expected atom counts for each residue.
    
    Raises:
        KeyError: If a residue type from sequence_list is not found in counts_map.
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
    Extracts the atom count from a tensor of partial 3D coordinates.
    
    This function returns the number of atoms represented in the input tensor. If the tensor has three dimensions 
    (with shape [B, N_atom, 3]), it returns the size of the second dimension. If it has two dimensions (with shape 
    [N_atom, 3]), it returns the size of the first dimension. A ValueError is raised for any other tensor shape.
    
    Args:
        partial_coords: A tensor containing partial 3D coordinates, expected to have shape [B, N_atom, 3] 
                        for batched data or [N_atom, 3] for a single instance.
    
    Returns:
        The number of atoms as an integer.
    
    Raises:
        ValueError: If partial_coords does not have 2 or 3 dimensions.
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
    Assigns contiguous atom indices to residues based on expected atom counts.
    
    Constructs a mapping in which each residue is assigned a sequence of consecutive
    atom indices. The indices start at 0 and are allocated according to the number of
    atoms expected for each residue.
    
    Args:
        expected_atom_counts: A list of integers representing the expected number of
            atoms for each residue.
    
    Returns:
        A list of lists, where each inner list contains the atom indices assigned to a
        corresponding residue.
    """
    residue_atom_map = []
    current_atom_idx = 0

    for atom_count in expected_atom_counts:
        atom_indices = list(range(current_atom_idx, current_atom_idx + atom_count))
        residue_atom_map.append(atom_indices)
        current_atom_idx += atom_count

    return residue_atom_map


def derive_residue_atom_map(
    sequence: Union[str, List[str]],
    partial_coords: Optional[torch.Tensor] = None,  # Shape [B, N_atom, 3] or [N_atom, 3]
    atom_metadata: Optional[Dict[str, Union[List[str], List[int], torch.Tensor]]] = None,  # e.g., {'atom_names': ['P', ...], 'residue_indices': [0, 0, ... 1, ...]}
    atom_counts_map: Optional[Dict[str, int]] = None  # Fallback: {'A': 22, 'U': 20, ...} derived from STANDARD_RNA_ATOMS
) -> ResidueAtomMap:
    """
    Derive a mapping from residues to global atom indices.
    
    This function groups atoms by their associated residue using one of three methods:
    1. If explicit atom metadata is provided (with a 'residue_indices' key), it uses this data.
    2. If partial coordinates are provided, it infers the total number of atoms from the tensor's shape and
       assumes contiguous ordering based on standard residue atom counts.
    3. Otherwise, it falls back to using a provided atom counts map alongside the sequence.
    
    Args:
        sequence (Union[str, List[str]]): The RNA sequence as a string (e.g., "AUCG") or a list of residues.
        partial_coords (Optional[torch.Tensor]): Atom coordinates that, if provided, help determine the total
            atom count. Expected shape is either [B, N_atom, 3] or [N_atom, 3].
        atom_metadata (Optional[Dict[str, Union[List[str], List[int], torch.Tensor]]]): Metadata linking atoms to
            residues. Must include 'residue_indices' that maps each atom to its residue index.
        atom_counts_map (Optional[Dict[str, int]]): A fallback mapping of residue types to their expected atom counts.
    
    Returns:
        ResidueAtomMap: A mapping where each residue index maps to a list of corresponding global atom indices.
    
    Raises:
        ValueError: If inputs are insufficient, inconsistent (e.g., a mismatch between sequence and atom counts),
            or if assumptions about contiguous ordering are violated.
        KeyError: If a residue in the sequence is not found in the atom counts map when using fallback mode.
    """
    # Prepare sequence and atom counts
    sequence_list, counts_map, n_residues = _prepare_sequence_and_counts(sequence, atom_counts_map)

    # If no residues, return empty map
    if n_residues == 0:
        logger.info("Empty sequence provided, returning empty residue-atom map.")
        return []

    # Method 1: Use atom_metadata if provided (most explicit and reliable)
    if atom_metadata is not None and 'residue_indices' in atom_metadata:
        logger.info("Deriving residue-atom map from explicit atom metadata.")
        residue_indices = atom_metadata['residue_indices']
        return _derive_map_from_metadata(residue_indices, sequence_list, n_residues)

    # Method 2: Use partial_coords shape and assume contiguous block ordering
    if partial_coords is not None:
        logger.info("Deriving residue-atom map from partial coordinates shape and sequence.")
        n_atoms = _get_atom_count_from_coords(partial_coords)
        expected_atom_counts = _calculate_expected_atom_counts(sequence_list, counts_map, n_atoms)
        return _create_residue_atom_map_from_counts(expected_atom_counts)

    # Method 3: Use atom_counts_map and sequence if coordinates and metadata are both missing
    logger.info("Deriving residue-atom map from sequence and standard atom counts.")
    expected_atom_counts = _calculate_expected_atom_counts(sequence_list, counts_map)
    return _create_residue_atom_map_from_counts(expected_atom_counts)

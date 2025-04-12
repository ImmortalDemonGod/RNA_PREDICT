"""
Validation utilities for tensor operations.

This module provides functions for validating tensor shapes, indices,
and other constraints to ensure correct operation of tensor utilities.
"""

import torch
from typing import List, Tuple

from .types import ResidueAtomMap, logger


def _validate_residue_count(
    n_residue: int,
    residue_atom_map: ResidueAtomMap,
) -> None:
    """
    Validate that residue count matches the length of residue_atom_map.

    Args:
        n_residue: Number of residues from embeddings
        residue_atom_map: Mapping from residue indices to atom indices

    Raises:
        ValueError: If residue_atom_map length doesn't match n_residue
    """
    if len(residue_atom_map) != n_residue:
        raise ValueError(f"residue_atom_map length ({len(residue_atom_map)}) does not match s_emb residue dimension ({n_residue}).")


def _check_atom_index_bounds(
    atom_idx: int,
    n_atom: int,
) -> None:
    """
    Verifies that the specified atom index is within the valid range [0, n_atom-1].
    
    Args:
        atom_idx: The index of the atom to validate.
        n_atom: The total number of atoms available.
    
    Raises:
        ValueError: If atom_idx is negative or not less than n_atom.
    """
    if atom_idx < 0 or atom_idx >= n_atom:
        raise ValueError(f"Atom index {atom_idx} is out of bounds [0, {n_atom-1}].")


def _count_atom_occurrences(
    residue_atom_map: ResidueAtomMap,
    n_atom: int,
) -> List[int]:
    """
    Count occurrences of each atom index in the residue-atom map.

    Args:
        residue_atom_map: Mapping from residue indices to atom indices
        n_atom: Total number of atoms

    Returns:
        List where index i contains the count of occurrences of atom i

    Raises:
        ValueError: If any atom index is out of bounds
    """
    atom_count = [0] * n_atom
    for atom_indices in residue_atom_map:
        for atom_idx in atom_indices:
            _check_atom_index_bounds(atom_idx, n_atom)
            atom_count[atom_idx] += 1
    return atom_count


def _compare_equal(count: int, target: int) -> bool:
    """
    Checks if the provided count is equal to the target value.
    
    Returns:
        bool: True if count equals target, False otherwise.
    """
    return count == target


def _compare_less(count: int, target: int) -> bool:
    """Compare if count is less than target."""
    return count < target


def _compare_greater(count: int, target: int) -> bool:
    """
    Determine whether count is greater than target.
    
    Returns:
        bool: True if count is strictly greater than target, otherwise False.
    """
    return count > target


def _compare_less_equal(count: int, target: int) -> bool:
    """
    Checks if count is less than or equal to target.
    
    Returns:
        bool: True if count is less than or equal to target, otherwise False.
    """
    return count <= target


def _compare_greater_equal(count: int, target: int) -> bool:
    """Compare if count is greater than or equal to target."""
    return count >= target


def _compare_counts(
    count: int,
    target: int,
    comparison: str,
) -> bool:
    """
    Compare a count value against a target using the specified operator.

    Args:
        count: The count value to compare
        target: The target value to compare against
        comparison: Comparison operator ("==", "<", ">", "<=", ">=")

    Returns:
        Result of the comparison

    Raises:
        ValueError: If comparison operator is not supported
    """
    comparisons = {
        "==": _compare_equal,
        "<": _compare_less,
        ">": _compare_greater,
        "<=": _compare_less_equal,
        ">=": _compare_greater_equal,
    }

    if comparison in comparisons:
        return comparisons[comparison](count, target)

    raise ValueError(f"Unsupported comparison operator: {comparison}")


def _find_atoms_with_count(
    atom_count: List[int],
    target_count: int,
    comparison: str = "==",
) -> List[int]:
    """
    Identifies atom indices whose occurrence counts meet a specified condition.
    
    This function iterates over the provided list of atom counts and returns the indices of atoms
    for which the count satisfies the comparison with the target count.
    
    Args:
        atom_count: A list of occurrence counts for each atom index.
        target_count: The value to compare against each atom's count.
        comparison: The comparison operator as a string (supported: "==", "<", ">", "<=", ">=").
    
    Returns:
        A list of atom indices where the count meets the specified comparison condition.
    """
    return [i for i, count in enumerate(atom_count)
            if _compare_counts(count, target_count, comparison)]


def validate_atom_indices(
    residue_atom_map: ResidueAtomMap,
    n_atom: int,
) -> None:
    """
    Validates that each atom index appears exactly once in the residue mapping.
    
    Checks that every atom index in the range [0, n_atom - 1] is represented once in
    residue_atom_map, raising a ValueError if any index is missing or duplicated.
        
    Args:
        residue_atom_map: Mapping from residue indices to atom indices.
        n_atom: Total number of atoms, defining the expected index range.
        
    Raises:
        ValueError: If any atom index is missing or appears more than once.
    """
    # Count occurrences of each atom index
    atom_count = _count_atom_occurrences(residue_atom_map, n_atom)

    # Check for missing atoms
    missing_atoms = _find_atoms_with_count(atom_count, 0)
    if missing_atoms:
        raise ValueError(f"residue_atom_map is missing atom indices: {missing_atoms}")

    # Check for duplicate atoms
    duplicate_atoms = _find_atoms_with_count(atom_count, 1, ">")
    if duplicate_atoms:
        raise ValueError(f"residue_atom_map has duplicate atom indices: {duplicate_atoms}")


def get_embedding_dimensions(
    s_emb: torch.Tensor,
) -> Tuple[int, int, bool]:
    """
    Extracts residue count, embedding size, and batch status from a tensor.
    
    The input tensor should be either 2D (n_residue x embedding_size) or 3D 
    (batch x n_residue x embedding_size). This function returns the number of residues, 
    the embedding size, and a boolean flag indicating whether the tensor is batched.
    
    Args:
        s_emb (torch.Tensor): The tensor containing residue embeddings.
    
    Returns:
        Tuple[int, int, bool]: A tuple (n_residue, embedding_size, is_batched) where:
            - n_residue is the number of residues.
            - embedding_size is the size of each residue's embedding.
            - is_batched is True if the tensor includes a batch dimension (3D), otherwise False.
    """
    # Check if s_emb is batched
    is_batched = s_emb.dim() == 3

    # Get N_residue and C_s
    if is_batched:
        _, n_residue, c_s = s_emb.shape
    else:
        n_residue, c_s = s_emb.shape

    return n_residue, c_s, is_batched


def get_atom_count_from_map(
    residue_atom_map: ResidueAtomMap,
) -> int:
    """
    Calculates the total number of atoms from a residue-to-atom mapping.
    
    This function flattens the given residue-to-atom map and determines the total number
    of unique atoms by returning the maximum atom index plus one, assuming zero-based indexing.
    An empty mapping returns 0.
    """
    all_atom_indices = []
    for atom_indices in residue_atom_map:
        all_atom_indices.extend(atom_indices)

    if not all_atom_indices:
        return 0

    return max(all_atom_indices) + 1


def get_dimensions(
    s_emb: torch.Tensor,
    residue_atom_map: ResidueAtomMap,
) -> Tuple[int, int, int, bool]:
    """
    Extract dimensions from residue embeddings and validate atom mapping.
    
    This function retrieves the number of residues, the embedding size, and the batched flag from
    the residue embeddings tensor. It then confirms that the residue count matches the length of the
    residue-to-atom mapping and calculates the total number of atoms from the mapping.
    
    Args:
        s_emb: A tensor containing residue embeddings.
        residue_atom_map: A mapping from residue indices to atom indices.
    
    Returns:
        A tuple (n_residue, n_atom, c_s, is_batched) where:
            n_residue: The number of residues.
            n_atom: The total number of atoms computed from the mapping.
            c_s: The size of each residue embedding.
            is_batched: A boolean indicating whether the tensor is batched.
    
    Raises:
        ValueError: If the number of residues from s_emb does not match the length of residue_atom_map.
    """
    # Get embedding dimensions
    n_residue, c_s, is_batched = get_embedding_dimensions(s_emb)

    # Validate residue count
    _validate_residue_count(n_residue, residue_atom_map)

    # Get atom count
    n_atom = get_atom_count_from_map(residue_atom_map)

    return n_residue, n_atom, c_s, is_batched

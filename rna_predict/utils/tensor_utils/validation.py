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
    Check if atom index is within bounds.

    Args:
        atom_idx: Atom index to check
        n_atom: Total number of atoms

    Raises:
        ValueError: If atom index is out of bounds
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
    Return True if count equals the target.
    
    Returns:
        bool: True if count is equal to target; otherwise, False.
    """
    return count == target


def _compare_less(count: int, target: int) -> bool:
    """Compare if count is less than target."""
    return count < target


def _compare_greater(count: int, target: int) -> bool:
    """
    Return True if count is greater than target, otherwise False.
    """
    return count > target


def _compare_less_equal(count: int, target: int) -> bool:
    """Compare if count is less than or equal to target."""
    return count <= target


def _compare_greater_equal(count: int, target: int) -> bool:
    """
    Return True if count is greater than or equal to target.
    
    This function compares two integer values and returns a boolean indicating
    whether count meets or exceeds the target value.
    """
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
    Find atom indices that match a specific count condition.

    Args:
        atom_count: List of atom occurrence counts
        target_count: Target count value to compare against
        comparison: Comparison operator ("==", "<", ">", "<=", ">=")

    Returns:
        List of atom indices that match the condition
    """
    return [i for i, count in enumerate(atom_count)
            if _compare_counts(count, target_count, comparison)]


def validate_atom_indices(
    residue_atom_map: ResidueAtomMap,
    n_atom: int,
) -> None:
    """
    Validates that each atom index in the residue-atom mapping appears exactly once.
    
    This function counts occurrences of each atom index (within the range defined by n_atom)
    in the residue_atom_map and verifies that no atom index is missing or duplicated.
    A ValueError is raised if any index is absent or occurs more than once.
    
    Args:
        residue_atom_map: A mapping from residue indices to atom indices.
        n_atom: Total number of atoms, defining the valid index range.
        
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
    Extract dimensions from a residue embeddings tensor.
    
    Determines if the tensor is batched by checking its dimensionality. For a batched
    tensor, the shape is assumed to be (batch, n_residue, c_s); for an unbatched tensor,
    the shape is (n_residue, c_s).
    
    Args:
        s_emb: Residue embeddings tensor.
    
    Returns:
        A tuple (n_residue, c_s, is_batched) where n_residue is the number of residues,
        c_s is the embedding size, and is_batched is True if the tensor includes a batch
        dimension, otherwise False.
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
    Compute total atom count from a residue-atom mapping.
    
    Determines the total number of atoms by finding the maximum atom index and adding one.
    Returns 0 if no atom indices are present.
    
    Args:
        residue_atom_map: An iterable where each element is a list of atom indices for a residue.
    
    Returns:
        The total number of atoms, computed as max(atom indices) + 1.
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
    Extract and validate dimensions from residue embeddings and residue-atom mapping.
    
    This function extracts the number of residues, the embedding size, and a batched flag from the
    residue embeddings tensor. It then verifies that the number of residues matches the length of
    the residue-to-atom mapping and computes the total number of atoms based on the mapping.
    
    Args:
        s_emb: Residue embeddings tensor which may include a batch dimension.
        residue_atom_map: Mapping from residue indices to atom indices.
    
    Returns:
        A tuple (n_residue, n_atom, c_s, is_batched) where:
          - n_residue is the number of residues,
          - n_atom is the total number of atoms (one more than the maximum atom index),
          - c_s is the size of the residue embeddings,
          - is_batched indicates whether the tensor includes a batch dimension.
    
    Raises:
        ValueError: If the length of residue_atom_map does not match the number of residues in s_emb.
    """
    # Get embedding dimensions
    n_residue, c_s, is_batched = get_embedding_dimensions(s_emb)

    # Validate residue count
    _validate_residue_count(n_residue, residue_atom_map)

    # Get atom count
    n_atom = get_atom_count_from_map(residue_atom_map)

    return n_residue, n_atom, c_s, is_batched

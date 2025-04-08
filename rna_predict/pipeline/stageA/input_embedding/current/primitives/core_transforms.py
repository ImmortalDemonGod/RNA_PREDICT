"""
Core data transformation module for neural network operations.

This module contains core functions for transforming, rearranging, and manipulating
tensor data for neural network operations, particularly for attention mechanisms.
"""

from dataclasses import dataclass
from typing import List, Tuple, Union, cast

import torch


@dataclass
class RearrangeConfig:
    """Configuration for tensor rearrangement operations."""

    n_queries: int
    n_keys: int
    compute_mask: bool = True


@dataclass
class TrunkInfo:
    """Information about trunk processing."""

    total_queries: int
    total_keys: int
    n_q_trunks: int
    n_k_trunks: int
    dim_q_list: List[int]
    dim_k_list: List[int]
    q_list: List[torch.Tensor]
    k_list: List[torch.Tensor]


@dataclass
class MaskCreationConfig:
    """Configuration for mask creation."""

    n_queries: int
    n_keys: int
    query_lists: List[torch.Tensor]
    key_lists: List[torch.Tensor]
    query_dims: List[int]
    key_dims: List[int]

    n_q_chunks: int
    n_k_chunks: int
    q_trunk_indices: List[int]
    n_q_per_chunk: int
    window_size: int
    original_query_length: int


@dataclass
class RearrangeInputConfig:
    """Configuration for rearrange input."""

    n_queries: int = 32
    n_keys: int = 128
    compute_mask: bool = True
    dim_q: Union[int, List[int]] = 1
    dim_k: Union[int, List[int]] = 1


def _validate_input_types(
    q: Union[torch.Tensor, List[torch.Tensor]],
    k: Union[torch.Tensor, List[torch.Tensor]],
) -> Tuple[bool, bool]:
    """
    Validate input types for tensor rearrangement.

    Args:
        q (Union[torch.Tensor, List[torch.Tensor]]): Query tensor or list of tensors
        k (Union[torch.Tensor, List[torch.Tensor]]): Key tensor or list of tensors

    Returns:
        Tuple[bool, bool]: Flags indicating if q and k are lists
    """
    q_is_list = isinstance(q, list)
    k_is_list = isinstance(k, list)

    # Type validations
    if q_is_list and not all(isinstance(t, torch.Tensor) for t in q):
        raise TypeError("All elements in q list must be torch.Tensor")

    if k_is_list and not all(isinstance(t, torch.Tensor) for t in k):
        raise TypeError("All elements in k list must be torch.Tensor")

    return q_is_list, k_is_list


def _prepare_tensor_lists(
    q: Union[torch.Tensor, List[torch.Tensor]],
    k: Union[torch.Tensor, List[torch.Tensor]],
    q_is_list: bool,
    k_is_list: bool,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Prepare tensor lists for rearrangement.

    Args:
        q (Union[torch.Tensor, List[torch.Tensor]]): Query tensor or list of tensors
        k (Union[torch.Tensor, List[torch.Tensor]]): Key tensor or list of tensors
        q_is_list (bool): Flag indicating if q is a list
        k_is_list (bool): Flag indicating if k is a list

    Returns:
        Tuple[List[torch.Tensor], List[torch.Tensor]]: Prepared lists of tensors
    """
    # Convert to lists if not already
    q_list = cast(List[torch.Tensor], q) if q_is_list else [cast(torch.Tensor, q)]
    k_list = cast(List[torch.Tensor], k) if k_is_list else [cast(torch.Tensor, k)]

    return q_list, k_list


def _validate_dimensions(
    dim_q_list: List[int],
    dim_k_list: List[int],
    q_list: List[torch.Tensor],
    k_list: List[torch.Tensor],
) -> None:
    """
    Validate dimensions match tensor lists.

    Args:
        dim_q_list (List[int]): List of query dimensions
        dim_k_list (List[int]): List of key dimensions
        q_list (List[torch.Tensor]): List of query tensors
        k_list (List[torch.Tensor]): List of key tensors

    Raises:
        ValueError: If dimensions don't match tensor lists
    """
    if len(dim_q_list) != len(q_list):
        raise ValueError(
            f"Length of dim_q ({len(dim_q_list)}) must match length of q ({len(q_list)})"
        )
    if len(dim_k_list) != len(k_list):
        raise ValueError(
            f"Length of dim_k ({len(dim_k_list)}) must match length of k ({len(k_list)})"
        )


def _calculate_trunk_info(
    q_list: List[torch.Tensor],
    k_list: List[torch.Tensor],
    dim_q_list: List[int],
    dim_k_list: List[int],
    n_queries: int,
    n_keys: int,
) -> TrunkInfo:
    """
    Calculate information about trunks.

    Args:
        q_list (List[torch.Tensor]): List of query tensors
        k_list (List[torch.Tensor]): List of key tensors
        dim_q_list (List[int]): List of query dimensions
        dim_k_list (List[int]): List of key dimensions
        n_queries (int): Number of queries per trunk
        n_keys (int): Number of keys per trunk

    Returns:
        TrunkInfo: Information about trunk processing
    """
    total_q = sum(q.shape[dim_q_list[i]] for i, q in enumerate(q_list))
    total_k = sum(k.shape[dim_k_list[i]] for i, k in enumerate(k_list))

    n_q_trunks = (total_q + n_queries - 1) // n_queries
    n_k_trunks = (total_k + n_keys - 1) // n_keys

    return TrunkInfo(
        total_queries=total_q,
        total_keys=total_k,
        n_q_trunks=n_q_trunks,
        n_k_trunks=n_k_trunks,
        dim_q_list=dim_q_list,
        dim_k_list=dim_k_list,
        q_list=q_list,
        k_list=k_list,
    )


def _apply_trunk_slices(
    tensor_list: List[torch.Tensor],
    dim_list: List[int],
    trunk_idx: int,
    n_entries: int,
) -> List[torch.Tensor]:
    """
    Apply trunk slices to a list of tensors.

    Args:
        tensor_list (List[torch.Tensor]): List of tensors to slice
        dim_list (List[int]): List of dimensions to slice on
        trunk_idx (int): Trunk index
        n_entries (int): Number of entries per trunk

    Returns:
        List[torch.Tensor]: List of sliced tensors
    """
    result_list = []

    for i, tensor in enumerate(tensor_list):
        # Calculate slice indices
        start_idx = trunk_idx * n_entries
        end_idx = min(start_idx + n_entries, tensor.shape[dim_list[i]])

        # Create slices list - initially all ':'
        slices = [slice(None)] * tensor.ndim

        # Set the specific dimension's slice
        slices[dim_list[i]] = slice(start_idx, end_idx)

        # Apply slices and append to result
        sliced_tensor = tensor[tuple(slices)]
        result_list.append(sliced_tensor)

    return result_list


def _handle_return_types(
    q_new: List[torch.Tensor],
    k_new: List[torch.Tensor],
    q_is_list: bool,
    k_is_list: bool,
) -> Tuple[
    Union[torch.Tensor, List[torch.Tensor]], Union[torch.Tensor, List[torch.Tensor]]
]:
    """
    Handle return types based on input types.

    Args:
        q_new (List[torch.Tensor]): Processed query tensors
        k_new (List[torch.Tensor]): Processed key tensors
        q_is_list (bool): Whether original query was a list
        k_is_list (bool): Whether original key was a list

    Returns:
        Tuple[Union[torch.Tensor, List[torch.Tensor]], Union[torch.Tensor, List[torch.Tensor]]]:
            Processed queries and keys with appropriate types
    """
    q_result: Union[torch.Tensor, List[torch.Tensor]]
    k_result: Union[torch.Tensor, List[torch.Tensor]]

    if not q_is_list and len(q_new) == 1:
        q_result = q_new[0]
    else:
        q_result = q_new

    if not k_is_list and len(k_new) == 1:
        k_result = k_new[0]
    else:
        k_result = k_new

    return q_result, k_result


def _create_zero_tensor(
    shape: List[int], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """
    Create a tensor filled with zeros.

    Args:
        shape (List[int]): Shape of the tensor
        dtype (torch.dtype): Data type of the tensor
        device (torch.device): Device to create tensor on

    Returns:
        torch.Tensor: Zero-filled tensor
    """
    return torch.zeros(shape, dtype=dtype, device=device)

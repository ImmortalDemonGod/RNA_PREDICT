# protenix/model/utils.py
# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
General utility functions for RNA structure prediction.
"""

from typing import Dict, List, Optional, Any, TypeVar

import numpy as np
import torch

T = TypeVar('T', bound=np.ndarray)

def sample_indices(
    n: int,
    sample_size: int,
    strategy: str = "random",
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    assert n >= 0, f"Number of items n must be non-negative, got {n}"
    assert sample_size >= 0, f"sample_size must be non-negative, got {sample_size}"
    assert strategy in ["random", "topk"], f"Invalid sampling strategy: {strategy}"
    assert sample_size <= n, f"Cannot sample {sample_size} items from {n} items"

    if strategy == "random":
        # Ensure n is positive for randperm
        if n <= 0:
            return torch.tensor([], dtype=torch.long, device=device)
        indices = torch.randperm(n=n, device=device)[:sample_size]
    elif strategy == "topk":
        indices = torch.arange(sample_size, device=device)
    else:
        raise ValueError(f"Invalid sampling strategy: {strategy}")
    return indices

def _should_return_original_dict(feat_dict: Dict[str, torch.Tensor], sample_size: int) -> bool:
    """
    Check if we should return the original dictionary without sampling.

    Args:
        feat_dict: Dictionary of features to sample from
        sample_size: Number of samples to take

    Returns:
        True if we should return the original dictionary, False otherwise
    """
    # Handle empty dictionary case
    if not feat_dict:
        return True

    # Handle case where 'msa' key is not present
    if "msa" not in feat_dict:
        return True

    # Get number of sequences from the 'msa' key
    n_seq = feat_dict["msa"].shape[0]

    # If sample_size >= n_seq, return the original dictionary
    if sample_size >= n_seq:
        return True

    return False


def _is_sequence_dimension_key(key: str) -> bool:
    """
    Check if a key has the sequence dimension as its first dimension.

    Args:
        key: Key to check

    Returns:
        True if the key has the sequence dimension, False otherwise
    """
    sequence_keys = {"msa", "has_deletion", "deletion_value", "xyz"}
    return key in sequence_keys


def sample_msa_feature_dict_random_without_replacement(
    feat_dict: Dict[str, torch.Tensor],
    sample_size: int,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Sample MSA features randomly without replacement.

    Args:
        feat_dict: Dictionary of features to sample from
        sample_size: Number of samples to take
        device: Device to use for sampling

    Returns:
        Dictionary with sampled features
    """
    # Check if we should return the original dictionary
    if _should_return_original_dict(feat_dict, sample_size):
        return feat_dict

    # Get number of sequences from the 'msa' key
    n_seq = feat_dict["msa"].shape[0]

    # Sample indices
    indices = sample_indices(n_seq, sample_size, strategy="random", device=device)

    # Create new dictionary with sampled features
    result = {}
    for k, v in feat_dict.items():
        # Skip None values for sequence dimension keys
        if v is None:
            continue

        if _is_sequence_dimension_key(k):
            # These keys have the same first dimension as the number of sequences
            result[k] = v[indices]
        else:
            # Other keys are passed through unchanged
            result[k] = v

    return result


def _convert_value_to_numpy(value) -> np.ndarray:
    """Convert a value to a numpy array.

    Args:
        value: Value to convert (float, int, torch.Tensor, or np.ndarray)

    Returns:
        np.ndarray: Converted value

    Raises:
        ValueError: If the value type is not supported
    """
    if isinstance(value, (float, int)):
        return np.array([value])
    elif isinstance(value, torch.Tensor):
        if value.dim() == 0:
            return np.array([value.item()])
        else:
            return value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        return value
    else:
        raise ValueError(f"Unsupported type for metric data: {type(value)}")


def _collect_values_from_dicts(dict_list: List[Dict[str, Any]]) -> Dict[str, List[np.ndarray]]:
    """Collect values from a list of dictionaries.

    Args:
        dict_list: List of dictionaries to collect values from

    Returns:
        dict: Dictionary with collected values as lists
    """
    merged_dict: Dict[str, List[np.ndarray]] = {}

    for x in dict_list:
        for k, v in x.items():
            if k not in merged_dict:
                merged_dict[k] = []
            merged_dict[k].append(_convert_value_to_numpy(v))

    return merged_dict


def _is_compatible_shape(shape1, shape2) -> bool:
    """Check if two shapes are compatible for reshaping.

    Args:
        shape1: First shape
        shape2: Second shape

    Returns:
        bool: True if shapes are compatible, False otherwise
    """
    return (
        shape1 == shape2
        or (shape1 == () and shape2 == (1,))
        or (shape1 == (1,) and shape2 == ())
    )


def _reshape_item_to_match(item: np.ndarray, target_shape) -> np.ndarray:
    """Reshape an item to match the target shape if possible.

    Args:
        item: Array to reshape
        target_shape: Target shape

    Returns:
        np.ndarray: Reshaped array
    """
    if item.shape == target_shape:
        return item
    elif item.shape == () and target_shape == (1,):
        return item.reshape(1)
    elif item.shape == (1,) and target_shape == ():
        return item.reshape(())
    else:
        return item  # Will be caught by compatibility check later


def _reshape_arrays_if_needed(arrays: List[np.ndarray], key: str) -> List[np.ndarray]:
    """Reshape arrays to make them compatible for concatenation.

    Args:
        arrays: List of arrays to reshape
        key: Dictionary key for error reporting

    Returns:
        list: List of reshaped arrays

    Raises:
        ValueError: If arrays have incompatible shapes
    """
    if not arrays:
        return arrays

    first_shape = arrays[0].shape
    if all(item.shape == first_shape for item in arrays):
        return arrays

    # Attempt to reshape if possible
    reshaped_arrays = []
    for item in arrays:
        # Check if shapes are compatible before reshaping
        if not _is_compatible_shape(item.shape, first_shape):
            raise ValueError(
                f"Incompatible shapes for key '{key}': {first_shape} vs {item.shape}"
            )

        # Reshape the item
        reshaped_item = _reshape_item_to_match(item, first_shape)
        reshaped_arrays.append(reshaped_item)

    return reshaped_arrays


def _concatenate_arrays(arrays: list, key: str) -> np.ndarray:
    """Concatenate arrays into a single array.

    Args:
        arrays: List of arrays to concatenate
        key: Dictionary key for error reporting

    Returns:
        np.ndarray: Concatenated array

    Raises:
        ValueError: If arrays cannot be concatenated
    """
    try:
        return np.concatenate(arrays)
    except ValueError as e:
        print(f"Error concatenating key '{key}': {e}")
        raise e


def simple_merge_dict_list(dict_list: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """
    Merge a list of dictionaries into a single dictionary.

    Args:
        dict_list (List[Dict[str, Any]]): List of dictionaries to merge.

    Returns:
        Dict[str, np.ndarray]: Merged dictionary where values are concatenated arrays.
    """
    # Collect values from dictionaries
    merged_dict = _collect_values_from_dicts(dict_list)
    result: Dict[str, np.ndarray] = {}

    # Process each key-value pair
    for k, v in list(merged_dict.items()):
        # Skip empty lists
        if not v:
            continue

        # Reshape arrays if needed
        reshaped_v = _reshape_arrays_if_needed(v, k)

        # Concatenate arrays
        result[k] = _concatenate_arrays(reshaped_v, k)

    return result

"""
Common utilities and types used across transformer components.
"""
from functools import partial
from typing import Optional, Union, Dict, List, TypedDict, Any, cast, Tuple, Callable, TypeVar

import torch

# Define type variable for better generic typing
T = TypeVar('T')


def make_typed_partial(
    func: Callable[..., T], 
    *args: Any, 
    **kwargs: Any
) -> Callable[..., T]:
    """
    Create a type-safe partial function.
    
    Args:
        func: The function to create a partial from
        args: Positional arguments to fix
        kwargs: Keyword arguments to fix
        
    Returns:
        A new callable with the specified arguments fixed
    """
    return cast(Callable[..., T], partial(func, *args, **kwargs))


class InputFeatureDict(TypedDict, total=False):
    """TypedDict for input feature dictionary used in transformer modules."""
    atom_to_token_idx: torch.Tensor
    ref_pos: torch.Tensor
    ref_charge: torch.Tensor
    ref_mask: torch.Tensor
    ref_element: torch.Tensor
    ref_atom_name_chars: torch.Tensor
    ref_space_uid: torch.Tensor
    restype: torch.Tensor


def safe_tensor_access(
    feature_dict: InputFeatureDict, 
    key: str
) -> torch.Tensor:
    """
    Safely access a tensor from the feature dictionary.
    
    Args:
        feature_dict: Dictionary containing features
        key: Key to access
        
    Returns:
        The tensor at the given key
        
    Raises:
        ValueError: If the value is not a tensor
    """
    value = feature_dict.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Expected tensor for key '{key}', got {type(value)}")
    return value


def validate_tensor_shape(
    tensor: torch.Tensor,
    expected_dims: Optional[int] = None,
    expected_last_dim: Optional[int] = None,
    name: str = "tensor"
) -> None:
    """
    Validate tensor shape against expected dimensions.
    
    Args:
        tensor: Tensor to validate
        expected_dims: Expected number of dimensions
        expected_last_dim: Expected size of last dimension
        name: Name of tensor for error messages
    
    Raises:
        ValueError: If validation fails
    """
    if expected_dims is not None and len(tensor.shape) != expected_dims:
        raise ValueError(
            f"Expected {name} to have {expected_dims} dimensions, "
            f"got {len(tensor.shape)} with shape {tensor.shape}"
        )
    
    if expected_last_dim is not None and tensor.shape[-1] != expected_last_dim:
        raise ValueError(
            f"Expected {name} to have last dimension {expected_last_dim}, "
            f"got {tensor.shape[-1]} with shape {tensor.shape}"
        ) 
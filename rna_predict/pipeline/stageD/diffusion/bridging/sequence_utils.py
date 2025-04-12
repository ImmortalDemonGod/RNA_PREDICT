"""
Sequence extraction and processing utilities for Stage D diffusion.

This module provides functions for extracting and processing sequence information
for the Stage D diffusion process.
"""

import logging
from typing import Any, Dict, List

import torch

logger = logging.getLogger(__name__)


# Helper functions for sequence extraction
def _convert_tensor_to_sequence(tensor_seq: torch.Tensor) -> List[str]:
    """Converts a tensor sequence to a list of strings.
    
    Moves the input tensor to the CPU, converts it to a list, and casts each element to a string.
    """
    return [str(s) for s in tensor_seq.cpu().tolist()]


def _convert_list_to_sequence(list_seq: list) -> List[str]:
    """
    Converts a list of elements to a list of strings.
    
    Each element in the input list is converted to its string representation using
    the built-in str() function, ensuring that all elements of the returned list are strings.
    """
    return [str(s) for s in list_seq]


def _convert_str_to_sequence(str_seq: str) -> List[str]:
    """Converts a string to a list of characters."""
    return list(str_seq)


# Helper 1: Try extracting sequence from input_features dictionary
def _try_extract_from_input_features(
    input_features: Dict[str, Any] | None,
) -> List[str] | None:
    """
    Extract a sequence from the input features.
    
    If the dictionary contains a "sequence" key, the function converts its value to a list of
    strings using an appropriate helper based on its type. Supported types include torch.Tensor,
    list, and str. If the key is absent or the value has an unsupported type, the function returns
    None.
        
    Args:
        input_features: A dictionary that may include sequence data under the "sequence" key.
        
    Returns:
        A list of strings representing the sequence if extraction is successful; otherwise, None.
    """
    # Early return if input_features is None or doesn't contain sequence
    if input_features is None or "sequence" not in input_features:
        return None

    seq_data = input_features["sequence"]

    # Handle different sequence data types
    if isinstance(seq_data, torch.Tensor):
        return _convert_tensor_to_sequence(seq_data)
    elif isinstance(seq_data, list):
        return _convert_list_to_sequence(seq_data)
    elif isinstance(seq_data, str):
        return _convert_str_to_sequence(seq_data)
    else:
        logger.warning(
            f"Unexpected type for 'sequence' in input_features: {type(seq_data)}. Skipping."
        )
        return None


# Helper 2: Try inferring sequence length from trunk embeddings
def _try_infer_from_embeddings(
    trunk_embeddings: Dict[str, torch.Tensor],
) -> List[str] | None:
    """
    Infers sequence length from trunk embeddings and returns a placeholder sequence.
    
    This function searches for candidate keys ("s_trunk", "s_inputs", "sing") in the provided
    dictionary. If a corresponding tensor with at least two dimensions is found, its second dimension
    is used as the sequence length, and a list of 'A' characters repeated for that length is returned.
    If a tensor has insufficient dimensions, a warning is logged and the key is skipped. If no valid
    tensor is found, the function returns None.
    
    Args:
        trunk_embeddings (Dict[str, torch.Tensor]): A dictionary mapping keys to embedding tensors.
            Expected keys are "s_trunk", "s_inputs", or "sing", where each tensor should have at least
            two dimensions with the second dimension representing the sequence length.
    
    Returns:
        List[str] | None: A placeholder sequence as a list of 'A' characters matching the inferred
        sequence length, or None if no valid embedding is found.
    """
    for key in ["s_trunk", "s_inputs", "sing"]:
        if key in trunk_embeddings and isinstance(trunk_embeddings[key], torch.Tensor):
            # Ensure tensor has at least 2 dimensions (Batch, Sequence, ...)
            if trunk_embeddings[key].ndim >= 2:
                n_residues = trunk_embeddings[key].shape[1]
                logger.warning(
                    f"No sequence provided. Inferring length {n_residues} from '{key}'. "
                    f"Using placeholder sequence. This may affect accuracy."
                )
                # Use 'A' as placeholder
                return ["A"] * n_residues
            else:
                logger.warning(
                    f"Trunk embedding '{key}' has insufficient dimensions ({trunk_embeddings[key].ndim}) to infer sequence length."
                )

    return None  # Could not infer length


def extract_sequence(
    sequence: List[str] | None,
    input_features: Dict[str, Any] | None,
    trunk_embeddings: Dict[str, torch.Tensor],
) -> List[str]:
    """
    Extracts a sequence from provided inputs with fallbacks.
    
    This function returns the explicitly provided sequence if available. If absent, it tries to extract the sequence
    from the input_features dictionary, and if that also fails, it infers a sequence from the trunk embeddings by generating
    a placeholder sequence. A ValueError is raised if none of these sources yield a valid sequence.
    
    Args:
        sequence: A list of sequence characters provided explicitly.
        input_features: A dictionary that may contain a sequence under the key 'sequence'.
        trunk_embeddings: A dictionary of trunk embeddings used to infer sequence length when other sources are absent.
    
    Returns:
        A list of sequence characters.
    
    Raises:
        ValueError: If a sequence cannot be derived from any of the provided sources.
    """
    # 1. Check explicitly provided sequence
    if sequence is not None:
        return sequence

    # 2. Try extracting from input_features using helper
    seq_from_features = _try_extract_from_input_features(input_features)
    if seq_from_features is not None:
        return seq_from_features

    # 3. Fallback: Try inferring from trunk embeddings using helper
    seq_from_embeddings = _try_infer_from_embeddings(trunk_embeddings)
    if seq_from_embeddings is not None:
        return seq_from_embeddings

    # 4. If no sequence source found, raise error
    raise ValueError(
        "Cannot derive sequence: Not provided explicitly, not found in "
        "input_features, and could not infer length from trunk_embeddings."
    )

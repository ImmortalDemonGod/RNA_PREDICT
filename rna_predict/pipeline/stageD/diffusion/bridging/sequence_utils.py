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
    """
    Converts a PyTorch tensor representing a sequence into a list of strings.
    
    The function moves the tensor to the CPU, converts it to a native Python list, and casts
    each element to its string representation.
    
    Returns:
        List[str]: A list containing the string representations of the tensor's elements.
    """
    return [str(s) for s in tensor_seq.cpu().tolist()]


def _convert_list_to_sequence(list_seq: list) -> List[str]:
    """
    Converts all elements in a list to their string representations.
    
    Returns a new list where each element from the input list is converted using the built-in str() function.
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
    Extracts a sequence from input features.
    
    This function looks for a "sequence" key in the provided dictionary and converts its 
    value into a list of strings using appropriate helper functions. It supports tensor, 
    list, and string representations. If the key is missing or the value is of an 
    unexpected type, the function logs a warning and returns None.
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
    """Infers sequence length from trunk embeddings.
    
    Searches for a valid tensor under potential keys in the trunk embeddings and checks that the
    tensor has at least two dimensions. If such a tensor is found, the sequence length is inferred
    from the tensor's second dimension and a placeholder sequence of repeated 'A' characters is
    returned. If no valid tensor is found, the function logs a warning and returns None.
    
    Returns:
        A list of 'A' characters with length determined by the inferred sequence, or None if
        inference fails.
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
    Extracts a sequence from available inputs with fallback options.
    
    If a sequence is explicitly provided, it is returned immediately. Otherwise, the function attempts to obtain the sequence from the input_features dictionary. If that fails, it infers a sequence based on the trunk_embeddings. If no valid sequence can be determined from any source, a ValueError is raised.
    
    Args:
        sequence: Optional list of residue types to use as the sequence.
        input_features: Optional dictionary that may contain sequence data.
        trunk_embeddings: Dictionary of tensors used to infer the sequence if not explicitly provided.
    
    Returns:
        A list of residue types representing the derived sequence.
    
    Raises:
        ValueError: If the sequence cannot be derived from any of the provided sources.
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

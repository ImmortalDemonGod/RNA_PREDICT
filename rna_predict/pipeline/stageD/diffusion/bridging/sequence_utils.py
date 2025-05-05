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
    """Converts a tensor sequence to a list of strings."""
    return [str(s) for s in tensor_seq.cpu().tolist()]


def _convert_list_to_sequence(list_seq: list) -> List[str]:
    """Ensures all elements in a list are strings."""
    return [str(s) for s in list_seq]


def _convert_str_to_sequence(str_seq: str) -> List[str]:
    """Converts a string to a list of characters."""
    return list(str_seq)


# Helper 1: Try extracting sequence from input_features dictionary
def _try_extract_from_input_features(
    input_features: Dict[str, Any] | None,
    *args, **kwargs
):
    """Attempts to extract sequence from the input_features dictionary."""
    # Early return if input_features is None or doesn't contain sequence
    if input_features is None or "sequence" not in input_features:
        return None

    seq_data = input_features["sequence"]
    # Handle different sequence data types
    if isinstance(seq_data, torch.Tensor):
        result = _convert_tensor_to_sequence(seq_data)
    elif isinstance(seq_data, list):
        result = _convert_list_to_sequence(seq_data)
    elif isinstance(seq_data, str):
        result = _convert_str_to_sequence(seq_data)
    else:
        logger.warning(
            f"Unexpected type for 'sequence' in input_features: {type(seq_data)}. Skipping.")
        return None
    print(f"[CASCADE-DEBUG][SEQ-EXTRACT] type={type(result)}, value={result}")
    return result


# Helper 2: Try inferring sequence length from trunk embeddings
def _try_infer_from_embeddings(
    trunk_embeddings: Dict[str, torch.Tensor],
) -> List[str] | None:
    """Attempts to infer sequence length from trunk embeddings and returns a placeholder."""
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
    sequence: List[str] | str | None,
    input_features: Dict[str, Any] | None,
    trunk_embeddings: Dict[str, torch.Tensor],
) -> str:
    """
    Extract sequence from available sources with fallbacks.

    Args:
        sequence: Explicitly provided sequence (string or list of chars)
        input_features: Input features dictionary that might contain sequence
        trunk_embeddings: Trunk embeddings that might be used to infer sequence length

    Returns:
        RNA sequence as a string (e.g., 'AUGC...')

    Raises:
        ValueError: If sequence cannot be derived from any source
    """
    # 1. Check explicitly provided sequence
    if sequence is not None:
        # If it's a list of single-character strings, join; else, assume already string
        if isinstance(sequence, list) and all(isinstance(x, str) and len(x) == 1 for x in sequence):
            return "".join(sequence)
        elif isinstance(sequence, str):
            return sequence
        else:
            raise ValueError(f"Unsupported sequence type: {type(sequence)}")

    # 2. Try extracting from input_features using helper
    seq_from_features = _try_extract_from_input_features(input_features)
    if seq_from_features is not None:
        if isinstance(seq_from_features, list) and all(isinstance(x, str) and len(x) == 1 for x in seq_from_features):
            return "".join(seq_from_features)
        elif isinstance(seq_from_features, str):
            return seq_from_features
        else:
            raise ValueError(f"Unsupported extracted sequence type: {type(seq_from_features)}")

    # 3. Fallback: Try inferring from trunk embeddings using helper
    seq_from_embeddings = _try_infer_from_embeddings(trunk_embeddings)
    if seq_from_embeddings is not None:
        if isinstance(seq_from_embeddings, list) and all(isinstance(x, str) and len(x) == 1 for x in seq_from_embeddings):
            return "".join(seq_from_embeddings)
        elif isinstance(seq_from_embeddings, str):
            return seq_from_embeddings
        else:
            raise ValueError(f"Unsupported inferred sequence type: {type(seq_from_embeddings)}")

    raise ValueError("If sequence cannot be derived from any source")

"""
Residue-to-atom bridging functions for Stage D diffusion.

This module provides functions for bridging between residue-level and atom-level
representations in the Stage D diffusion process.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch

from rna_predict.utils.tensor_utils import derive_residue_atom_map, residue_to_atoms

from ..utils.tensor_utils import normalize_tensor_dimensions
from .sequence_utils import extract_sequence

logger = logging.getLogger(__name__)


@dataclass
class BridgingInput:
    """Data class for inputs to the residue-to-atom bridging function."""

    partial_coords: torch.Tensor
    trunk_embeddings: Dict[str, torch.Tensor]
    input_features: Dict[str, Any] | None
    sequence: List[str] | None


# Helper function for processing single embeddings
def _process_single_embedding(
    value: torch.Tensor,
    residue_atom_map: List[List[int]],
    key: str,
    debug_logging: bool,
) -> torch.Tensor:
    """Applies residue-to-atom bridging for single embeddings."""
    bridged_value = residue_to_atoms(value, residue_atom_map)
    if debug_logging:
        logger.debug(
            f"Bridged {key} from shape {value.shape} to {bridged_value.shape} "
            f"using residue-to-atom mapping"
        )
    return bridged_value


# Helper function for processing pair embeddings
def _process_pair_embedding(value: torch.Tensor, key: str) -> torch.Tensor:
    """Handles pair embeddings (currently logs a warning)."""
    logger.warning(
        f"Pair embedding bridging not implemented. The tensor {key} with shape {value.shape} "
        f"may cause issues in Stage D if its dimensions don't match atom counts."
    )
    return value


# Helper function to process the 'deletion_mean' tensor specifically
def _process_deletion_mean(value: torch.Tensor) -> torch.Tensor:
    """Ensure deletion_mean tensor has the shape [B, N, 1]."""
    if value.dim() == 2:  # If 2D [B, N]
        return value.unsqueeze(-1)  # Make it [B, N, 1]
    elif value.dim() == 3 and value.shape[-1] != 1:  # If 3D but wrong last dim
        # Ensure returning a tensor with the last dimension as 1
        return value[..., :1]  # Take first channel, ensures last dim is 1
    # Return as is if already correct shape ([B, N, 1]) or not 2D/3D
    return value


@dataclass
class EmbeddingProcessContext:
    """Context for processing trunk embeddings."""
    residue_atom_map: List[List[int]]
    batch_size: int
    debug_logging: bool


# Helper function to process a single trunk embedding item
def _process_one_trunk_embedding(
    key: str,
    value: Any,  # Can be Tensor or other type
    context: EmbeddingProcessContext,
) -> Any:  # Return type matches input 'value' or processed Tensor
    """Processes a single key-value pair from trunk_embeddings dictionary."""
    if not isinstance(value, torch.Tensor):
        # Keep non-tensor values as is
        return value

    # Normalize dimensions
    temp_value = normalize_tensor_dimensions(value, context.batch_size)

    # Apply bridging based on tensor type using helper functions
    if key in ["s_trunk", "s_inputs", "sing"]:
        processed_value = _process_single_embedding(
            temp_value, context.residue_atom_map, key, context.debug_logging
        )
    elif key == "pair":
        processed_value = _process_pair_embedding(temp_value, key)
    else:
        # Keep other tensors as is (no bridging needed)
        processed_value = temp_value

    return processed_value


def process_trunk_embeddings(
    trunk_embeddings: Dict[str, torch.Tensor],
    residue_atom_map: List[List[int]],
    batch_size: int,
    debug_logging: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Process trunk embeddings to bridge residue-level to atom-level representations.

    Args:
        trunk_embeddings: Dictionary of trunk embeddings (residue-level)
        residue_atom_map: Mapping from residue indices to atom indices
        batch_size: Batch size for tensor dimension validation
        debug_logging: Whether to enable debug logging

    Returns:
        Dictionary of bridged trunk embeddings
    """
    bridged_trunk_embeddings = {}

    # Create processing context
    context = EmbeddingProcessContext(
        residue_atom_map=residue_atom_map,
        batch_size=batch_size,
        debug_logging=debug_logging
    )

    # Process each tensor in trunk_embeddings
    for key, value in trunk_embeddings.items():
        bridged_trunk_embeddings[key] = _process_one_trunk_embedding(
            key, value, context
        )

    # Convert 'sing' to 's_inputs' if 'sing' exists and 's_inputs' doesn't
    if (
        "sing" in bridged_trunk_embeddings
        and "s_inputs" not in bridged_trunk_embeddings
    ):
        bridged_trunk_embeddings["s_inputs"] = bridged_trunk_embeddings["sing"]
        if debug_logging:
            logger.debug("Converted 'sing' to 's_inputs' in trunk_embeddings")

    return bridged_trunk_embeddings


def process_input_features(
    input_features: Dict[str, Any],
    partial_coords: torch.Tensor,
) -> Dict[str, Any]:
    """
    Process input features to ensure tensor shapes are compatible.

    Args:
        input_features: Dictionary of input features
        partial_coords: Partial coordinates tensor

    Returns:
        Dictionary of processed input features
    """
    fixed_input_features = {}

    for key, value in input_features.items():
        if not isinstance(value, torch.Tensor):
            fixed_input_features[key] = value
            continue  # Skip non-tensors

        # Handle deletion_mean shape specifically using helper
        if key == "deletion_mean":
            fixed_input_features[key] = _process_deletion_mean(value)
        else:
            # Keep other tensors as is for now
            fixed_input_features[key] = value

    # Ensure ref_pos uses the partial_coords
    fixed_input_features["ref_pos"] = partial_coords

    return fixed_input_features


def bridge_residue_to_atom(
    bridging_input: BridgingInput,
    debug_logging: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Bridge residue-level embeddings to atom-level embeddings for compatibility with Stage D.

    This function replaces the previous ad-hoc shape fixing approach with a systematic
    residue-to-atom bridging mechanism. It uses the sequence information and partial coordinates
    to derive a mapping between residues and their constituent atoms, then applies this mapping
    to expand residue-level embeddings to atom-level.

    Args:
        bridging_input: Dataclass containing input tensors and metadata.
        debug_logging: Whether to enable debug logging

    Returns:
        Tuple of (partial_coords, bridged_trunk_embeddings, fixed_input_features)
    """
    # Use local variables for clarity, extracted from input object
    partial_coords = bridging_input.partial_coords
    trunk_embeddings = bridging_input.trunk_embeddings
    input_features = (
        bridging_input.input_features
        if bridging_input.input_features is not None
        else {}
    )
    sequence = bridging_input.sequence

    # Get batch size from partial_coords
    batch_size = partial_coords.shape[0]

    # Extract sequence with fallbacks
    # Note: sequence_list will correctly handle sequence being None within extract_sequence
    sequence_list = extract_sequence(sequence, input_features, trunk_embeddings)

    # Derive the residue-to-atom mapping
    logger.info(
        f"Deriving residue-to-atom mapping for sequence of length {len(sequence_list)}"
    )
    residue_atom_map = derive_residue_atom_map(
        sequence=sequence_list,
        partial_coords=partial_coords,
        atom_metadata=input_features.get("atom_metadata"),
    )

    # Process trunk embeddings
    bridged_trunk_embeddings = process_trunk_embeddings(
        trunk_embeddings, residue_atom_map, batch_size, debug_logging
    )

    # Process input features
    fixed_input_features = process_input_features(input_features, partial_coords)

    # Return original coords, bridged embeddings, and fixed features
    return partial_coords, bridged_trunk_embeddings, fixed_input_features

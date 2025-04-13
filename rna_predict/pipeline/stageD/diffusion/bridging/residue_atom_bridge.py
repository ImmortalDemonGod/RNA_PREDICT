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
    """
    Bridges a residue-level embedding to its atom-level representation.
    
    Applies the residue-to-atom mapping to convert the input tensor into an atom-level
    embedding. If debug logging is enabled, logs the transformation detailing the change
    in tensor shape along with the provided key.
    
    Args:
        value: A tensor containing residue-level embeddings.
        residue_atom_map: A list of lists mapping residue indices to corresponding atom indices.
        key: A label for the type of embedding, used in logging.
        debug_logging: A flag indicating whether to log detailed debug information.
    
    Returns:
        A tensor representing the atom-level embedding.
    """
    bridged_value = residue_to_atoms(value, residue_atom_map)
    if debug_logging:
        logger.debug(
            f"Bridged {key} from shape {value.shape} to {bridged_value.shape} "
            f"using residue-to-atom mapping"
        )
    return bridged_value


# Helper function for processing pair embeddings
def _process_pair_embedding(value: torch.Tensor, key: str) -> torch.Tensor:
    """
    Bridge pair embeddings by logging a warning and returning the input tensor unchanged.
    
    Since pair embedding bridging is not implemented, this function logs a warning that the provided
    tensor (identified by its key) may not match expected atom counts. It then returns the original tensor,
    acting as a placeholder for a future implementation.
    
    Args:
        value (torch.Tensor): Tensor containing pair embeddings.
        key (str): Identifier for the embedding tensor.
    
    Returns:
        torch.Tensor: The unmodified input tensor.
    """
    logger.warning(
        f"Pair embedding bridging not implemented. The tensor {key} with shape {value.shape} "
        f"may cause issues in Stage D if its dimensions don't match atom counts."
    )
    return value


# Helper function to process the 'deletion_mean' tensor specifically
def _process_deletion_mean(value: torch.Tensor) -> torch.Tensor:
    """
    Adjusts the deletion_mean tensor shape to [B, N, 1].
    
    If the tensor is two-dimensional ([B, N]), it is expanded by adding a singleton dimension to form [B, N, 1]. 
    If the tensor is three-dimensional with a last dimension not equal to 1, only the first channel is retained 
    to ensure the output has the correct shape. Tensors already conforming to [B, N, 1] or with different dimensions 
    are returned unchanged.
    """
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
    """
    Bridges a trunk embedding from residue- to atom-level when applicable.
    
    If the input value is a tensor, its dimensions are normalized with respect to the batch size from the context. Based on the embedding key, the function either applies residue-to-atom bridging (for single embeddings) or processes pair embeddings, while leaving other tensor values unmodified. Non-tensor values are returned unchanged.
    
    Args:
        key: Identifier for the embedding type (e.g., "s_trunk", "s_inputs", "sing", or "pair").
        value: The trunk embedding, which may be a tensor or another type.
        context: A processing context containing the residue-to-atom mapping, batch size, and debug logging flag.
    
    Returns:
        The processed tensor with bridged representation if applicable, or the original value.
    """
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
    Bridges trunk embeddings from residue-level to atom-level.
    
    Processes each trunk embedding tensor using a residue-to-atom mapping and validates
    tensor dimensions against the specified batch size. If the key 'sing' is present without
    an accompanying 's_inputs', its value is duplicated as 's_inputs' to standardize the output.
    
    Args:
        trunk_embeddings: Dictionary mapping embedding names to residue-level tensors.
        residue_atom_map: Mapping from residue indices to lists of corresponding atom indices.
        batch_size: Batch size used for tensor dimension validation.
        debug_logging: Optional; enables detailed debug logging during processing.
    
    Returns:
        Dictionary mapping embedding names to their respective atom-level tensors.
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
    Process input features, adjusting tensor shapes and setting reference positions.
    
    This function processes a dictionary of input features by applying necessary tensor
    transformations. For tensor values under the key "deletion_mean", it adjusts the shape
    using a helper function to ensure compatibility. Non-tensor values are passed through
    unchanged. Additionally, the function assigns the provided partial coordinates to the
    "ref_pos" key to maintain consistent reference positions for downstream processing.
    
    Args:
        input_features: Dictionary of input feature values, where some may be tensors.
        partial_coords: Tensor containing partial coordinates used to update the reference position.
    
    Returns:
        A dictionary of processed input features with adjusted tensor shapes and updated "ref_pos".
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
    Bridges residue-level embeddings to atom-level representations.
    
    This function derives a residue-to-atom mapping from the provided partial coordinates and sequence
    information, then applies the mapping to convert trunk embeddings to atom-level representations.
    It also processes input features to ensure their tensor shapes are compatible with the Stage D
    diffusion process.
    
    Args:
        bridging_input: A BridgingInput instance containing partial coordinates, trunk embeddings, and optional
                        input features and sequence data.
        debug_logging: Flag to enable debug logging.
    
    Returns:
        A tuple containing:
          - the original partial coordinates,
          - a dictionary of bridged trunk embeddings,
          - a dictionary of processed input features.
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

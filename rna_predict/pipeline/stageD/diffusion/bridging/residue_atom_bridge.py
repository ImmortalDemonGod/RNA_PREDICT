"""
Residue-to-atom bridging functions for Stage D diffusion.

This module provides functions for bridging between residue-level and atom-level
representations in the Stage D diffusion process.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List
import torch
import traceback
import psutil
import os
import snoop
from rna_predict.utils.tensor_utils import derive_residue_atom_map, residue_to_atoms
from rna_predict.utils.shape_utils import adjust_tensor_feature_dim

from ..utils.tensor_utils import normalize_tensor_dimensions
from .sequence_utils import extract_sequence
from .hybrid_bridging_template import hybrid_bridging_sparse_to_dense

# Initialize logger for Stage D bridging
logger = logging.getLogger("rna_predict.pipeline.stageD.diffusion.bridging.residue_atom_bridge")


def log_mem(stage):
    process = psutil.Process(os.getpid())
    print(f"[MEMORY-LOG][BRIDGE][{stage}] Memory usage: {process.memory_info().rss / (1024 * 1024):.2f} MB")


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
    # Check for N_sample dimension (dim=1 in [B, N_sample, N_res, C])
    has_n_sample = value.dim() == 4

    if has_n_sample:
        # Handle N_sample dimension by processing each sample separately
        batch_size, n_sample, n_res, feat_dim = value.shape
        # Create output tensor with atom dimension
        n_atoms = sum(len(atoms) for atoms in residue_atom_map)
        bridged_value = torch.zeros(
            batch_size, n_sample, n_atoms, feat_dim,
            device=value.device, dtype=value.dtype
        )

        # Process each sample separately
        for i in range(n_sample):
            # Extract this sample's tensor [B, N_res, C]
            sample_tensor = value[:, i, :, :]
            # Bridge to atom level
            sample_bridged = residue_to_atoms(sample_tensor, residue_atom_map)
            # Store in the output tensor
            bridged_value[:, i, :, :] = sample_bridged
    else:
        # Standard processing for tensors without N_sample dimension
        bridged_value = residue_to_atoms(value, residue_atom_map)

    if debug_logging:
        logger.debug(
            f"Bridged {key} from shape {value.shape} to {bridged_value.shape} "
            f"using residue-to-atom mapping (has_n_sample={has_n_sample})"
        )
        # SYSTEMATIC DEBUGGING: Print tensor shapes for hypothesis testing
        print(f"[DEBUG][BRIDGE][_process_single_embedding] key={key} value.shape={value.shape} bridged_value.shape={bridged_value.shape}")
    return bridged_value


# Helper function for processing pair embeddings
# Implements bridging for pair embeddings to expand residue-level pair embeddings to atom-level pair embeddings
def _process_pair_embedding(value: torch.Tensor, residue_atom_map: list, debug_logging: bool, key: str) -> torch.Tensor:
    """
    Expands residue-level pair embeddings to atom-level pair embeddings.
    Args:
        value (torch.Tensor): Residue-level pair embeddings. Shape [B, N_sample, N_residue, N_residue, C] or [B, N_residue, N_residue, C] or [N_residue, N_residue, C]
        residue_atom_map (List[List[int]]): Mapping from residue indices to atom indices
        debug_logging (bool): Whether to log debug info
        key (str): Name of the embedding
    Returns:
        torch.Tensor: Atom-level pair embeddings with corresponding shape
    """
    if debug_logging:
        logger.debug(
            "[_process_pair_embedding] %s shape=%s",
            key,
            getattr(value, "shape", None),
        )

    # Early return for non-tensor inputs
    if not isinstance(value, torch.Tensor):
        logger.warning(f"Pair embedding for {key} is not a tensor. Returning as is.")
        return value

    n_res = len(residue_atom_map)
    # Determine atom count
    atom_indices = [atom for sublist in residue_atom_map for atom in sublist]
    n_atom = max(atom_indices) + 1 if atom_indices else 0

    # Check for N_sample dimension (dim=1 in [B, N_sample, N_res, N_res, C])
    has_n_sample = value.dim() == 5

    if has_n_sample:
        # Handle tensor with N_sample dimension [B, N_sample, N_res, N_res, C]
        B, N_sample, N_res1, N_res2, C = value.shape

        if N_res1 != n_res or N_res2 != n_res:
            logger.warning(f"Shape mismatch in pair embedding bridging for {key}: value.shape={value.shape}, n_res={n_res}")
            return value

        # Create output tensor with atom dimensions
        out = value.new_zeros((B, N_sample, n_atom, n_atom, C))

        # Process each sample separately
        for s in range(N_sample):
            # For each residue pair, copy the embedding to all corresponding atom pairs
            for i, atom_indices_i in enumerate(residue_atom_map):
                for j, atom_indices_j in enumerate(residue_atom_map):
                    # Broadcast the residue-pair embedding to all atom-pairs
                    out[:, s, atom_indices_i, :][:, :, :, atom_indices_j, :] = value[:, s, i:i+1, j:j+1, :]

        if debug_logging:
            logger.debug(f"Bridged pair embedding {key} from shape {value.shape} to {out.shape} (with N_sample dimension)")
        return out

    # Handle standard batched tensor [B, N_res, N_res, C]
    elif value.dim() == 4:
        B, N_res1, N_res2, C = value.shape

        if N_res1 != n_res or N_res2 != n_res:
            logger.warning(f"Shape mismatch in pair embedding bridging for {key}: value.shape={value.shape}, n_res={n_res}")
            return value

        # Expand to atom-level
        out = value.new_zeros((B, n_atom, n_atom, C))

        # For each residue pair, copy the embedding to all corresponding atom pairs
        for i, atom_indices_i in enumerate(residue_atom_map):
            for j, atom_indices_j in enumerate(residue_atom_map):
                out[:, atom_indices_i, :][:, :, atom_indices_j, :] = value[:, i:i+1, j:j+1, :]

        if debug_logging:
            logger.debug(f"Bridged pair embedding {key} from shape {value.shape} to {out.shape}")
        return out

    # Handle unbatched tensor [N_res, N_res, C]
    else:
        N_res1, N_res2, C = value.shape

        if N_res1 != n_res or N_res2 != n_res:
            logger.warning(f"Shape mismatch in pair embedding bridging for {key}: value.shape={value.shape}, n_res={n_res}")
            return value

        out = value.new_zeros((n_atom, n_atom, C))

        # For each residue pair, copy the embedding to all corresponding atom pairs
        for i, atom_indices_i in enumerate(residue_atom_map):
            for j, atom_indices_j in enumerate(residue_atom_map):
                out[atom_indices_i, :][:, atom_indices_j, :] = value[i:i+1, j:j+1, :]

        if debug_logging:
            logger.debug(f"Bridged pair embedding {key} from shape {value.shape} to {out.shape}")
        return out


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
    config: Any,  # Accepts either config object or DictConfig
) -> Any:  # Return type matches input 'value' or processed Tensor
    """Processes a single key-value pair from trunk_embeddings dictionary."""
    if not isinstance(value, torch.Tensor):
        # Keep non-tensor values as is
        return value

    # Normalize dimensions - check for N_sample dimension
    has_n_sample = value.dim() >= 4 and key in ["s_trunk", "s_inputs", "sing"] or value.dim() >= 5 and key in ["pair", "z_trunk"]
    temp_value = normalize_tensor_dimensions(value, context.batch_size, key=key, preserve_n_sample=has_n_sample)
    if context.debug_logging:
        logger.debug(
            "[_process_one_trunk_embedding] %s shape=%s, has_n_sample=%s",
            key,
            getattr(temp_value, "shape", None),
            has_n_sample
        )

    # CRITICAL FIX: Slice residue dimension if it exceeds residue_atom_map length
    if key in ["s_trunk", "s_inputs", "sing"] and len(context.residue_atom_map) > 0:
        expected_residue_count = len(context.residue_atom_map)
        actual_residue_count = temp_value.shape[1] if temp_value.dim() >= 2 else 0
        if actual_residue_count > expected_residue_count:
            logger.warning(
                f"[SLICING FIX] {key} residue dimension {actual_residue_count} > residue_atom_map {expected_residue_count}. Slicing tensor before bridging."
            )
            # Slice to only the valid residues
            if temp_value.dim() == 3:
                temp_value = temp_value[:, :expected_residue_count, :]
            elif temp_value.dim() == 2:
                temp_value = temp_value[:expected_residue_count, :]
            if context.debug_logging:
                logger.debug(
                    f"[_process_one_trunk_embedding] Sliced {key} to {temp_value.shape}"
                )

    # CRITICAL FIX: Check for residue dimension mismatch and reshape if needed
    if key in ["s_trunk", "s_inputs", "sing"] and len(context.residue_atom_map) > 0:
        expected_residue_count = len(context.residue_atom_map)
        actual_residue_count = temp_value.shape[1] if temp_value.dim() >= 2 else 0

        if actual_residue_count != expected_residue_count:
            logger.warning(
                f"[RESHAPE WARNING] {key} residue dimension mismatch: "
                f"tensor has {actual_residue_count} residues, but residue_atom_map has {expected_residue_count} residues. "
                f"Reshaping tensor to match residue_atom_map."
            )

            # Reshape the tensor to match the expected residue count
            if temp_value.dim() == 3:  # [batch_size, n_residue, feature_dim]
                batch_size, _, feature_dim = temp_value.shape
                # Create a new tensor with the correct shape
                reshaped_value = torch.zeros(
                    batch_size, expected_residue_count, feature_dim,
                    dtype=temp_value.dtype, device=temp_value.device
                )

                # Copy data for the common residues (up to the minimum of the two dimensions)
                min_residues = min(actual_residue_count, expected_residue_count)
                reshaped_value[:, :min_residues, :] = temp_value[:, :min_residues, :]

                # Replace the original tensor with the reshaped one
                temp_value = reshaped_value

                if context.debug_logging:
                    logger.debug(
                        f"[_process_one_trunk_embedding] Reshaped {key} to {temp_value.shape}"
                    )
            elif temp_value.dim() == 2:  # [n_residue, feature_dim]
                _, feature_dim = temp_value.shape
                # Create a new tensor with the correct shape
                reshaped_value = torch.zeros(
                    expected_residue_count, feature_dim,
                    dtype=temp_value.dtype, device=temp_value.device
                )

                # Copy data for the common residues (up to the minimum of the two dimensions)
                min_residues = min(actual_residue_count, expected_residue_count)
                reshaped_value[:min_residues, :] = temp_value[:min_residues, :]

                # Replace the original tensor with the reshaped one
                temp_value = reshaped_value

                if context.debug_logging:
                    logger.debug(
                        f"[_process_one_trunk_embedding] Reshaped {key} to {temp_value.shape}"
                    )


    # Apply feature dimension adjustment if needed
    if key in ["s_trunk", "s_inputs", "sing"]:
        # SYSTEMATIC DEBUGGING: Print config structure for debugging
        logger.debug(f"[_process_one_trunk_embedding] Processing key: {key}")
        logger.debug(f"[_process_one_trunk_embedding] Config type: {type(config)}")

        # Get expected feature dimensions from config - try all possible paths
        expected_dim = None
        feature_dimensions = None

        # Path 1: Check if config has diffusion.feature_dimensions (Hydra structure)
        if hasattr(config, 'diffusion') and hasattr(config.diffusion, 'feature_dimensions'):
            feature_dimensions = config.diffusion.feature_dimensions
            logger.debug("[_process_one_trunk_embedding] Found feature_dimensions in config.diffusion")

        # Path 2: Check if config has feature_dimensions directly
        elif hasattr(config, 'feature_dimensions'):
            feature_dimensions = config.feature_dimensions
            logger.debug("[_process_one_trunk_embedding] Found feature_dimensions directly in config")

        # Path 3: Check if config is a dict with diffusion.feature_dimensions
        elif isinstance(config, dict) and 'diffusion' in config and 'feature_dimensions' in config['diffusion']:
            feature_dimensions = config['diffusion']['feature_dimensions']
            logger.debug("[_process_one_trunk_embedding] Found feature_dimensions in config['diffusion']")

        # Path 4: Check if config is a dict with feature_dimensions directly
        elif isinstance(config, dict) and 'feature_dimensions' in config:
            feature_dimensions = config['feature_dimensions']
            logger.debug("[_process_one_trunk_embedding] Found feature_dimensions directly in config dict")

        # Path 5: Check if config has diffusion_config.diffusion.feature_dimensions
        elif hasattr(config, 'diffusion_config') and hasattr(config.diffusion_config, 'diffusion') and \
             hasattr(config.diffusion_config.diffusion, 'feature_dimensions'):
            feature_dimensions = config.diffusion_config.diffusion.feature_dimensions
            logger.debug("[_process_one_trunk_embedding] Found feature_dimensions in config.diffusion_config.diffusion")

        # Path 6: Check if config has diffusion_config.feature_dimensions
        elif hasattr(config, 'diffusion_config') and hasattr(config.diffusion_config, 'feature_dimensions'):
            feature_dimensions = config.diffusion_config.feature_dimensions
            logger.debug("[_process_one_trunk_embedding] Found feature_dimensions in config.diffusion_config")

        # Path 7: Check if config has diffusion_config dict with diffusion.feature_dimensions
        elif hasattr(config, 'diffusion_config') and isinstance(config.diffusion_config, dict) and \
             'diffusion' in config.diffusion_config and 'feature_dimensions' in config.diffusion_config['diffusion']:
            feature_dimensions = config.diffusion_config['diffusion']['feature_dimensions']
            logger.debug("[_process_one_trunk_embedding] Found feature_dimensions in config.diffusion_config['diffusion']")

        # Path 8: Check if config has diffusion_config dict with feature_dimensions
        elif hasattr(config, 'diffusion_config') and isinstance(config.diffusion_config, dict) and \
             'feature_dimensions' in config.diffusion_config:
            feature_dimensions = config.diffusion_config['feature_dimensions']
            logger.debug("[_process_one_trunk_embedding] Found feature_dimensions in config.diffusion_config dict")

        # Path 9: Check if config has model.stageD.diffusion.feature_dimensions (full Hydra path)
        elif hasattr(config, 'model') and hasattr(config.model, 'stageD') and \
             hasattr(config.model.stageD, 'diffusion') and hasattr(config.model.stageD.diffusion, 'feature_dimensions'):
            feature_dimensions = config.model.stageD.diffusion.feature_dimensions
            logger.debug("[_process_one_trunk_embedding] Found feature_dimensions in config.model.stageD.diffusion")

        # If feature_dimensions was found, extract the expected dimension
        if feature_dimensions is not None:
            # Try to get the dimension based on the key
            if key == "s_trunk":
                # Try multiple possible keys for s_trunk dimension
                expected_dim = getattr(feature_dimensions, 'c_s', None) if hasattr(feature_dimensions, 'c_s') else None
                if expected_dim is None and hasattr(feature_dimensions, 'get'):
                    expected_dim = feature_dimensions.get('c_s')
                if expected_dim is None and hasattr(feature_dimensions, 's_trunk'):
                    expected_dim = getattr(feature_dimensions, 's_trunk')
                if expected_dim is None and hasattr(feature_dimensions, 'get'):
                    expected_dim = feature_dimensions.get('s_trunk')
            elif key == "s_inputs":
                # Try multiple possible keys for s_inputs dimension
                expected_dim = getattr(feature_dimensions, 'c_s_inputs', None) if hasattr(feature_dimensions, 'c_s_inputs') else None
                if expected_dim is None and hasattr(feature_dimensions, 'get'):
                    expected_dim = feature_dimensions.get('c_s_inputs')
                if expected_dim is None and hasattr(feature_dimensions, 's_inputs'):
                    expected_dim = getattr(feature_dimensions, 's_inputs')
                if expected_dim is None and hasattr(feature_dimensions, 'get'):
                    expected_dim = feature_dimensions.get('s_inputs')
            elif key == "sing":
                # Try multiple possible keys for sing dimension
                expected_dim = getattr(feature_dimensions, 'c_sing', None) if hasattr(feature_dimensions, 'c_sing') else None
                if expected_dim is None and hasattr(feature_dimensions, 'get'):
                    expected_dim = feature_dimensions.get('c_sing')
                if expected_dim is None and hasattr(feature_dimensions, 'sing'):
                    expected_dim = getattr(feature_dimensions, 'sing')
                if expected_dim is None and hasattr(feature_dimensions, 'get'):
                    expected_dim = feature_dimensions.get('sing')

            # Log the found dimension
            logger.debug(f"[_process_one_trunk_embedding] For key {key}, found expected_dim={expected_dim}")

        # Raise error if expected dimension is missing
        if expected_dim is None:
            # As a last resort, use hardcoded values based on the Hydra config
            if key == "s_trunk":
                expected_dim = 384  # Default from stageD_diffusion.yaml
                logger.warning(f"[_process_one_trunk_embedding] Using hardcoded value {expected_dim} for {key}")
            elif key == "s_inputs":
                expected_dim = 449  # Default from stageD_diffusion.yaml
                logger.warning(f"[_process_one_trunk_embedding] Using hardcoded value {expected_dim} for {key}")
            elif key == "sing":
                expected_dim = 384  # Default from stageD_diffusion.yaml
                logger.warning(f"[_process_one_trunk_embedding] Using hardcoded value {expected_dim} for {key}")
            else:
                # If we still don't have a value, raise an error with detailed information
                raise ValueError(
                    f"[BRIDGE ERROR][CONFIG] Missing expected feature dimension for {key} in config. "
                    "Please ensure 'feature_dimensions' is properly configured in your Hydra config. "
                    "Check model.stageD.diffusion.feature_dimensions in your configuration."
                )

        # Only adjust if the actual dimension doesn't match
        if temp_value.shape[-1] != expected_dim:
            temp_value = adjust_tensor_feature_dim(temp_value, expected_dim, key)
            if context.debug_logging:
                logger.debug(f"Adjusted {key} feature dimension to {expected_dim}. New shape: {temp_value.shape}")

    # Apply bridging based on tensor type using helper functions
    if key in ["s_trunk", "s_inputs", "sing"]:
        processed_value = _process_single_embedding(
            temp_value, context.residue_atom_map, key, context.debug_logging
        )
    elif key in ["pair", "z_trunk"]:
        if context.debug_logging:
            logger.debug(
                "[_process_one_trunk_embedding] %s shape=%s",
                key,
                getattr(temp_value, "shape", None),
            )
        processed_value = _process_pair_embedding(temp_value, context.residue_atom_map, context.debug_logging, key)
    else:
        # Keep other tensors as is (no bridging needed)
        processed_value = temp_value

    return processed_value


def process_trunk_embeddings(
    trunk_embeddings: Dict[str, torch.Tensor],
    residue_atom_map: List[List[int]],
    batch_size: int,
    debug_logging: bool = False,
    config: Any = None,  # Accepts either config object or DictConfig
) -> Dict[str, torch.Tensor]:
    """
    Process trunk embeddings to bridge residue-level to atom-level representations.

    Args:
        trunk_embeddings: Dictionary of trunk embeddings (residue-level)
        residue_atom_map: Mapping from residue indices to atom indices
        batch_size: Batch size for tensor dimension validation
        debug_logging: Whether to enable debug logging
        config: Configuration object or DictConfig

    Returns:
        Dictionary of bridged trunk embeddings
    """
    print("[DEBUG] process_trunk_embeddings: trunk_embeddings keys=", list(trunk_embeddings.keys()))
    for key, value in trunk_embeddings.items():
        print(f"[DEBUG] process_trunk_embeddings: key={key}, value.shape={getattr(value, 'shape', None)}")
    print("[DEBUG] process_trunk_embeddings: residue_atom_map length=", len(residue_atom_map))
    bridged_trunk_embeddings = {}

    # Create processing context
    context = EmbeddingProcessContext(
        residue_atom_map=residue_atom_map,
        batch_size=batch_size,
        debug_logging=debug_logging
    )

    # Before processing trunk embeddings, log debug info
    if debug_logging:
        logger.debug(f"[DEBUG][StageD] residue_atom_map shape: {getattr(residue_atom_map, 'shape', type(residue_atom_map))}")
        logger.debug(f"[DEBUG][StageD] residue_atom_map (first 10): {residue_atom_map[:10] if hasattr(residue_atom_map, '__getitem__') else residue_atom_map}")
        if 's_emb' in trunk_embeddings:
            s_emb = trunk_embeddings['s_emb']
            logger.debug(f"[DEBUG][StageD] s_emb shape: {getattr(s_emb, 'shape', type(s_emb))}")
            if hasattr(s_emb, 'shape') and len(s_emb.shape) > 1:
                logger.debug(f"[DEBUG][StageD] s_emb (first residue): {s_emb[0]}")
        atom_metadata = trunk_embeddings.get('atom_metadata')
        if atom_metadata is not None and 'residue_indices' in atom_metadata:
            logger.debug(f"[DEBUG][StageD] atom_metadata['residue_indices'] (len): {len(atom_metadata['residue_indices'])}")
            logger.debug(f"[DEBUG][StageD] atom_metadata['residue_indices'] (first 10): {atom_metadata['residue_indices'][:10]}")

    # Process each tensor in trunk_embeddings
    for key, value in trunk_embeddings.items():
        bridged_trunk_embeddings[key] = _process_one_trunk_embedding(
            key, value, context, config
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
    input_features: Dict[str, Any] | None,
    partial_coords: torch.Tensor,
    residue_atom_map: List[List[int]],
    batch_size: int,
) -> Dict[str, Any]:
    """
    Process input features to ensure tensor shapes are compatible.

    Args:
        input_features: Dictionary of input features
        partial_coords: Partial coordinates tensor
        residue_atom_map: Mapping from residue indices to atom indices
        batch_size: Batch size for tensor dimension validation

    Returns:
        Dictionary of processed input features
    """
    fixed_input_features = {}

    # Handle None input_features
    if input_features is None:
        input_features = {}

    # Import torch here to avoid UnboundLocalError
    import torch

    # Calculate total number of atoms and residues
    n_atoms = sum(len(atoms) for atoms in residue_atom_map)
    n_residues = len(residue_atom_map)

    # Identify residue-level tensors that need to be expanded to atom-level
    residue_level_keys = ["ref_charge", "ref_element", "ref_atom_name_chars", "ref_mask", "restype", "profile", "deletion_mean"]

    for key, value in input_features.items():
        if not isinstance(value, torch.Tensor):
            fixed_input_features[key] = value
            continue  # Skip non-tensors

        # Handle deletion_mean shape specifically using helper
        if key == "deletion_mean":
            value = _process_deletion_mean(value)

        # Check if this is a residue-level tensor that needs expansion
        if key in residue_level_keys and value.shape[1] == n_residues:
            # This is a residue-level tensor that needs to be expanded to atom-level
            print(f"[DEBUG][BRIDGE] Expanding residue-level tensor {key} from shape {value.shape} to atom-level")
            B, _, *rest_dims = value.shape
            rest_shape = tuple(rest_dims)

            # Create a new tensor with atom-level dimensions
            expanded_value = torch.zeros((B, n_atoms, *rest_shape), device=value.device, dtype=value.dtype)

            # Fill in the expanded tensor by repeating each residue's values for its atoms
            atom_counter = 0
            for res_idx, atom_indices in enumerate(residue_atom_map):
                n_atoms_in_res = len(atom_indices)
                if n_atoms_in_res == 0:
                    continue

                # Expand the residue's values to all its atoms
                if len(rest_shape) == 0:  # No extra dimensions
                    expanded_value[:, atom_counter:atom_counter+n_atoms_in_res] = value[:, res_idx:res_idx+1].expand(B, n_atoms_in_res)
                else:  # Has extra dimensions
                    expanded_value[:, atom_counter:atom_counter+n_atoms_in_res] = value[:, res_idx:res_idx+1].expand(B, n_atoms_in_res, *rest_shape)

                atom_counter += n_atoms_in_res

            fixed_input_features[key] = expanded_value
            print(f"[DEBUG][BRIDGE] Expanded {key} to shape {expanded_value.shape}")
        else:
            # Keep other tensors as is for now
            fixed_input_features[key] = value

    # Ensure ref_pos uses the partial_coords
    fixed_input_features["ref_pos"] = partial_coords

    # CRITICAL FIX: Create atom_to_token_idx mapping from residue_atom_map
    # This is essential for the diffusion model to correctly map between atom and residue representations
    # Always recreate atom_to_token_idx to ensure it matches the current residue_atom_map
    # Create a tensor that maps each atom to its corresponding residue index
    total_atoms = sum(len(atoms) for atoms in residue_atom_map)
    residue_count = len(residue_atom_map)
    print(f"[DEBUG][BRIDGE] residue_atom_map lens={[len(x) for x in residue_atom_map]} total_atoms={total_atoms} residue_count={residue_count}")
    atom_to_token_idx = torch.zeros(
        batch_size,
        total_atoms,
        dtype=torch.long,
        device=partial_coords.device,
    )

    # Fill in the mapping
    for residue_idx, atom_indices in enumerate(residue_atom_map):
        for atom_idx in atom_indices:
            atom_to_token_idx[:, atom_idx] = residue_idx

    # DEBUG: Print atom_to_token_idx shape after construction
    print(f"[DEBUG][BRIDGE] atom_to_token_idx.shape={atom_to_token_idx.shape}")

    # UNCONDITIONAL DEBUG: Print all keys and their shapes in fixed_input_features
    print("[DEBUG][BRIDGE] fixed_input_features keys and shapes:")
    for k, v in fixed_input_features.items():
        print(f"[DEBUG][BRIDGE]   {k}: shape={getattr(v, 'shape', 'N/A')} type={type(v)}")

    # Add to input features - always overwrite with the correct mapping
    fixed_input_features["atom_to_token_idx"] = atom_to_token_idx
    logger.info(f"Created atom_to_token_idx mapping with shape {atom_to_token_idx.shape}")

    return fixed_input_features


@snoop
def bridge_residue_to_atom(
    bridging_input: BridgingInput,
    config: Any,  # Accepts either config object or DictConfig
    debug_logging: bool = False,
):
    log_mem("ENTRY")
    # --- CONFIG VALIDATION PATCH: Ensure feature_dimensions can be found in the config ---
    print(f"[DEBUG][BRIDGE][CONFIG STRUCTURE] config type: {type(config)}; keys: {list(config.keys()) if hasattr(config, 'keys') else dir(config)}")
    feature_dimensions = None
    # Try to get feature_dimensions from config (robust to both dict and OmegaConf)
    # Path 1: Check if config has diffusion.feature_dimensions (Hydra structure)
    if hasattr(config, 'diffusion') and hasattr(config.diffusion, 'feature_dimensions'):
        feature_dimensions = config.diffusion.feature_dimensions
        logger.debug("[bridge_residue_to_atom] Found feature_dimensions in config.diffusion")

    # Path 2: Check if config has feature_dimensions directly
    elif hasattr(config, 'feature_dimensions'):
        feature_dimensions = config.feature_dimensions
        logger.debug("[bridge_residue_to_atom] Found feature_dimensions directly in config")

    # Path 3: Check if config is a dict with diffusion.feature_dimensions
    elif isinstance(config, dict) and 'diffusion' in config and 'feature_dimensions' in config['diffusion']:
        feature_dimensions = config['diffusion']['feature_dimensions']
        logger.debug("[bridge_residue_to_atom] Found feature_dimensions in config['diffusion']")

    # Path 4: Check if config is a dict with feature_dimensions directly
    elif isinstance(config, dict) and 'feature_dimensions' in config:
        feature_dimensions = config['feature_dimensions']
        logger.debug("[bridge_residue_to_atom] Found feature_dimensions directly in config dict")

    # Path 5: Check if config has model.stageD.diffusion.feature_dimensions (full Hydra path)
    elif hasattr(config, 'model') and hasattr(config.model, 'stageD') and \
            hasattr(config.model.stageD, 'diffusion') and hasattr(config.model.stageD.diffusion, 'feature_dimensions'):
        feature_dimensions = config.model.stageD.diffusion.feature_dimensions
        logger.debug("[bridge_residue_to_atom] Found feature_dimensions in config.model.stageD.diffusion")

    # If we still don't have feature_dimensions, raise a ValueError
    if feature_dimensions is None:
        logger.error(
            "[BRIDGE ERROR][CONFIG] Could not find feature_dimensions in config. "
            "This is a required configuration section. Please check your Hydra configuration.")
        raise ValueError(
            "Configuration missing required 'feature_dimensions' section. "
            "Please ensure this is properly configured in your Hydra config.")
    # Ensure 's_inputs' is present in feature_dimensions
    s_inputs_found = False
    if hasattr(feature_dimensions, 's_inputs'):
        s_inputs_found = True
    elif hasattr(feature_dimensions, 'get') and feature_dimensions.get('s_inputs') is not None:
        s_inputs_found = True
    elif hasattr(feature_dimensions, 'c_s_inputs'):
        s_inputs_found = True
    elif hasattr(feature_dimensions, 'get') and feature_dimensions.get('c_s_inputs') is not None:
        s_inputs_found = True

    if not s_inputs_found:
        logger.error(
            "[BRIDGE ERROR][CONFIG] 's_inputs' missing from feature_dimensions. "
            "This is a required field.")
        raise ValueError(
            "[BRIDGE ERROR][CONFIG] 's_inputs' missing from Stage D diffusion feature_dimensions config. "
            "This is a required field. Please check your Hydra config for model.stageD.diffusion.feature_dimensions.s_inputs.")
    # --- END PATCH ---
    # --- PATCH: Guard against double-bridging or atom-level input (moved to top, robust) ---
    trunk_embeddings = bridging_input.trunk_embeddings
    sequence = bridging_input.sequence
    input_features = bridging_input.input_features or {}
    atom_metadata = input_features.get("atom_metadata") if input_features else None
    residue_count = None
    if sequence is not None:
        residue_count = len(sequence)
    elif atom_metadata and "residue_indices" in atom_metadata:
        residue_count = len(set(atom_metadata["residue_indices"]))
    if residue_count is None:
        raise ValueError(
            "[BRIDGE ERROR][UNIQUE_CODE_002] Cannot determine residue count for bridging. Provide sequence or atom_metadata."
        )
    if trunk_embeddings.get("s_trunk") is not None:
        s_emb = trunk_embeddings["s_trunk"]
        if s_emb.shape[1] != residue_count:
            raise ValueError(
                f"[BRIDGE ERROR][UNIQUE_CODE_001] s_emb.shape[1] = {s_emb.shape[1]} does not match residue count ({residue_count}). "
                "This likely means atom-level embeddings were passed to the bridging function, which expects residue-level. "
                "Check the pipeline for double-bridging or misrouted tensors."
            )
    # SYSTEMATIC DEBUGGING: Log sequence and mapping lengths
    sequence = bridging_input.sequence
    trunk_embeddings = bridging_input.trunk_embeddings
    input_features = bridging_input.input_features or {}
    partial_coords = bridging_input.partial_coords
    if debug_logging:
        logger.debug(f"[bridge_residue_to_atom] sequence: {sequence}")
        logger.debug(f"[bridge_residue_to_atom] sequence length: {len(sequence) if sequence is not None else 'None'}")
        logger.debug(f"[bridge_residue_to_atom] trunk_embeddings keys: {list(trunk_embeddings.keys())}")
        for k, v in trunk_embeddings.items():
            if v is not None:
                logger.debug(f"[bridge_residue_to_atom] trunk_embeddings[{k}].shape: {v.shape}")
            else:
                logger.warning(f"[bridge_residue_to_atom] trunk_embeddings[{k}] is None!")
        logger.debug(f"[bridge_residue_to_atom] atom_metadata: {input_features.get('atom_metadata') if input_features else None}")
    # Instrumentation for systematic debugging
    if debug_logging:
        logger.debug("[BRIDGE DEBUG] bridge_residue_to_atom called")
        logger.debug(f"[BRIDGE DEBUG] sequence: {sequence if 'sequence' in locals() else 'N/A'}")
        logger.debug(f"[BRIDGE DEBUG] trunk_embeddings keys: {list(trunk_embeddings.keys()) if 'trunk_embeddings' in locals() else 'N/A'}")
        for k, v in (trunk_embeddings.items() if 'trunk_embeddings' in locals() else []):
            if v is not None:
                logger.debug(f"[BRIDGE DEBUG] trunk_embeddings[{k}].shape: {v.shape}")
            else:
                logger.warning(f"[BRIDGE DEBUG] trunk_embeddings[{k}] is None!")
        logger.debug("(stack trace omitted for performance)")
    # === SYSTEMATIC DEBUG: Log trunk_embeddings keys at entry ===
    print(f"[BRIDGE DEBUG] Input trunk_embeddings keys: {list(trunk_embeddings.keys())}")
    if "s_trunk" in trunk_embeddings:
        print(f"[BRIDGE DEBUG] s_trunk present, shape: {getattr(trunk_embeddings['s_trunk'], 'shape', 'N/A')}")
    else:
        print("[BRIDGE DEBUG] s_trunk MISSING in input trunk_embeddings!")
    # --- original code continues ---
    sequence_list = extract_sequence(sequence, input_features, trunk_embeddings)
    # PATCH: SYSTEMATIC DEBUGGING FOR ATOM COUNT CONSISTENCY
    # Print atom_metadata, atom_names, residue_indices, and partial_coords shape if available
    if atom_metadata is not None:
        atom_names = atom_metadata.get("atom_names", [])
        residue_indices = atom_metadata.get("residue_indices", [])
        print(f"[BRIDGE DEBUG] atom_metadata present: len(atom_names)={len(atom_names)}, len(residue_indices)={len(residue_indices)}")
    if bridging_input.partial_coords is not None:
        print(f"[BRIDGE DEBUG] partial_coords.shape={bridging_input.partial_coords.shape}")

    # --- PATCH: Always derive residue_atom_map from atom_metadata if available ---
    from rna_predict.utils.tensor_utils import derive_residue_atom_map
    # Use atom_metadata if present, else fallback
    residue_atom_map = derive_residue_atom_map(
        sequence=sequence,
        partial_coords=bridging_input.partial_coords,
        atom_metadata=atom_metadata
    )
    n_atoms_from_map = sum(len(x) for x in residue_atom_map)
    print(f"[BRIDGE DEBUG] residue_atom_map: {len(residue_atom_map)} residues, total atoms from map: {n_atoms_from_map}")
    # Defensive: If atom_names or residue_indices present, check length matches n_atoms_from_map
    if atom_metadata is not None:
        if "atom_names" in atom_metadata and len(atom_metadata["atom_names"]) != n_atoms_from_map:
            raise ValueError(f"[BRIDGE ERROR] atom_names length ({len(atom_metadata['atom_names'])}) does not match total atoms from residue_atom_map ({n_atoms_from_map})")
        if "residue_indices" in atom_metadata and len(atom_metadata["residue_indices"]) != n_atoms_from_map:
            raise ValueError(f"[BRIDGE ERROR] residue_indices length ({len(atom_metadata['residue_indices'])}) does not match total atoms from residue_atom_map ({n_atoms_from_map})")
    # Defensive: If partial_coords present, check shape matches
    if bridging_input.partial_coords is not None:
        n_atoms_coords = bridging_input.partial_coords.shape[-2]
        if n_atoms_coords != n_atoms_from_map:
            raise ValueError(f"[BRIDGE ERROR] partial_coords.shape[-2] ({n_atoms_coords}) does not match total atoms from residue_atom_map ({n_atoms_from_map})")
    # Continue with the rest of the function as before
    log_mem("After residue-to-atom mapping")
    if debug_logging:
        logger.debug(f"[bridge_residue_to_atom] residue_atom_map length: {len(residue_atom_map)}")
        logger.debug(f"[bridge_residue_to_atom] residue_atom_map: {residue_atom_map}")
    # --- PATCH: Config-driven max_len for bridging ---
    from omegaconf import DictConfig
    cfg = None
    # Accept both DictConfig and dataclass configs
    if isinstance(config, DictConfig):
        cfg = config
        # OmegaConf: attribute or key access
        try:
            seq_len = cfg.model.stageD.diffusion.test_residues_per_batch
        except Exception:
            seq_len = 25
    else:
        # Dataclass or custom config: attribute access
        try:
            seq_len = config.model.stageD.diffusion.test_residues_per_batch
        except Exception:
            seq_len = 25
    max_len = seq_len
    if debug_logging:
        logger.debug(f"[bridge_residue_to_atom] apply_memory_preprocess={False}, max_len={max_len}")

    # Keep original trunk_embeddings
    trunk_embeddings = bridging_input.trunk_embeddings
    # Process trunk embeddings
    batch_size = partial_coords.shape[0]
    bridged_trunk_embeddings = process_trunk_embeddings(
        trunk_embeddings, residue_atom_map, batch_size, debug_logging, config
    )
    # Process input features
    fixed_input_features = process_input_features(input_features, partial_coords, residue_atom_map, batch_size)
    # SYSTEMATIC DEBUGGING: Print all keys and check for atom_to_token_idx
    print("[DEBUG][BRIDGE][bridge_residue_to_atom] fixed_input_features keys:", list(fixed_input_features.keys()))
    if "atom_to_token_idx" in fixed_input_features:
        print("[DEBUG][BRIDGE][bridge_residue_to_atom] atom_to_token_idx shape:", getattr(fixed_input_features["atom_to_token_idx"], "shape", type(fixed_input_features["atom_to_token_idx"])))
    else:
        print("[ERROR][BRIDGE][bridge_residue_to_atom] atom_to_token_idx is MISSING in fixed_input_features!")
    output_embeddings = {"trunk_embeddings": bridged_trunk_embeddings, "input_features": fixed_input_features}
    # === SYSTEMATIC DEBUG: Log output keys ===
    print(f"[BRIDGE DEBUG] Output embeddings keys: {list(output_embeddings.keys())}")
    if "s_trunk" in output_embeddings["trunk_embeddings"]:
        print(f"[BRIDGE DEBUG] s_trunk present in output, shape: {getattr(output_embeddings['trunk_embeddings']['s_trunk'], 'shape', 'N/A')}")
    else:
        print("[BRIDGE DEBUG] s_trunk MISSING in output embeddings!")
    # --- PATCH: Always propagate s_trunk if present in input ---
    if "s_trunk" not in trunk_embeddings and "s_inputs" in trunk_embeddings:
        print("[BRIDGE PATCH] s_trunk missing, only s_inputs present. Downstream code may fail if s_trunk is required.")
    # If s_trunk is present in input, propagate to output (replicate to atom-level if necessary)
    if "s_trunk" in trunk_embeddings:
        s_trunk_value = trunk_embeddings["s_trunk"]
        if s_trunk_value.shape[1] == residue_count:
            # Residue-level: propagate as is
            output_embeddings["trunk_embeddings"]["s_trunk"] = s_trunk_value
            print(f"[BRIDGE PATCH] Propagated residue-level s_trunk shape: {s_trunk_value.shape}")
        else:
            # Atom-level: log warning, propagate as is
            output_embeddings["trunk_embeddings"]["s_trunk"] = s_trunk_value
            print(f"[BRIDGE PATCH] Propagated atom-level s_trunk shape: {s_trunk_value.shape}")
    elif "s_inputs" in trunk_embeddings:
        s_inputs_value = trunk_embeddings["s_inputs"]
        import torch
        B, n_second_dim, c = s_inputs_value.shape
        n_residues = len(residue_atom_map)
        device = s_inputs_value.device
        # PATCH: handle both residue-level and atom-level s_inputs
        if n_second_dim == n_residues:
            # s_inputs is residue-level: expand to atom-level first
            n_atoms = sum(len(atom_indices) for atom_indices in residue_atom_map)
            # Build [B, n_atoms, c] by repeating each residue embedding for its atoms
            s_inputs_atom = torch.zeros((B, n_atoms, c), device=device, dtype=s_inputs_value.dtype)
            atom_counter = 0
            for res_idx, atom_indices in enumerate(residue_atom_map):
                n_atoms_in_res = len(atom_indices)
                if n_atoms_in_res == 0:
                    continue
                s_inputs_atom[:, atom_counter:atom_counter+n_atoms_in_res, :] = s_inputs_value[:, res_idx:res_idx+1, :].expand(B, n_atoms_in_res, c)
                atom_counter += n_atoms_in_res
            # Now average over atoms for each residue (which just gives back the original s_inputs_value)
            s_trunk_residue = s_inputs_value
            print(f"[BRIDGE PATCH] s_inputs was residue-level, expanded to atom-level for mapping. s_inputs_atom.shape={s_inputs_atom.shape}")
        else:
            # s_inputs is atom-level: average over atoms for each residue
            s_trunk_residue = torch.zeros((B, n_residues, c), device=device, dtype=s_inputs_value.dtype)
            for res_idx, atom_indices in enumerate(residue_atom_map):
                if len(atom_indices) == 0:
                    continue
                atom_indices_tensor = torch.tensor(atom_indices, device=device)
                s_trunk_residue[:, res_idx, :] = s_inputs_value.index_select(1, atom_indices_tensor).mean(dim=1)
            print(f"[BRIDGE PATCH] s_inputs was atom-level, averaged to residue-level. s_trunk_residue.shape={s_trunk_residue.shape}")
        output_embeddings["trunk_embeddings"]["s_trunk"] = s_trunk_residue
        print(f"[BRIDGE PATCH] Set s_trunk in output_embeddings: {s_trunk_residue.shape}")
    else:
        print("[BRIDGE PATCH] s_trunk and s_inputs both missing in input trunk_embeddings!")
    # --- Ensure s_trunk and s_inputs are both atom-level for Stage D diffusion ---
    # SYSTEMATIC DEBUG: Print initial shapes
    s_trunk = output_embeddings["trunk_embeddings"].get("s_trunk")
    s_inputs = output_embeddings["trunk_embeddings"].get("s_inputs")
    # Import torch here to avoid UnboundLocalError
    import torch

    residue_atom_map = derive_residue_atom_map(
        sequence,
        partial_coords=partial_coords,
        atom_metadata=input_features.get("atom_metadata") if input_features else None,
    )
    n_atoms = sum(len(atom_indices) for atom_indices in residue_atom_map)
    n_residues = len(residue_atom_map)
    # Expand s_trunk to atom-level if needed
    if s_trunk is not None:
        if s_trunk.shape[1] == n_residues:
            # Expand residue-level s_trunk to atom-level
            B, _, c = s_trunk.shape
            device = s_trunk.device
            s_trunk_atom = torch.zeros((B, n_atoms, c), device=device, dtype=s_trunk.dtype)
            atom_counter = 0
            for res_idx, atom_indices in enumerate(residue_atom_map):
                n_atoms_in_res = len(atom_indices)
                if n_atoms_in_res == 0:
                    continue
                s_trunk_atom[:, atom_counter:atom_counter+n_atoms_in_res, :] = s_trunk[:, res_idx:res_idx+1, :].expand(B, n_atoms_in_res, c)
                atom_counter += n_atoms_in_res
            output_embeddings["trunk_embeddings"]["s_trunk"] = s_trunk_atom
            print(f"[BRIDGE PATCH][ATOM-LEVEL] s_trunk expanded to atom-level: {s_trunk_atom.shape}")
        elif s_trunk.shape[1] == n_atoms:
            print(f"[BRIDGE PATCH][ATOM-LEVEL] s_trunk already atom-level: {s_trunk.shape}")
        else:
            print(f"[BRIDGE PATCH][ATOM-LEVEL][ERROR] s_trunk shape unexpected: {s_trunk.shape}, expected {n_residues} or {n_atoms} in dim 1")
    else:
        print("[BRIDGE PATCH][ATOM-LEVEL][ERROR] s_trunk missing from trunk_embeddings!")
    # Expand s_inputs to atom-level if needed (should already be handled, but double-check)
    if s_inputs is not None:
        if s_inputs.shape[1] == n_residues:
            # Expand residue-level s_inputs to atom-level
            B, _, c = s_inputs.shape
            device = s_inputs.device
            # Use the torch import from above
            s_inputs_atom = torch.zeros((B, n_atoms, c), device=device, dtype=s_inputs.dtype)
            atom_counter = 0
            for res_idx, atom_indices in enumerate(residue_atom_map):
                n_atoms_in_res = len(atom_indices)
                if n_atoms_in_res == 0:
                    continue
                s_inputs_atom[:, atom_counter:atom_counter+n_atoms_in_res, :] = s_inputs[:, res_idx:res_idx+1, :].expand(B, n_atoms_in_res, c)
                atom_counter += n_atoms_in_res
            output_embeddings["trunk_embeddings"]["s_inputs"] = s_inputs_atom
            print(f"[BRIDGE PATCH][ATOM-LEVEL] s_inputs expanded to atom-level: {s_inputs_atom.shape}")
        elif s_inputs.shape[1] == n_atoms:
            print(f"[BRIDGE PATCH][ATOM-LEVEL] s_inputs already atom-level: {s_inputs.shape}")
        else:
            print(f"[BRIDGE PATCH][ATOM-LEVEL][ERROR] s_inputs shape unexpected: {s_inputs.shape}, expected {n_residues} or {n_atoms} in dim 1")
    else:
        print("[BRIDGE PATCH][ATOM-LEVEL][ERROR] s_inputs missing from trunk_embeddings!")
    # SYSTEMATIC DEBUG: Print shapes before returning
    print(f"[BRIDGE PATCH][ATOM-LEVEL][FINAL] s_trunk: {output_embeddings['trunk_embeddings'].get('s_trunk', None).shape if output_embeddings['trunk_embeddings'].get('s_trunk', None) is not None else 'MISSING'}")
    print(f"[BRIDGE PATCH][ATOM-LEVEL][FINAL] s_inputs: {output_embeddings['trunk_embeddings'].get('s_inputs', None).shape if output_embeddings['trunk_embeddings'].get('s_inputs', None) is not None else 'MISSING'}")
    return partial_coords, output_embeddings["trunk_embeddings"], output_embeddings["input_features"]

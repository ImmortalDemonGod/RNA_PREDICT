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
    bridged_value = residue_to_atoms(value, residue_atom_map)
    if debug_logging:
        logger.debug(
            f"Bridged {key} from shape {value.shape} to {bridged_value.shape} "
            f"using residue-to-atom mapping"
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
        value (torch.Tensor): Residue-level pair embeddings. Shape [B, N_residue, N_residue, C] or [N_residue, N_residue, C]
        residue_atom_map (List[List[int]]): Mapping from residue indices to atom indices
        debug_logging (bool): Whether to log debug info
        key (str): Name of the embedding
    Returns:
        torch.Tensor: Atom-level pair embeddings. Shape [B, N_atom, N_atom, C] or [N_atom, N_atom, C]
    """
    import torch
    if debug_logging:
        logger.debug(
            "[_process_pair_embedding] %s shape=%s",
            key,
            getattr(value, "shape", None),
        )
    # Validate input
    if not isinstance(value, torch.Tensor):
        logger.warning(f"Pair embedding for {key} is not a tensor. Returning as is.")
        return value
    n_res = len(residue_atom_map)
    # Determine atom count
    atom_indices = [atom for sublist in residue_atom_map for atom in sublist]
    n_atom = max(atom_indices) + 1 if atom_indices else 0
    # Handle batch dim
    is_batched = value.dim() == 4
    if is_batched:
        B, N_res1, N_res2, C = value.shape
        if N_res1 != n_res or N_res2 != n_res:
            logger.warning(f"Shape mismatch in pair embedding bridging for {key}: value.shape={value.shape}, n_res={n_res}")
            return value
        # Expand to atom-level
        out = value.new_zeros((B, n_atom, n_atom, C))
        for i, atom_indices_i in enumerate(residue_atom_map):
            for j, atom_indices_j in enumerate(residue_atom_map):
                out[:, atom_indices_i, :][:, :, atom_indices_j, :] = value[:, i:i+1, j:j+1, :]
        if debug_logging:
            logger.debug(f"Bridged pair embedding {key} from shape {value.shape} to {out.shape}")
        return out
    else:
        N_res1, N_res2, C = value.shape
        if N_res1 != n_res or N_res2 != n_res:
            logger.warning(f"Shape mismatch in pair embedding bridging for {key}: value.shape={value.shape}, n_res={n_res}")
            return value
        out = value.new_zeros((n_atom, n_atom, C))
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

    # Normalize dimensions
    temp_value = normalize_tensor_dimensions(value, context.batch_size, key=key)
    if context.debug_logging:
        logger.debug(
            "[_process_one_trunk_embedding] %s shape=%s",
            key,
            getattr(temp_value, "shape", None),
        )

    # Apply feature dimension adjustment if needed
    if key in ["s_trunk", "s_inputs", "sing"]:
        # SYSTEMATIC DEBUGGING: Print diffusion_config structure FIRST
        if hasattr(config, 'diffusion_config'):
            print(f"[DEBUG][_process_one_trunk_embedding] diffusion_config type: {type(config.diffusion_config)}")
            print(f"[DEBUG][_process_one_trunk_embedding] diffusion_config dir: {dir(config.diffusion_config)}")
            if hasattr(config.diffusion_config, 'keys'):
                try:
                    print(f"[DEBUG][_process_one_trunk_embedding] diffusion_config.keys(): {list(config.diffusion_config.keys())}")
                except Exception as e:
                    print(f"[DEBUG][_process_one_trunk_embedding] diffusion_config.keys() error: {e}")
            else:
                print("[DEBUG][_process_one_trunk_embedding] diffusion_config has no 'keys' attribute")
            if hasattr(config.diffusion_config, 'feature_dimensions'):
                print(f"[DEBUG][_process_one_trunk_embedding] diffusion_config.feature_dimensions: {config.diffusion_config.feature_dimensions}")
        # Get expected feature dimensions from config
        expected_dim = None
        # PATCH: handle nested feature_dimensions in config.diffusion_config['diffusion']
        if hasattr(config, 'diffusion_config') and isinstance(config.diffusion_config, dict):
            if 'diffusion' in config.diffusion_config and \
               'feature_dimensions' in config.diffusion_config['diffusion']:
                feature_dimensions = config.diffusion_config['diffusion']['feature_dimensions']
                print(f"[DEBUG][_process_one_trunk_embedding] feature_dimensions (from diffusion_config['diffusion']): {feature_dimensions}")
                if key == "s_trunk":
                    expected_dim = feature_dimensions.get('c_s')
                elif key == "s_inputs":
                    expected_dim = feature_dimensions.get('c_s_inputs')
                elif key == "sing":
                    expected_dim = feature_dimensions.get('c_sing')
            elif 'feature_dimensions' in config.diffusion_config:
                feature_dimensions = config.diffusion_config['feature_dimensions']
                print(f"[DEBUG][_process_one_trunk_embedding] feature_dimensions (from diffusion_config): {feature_dimensions}")
                if key == "s_trunk":
                    expected_dim = feature_dimensions.get('c_s')
                elif key == "s_inputs":
                    expected_dim = feature_dimensions.get('c_s_inputs')
                elif key == "sing":
                    expected_dim = feature_dimensions.get('c_sing')
        elif hasattr(config, 'feature_dimensions'):
            feature_dimensions = config.feature_dimensions
            print(f"[DEBUG][_process_one_trunk_embedding] feature_dimensions (from config): {feature_dimensions}")
            if key == "s_trunk":
                expected_dim = feature_dimensions.get('c_s')
            elif key == "s_inputs":
                expected_dim = feature_dimensions.get('c_s_inputs')
            elif key == "sing":
                expected_dim = feature_dimensions.get('c_sing')

        # Raise error if expected dimension is missing
        if expected_dim is None:
            raise ValueError(
                f"Missing expected feature dimension for {key} in config. "
                "Please ensure 'feature_dimensions' is properly configured."
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

    # CRITICAL FIX: Create atom_to_token_idx mapping from residue_atom_map
    # This is essential for the diffusion model to correctly map between atom and residue representations
    if "atom_to_token_idx" not in fixed_input_features:
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

        # Attempt to find an atom-level tensor for alignment
        n_valid_atoms = None
        c_l_tensor = fixed_input_features.get('c_l', None)
        if c_l_tensor is not None:
            n_valid_atoms = c_l_tensor.shape[1]
            print(f"[DEBUG][BRIDGE] fixed_input_features['c_l'].shape={c_l_tensor.shape}")
        else:
            print("[DEBUG][BRIDGE] WARNING: 'c_l' not found in fixed_input_features.")

        # If atom_to_token_idx has more atoms than c_l, slice it
        if n_valid_atoms is not None and total_atoms > n_valid_atoms:
            print(f"[DEBUG][BRIDGE] Slicing atom_to_token_idx from {total_atoms} to {n_valid_atoms}")
            atom_to_token_idx = atom_to_token_idx[:, :n_valid_atoms]

        # DEBUG: Print atom_to_token_idx shape after possible slicing
        print(f"[DEBUG][BRIDGE] (post-slice) atom_to_token_idx.shape={atom_to_token_idx.shape}")

        # Add to input features
        fixed_input_features["atom_to_token_idx"] = atom_to_token_idx
        logger.info(f"Created atom_to_token_idx mapping with shape {atom_to_token_idx.shape}")

    return fixed_input_features

def bridge_residue_to_atom(
    bridging_input: BridgingInput,
    config: Any,  # Accepts either config object or DictConfig
    debug_logging: bool = False,
):
    log_mem("ENTRY")
    # --- CONFIG VALIDATION PATCH: Ensure 's_inputs' is present in feature_dimensions ---
    feature_dimensions = None
    # Try to get feature_dimensions from config (robust to both dict and OmegaConf)
    if hasattr(config, 'diffusion') and hasattr(config.diffusion, 'feature_dimensions'):
        feature_dimensions = config.diffusion.feature_dimensions
    elif hasattr(config, 'diffusion') and isinstance(config.diffusion, dict):
        feature_dimensions = config.diffusion.get('feature_dimensions', None)
    elif hasattr(config, 'feature_dimensions'):
        feature_dimensions = config.feature_dimensions
    elif isinstance(config, dict):
        # DictConfig or plain dict
        if 'diffusion' in config and 'feature_dimensions' in config['diffusion']:
            feature_dimensions = config['diffusion']['feature_dimensions']
        elif 'feature_dimensions' in config:
            feature_dimensions = config['feature_dimensions']
    if feature_dimensions is None or 's_inputs' not in feature_dimensions:
        raise ValueError(
            "[BRIDGE ERROR][CONFIG] 's_inputs' missing from Stage D diffusion feature_dimensions config. "
            "This is a required field. Please check your Hydra config for model.stageD.diffusion.feature_dimensions.s_inputs."
        )
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
            logger.debug(f"[bridge_residue_to_atom] trunk_embeddings[{k}].shape: {v.shape}")
        logger.debug(f"[bridge_residue_to_atom] atom_metadata: {input_features.get('atom_metadata') if input_features else None}")
    # Instrumentation for systematic debugging
    if debug_logging:
        logger.debug("[BRIDGE DEBUG] bridge_residue_to_atom called")
        logger.debug(f"[BRIDGE DEBUG] sequence: {sequence if 'sequence' in locals() else 'N/A'}")
        logger.debug(f"[BRIDGE DEBUG] trunk_embeddings keys: {list(trunk_embeddings.keys()) if 'trunk_embeddings' in locals() else 'N/A'}")
        for k, v in (trunk_embeddings.items() if 'trunk_embeddings' in locals() else []):
            logger.debug(f"[BRIDGE DEBUG] trunk_embeddings[{k}].shape: {v.shape}")
        logger.debug("(stack trace omitted for performance)")
    # --- original code continues ---
    sequence_list = extract_sequence(sequence, input_features, trunk_embeddings)
    residue_atom_map = derive_residue_atom_map(
        sequence_list,
        partial_coords=partial_coords,
        atom_metadata=atom_metadata,
    )
    # --- INTEGRATION: Use hybrid_bridging_sparse_to_dense for Stage C output bridging ---
    if trunk_embeddings.get("stage_c_output") is not None:
        stage_c_output = trunk_embeddings["stage_c_output"]
        n_residues = residue_count
        # PATCH: Strictly require canonical_atom_count from Hydra config
        if hasattr(config, "get"):
            canonical_atom_count = config.get("canonical_atom_count", None)
        else:
            canonical_atom_count = getattr(config, "canonical_atom_count", None)
        if canonical_atom_count is None:
            raise ValueError(
                "[BRIDGE ERROR][HYDRA_CONF] canonical_atom_count must be set in the Hydra config for Stage D. "
                "Do not rely on fallback values. Ensure atoms_per_residue is set and propagated via config."
            )
        print(f"[DEBUG][BRIDGE][hybrid_bridging_sparse_to_dense] canonical_atom_count (from config): {canonical_atom_count}")
        logger.info(f"[DEBUG][BRIDGE][hybrid_bridging_sparse_to_dense] canonical_atom_count (from config): {canonical_atom_count}")
        feature_dim = stage_c_output.shape[-1]
        batch_size = stage_c_output.shape[0] if stage_c_output.ndim == 4 else 1
        device = stage_c_output.device
        dense_atoms, mask = hybrid_bridging_sparse_to_dense(
            stage_c_output=stage_c_output,
            residue_atom_map=residue_atom_map,
            n_residues=n_residues,
            canonical_atom_count=canonical_atom_count,
            feature_dim=feature_dim,
            batch_size=batch_size,
            fill_value=0.0,
            device=device
        )
        print(f"[DEBUG][BRIDGE][hybrid_bridging_sparse_to_dense] dense_atoms.shape={dense_atoms.shape} atom_mask.shape={mask.shape}")
        logger.info(f"[DEBUG][BRIDGE][hybrid_bridging_sparse_to_dense] dense_atoms.shape={dense_atoms.shape} atom_mask.shape={mask.shape}")
        # PATCH: Flatten dense_atoms and atom_mask to [B, n_residues*canonical_atom_count, ...]
        B, n_residues, n_atoms_per_res, F = dense_atoms.shape
        dense_atoms_flat = dense_atoms.reshape(B, n_residues * n_atoms_per_res, F)
        atom_mask_flat = mask.reshape(B, n_residues * n_atoms_per_res)
        print(f"[DEBUG][BRIDGE][flatten] dense_atoms_flat.shape={dense_atoms_flat.shape} atom_mask_flat.shape={atom_mask_flat.shape}")
        logger.info(f"[DEBUG][BRIDGE][flatten] dense_atoms_flat.shape={dense_atoms_flat.shape} atom_mask_flat.shape={atom_mask_flat.shape}")
        trunk_embeddings["dense_atoms"] = dense_atoms_flat
        trunk_embeddings["atom_mask"] = atom_mask_flat
        # PATCH: Ensure the encoder uses the correct atom-level tensor for c_l
        trunk_embeddings["c_l"] = dense_atoms_flat
        for k, v in trunk_embeddings.items():
            if hasattr(v, 'shape'):
                print(f"[DEBUG][BRIDGE][trunk_embeddings] {k}: {v.shape}")
                logger.info(f"[DEBUG][BRIDGE][trunk_embeddings] {k}: {v.shape}")
    # Continue with rest of bridging logic as before
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

    # --- PATCH: Only call preprocess_inputs if flag is True ---
    if False:
        if debug_logging:
            logger.debug("[bridge_residue_to_atom] Calling preprocess_inputs!")
        from rna_predict.pipeline.stageD.memory_optimization.memory_fix import preprocess_inputs
        # Process both coords and trunk_embeddings
        processed_coords, trunk_embeddings = preprocess_inputs(
            bridging_input.partial_coords,
            bridging_input.trunk_embeddings,
            max_seq_len=max_len
        )
        # Update partial_coords with processed_coords
        partial_coords = processed_coords
    else:
        # Keep original trunk_embeddings
        trunk_embeddings = bridging_input.trunk_embeddings
    # Process trunk embeddings
    batch_size = partial_coords.shape[0]
    bridged_trunk_embeddings = process_trunk_embeddings(
        trunk_embeddings, residue_atom_map, batch_size, debug_logging, config
    )
    # Process input features
    fixed_input_features = process_input_features(input_features, partial_coords, residue_atom_map, batch_size)
    # Return original coords, bridged embeddings, and fixed features
    logger.info("bridge_residue_to_atom completed successfully.")
    log_mem("EXIT")
    return partial_coords, bridged_trunk_embeddings, fixed_input_features


class ResidueToAtomsConfig:
    def __init__(self, s_emb, residue_atom_map):
        logger.debug("[BRIDGE DEBUG] ResidueToAtomsConfig __init__ called")
        logger.debug(f"[BRIDGE DEBUG] s_emb.shape = {getattr(s_emb, 'shape', 'N/A')}")
        logger.debug(f"[BRIDGE DEBUG] len(residue_atom_map) = {len(residue_atom_map) if hasattr(residue_atom_map, '__len__') else 'N/A'}")
        logger.debug(f"[BRIDGE DEBUG] residue_atom_map (first 2) = {residue_atom_map[:2] if hasattr(residue_atom_map, '__getitem__') else 'N/A'}")
        logger.debug("[BRIDGE DEBUG] Call stack:")
        if logger.isEnabledFor(logging.DEBUG):
            traceback.print_stack()
        # ...existing code...

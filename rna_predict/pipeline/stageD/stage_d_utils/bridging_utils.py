"""
Bridging and shape/feature validation utilities for Stage D pipeline.
Extracted from run_stageD.py for code quality and modularity.
"""
from typing import Any, Dict, Optional
import torch
import logging

logger = logging.getLogger(__name__)

def check_and_bridge_embeddings(
    trunk_embeddings: Dict[str, torch.Tensor],
    features: Dict[str, torch.Tensor],
    input_feature_dict: Dict[str, Any],
    coords: torch.Tensor,
    atom_metadata: Optional[Dict[str, Any]] = None,
    debug_logging: bool = False,
) -> None:
    """
    Defensive shape checks and bridging for atom count and feature mismatches.
    Mutates trunk_embeddings and features in-place as needed.
    Raises ValueError if mismatches are not bridgeable.
    """
    n_atoms = (
        coords.shape[1] if hasattr(coords, "shape") and len(coords.shape) > 1 else None
    )
    err_prefix = "[ERR-STAGED-SHAPE-001]"
    if n_atoms is not None:
        skip_keys = ["atom_metadata", "sequence", "ref_space_uid"]
        # Check trunk_embeddings
        for key, value in list(trunk_embeddings.items()):
            if key in skip_keys or not isinstance(value, torch.Tensor):
                continue
            if value.shape[1] != n_atoms:
                if key in ["s_trunk", "s_inputs", "pair"]:
                    continue
                raise ValueError(
                    f"{err_prefix} trunk_embeddings['{key}'] atom dim ({value.shape[1]}) != n_atoms ({n_atoms})"
                )
        # Check features
        for key, value in list(features.items()):
            if key in skip_keys or not isinstance(value, torch.Tensor):
                continue
            if value.shape[1] != n_atoms:
                if key in ["s_trunk", "s_inputs", "pair"]:
                    continue
                raise ValueError(
                    f"{err_prefix} features['{key}'] atom dim ({value.shape[1]}) != n_atoms ({n_atoms})"
                )
    # Copy all features to input_feature_dict to ensure they're available to the diffusion model
    for key, value in features.items():
        if key not in input_feature_dict:
            input_feature_dict[key] = value

    # Ensure 'atom_to_token_idx' is present in input_feature_dict
    if 'atom_to_token_idx' not in input_feature_dict:
        # Try to get from atom_metadata if possible
        if atom_metadata is not None and 'residue_indices' in atom_metadata:
            residue_indices = atom_metadata['residue_indices']
            if isinstance(residue_indices, torch.Tensor):
                input_feature_dict['atom_to_token_idx'] = residue_indices.unsqueeze(0)
            else:
                input_feature_dict['atom_to_token_idx'] = torch.tensor(
                    residue_indices, device=coords.device if hasattr(coords, 'device') else 'cpu', dtype=torch.long
                ).unsqueeze(0)
        else:
            # Fallback: map each atom to its own index (should not happen in normal pipeline)
            n_atoms = coords.shape[1] if hasattr(coords, 'shape') and len(coords.shape) > 1 else 0
            input_feature_dict['atom_to_token_idx'] = torch.arange(n_atoms, device=coords.device if hasattr(coords, 'device') else 'cpu').long().unsqueeze(0)

    if debug_logging:
        logger.debug("[DEBUG][check_and_bridge_embeddings] BEFORE bridging: atom_to_token_idx type: %s shape: %s", type(input_feature_dict.get("atom_to_token_idx")), getattr(input_feature_dict.get("atom_to_token_idx"), 'shape', None))

    # CRITICAL FIX: Check for token dimension mismatch between s_trunk and s_inputs
    # If s_trunk is at atom level but s_inputs is at residue level, bridge s_inputs to atom level
    if ('s_trunk' in trunk_embeddings and 's_inputs' in trunk_embeddings and
        isinstance(trunk_embeddings['s_trunk'], torch.Tensor) and
        isinstance(trunk_embeddings['s_inputs'], torch.Tensor)):

        s_trunk = trunk_embeddings['s_trunk']
        s_inputs = trunk_embeddings['s_inputs']

        # Check if there's a token dimension mismatch
        if s_trunk.shape[1] != s_inputs.shape[1]:
            print(f"[BRIDGE-FIX] Detected token dimension mismatch: s_trunk.shape={s_trunk.shape}, s_inputs.shape={s_inputs.shape}")

            # If atom_to_token_idx is available, use it to bridge s_inputs to atom level
            if 'atom_to_token_idx' in input_feature_dict:
                atom_to_token_idx = input_feature_dict['atom_to_token_idx']
                if isinstance(atom_to_token_idx, torch.Tensor):
                    # Get device from existing tensors
                    device = s_inputs.device

                    # Create a new tensor to hold atom-level s_inputs
                    batch_size = s_inputs.shape[0]
                    n_atoms = s_trunk.shape[1]
                    c_s_inputs = s_inputs.shape[2]
                    s_inputs_atom = torch.zeros((batch_size, n_atoms, c_s_inputs), device=device, dtype=s_inputs.dtype)

                    # Map residue-level features to atom-level using atom_to_token_idx
                    for b in range(batch_size):
                        for atom_idx in range(n_atoms):
                            # SYSTEMATIC DEBUGGING: Print atom_to_token_idx inside bridging loop
                            if debug_logging and b == 0 and atom_idx == 0:
                                logger.debug("[DEBUG][check_and_bridge_embeddings] INSIDE bridging: atom_to_token_idx type: %s shape: %s", type(atom_to_token_idx), getattr(atom_to_token_idx, 'shape', None))
                            # Extract residue index safely from atom_to_token_idx
                            if atom_to_token_idx.dim() > 1:
                                residue_idx = atom_to_token_idx[0, atom_idx].item()
                            else:
                                residue_idx = atom_to_token_idx[atom_idx].item()
                            if residue_idx < s_inputs.shape[1]:
                                s_inputs_atom[b, atom_idx] = s_inputs[b, int(residue_idx)]

                    # Replace the residue-level s_inputs with atom-level s_inputs
                    trunk_embeddings['s_inputs'] = s_inputs_atom
                    if debug_logging:
                        logger.debug(f"[BRIDGE-FIX] Successfully bridged s_inputs from residue level to atom level: {s_inputs.shape} -> {s_inputs_atom.shape}")
                else:
                    if debug_logging:
                        logger.debug("[BRIDGE-FIX] Warning: atom_to_token_idx is not a tensor, cannot bridge s_inputs")

    if debug_logging:
        logger.debug("[DEBUG][check_and_bridge_embeddings] AFTER bridging: atom_to_token_idx type: %s shape: %s", type(input_feature_dict.get("atom_to_token_idx")), getattr(input_feature_dict.get("atom_to_token_idx"), 'shape', None))

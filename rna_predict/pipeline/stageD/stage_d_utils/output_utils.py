"""
Diffusion model invocation and output handling utilities for Stage D pipeline.
Extracted from run_stageD.py for modularity and testability.
"""
from typing import Union, Tuple
import torch
from rna_predict.pipeline.stageD.context import StageDContext
import logging

logger = logging.getLogger(__name__)

class DiffusionInferenceError(Exception):
    """Custom exception for diffusion inference failures."""
    pass


def run_diffusion_and_handle_output(
    context: StageDContext
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Calls the diffusion model and handles output for Stage D.
    Uses StageDContext for argument bundling.
    """
    # --- ACTUAL LOGIC: Call the diffusion manager/model ---
    # Assumes context has all required fields (trunk_embeddings, features, input_feature_dict, etc.)
    from rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager import ProtenixDiffusionManager
    # Instantiate the diffusion manager
    diffusion_manager = ProtenixDiffusionManager(cfg=context.cfg)
    # Prepare inputs for the diffusion call
    coords = context.coords
    trunk_embeddings = context.trunk_embeddings
    input_feature_dict = context.input_feature_dict
    unified_latent = context.unified_latent
    # --- PATCH: Enforce max_atoms from Hydra config ---
    # Hydra best practice: always access config via context.cfg
    if hasattr(context.cfg, 'model') and hasattr(context.cfg.model, 'stageD') and hasattr(context.cfg.model.stageD, 'diffusion'):
        max_atoms = getattr(context.cfg.model.stageD.diffusion, 'max_atoms', 4096)
    else:
        max_atoms = 4096  # fallback
    orig_shape = coords.shape
    # Pad or truncate coords to [batch, max_atoms, 3]
    if coords.shape[1] < max_atoms:
        pad_size = max_atoms - coords.shape[1]
        pad = torch.zeros((coords.shape[0], pad_size, coords.shape[2]), dtype=coords.dtype, device=coords.device)
        coords_patched = torch.cat([coords, pad], dim=1)
    elif coords.shape[1] > max_atoms:
        coords_patched = coords[:, :max_atoms, :]
    else:
        coords_patched = coords
    logger.debug(f"[PATCH][coords_init] original shape: {orig_shape}, patched shape: {coords_patched.shape}, max_atoms={max_atoms}")

    # --- PATCH: Pad/truncate s_trunk and s_inputs to max_atoms ---
    s_trunk = trunk_embeddings.get('s_trunk', None)
    s_inputs = trunk_embeddings.get('s_inputs', None)
    if s_trunk is not None:
        orig_s_trunk_shape = s_trunk.shape
        if s_trunk.shape[1] < max_atoms:
            pad_size = max_atoms - s_trunk.shape[1]
            pad = torch.zeros((s_trunk.shape[0], pad_size, s_trunk.shape[2]), dtype=s_trunk.dtype, device=s_trunk.device)
            s_trunk_patched = torch.cat([s_trunk, pad], dim=1)
        elif s_trunk.shape[1] > max_atoms:
            s_trunk_patched = s_trunk[:, :max_atoms, :]
        else:
            s_trunk_patched = s_trunk
        trunk_embeddings['s_trunk'] = s_trunk_patched
        logger.debug(f"[PATCH][s_trunk] original shape: {orig_s_trunk_shape}, patched shape: {s_trunk_patched.shape}, max_atoms={max_atoms}")
    if s_inputs is not None:
        orig_s_inputs_shape = s_inputs.shape
        if s_inputs.shape[1] < max_atoms:
            pad_size = max_atoms - s_inputs.shape[1]
            pad = torch.zeros((s_inputs.shape[0], pad_size, s_inputs.shape[2]), dtype=s_inputs.dtype, device=s_inputs.device)
            s_inputs_patched = torch.cat([s_inputs, pad], dim=1)
        elif s_inputs.shape[1] > max_atoms:
            s_inputs_patched = s_inputs[:, :max_atoms, :]
        else:
            s_inputs_patched = s_inputs
        trunk_embeddings['s_inputs'] = s_inputs_patched
        logger.debug(f"[PATCH][s_inputs] original shape: {orig_s_inputs_shape}, patched shape: {s_inputs_patched.shape}, max_atoms={max_atoms}")

    # Call the diffusion manager (inference or train mode)
    try:
        # Convert trunk_embeddings to Dict[str, Any] for compatibility
        result = diffusion_manager.multi_step_inference(
            coords_init=coords_patched,
            trunk_embeddings=trunk_embeddings if trunk_embeddings is not None else {},
            override_input_features=input_feature_dict,
            unified_latent=unified_latent,
        )
    except Exception as err:
        logger.exception("Diffusion inference failed for Stage D context (sequence: %s)", getattr(context, 'sequence', None))
        raise DiffusionInferenceError("Failed to run diffusion inference") from err
    # Store the result in the context for later retrieval
    context.result = result
    return result

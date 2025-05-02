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
    # Call the diffusion manager (inference or train mode)
    # FIXME: permanent use of the ProtenixDiffusionManager inference API
    try:
        # Convert trunk_embeddings to Dict[str, Any] for compatibility
        result = diffusion_manager.multi_step_inference(
            coords_init=coords,
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

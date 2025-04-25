"""
Diffusion model invocation and output handling utilities for Stage D pipeline.
Extracted from run_stageD.py for modularity and testability.
"""
from typing import Union, Tuple
import torch
from .context import StageDContext

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
    # Call the diffusion manager (inference or train mode)
    # Always use multi_step_inference for now, as the train method is not implemented
    # PATCH: Use the correct inference API for ProtenixDiffusionManager
    result = diffusion_manager.multi_step_inference(
        coords_init=coords,
        trunk_embeddings=trunk_embeddings,
        override_input_features=input_feature_dict,
    )
    # Store the result in the context for later retrieval
    context.result = result
    return result

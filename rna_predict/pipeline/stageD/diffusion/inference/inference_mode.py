"""
Inference mode functions for Stage D diffusion.

This module provides functions for running inference in the Stage D diffusion process.
"""

import logging
import torch
from typing import Any, Dict
from dataclasses import dataclass

from rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager import (
    ProtenixDiffusionManager,
)

logger = logging.getLogger(__name__)



@dataclass
class InferenceContext:
    """Context for inference operations."""
    diffusion_manager: ProtenixDiffusionManager
    partial_coords: torch.Tensor
    trunk_embeddings_internal: Dict[str, torch.Tensor]
    original_trunk_embeddings_ref: Dict[str, torch.Tensor]
    diffusion_config: Dict[str, Any]
    input_features: Dict[str, Any]
    device: str


def run_inference_mode(
    context: InferenceContext,
) -> torch.Tensor:
    """
    Run diffusion in inference mode.

    Args:
        context: Inference context with all required parameters

    Returns:
        Refined coordinates tensor
    """
    # Note: We no longer need to pass inference_params directly to multi_step_inference
    # as it now reads parameters from the manager's internal config

    # Pass the internal (potentially fixed) copy to the manager
    coords = context.diffusion_manager.multi_step_inference(
        coords_init=context.partial_coords.to(context.device),
        trunk_embeddings=context.trunk_embeddings_internal,
        override_input_features=context.input_features
    )

    # Update the original trunk_embeddings dict with cached s_inputs if it was added
    if (
        "s_inputs" in context.trunk_embeddings_internal
        and "s_inputs" not in context.original_trunk_embeddings_ref
    ):
        logger.debug("Copying cached 's_inputs' back to original dictionary.")
        context.original_trunk_embeddings_ref["s_inputs"] = context.trunk_embeddings_internal[
            "s_inputs"
        ]

    return coords

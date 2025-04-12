"""
Training mode functions for Stage D diffusion.

This module provides functions for running training in the Stage D diffusion process.
"""

import logging
import torch
from typing import Any, Dict, Tuple

from rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager import (
    ProtenixDiffusionManager,
)
from rna_predict.pipeline.stageD.diffusion.utils.embedding_utils import (
    ensure_s_inputs,
    ensure_z_trunk,
)

logger = logging.getLogger(__name__)


from dataclasses import dataclass


@dataclass
class TrainingContext:
    """Context for training operations."""
    diffusion_manager: ProtenixDiffusionManager
    partial_coords: torch.Tensor
    trunk_embeddings_internal: Dict[str, torch.Tensor]
    original_trunk_embeddings_ref: Dict[str, torch.Tensor]
    diffusion_config: Dict[str, Any]
    input_features: Dict[str, Any]
    device: str


def run_training_mode(
    context: TrainingContext,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Executes a diffusion training step using the specified training context.
    
    This function prepares label and embedding contexts from the provided partial
    coordinates, diffusion configuration, and input features, ensuring that the
    required embeddings exist. It then runs a training step via the diffusion manager
    and processes the resulting sigma value to ensure it is a scalar tensor.
    
    Args:
        context: TrainingContext encapsulating parameters and embeddings for diffusion
            training, including configuration settings and target device.
    
    Returns:
        A tuple containing:
            - x_denoised: The denoised output tensor from the training step.
            - sigma: A scalar tensor representing the sigma value.
            - x_gt_augment: The augmented ground truth tensor.
    """
    # Create label dictionary
    label_dict = {
        "coordinate": context.partial_coords.to(context.device),
        "coordinate_mask": torch.ones_like(context.partial_coords[..., 0], device=context.device),
    }

    # Create embedding context for ensure_s_inputs and ensure_z_trunk
    from rna_predict.pipeline.stageD.diffusion.utils.embedding_utils import EmbeddingContext
    embedding_context = EmbeddingContext(
        diffusion_config=context.diffusion_config,
        device=context.device
    )

    # Ensure required embeddings exist
    ensure_s_inputs(
        context.trunk_embeddings_internal,
        context.original_trunk_embeddings_ref,
        context.input_features,
        embedding_context,
    )
    ensure_z_trunk(
        context.trunk_embeddings_internal,
        context.original_trunk_embeddings_ref,
        embedding_context,
    )

    # Run training step using the internal copy
    x_denoised_tuple = context.diffusion_manager.train_diffusion_step(
        label_dict=label_dict,
        input_feature_dict=context.input_features,
        s_inputs=context.trunk_embeddings_internal["s_inputs"],
        s_trunk=context.trunk_embeddings_internal["s_trunk"],
        z_trunk=context.trunk_embeddings_internal["pair"],
        sampler_params={"sigma_data": context.diffusion_config["sigma_data"]},
        N_sample=1,
    )

    # Unpack the results - x_gt_augment, x_denoised, sigma
    x_gt_augment, x_denoised, sigma = x_denoised_tuple
    # Ensure sigma is a scalar tensor
    if sigma.dim() > 0:
        sigma = sigma.mean().squeeze()  # Take mean and remove all dimensions

    return x_denoised, sigma, x_gt_augment

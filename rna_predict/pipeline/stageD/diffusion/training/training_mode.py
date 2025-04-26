"""
Training mode functions for Stage D diffusion.

This module provides functions for running training in the Stage D diffusion process.
"""

import logging
import torch
from typing import Any, Dict, Tuple
from dataclasses import dataclass

from rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager import (
    ProtenixDiffusionManager,
    DiffusionStepInput
)
from rna_predict.pipeline.stageD.diffusion.utils.embedding_utils import (
    ensure_s_inputs,
    ensure_z_trunk,
)

logger = logging.getLogger(__name__)



@dataclass
class TrainingContext:
    """Context for training operations."""
    diffusion_manager: ProtenixDiffusionManager
    partial_coords: torch.Tensor
    trunk_embeddings_internal: Dict[str, torch.Tensor]
    diffusion_config: Dict[str, Any]
    input_features: Dict[str, Any]
    device: str
    original_trunk_embeddings_ref: Dict[str, torch.Tensor]


def run_training_mode(
    context: TrainingContext,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run diffusion in training mode.

    Args:
        context: Training context with all required parameters

    Returns:
        Tuple of (x_denoised, sigma, x_gt_augment)
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
    step_input = DiffusionStepInput(
        label_dict=label_dict,
        input_feature_dict=context.input_features,
        s_inputs=context.trunk_embeddings_internal["s_inputs"],
        s_trunk=context.trunk_embeddings_internal["s_trunk"],
        z_trunk=context.trunk_embeddings_internal["pair"]
    )
    x_denoised_tuple = context.diffusion_manager.train_diffusion_step(step_input)

    # Unpack the results - x_gt_augment, x_denoised, sigma
    x_gt_augment, x_denoised, sigma = x_denoised_tuple
    # Ensure sigma is a scalar tensor
    if sigma.dim() > 0:
        sigma = sigma.mean().squeeze()  # Take mean and remove all dimensions

    # --- PATCH: Robust shape handling for x_denoised ---
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"[StageD][run_training_mode] x_denoised output shape BEFORE adjustment: {x_denoised.shape}")
    # Accept [1, 25, 3] or [1, 1, 25, 3]. Squeeze if needed.
    if x_denoised.dim() == 4 and x_denoised.shape[1] == 1:
        logger.debug("[StageD][run_training_mode] Squeezing extra dimension from x_denoised.")
        x_denoised = x_denoised.squeeze(1)
    if x_denoised.dim() != 3:
        raise AssertionError(f"[ERR-STAGED-TRAIN-SHAPE] UNIQUE ERROR: x_denoised must have 3 dims after adjustment, got {x_denoised.shape}")
    if x_denoised.shape[0] != 1:
        raise AssertionError(f"[ERR-STAGED-TRAIN-SHAPE] UNIQUE ERROR: Batch size must be 1, got {x_denoised.shape}")
    if x_denoised.shape[2] != 3:
        raise AssertionError(f"[ERR-STAGED-TRAIN-SHAPE] UNIQUE ERROR: Last dim must be 3, got {x_denoised.shape}")
    logger.debug(f"[StageD][run_training_mode] x_denoised output shape AFTER adjustment: {x_denoised.shape}")

    return x_denoised, sigma, x_gt_augment

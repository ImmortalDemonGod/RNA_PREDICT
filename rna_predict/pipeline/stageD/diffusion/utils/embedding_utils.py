"""
Embedding utilities for Stage D diffusion.

This module provides functions for handling embeddings in the Stage D diffusion process.
"""

import logging
import torch
from typing import Any, Dict, Union
from dataclasses import dataclass
from .config_utils import get_embedding_dimension
from .config_types import DiffusionConfig

logger = logging.getLogger(__name__)



@dataclass
class EmbeddingContext:
    """Context for embedding operations."""
    diffusion_config: Union[DiffusionConfig, Dict[str, Any]]
    device: str


def ensure_s_inputs(
    trunk_embeddings_internal: Dict[str, torch.Tensor],
    original_trunk_embeddings_ref: Dict[str, torch.Tensor],
    input_features: Dict[str, Any],
    context: EmbeddingContext,
) -> None:
    """
    Ensure s_inputs exists in trunk_embeddings_internal, creating it if needed.

    Args:
        trunk_embeddings_internal: Processed trunk embeddings
        original_trunk_embeddings_ref: Original trunk embeddings reference
        input_features: Input features dictionary
        context: Embedding context with configuration and device
    """
    s_inputs = trunk_embeddings_internal.get("s_inputs")
    if s_inputs is None:
        # Try to get s_inputs from input_features (using 'sing' as fallback key)
        s_inputs = input_features.get("sing")
        if s_inputs is None:
            logger.warning(
                "'s_inputs' not found in trunk_embeddings or input_features ('sing') for training mode. Using fallback."
            )
            # Create a fallback s_inputs with the right shape
            n_tokens = trunk_embeddings_internal["s_trunk"].shape[1]
            # Handle both DiffusionConfig and dict types
            if hasattr(context.diffusion_config, 'diffusion_config'):
                # It's a DiffusionConfig object
                c_s_inputs_dim = get_embedding_dimension(
                    context.diffusion_config, "c_s_inputs", 449
                )
            else:
                # It's a dict
                conditioning_config = context.diffusion_config.get("conditioning", {})
                c_s_inputs_dim = context.diffusion_config.get(
                    "c_s_inputs", conditioning_config.get("c_s_inputs", 449)
                )
            s_inputs = torch.zeros((1, n_tokens, c_s_inputs_dim), device=context.device)

        # Update the internal copy
        trunk_embeddings_internal["s_inputs"] = s_inputs
        # Also update the original reference if it wasn't there initially
        if "s_inputs" not in original_trunk_embeddings_ref:
            logger.debug(
                "Copying generated 's_inputs' back to original dictionary (train mode)."
            )
            original_trunk_embeddings_ref["s_inputs"] = s_inputs


def ensure_z_trunk(
    trunk_embeddings_internal: Dict[str, torch.Tensor],
    original_trunk_embeddings_ref: Dict[str, torch.Tensor],
    context: EmbeddingContext,
) -> None:
    """
    Ensure z_trunk (pair) exists in trunk_embeddings_internal, creating it if needed.

    Args:
        trunk_embeddings_internal: Processed trunk embeddings
        original_trunk_embeddings_ref: Original trunk embeddings reference
        context: Embedding context with configuration and device
    """
    z_trunk = trunk_embeddings_internal.get("pair")
    if z_trunk is None:
        logger.warning("Fallback: Creating dummy 'z_trunk' for training.")
        n_tokens = trunk_embeddings_internal["s_trunk"].shape[1]
        # Handle both DiffusionConfig and dict types
        if hasattr(context.diffusion_config, 'diffusion_config'):
            # It's a DiffusionConfig object
            c_z_dim = get_embedding_dimension(context.diffusion_config, "c_z", 128)
        else:
            # It's a dict
            conditioning_config = context.diffusion_config.get("conditioning", {})
            c_z_dim = context.diffusion_config.get(
                "c_z", conditioning_config.get("c_z", 128)
            )
        z_trunk = torch.zeros((1, n_tokens, n_tokens, c_z_dim), device=context.device)

        # Update the internal copy
        trunk_embeddings_internal["pair"] = z_trunk
        # Also update the original reference if it wasn't there initially
        if "pair" not in original_trunk_embeddings_ref:
            logger.debug(
                "Copying generated 'pair' back to original dictionary (train mode)."
            )
            original_trunk_embeddings_ref["pair"] = z_trunk

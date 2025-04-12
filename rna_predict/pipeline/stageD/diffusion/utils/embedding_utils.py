"""
Embedding utilities for Stage D diffusion.

This module provides functions for handling embeddings in the Stage D diffusion process.
"""

import logging
import torch
from typing import Any, Dict

from .config_utils import get_embedding_dimension

logger = logging.getLogger(__name__)


from dataclasses import dataclass


@dataclass
class EmbeddingContext:
    """Context for embedding operations."""
    diffusion_config: Dict[str, Any]
    device: str


def ensure_s_inputs(
    trunk_embeddings_internal: Dict[str, torch.Tensor],
    original_trunk_embeddings_ref: Dict[str, torch.Tensor],
    input_features: Dict[str, Any],
    context: EmbeddingContext,
) -> None:
    """
    Ensure the "s_inputs" tensor is present in trunk embeddings.
    
    If "s_inputs" is missing in the processed embeddings, the function first attempts to retrieve it from the input
    features using the key "sing". If absent, it logs a warning and creates a fallback tensor of zeros with a shape
    determined by the token count in "s_trunk" and an embedding dimension from the diffusion configuration. The tensor
    is then added to both the internal embeddings and the original reference.
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
            c_s_inputs_dim = get_embedding_dimension(
                context.diffusion_config, "c_s_inputs", 449
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
    Ensures the 'z_trunk' embedding is present in the trunk embeddings.
    
    If the trunk embeddings dictionary does not contain the key 'pair' (used for the
    'z_trunk' embedding), this function creates a fallback tensor of zeros. The tensor
    has shape (1, n_tokens, n_tokens, c_z_dim), where n_tokens is derived from the 's_trunk'
    tensor's second dimension and c_z_dim is obtained from the diffusion configuration (defaulting
    to 128 if unspecified). The generated tensor is then added to both the internal embeddings
    dictionary and the original reference.
    """
    z_trunk = trunk_embeddings_internal.get("pair")
    if z_trunk is None:
        logger.warning("Fallback: Creating dummy 'z_trunk' for training.")
        n_tokens = trunk_embeddings_internal["s_trunk"].shape[1]
        c_z_dim = get_embedding_dimension(context.diffusion_config, "c_z", 128)
        z_trunk = torch.zeros((1, n_tokens, n_tokens, c_z_dim), device=context.device)

        # Update the internal copy
        trunk_embeddings_internal["pair"] = z_trunk
        # Also update the original reference if it wasn't there initially
        if "pair" not in original_trunk_embeddings_ref:
            logger.debug(
                "Copying generated 'pair' back to original dictionary (train mode)."
            )
            original_trunk_embeddings_ref["pair"] = z_trunk

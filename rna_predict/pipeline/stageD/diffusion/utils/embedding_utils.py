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
    Ensure that the 's_inputs' tensor exists in the trunk embeddings dictionary.
    
    If 's_inputs' is missing from trunk_embeddings_internal, the function attempts to
    retrieve it from input_features using the key 'sing'. If it is still not found, a fallback
    zero tensor is created with a shape based on the number of tokens in 's_trunk' and an
    embedding dimension obtained from the context's diffusion configuration. The resulting
    tensor is then added to both trunk_embeddings_internal and original_trunk_embeddings_ref.
      
    Args:
        trunk_embeddings_internal: Dictionary containing current trunk embeddings.
        original_trunk_embeddings_ref: Reference dictionary for the original trunk embeddings.
        input_features: Dictionary of input features, with a fallback key 'sing' for 's_inputs'.
        context: Embedding context containing diffusion configuration and device information.
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
    Ensures that the 'pair' tensor (z_trunk) exists.
    
    If the 'pair' key is missing from trunk_embeddings_internal, a fallback zero tensor is created.
    The tensor has a shape of (1, n_tokens, n_tokens, c_z_dim), where n_tokens is obtained from the
    second dimension of the 's_trunk' tensor and c_z_dim is determined from the diffusion configuration
    (using 128 as the default value). The created tensor is added to trunk_embeddings_internal and,
    if not already present, to original_trunk_embeddings_ref.
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

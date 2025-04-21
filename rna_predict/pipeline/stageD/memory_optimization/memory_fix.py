"""
Memory optimization utility functions for stageD.
"""

import torch
import gc
from typing import Dict, Tuple, Any
import warnings

def clear_memory():
    """Clear memory by running garbage collection and emptying CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def preprocess_inputs(
    partial_coords: torch.Tensor,
    trunk_embeddings: Dict[str, Any], # Allow non-tensors
    max_seq_len: int = 25
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Preprocess inputs to potentially reduce memory usage by truncating sequence length.
       Note: This is a basic truncation and might be unsuitable for all use cases.
             Consider making its use configurable.

    Args:
        partial_coords: Input coordinates tensor.
        trunk_embeddings: Dictionary of trunk embeddings (can contain non-tensors).
        max_seq_len: The maximum sequence length to truncate to.

    Returns:
        Tuple of (processed_coords, processed_embeddings)
    """
    processed_coords = partial_coords # Default to original
    # Assuming coords shape is [B, N_atoms, 3] or similar where dim 1 relates to seq len
    if partial_coords.ndim >= 3 and partial_coords.shape[1] > max_seq_len:
         processed_coords = partial_coords[:, :max_seq_len, :]
    elif partial_coords.ndim == 2 and partial_coords.shape[0] > max_seq_len: # Handle case like [N_atoms, 3]
         warnings.warn(f"Preprocessing 2D coords, truncating dim 0 to {max_seq_len} atoms. Ensure this is intended.")
         # This simple truncation might be incorrect if N_atoms doesn't scale linearly with seq len
         processed_coords = partial_coords[:max_seq_len, :]

    processed_embeddings = {}
    for key, tensor in trunk_embeddings.items():
         if not isinstance(tensor, torch.Tensor):
              processed_embeddings[key] = tensor # Keep non-tensor items
              continue

         processed_tensor = tensor # Default to original
         # Assuming seq_len dimension is typically 1 for non-pair, 1 and 2 for pair
         if key == "pair":
              # For pair embeddings [B, N, N, C]
              if tensor.ndim >= 4 and tensor.shape[1] > max_seq_len:
                   processed_tensor = tensor[:, :max_seq_len, :max_seq_len, :]
         else:
              # For other embeddings [B, N, C] or [N, C]
              if tensor.ndim >= 3 and tensor.shape[1] > max_seq_len:
                   processed_tensor = tensor[:, :max_seq_len, :]
              elif tensor.ndim == 2 and tensor.shape[0] > max_seq_len: # Handle case like [N, C]
                   warnings.warn(f"Preprocessing 2D embedding '{key}', truncating dim 0 to {max_seq_len}. Ensure this relates to sequence length.")
                   processed_tensor = tensor[:max_seq_len, :]
         processed_embeddings[key] = processed_tensor

    return processed_coords, processed_embeddings

# Removed apply_memory_fixes function. Memory settings should be controlled via Hydra config.
# Removed run_stageD_with_memory_fixes function. Core logic moved to run_stageD.py.
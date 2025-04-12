"""
Tensor operations for RNA structure prediction.
"""

from typing import Optional

import einops
import torch


def chain2atoms(
    x: torch.Tensor, mask: Optional[torch.Tensor] = None, c: int = 3
) -> torch.Tensor:
    """Expand from (L, other) to (L, C, other).

    Args:
        x: Input tensor of shape (L, other)
        mask: Optional boolean mask of shape (L,)
        c: Number of atoms to expand to

    Returns:
        torch.Tensor: Shape (L, C, other) or (masked_size, C, other) if mask provided
    """
    wrap = einops.repeat(x, "l ... -> l c ...", c=c)
    if mask is not None:
        return wrap[mask]
    return wrap


def process_coordinates(noised_coords: torch.Tensor, scaffolds: dict) -> torch.Tensor:
    """
    Process coordinates with explicit imports.

    Args:
        noised_coords: Coordinates to process
        scaffolds: Dictionary of scaffold information

    Returns:
        torch.Tensor: Processed coordinates
    """
    from rna_predict.pipeline.stageC.mp_nerf.proteins import sidechain_fold

    noised_coords = einops.rearrange(
        noised_coords, "(l c) d -> l c d", c=14
    )
    noised_coords, _ = sidechain_fold(
        wrapper=noised_coords.cpu(), **scaffolds, c_beta=False
    )
    return noised_coords

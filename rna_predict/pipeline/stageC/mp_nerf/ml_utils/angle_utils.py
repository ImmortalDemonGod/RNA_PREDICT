"""
Angle calculation utilities for RNA structure prediction.
"""

import torch
from typing import Optional

from rna_predict.pipeline.stageC.mp_nerf.utils import to_zero_two_pi


def torsion_angle_loss(
    pred_torsions: torch.Tensor, 
    true_torsions: torch.Tensor, 
    coeff: float = 2.0, 
    angle_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Computes a loss on the angles as the cosine of the difference.
    Due to angle periodicity, calculate the disparity on both sides
    
    Args:
        pred_torsions: ( (B), L, X ) float. Predicted torsion angles.(-pi, pi)
                                   Same format as sidechainnet.
        true_torsions: ( (B), L, X ) true torsion angles. (-pi, pi)
        coeff: float. weight coefficient
        angle_mask: ((B), L, (X)) bool. Masks the non-existing angles.

    Returns:
        torch.Tensor: ( (B), L, 6 ) cosine difference
    """
    l_normal = torch.cos(pred_torsions - true_torsions)
    l_cycle = torch.cos(to_zero_two_pi(pred_torsions) - to_zero_two_pi(true_torsions))
    maxi = torch.max(l_normal, l_cycle)
    # Handle potential NaNs from invalid inputs (e.g., inf)
    # Replace NaN with 1.0 (max cosine similarity) -> zero loss for these entries
    maxi = torch.nan_to_num(maxi, nan=1.0)
    if angle_mask is not None:
        # Ensure angle_mask is boolean before indexing
        maxi[angle_mask.bool()] = 1.0
    return coeff * (1 - maxi)

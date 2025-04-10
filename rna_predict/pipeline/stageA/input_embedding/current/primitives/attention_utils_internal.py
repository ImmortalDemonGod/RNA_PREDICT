"""
Internal utilities for attention mechanisms.

This module contains internal utility functions for attention mechanisms,
including tensor preparation, gating, and output processing.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


def prep_qkv(
    q_x: torch.Tensor, 
    kv_x: torch.Tensor, 
    to_q: nn.Module,
    to_k: nn.Module,
    to_v: nn.Module,
    num_heads: int,
    head_dim: int,
    apply_scale: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare query, key and value tensors for attention.

    Args:
        q_x (torch.Tensor): query input
        kv_x (torch.Tensor): key-value input
        to_q (nn.Module): query projection module
        to_k (nn.Module): key projection module
        to_v (nn.Module): value projection module
        num_heads (int): number of attention heads
        head_dim (int): dimension of each head
        apply_scale (bool): whether to scale the query

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: processed query, key and value tensors
    """
    # Project inputs to attention space
    q = to_q(q_x)
    k = to_k(kv_x)
    v = to_v(kv_x)

    # Apply scaling if requested
    if apply_scale:
        scaling_factor = (head_dim**-0.5) if apply_scale else 1.0
        q = q * scaling_factor

    # Reshape for multi-head attention
    q = q.reshape(*q.shape[:-1], num_heads, head_dim)
    k = k.reshape(*k.shape[:-1], num_heads, head_dim)
    v = v.reshape(*v.shape[:-1], num_heads, head_dim)

    return q, k, v


def has_gating(
    gating: bool, 
    gating_linear: Optional[nn.Linear], 
    gating_bias: Optional[nn.Parameter]
) -> bool:
    """
    Check if gating is enabled and properly initialized.
    
    Args:
        gating (bool): Whether gating is enabled
        gating_linear (Optional[nn.Linear]): Gating linear layer
        gating_bias (Optional[nn.Parameter]): Gating bias parameter
        
    Returns:
        bool: Whether gating can be applied
    """
    return (
        gating
        and gating_linear is not None
        and gating_bias is not None
    )


def apply_gating(
    o: torch.Tensor, 
    q_x: torch.Tensor, 
    gating: bool,
    gating_linear: Optional[nn.Linear], 
    gating_bias: Optional[nn.Parameter]
) -> torch.Tensor:
    """
    Apply gating to the output if enabled.
    
    Args:
        o (torch.Tensor): Output tensor
        q_x (torch.Tensor): Original query input
        gating (bool): Whether gating is enabled
        gating_linear (Optional[nn.Linear]): Gating linear layer
        gating_bias (Optional[nn.Parameter]): Gating bias parameter
        
    Returns:
        torch.Tensor: Gated output
    """
    if not has_gating(gating, gating_linear, gating_bias):
        return o

    # These assertions help mypy understand that the attributes are not None
    assert gating_linear is not None, "Gating linear should not be None"
    assert gating_bias is not None, "Gating bias should not be None"

    g = gating_linear(q_x)
    g = g + gating_bias
    g = torch.sigmoid(g)
    return o * g


def wrap_up(
    o: torch.Tensor, 
    q_x: torch.Tensor, 
    c_hidden: int,
    to_out: nn.Linear,
    gating: bool,
    gating_linear: Optional[nn.Linear], 
    gating_bias: Optional[nn.Parameter]
) -> torch.Tensor:
    """
    Process the output of attention.

    Args:
        o (torch.Tensor): attention output
        q_x (torch.Tensor): original query input
        c_hidden (int): hidden dimension
        to_out (nn.Linear): output projection
        gating (bool): whether gating is enabled
        gating_linear (Optional[nn.Linear]): gating linear layer
        gating_bias (Optional[nn.Parameter]): gating bias parameter

    Returns:
        torch.Tensor: final processed output
    """
    # Reshape from multi-head back to batch
    o = o.reshape(*o.shape[:-2], c_hidden)

    # Project to output dimension
    o = to_out(o)

    # Apply gating if enabled
    return apply_gating(o, q_x, gating, gating_linear, gating_bias)

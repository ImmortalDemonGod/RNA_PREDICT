"""
Internal utilities for attention mechanisms.

This module contains internal utility functions for attention mechanisms,
including tensor preparation, gating, and output processing.
"""

from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn as nn


class TensorInputs(NamedTuple):
    """
    Base class for tensor inputs.
    """

    q_x: torch.Tensor
    kv_x: torch.Tensor


class ProjectionModules(NamedTuple):
    """
    Projection modules for attention.
    """

    to_q: nn.Module
    to_k: nn.Module
    to_v: nn.Module


class HeadConfig(NamedTuple):
    """
    Configuration for attention heads.
    """

    num_heads: int
    head_dim: int
    apply_scale: bool = True


class PrepQKVParams(NamedTuple):
    """
    Parameter object for preparing query, key, and value tensors.
    """

    tensors: TensorInputs
    modules: ProjectionModules
    config: HeadConfig


def prep_qkv(params: PrepQKVParams) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare query, key and value tensors for attention.

    Args:
        params: Parameter object containing tensors, modules, and configuration

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: processed query, key and value tensors
    """
    # Project inputs to attention space
    q = params.modules.to_q(params.tensors.q_x)
    k = params.modules.to_k(params.tensors.kv_x)
    v = params.modules.to_v(params.tensors.kv_x)

    # Apply scaling if requested
    if params.config.apply_scale:
        scaling_factor = (
            (params.config.head_dim**-0.5) if params.config.apply_scale else 1.0
        )
        q = q * scaling_factor

    # Reshape for multi-head attention
    q = q.reshape(*q.shape[:-1], params.config.num_heads, params.config.head_dim)
    k = k.reshape(*k.shape[:-1], params.config.num_heads, params.config.head_dim)
    v = v.reshape(*v.shape[:-1], params.config.num_heads, params.config.head_dim)

    return q, k, v


def has_gating(
    gating: bool,
    gating_linear: Optional[nn.Linear],
    gating_bias: Optional[nn.Parameter],
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
    return gating and gating_linear is not None and gating_bias is not None


class GatingTensors(NamedTuple):
    """
    Tensors for gating operations.
    """

    o: torch.Tensor
    q_x: torch.Tensor


class GatingModules(NamedTuple):
    """
    Modules for gating operations.
    """

    gating_linear: Optional[nn.Linear] = None
    gating_bias: Optional[nn.Parameter] = None


class GatingParams(NamedTuple):
    """
    Parameter object for gating operations.
    """

    tensors: GatingTensors
    modules: GatingModules
    gating: bool


def apply_gating(params: GatingParams) -> torch.Tensor:
    """

    Applies gating modulation to the output tensor based on the query input.
    
    If gating is enabled and both gating_linear and gating_bias are provided, this
    function computes a gating signal using a linear transformation of the query,
    adds the bias, applies a sigmoid activation, and multiplies it element-wise
    with the output tensor. If gating is disabled or the gating parameters are missing,
    the original output tensor is returned.
    
    Args:
        o (torch.Tensor): The output tensor to potentially modulate.
        q_x (torch.Tensor): The query tensor used for computing the gating signal.
        gating (bool): Whether to apply gating.
        gating_linear (Optional[nn.Linear]): Linear layer for computing the gating signal.
        gating_bias (Optional[nn.Parameter]): Bias parameter for the gating computation.
    

    Apply gating to the output if enabled.

    Args:
        params: Parameter object containing tensors, modules, and gating flag

    Returns:
        torch.Tensor: The gated output tensor if gating is applied; otherwise, the original output.
    """
    if not has_gating(
        params.gating, params.modules.gating_linear, params.modules.gating_bias
    ):
        return params.tensors.o

    # These assertions help mypy understand that the attributes are not None
    assert params.modules.gating_linear is not None, "Gating linear should not be None"
    assert params.modules.gating_bias is not None, "Gating bias should not be None"

    g = params.modules.gating_linear(params.tensors.q_x)
    g = g + params.modules.gating_bias
    g = torch.sigmoid(g)
    return params.tensors.o * g


class WrapUpConfig(NamedTuple):
    """
    Configuration for wrapping up attention output.
    """

    c_hidden: int
    gating: bool


class WrapUpModules(NamedTuple):
    """
    Modules for wrapping up attention output.
    """

    to_out: nn.Linear
    gating_linear: Optional[nn.Linear] = None
    gating_bias: Optional[nn.Parameter] = None


class WrapUpParams(NamedTuple):
    """
    Parameter object for wrapping up attention output.
    """

    tensors: GatingTensors
    config: WrapUpConfig
    modules: WrapUpModules


def wrap_up(params: WrapUpParams) -> torch.Tensor:
    """
    Finalize multi-head attention output by reshaping, projecting, and optionally applying gating.
    
    This function reshapes the multi-head attention output tensor to match the original hidden
    dimension, applies an output projection via a linear layer, and, if gating is enabled, modulates
    the projected output using the provided gating linear layer and bias.
    
    Args:

        o (torch.Tensor): Attention output tensor in multi-head format.
        q_x (torch.Tensor): Original query tensor.
        c_hidden (int): Hidden dimension size used for reshaping the tensor.
        to_out (nn.Linear): Linear layer used to project the reshaped tensor to the output dimension.
        gating (bool): Flag indicating whether to apply gating.
        gating_linear (Optional[nn.Linear]): Gating linear layer used to compute the gating signal, if gating is enabled.
        gating_bias (Optional[nn.Parameter]): Bias parameter for the gating signal, if gating is enabled.
    

        params: Parameter object containing tensors, configuration, and modules

    Returns:
        torch.Tensor: Final processed attention output tensor.
    """
    # Reshape from multi-head back to batch
    o = params.tensors.o.reshape(*params.tensors.o.shape[:-2], params.config.c_hidden)

    # Project to output dimension
    o = params.modules.to_out(o)

    # Apply gating if enabled
    gating_tensors = GatingTensors(o=o, q_x=params.tensors.q_x)
    gating_modules = GatingModules(
        gating_linear=params.modules.gating_linear,
        gating_bias=params.modules.gating_bias,
    )
    gating_params = GatingParams(
        tensors=gating_tensors, modules=gating_modules, gating=params.config.gating
    )
    return apply_gating(gating_params)

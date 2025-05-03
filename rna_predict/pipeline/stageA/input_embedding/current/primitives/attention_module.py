"""
Attention module implementation.

This module contains the main Attention class that implements multi-head attention
with support for various attention mechanisms.
"""

from typing import NamedTuple, Optional

import torch
import torch.nn as nn
from torch.nn import Linear

from .attention_core import (
    AttentionConfig,
    ForwardInputs,
    ProcessDifferentQueryInputs,
    ProcessQueryInputs,
)
from .attention_processing import (
    SmallTensorInputs as ProcessingSmallTensorInputs,
)
from .attention_processing import (
    process_different_query_keyvalue,
    process_same_query_keyvalue,
)
from .attention_processing import (
    process_small_tensors as process_small_tensors_func,
)
from .attention_utils_internal import (
    GatingTensors,
    HeadConfig,
    PrepQKVParams,
    ProjectionModules,
    TensorInputs,
    WrapUpConfig,
    WrapUpModules,
    WrapUpParams,
    prep_qkv,
    wrap_up,
)
from .linear_primitives import LinearNoBias


class Attention(nn.Module):
    """
    Attention module with support for multi-head attention
    """

    def __init__(self, config: AttentionConfig) -> None:
        """
        Initialize the attention module with configuration.

        Args:
            config (AttentionConfig): Configuration parameters for the attention module
        """
        super().__init__()
        self.c_q = config.c_q
        self.c_k = config.c_k
        self.c_v = config.c_v
        self.c_hidden = config.c_hidden
        self.num_heads = config.num_heads
        self.head_dim = config.c_hidden // config.num_heads
        self.gating = config.gating
        self.local_attention_method = config.local_attention_method
        self.use_efficient_implementation = config.use_efficient_implementation
        self.attn_weight_dropout_p = config.attn_weight_dropout_p

        # Linear layers for Q, K, V projections
        self.to_q = (
            Linear(config.c_q, config.c_hidden, bias=config.q_linear_bias)
            if config.q_linear_bias
            else LinearNoBias(config.c_q, config.c_hidden)
        )
        self.to_k = LinearNoBias(config.c_k, config.c_hidden)
        self.to_v = LinearNoBias(config.c_v, config.c_hidden)

        # Output projection
        self.to_out = Linear(config.c_hidden, config.c_q)

        # Optional gating parameters
        self.gating_bias: Optional[nn.Parameter] = None
        self.gating_linear: Optional[Linear] = None

        if self.gating:
            self.gating_bias = nn.Parameter(torch.ones((config.c_q)))
            self.gating_linear = Linear(config.c_q, config.c_q)

        # Initialize parameters
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Initialize weights and biases to zeros."""
        nn.init.zeros_(self.to_q.weight)
        nn.init.zeros_(self.to_k.weight)
        nn.init.zeros_(self.to_v.weight)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        if self.gating and self.gating_linear is not None:
            nn.init.zeros_(self.gating_linear.weight)
            nn.init.zeros_(self.gating_linear.bias)

    def forward(self, inputs: ForwardInputs) -> torch.Tensor:
        """
        Forward pass of the attention module.

        Args:
            inputs (ForwardInputs): Input parameters for attention

        Returns:
            torch.Tensor: output tensor
        """
        # Prepare query, key, value
        tensor_inputs = TensorInputs(q_x=inputs.q_x, kv_x=inputs.kv_x)
        projection_modules = ProjectionModules(
            to_q=self.to_q, to_k=self.to_k, to_v=self.to_v
        )
        head_config = HeadConfig(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            apply_scale=(inputs.q_x is inputs.kv_x and inputs.q_x.ndim == 3),
        )
        prep_params = PrepQKVParams(
            tensors=tensor_inputs, modules=projection_modules, config=head_config
        )
        q, k, v = prep_qkv(prep_params)

        # Handle case when query and key/value are the same
        if inputs.q_x is inputs.kv_x and inputs.q_x.ndim == 3:
            process_inputs = ProcessQueryInputs(
                q=q,
                k=k,
                v=v,
                q_x=inputs.q_x,
                attn_bias=inputs.attn_bias,
                inplace_safe=inputs.inplace_safe,
            )
            attn_output = process_same_query_keyvalue(
                process_inputs,
                self.num_heads,
                self.attn_weight_dropout_p,
                self.use_efficient_implementation,
            )
        else:
            # Different query/key-value processing
            diff_process_inputs = ProcessDifferentQueryInputs(
                q=q,
                k=k,
                v=v,
                q_x=inputs.q_x,
                attn_bias=inputs.attn_bias,
                trunked_attn_bias=inputs.trunked_attn_bias,
                n_queries=inputs.n_queries,
                n_keys=inputs.n_keys,
                inf=inputs.inf,
                inplace_safe=inputs.inplace_safe,
                chunk_size=inputs.chunk_size,
            )
            attn_output = process_different_query_keyvalue(
                diff_process_inputs,
                self.use_efficient_implementation,
                self.attn_weight_dropout_p,
                self.local_attention_method,
            )

        # Create parameter objects for wrap_up
        gating_tensors = GatingTensors(o=attn_output, q_x=inputs.q_x)
        wrap_config = WrapUpConfig(c_hidden=self.c_hidden, gating=self.gating)
        wrap_modules = WrapUpModules(
            to_out=self.to_out,
            gating_linear=self.gating_linear,
            gating_bias=self.gating_bias,
        )
        wrap_params = WrapUpParams(
            tensors=gating_tensors, config=wrap_config, modules=wrap_modules
        )
        return wrap_up(wrap_params)

    # Use the imported ProcessingSmallTensorInputs class instead of defining our own

    class SmallTensorArgs(NamedTuple):
        """
        Arguments for small tensor processing.
        """

        q: torch.Tensor
        k: torch.Tensor
        v: torch.Tensor
        bias: Optional[torch.Tensor] = None
        mask: Optional[torch.Tensor] = None

    def process_small_tensors(self, *args, **kwargs) -> torch.Tensor:
        """
        Process attention for small tensors that fit in memory.

        Args:
            q: Query tensor [..., N_q, d]
            k: Key tensor [..., N_k, d]
            v: Value tensor [..., N_k, d]
            bias: Optional attention bias [..., N_q, N_k]
            mask: Optional attention mask [..., N_q, N_k]

        Returns:
            Output tensor [..., N_q, d]
        """
        # Extract arguments
        if len(args) >= 3:
            q, k, v = args[0], args[1], args[2]
            bias = args[3] if len(args) > 3 else kwargs.get("bias")
            mask = args[4] if len(args) > 4 else kwargs.get("mask")
        else:
            q = kwargs.get("q")
            k = kwargs.get("k")
            v = kwargs.get("v")
            bias = kwargs.get("bias")
            mask = kwargs.get("mask")

        # Create a parameter object and delegate to the function
        tensor_args = self.SmallTensorArgs(q=q, k=k, v=v, bias=bias, mask=mask)
        return self._process_small_tensors_impl(tensor_args)

    def _process_small_tensors_impl(self, args: SmallTensorArgs) -> torch.Tensor:
        """
        Implementation helper to reduce function argument count.

        Args:
            args: Named tuple containing all tensor arguments

        Returns:
            Processed attention output
        """
        inputs = ProcessingSmallTensorInputs(
            q=args.q, k=args.k, v=args.v, bias=args.bias, mask=args.mask
        )
        return process_small_tensors_func(inputs)

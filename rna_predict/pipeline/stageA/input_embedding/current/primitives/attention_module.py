"""
Attention module implementation.

This module contains the main Attention class that implements multi-head attention
with support for various attention mechanisms.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.nn import Linear

from .linear_primitives import LinearNoBias
from .attention_core import (
    AttentionConfig,
    ProcessQueryInputs,
    ProcessDifferentQueryInputs,
    ForwardInputs,
)
from .attention_processing import (
    process_same_query_keyvalue,
    process_different_query_keyvalue,
    process_small_tensors as process_small_tensors_func
)
from .attention_utils_internal import (
    prep_qkv,
    wrap_up,
)


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
        super(Attention, self).__init__()
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
        """
        Zero out the parameters of the attention module.
        
        Initializes the weights and biases for the query, key, value, and output
        projection layers to zero. If gating is enabled and a gating linear layer is
        present, its weights and biases are also set to zero.
        """
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
        Computes multi-head attention output for the provided inputs.
        
        This method projects the query, key, and value tensors from the input and selects an attention
        computation pathway based on whether the query tensor is shared with the key/value tensor.
        After computing the attention, it applies an output projection and, if enabled, an optional
        gating mechanism.
        
        Args:
            inputs (ForwardInputs): A container with all necessary tensors and settings for the attention computation.
        
        Returns:
            torch.Tensor: The resulting attention output.
        """
        # Prepare query, key, value
        q, k, v = prep_qkv(
            inputs.q_x,
            inputs.kv_x,
            self.to_q,
            self.to_k,
            self.to_v,
            self.num_heads,
            self.head_dim,
            apply_scale=(inputs.q_x is inputs.kv_x and inputs.q_x.ndim == 3),
        )

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
                self.use_efficient_implementation
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
                self.local_attention_method
            )

        return wrap_up(
            attn_output,
            inputs.q_x,
            self.c_hidden,
            self.to_out,
            self.gating,
            self.gating_linear,
            self.gating_bias
        )

    def process_small_tensors(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        return process_small_tensors_func(q, k, v, bias, mask)

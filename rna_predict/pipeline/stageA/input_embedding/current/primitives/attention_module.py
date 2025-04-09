"""
Attention module implementation.

This module contains the main Attention class that implements multi-head attention
with support for various attention mechanisms.
"""

import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

from .linear_primitives import LinearNoBias
from .attention_core import (
    AttentionConfig,
    AttentionInputs,
    ProcessQueryInputs,
    ProcessDifferentQueryInputs,
    ForwardInputs,
    attention,
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
        """Initialize weights and biases to zeros."""
        nn.init.zeros_(self.to_q.weight)
        nn.init.zeros_(self.to_k.weight)
        nn.init.zeros_(self.to_v.weight)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        if self.gating and self.gating_linear is not None:
            nn.init.zeros_(self.gating_linear.weight)
            nn.init.zeros_(self.gating_linear.bias)

    def _has_gating(self) -> bool:
        """Check if gating is enabled and properly initialized."""
        return (
            self.gating
            and self.gating_linear is not None
            and self.gating_bias is not None
        )

    def _apply_gating(self, o: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
        """Apply gating to the output if enabled."""
        if not self._has_gating():
            return o

        # These assertions help mypy understand that the attributes are not None
        assert self.gating_linear is not None, "Gating linear should not be None"
        assert self.gating_bias is not None, "Gating bias should not be None"

        g = self.gating_linear(q_x)
        g = g + self.gating_bias
        g = torch.sigmoid(g)
        return o * g

    def _wrap_up(self, o: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
        """
        Process the output of attention.

        Args:
            o (torch.Tensor): attention output
            q_x (torch.Tensor): original query input

        Returns:
            torch.Tensor: final processed output
        """
        # Reshape from multi-head back to batch
        o = o.reshape(*o.shape[:-2], self.c_hidden)

        # Project to output dimension
        o = self.to_out(o)

        # Apply gating if enabled
        return self._apply_gating(o, q_x)

    def _prep_qkv(
        self, q_x: torch.Tensor, kv_x: torch.Tensor, apply_scale: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare query, key and value tensors for attention.

        Args:
            q_x (torch.Tensor): query input
            kv_x (torch.Tensor): key-value input
            apply_scale (bool): whether to scale the query

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: processed query, key and value tensors
        """
        # Project inputs to attention space
        q = self.to_q(q_x)
        k = self.to_k(kv_x)
        v = self.to_v(kv_x)

        # Apply scaling if requested
        if apply_scale:
            scaling_factor = (self.head_dim**-0.5) if apply_scale else 1.0
            q = q * scaling_factor

        # Reshape for multi-head attention
        q = q.reshape(*q.shape[:-1], self.num_heads, self.head_dim)
        k = k.reshape(*k.shape[:-1], self.num_heads, self.head_dim)
        v = v.reshape(*v.shape[:-1], self.num_heads, self.head_dim)

        return q, k, v

    def _process_same_query_keyvalue(self, inputs: ProcessQueryInputs) -> torch.Tensor:
        """
        Process attention when query and key/value are the same.

        Args:
            inputs (ProcessQueryInputs): Input parameters

        Returns:
            torch.Tensor: processed attention output
        """
        # Move head dimension for standard attention calculation
        # [..., n_q, h, d_h] -> [..., h, n_q, d_h]
        q = inputs.q.transpose(-2, -3)
        k = inputs.k.transpose(-2, -3)
        v = inputs.v.transpose(-2, -3)

        # Use efficient SDPA if available
        if self.use_efficient_implementation:
            try:
                o = F.scaled_dot_product_attention(
                    query=q,
                    key=k,
                    value=v,
                    attn_mask=inputs.attn_bias,
                    dropout_p=self.attn_weight_dropout_p,
                )
                # [..., h, n_q, d_h] -> [..., n_q, h, d_h]
                o = o.transpose(-2, -3)
                return self._wrap_up(o, inputs.q_x)
            except RuntimeError:
                # Fall back to manual implementation
                pass

        # Otherwise use batch matmul
        bsz = q.shape[0]
        q = q.reshape(-1, *q.shape[-2:])
        k = k.reshape(-1, *k.shape[-2:])
        v = v.reshape(-1, *v.shape[-2:])

        # Get attention scores
        reshaped_bias = None
        if inputs.attn_bias is not None:
            # Original bias shape: [B, H, N_q, N_kv]
            # Target shape for addition to attn_weight [B*H, N_q, N_kv]
            try:
                # First ensure bias has correct number of heads
                if inputs.attn_bias.shape[1] != self.num_heads:
                    # If bias has wrong number of heads, expand/repeat to match
                    bias = inputs.attn_bias.unsqueeze(1)  # [B, 1, N_q, N_kv]
                    bias = bias.expand(-1, self.num_heads, -1, -1)  # [B, H, N_q, N_kv]
                else:
                    bias = inputs.attn_bias

                # Now reshape to match attention weights
                reshaped_bias = bias.reshape(-1, *bias.shape[-2:])  # [B*H, N_q, N_kv]
            except RuntimeError as e:
                warnings.warn(
                    f"Could not reshape attn_bias from {inputs.attn_bias.shape} to match attention weights. Error: {str(e)}"
                )
                reshaped_bias = None  # Skip bias if reshape fails

        attention_inputs = AttentionInputs(
            q=q,
            k=k,
            v=v,
            attn_bias=reshaped_bias,
            use_efficient_implementation=self.use_efficient_implementation,
            attn_weight_dropout_p=self.attn_weight_dropout_p,
            inplace_safe=inputs.inplace_safe,
        )

        attn_output = attention(attention_inputs)

        # Reshape back
        h = self.num_heads
        attn_output = attn_output.reshape(bsz, h, *attn_output.shape[-2:])

        # [..., h, n_q, d_h] -> [..., n_q, h, d_h]
        attn_output = attn_output.transpose(-2, -3)

        return attn_output

    def _process_different_query_keyvalue(
        self, inputs: ProcessDifferentQueryInputs
    ) -> torch.Tensor:
        """
        Process attention when query and key/value are different.

        Args:
            inputs (ProcessDifferentQueryInputs): Input parameters

        Returns:
            torch.Tensor: processed attention output
        """
        # For other cases, import the local attention function
        from .attention_utils import LocalAttentionInputs, _local_attention

        # Use efficient implementation if available
        if "global_attention_with_bias" in self.local_attention_method:
            # Handle n_queries and n_keys for type compatibility
            actual_n_queries = (
                inputs.n_queries if inputs.n_queries is not None else inputs.q.size(-2)
            )
            actual_n_keys = (
                inputs.n_keys if inputs.n_keys is not None else inputs.k.size(-2)
            )

            # Create the LocalAttentionInputs instance
            local_attn_inputs = LocalAttentionInputs(
                q=inputs.q,
                k=inputs.k,
                v=inputs.v,
                n_queries=actual_n_queries,
                n_keys=actual_n_keys,
                attn_bias=inputs.attn_bias,
                trunked_attn_bias=inputs.trunked_attn_bias,
                inf=inputs.inf if inputs.inf is not None else 1e10,
                use_efficient_implementation=self.use_efficient_implementation,
                attn_weight_dropout_p=self.attn_weight_dropout_p,
                inplace_safe=inputs.inplace_safe,
                chunk_size=inputs.chunk_size,
            )

            # This implementation requires advanced handling, use _local_attention with dataclass
            o = _local_attention(local_attn_inputs)
        else:
            # Simple attention without special handling
            attention_inputs = AttentionInputs(
                q=inputs.q,
                k=inputs.k,
                v=inputs.v,
                attn_bias=inputs.attn_bias,
                use_efficient_implementation=self.use_efficient_implementation,
                attn_weight_dropout_p=self.attn_weight_dropout_p,
                inplace_safe=inputs.inplace_safe,
            )
            o = attention(attention_inputs)

        return o

    def forward(self, inputs: ForwardInputs) -> torch.Tensor:
        """
        Forward pass of the attention module.

        Args:
            inputs (ForwardInputs): Input parameters for attention

        Returns:
            torch.Tensor: output tensor
        """
        # Prepare query, key, value
        q, k, v = self._prep_qkv(
            inputs.q_x,
            inputs.kv_x,
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
            attn_output = self._process_same_query_keyvalue(process_inputs)
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
            attn_output = self._process_different_query_keyvalue(diff_process_inputs)

        return self._wrap_up(attn_output, inputs.q_x)

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
        # Create a parameter object to reduce the number of arguments
        class SmallTensorParams:
            def __init__(self, q, k, v, bias, mask):
                self.q = q
                self.k = k
                self.v = v
                self.bias = bias
                self.mask = mask
        
        params = SmallTensorParams(q, k, v, bias, mask)
        
        # Ensure all tensors have same batch dimensions
        batch_dims = params.q.shape[:-2]
        for t_name in ['k', 'v']:
            t = getattr(params, t_name)
            if t.shape[:-2] != batch_dims:
                setattr(params, t_name, t.expand(*batch_dims, *t.shape[-2:]))

        # Compute attention scores
        scores = torch.matmul(params.q, params.k.transpose(-2, -1)) / math.sqrt(params.q.size(-1))

        # Add bias if provided
        if params.bias is not None:
            from rna_predict.utils.shape_utils import adjust_attention_bias

            # Adjust bias to match scores shape
            adjusted_bias = adjust_attention_bias(
                params.bias, scores.shape, tensor_name="attention_bias"
            )
            scores = scores + adjusted_bias

        # Apply mask if provided
        if params.mask is not None:
            # Ensure mask has compatible shape
            if params.mask.shape[-2:] != scores.shape[-2:]:
                params.mask = params.mask.expand(*scores.shape[:-2], *params.mask.shape[-2:])
            scores = scores.masked_fill(~params.mask, float("-inf"))

        # Apply attention
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, params.v)

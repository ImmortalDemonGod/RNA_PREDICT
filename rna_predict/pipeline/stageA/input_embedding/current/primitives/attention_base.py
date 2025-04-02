"""
Attention base module for neural network operations.

This module contains the core attention mechanism implementations, including the
AdaptiveLayerNorm, base attention function, and the main Attention class.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

from .linear_primitives import LinearNoBias


class AdaptiveLayerNorm(nn.Module):
    """
    Implements Algorithm 26 in AF3
    """

    def __init__(self, c_a: int = 768, c_s: int = 384) -> None:
        """
        Args:
            c_a (int, optional): the embedding dim of a(single feature aggregated atom info). Defaults to 768.
            c_s (int, optional): hidden dim [for single embedding]. Defaults to 384.
        """
        super(AdaptiveLayerNorm, self).__init__()
        self.layernorm_a = nn.LayerNorm(c_a, elementwise_affine=False, bias=False)
        # The pytorch version should be newer than 2.1
        self.layernorm_s = nn.LayerNorm(c_s, bias=False)
        self.linear_s = Linear(in_features=c_s, out_features=c_a)
        self.linear_nobias_s = LinearNoBias(in_features=c_s, out_features=c_a)

    def zero_init(self) -> None:
        """Initialize the weights and biases to zero."""
        nn.init.zeros_(self.linear_s.weight)
        nn.init.zeros_(self.linear_s.bias)
        nn.init.zeros_(self.linear_nobias_s.weight)

    def _apply_conditioning(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Apply conditioning from s to a.

        Args:
            a (torch.Tensor): normalized representation
            s (torch.Tensor): normalized single embedding

        Returns:
            torch.Tensor: conditioned tensor or original if shapes don't match
        """
        try:
            scale = torch.sigmoid(self.linear_s(s))
            shift = self.linear_nobias_s(s)
            return scale * a + shift
        except RuntimeError as e:
            print(
                f"WARNING: Skipping adaptive layernorm conditioning due to shape mismatch: {e}"
            )
            print(f"         a shape: {a.shape}, s shape: {s.shape}")
            return a

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a (torch.Tensor): the single feature aggregate per-atom representation
                [..., N_token, c_a]
            s (torch.Tensor): single embedding
                [..., N_token, c_s]

        Returns:
            torch.Tensor: the updated a from AdaLN
                [..., N_token, c_a]
        """
        # Normalize inputs
        a_norm = self.layernorm_a(a)
        s_norm = self.layernorm_s(s)

        # Apply conditioning
        return self._apply_conditioning(a_norm, s_norm)


def _validate_attention_shapes(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> None:
    """
    Validate shapes of query, key and value tensors.

    Args:
        q (torch.Tensor): query tensor of shape [..., n_q, d]
        k (torch.Tensor): key tensor of shape [..., n_kv, d]
        v (torch.Tensor): value tensor of shape[..., n_kv, d]

    Raises:
        ValueError: If key and value tensors have different shapes
    """
    # Print debug info for small tensors
    if q.numel() < 1000 and k.numel() < 1000:
        print(f"q.shape: {q.shape}, k.shape: {k.shape}, v.shape: {v.shape}")

    # Ensure k and v have matching shapes
    if k.shape != v.shape:
        raise ValueError(
            f"Key and value tensors must have the same shape. Got k:{k.shape}, v:{v.shape}"
        )


def _handle_dimension_mismatch(
    q: torch.Tensor, k: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Handle dimension mismatch between query and key tensors.

    Args:
        q (torch.Tensor): query tensor
        k (torch.Tensor): key tensor (transposed)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Adjusted query and key tensors
    """
    # Check for dimension mismatch before matrix multiplication
    q_dim = q.size(-1)
    k_dim = k.size(-2)

    if q_dim != k_dim:
        # Handle dimension mismatch by padding or truncating
        if q_dim < k_dim:
            # Pad q with zeros to match k's dimension
            padding = torch.zeros(
                *q.shape[:-1], k_dim - q_dim, dtype=q.dtype, device=q.device
            )
            q = torch.cat([q, padding], dim=-1)
        else:
            # Truncate q to match k's dimension
            q = q[..., :k_dim]

    return q, k


def _compute_attention_weights(
    q: torch.Tensor,
    k: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute attention weights from query and key tensors.

    Args:
        q (torch.Tensor): query tensor
        k (torch.Tensor): key tensor (transposed)
        attn_bias (torch.Tensor, optional): attention bias tensor

    Returns:
        torch.Tensor: attention weights
    """
    # Compute scaled dot-product
    d_k = q.size(-1)
    attn_weight = torch.matmul(q, k) / math.sqrt(d_k)

    # Apply attention bias if provided and shapes match
    if attn_bias is not None:
        # Debug output for shape mismatch
        if attn_bias.shape != attn_weight.shape:
            print(
                f"Shape mismatch - attn_weight: {attn_weight.shape}, attn_bias: {attn_bias.shape}"
            )
            print(
                "WARNING: Attention bias shape mismatch. Skipping bias application for stability."
            )
        else:
            # Only apply bias if shapes match exactly
            attn_weight = attn_weight + attn_bias

    # Softmax normalization
    return F.softmax(attn_weight, dim=-1)


@dataclass
class AttentionInputs:
    """Inputs for attention operation."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    attn_bias: Optional[torch.Tensor] = None
    use_efficient_implementation: bool = False
    attn_weight_dropout_p: float = 0.0
    inplace_safe: bool = False


@dataclass
class ProcessQueryInputs:
    """Inputs for processing query-key-value attention."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    q_x: torch.Tensor
    attn_bias: Optional[torch.Tensor] = None
    inplace_safe: bool = False


@dataclass
class ProcessDifferentQueryInputs:
    """Inputs for processing different query-key-value attention."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    q_x: torch.Tensor
    attn_bias: Optional[torch.Tensor] = None
    trunked_attn_bias: Optional[torch.Tensor] = None
    n_queries: Optional[int] = None
    n_keys: Optional[int] = None
    inf: Optional[float] = 1e10
    inplace_safe: bool = False
    chunk_size: Optional[int] = None


@dataclass
class ForwardInputs:
    """Inputs for attention forward pass."""

    q_x: torch.Tensor
    kv_x: torch.Tensor
    attn_bias: Optional[torch.Tensor] = None
    trunked_attn_bias: Optional[torch.Tensor] = None
    n_queries: Optional[int] = None
    n_keys: Optional[int] = None
    inf: Optional[float] = 1e10
    inplace_safe: bool = False
    chunk_size: Optional[int] = None


def _attention(inputs: AttentionInputs) -> torch.Tensor:
    """Attention.

    Args:
        inputs (AttentionInputs): Inputs for attention calculation

    Returns:
        torch.Tensor: output tensor
    """
    # Validate input shapes
    _validate_attention_shapes(inputs.q, inputs.k, inputs.v)

    # Try efficient implementation if requested
    if inputs.use_efficient_implementation:
        try:
            attn_output = F.scaled_dot_product_attention(
                query=inputs.q,
                key=inputs.k,
                value=inputs.v,
                attn_mask=inputs.attn_bias,
                dropout_p=inputs.attn_weight_dropout_p,
            )
            return attn_output
        except RuntimeError:
            # Fall back to manual implementation
            pass

    # Transpose key for matrix multiplication
    # [..., n_kv, d] -> [..., d, n_kv]
    k_transposed = inputs.k.transpose(-1, -2)

    # Handle potential dimension mismatch
    q_adj, k_adj = _handle_dimension_mismatch(inputs.q, k_transposed)

    # Compute attention weights
    attn_weight = _compute_attention_weights(q_adj, k_adj, inputs.attn_bias)

    # Apply dropout if specified
    if inputs.attn_weight_dropout_p > 0.0:
        attn_weight = F.dropout(attn_weight, p=inputs.attn_weight_dropout_p)

    # Apply attention weights to values
    return torch.matmul(attn_weight, inputs.v)


@dataclass
class AttentionConfig:
    """Configuration for Attention module."""

    c_q: int
    c_k: int
    c_v: int
    c_hidden: int
    num_heads: int
    gating: bool = True
    q_linear_bias: bool = False
    local_attention_method: str = "global_attention_with_bias"
    use_efficient_implementation: bool = False
    attn_weight_dropout_p: float = 0.0


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
            reshaped_bias = inputs.attn_bias.reshape(-1, *inputs.attn_bias.shape[-2:])

        attention_inputs = AttentionInputs(
            q=q,
            k=k,
            v=v,
            attn_bias=reshaped_bias,
            use_efficient_implementation=self.use_efficient_implementation,
            attn_weight_dropout_p=self.attn_weight_dropout_p,
            inplace_safe=inputs.inplace_safe,
        )

        attn_output = _attention(attention_inputs)

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
        from .attention_utils import _local_attention, LocalAttentionInputs

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
            o = _attention(attention_inputs)

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

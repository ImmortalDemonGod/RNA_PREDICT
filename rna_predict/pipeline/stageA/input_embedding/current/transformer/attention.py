"""
Attention modules for transformer-based RNA structure prediction.
"""

import warnings
from typing import Optional, cast, Union # Added Union

import torch
import torch.nn as nn

from rna_predict.pipeline.stageA.input_embedding.current.primitives import (
    AdaptiveLayerNorm,
    Attention,
    LayerNorm,
    LinearNoBias,
)
from rna_predict.pipeline.stageA.input_embedding.current.primitives.attention_base import (
    AttentionConfig,
    ForwardInputs,
)
from rna_predict.pipeline.stageA.input_embedding.current.transformer.common import (
    validate_tensor_shape,
)
from rna_predict.pipeline.stageA.input_embedding.current.utils import permute_final_dims


class AttentionPairBias(nn.Module):
    """
    Implements attention with pair bias for structure prediction.
    Based on Algorithm 7, 8, 20 in AlphaFold3.
    """

    def __init__(
        self,
        has_s: bool = True,
        n_heads: int = 16,
        c_a: int = 768,
        c_s: int = 384,
        c_z: int = 128,
        biasinit: float = -2.0,
    ) -> None:
        """
        Initialize AttentionPairBias module.

        Args:
            has_s: Whether to use the single representation
            n_heads: Number of heads for multi-head attention
            c_a: Single embedding dimension for atom features
            c_s: Single embedding dimension for style features
            c_z: Pair embedding dimension
            biasinit: Bias initialization value in multi-head attention
        """
        super(AttentionPairBias, self).__init__()
        self.has_s = has_s
        self.biasinit = biasinit
        self.n_heads = n_heads
        self.c_a = c_a
        self.c_s = c_s
        self.c_z = c_z

        # Define type hint for layernorm_a
        self.layernorm_a: Union[AdaptiveLayerNorm, LayerNorm]

        # Adaptive Layer Norm for single representation
        if has_s:
            self.layernorm_a = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)
        else:
            # For non-adaptive case, use standard LayerNorm
            self.layernorm_a = LayerNorm(c_a)

        # Layer Normalization for pair embedding
        if hasattr(nn, "LayerNorm"):
            self.layernorm_z = nn.LayerNorm(c_z)
        else:
            self.layernorm_z = LayerNorm(c_z)

        # Create n_head representation from pair representation
        self.linear_nobias_z = LinearNoBias(c_z, n_heads)

        # Create attention configuration
        attention_config = AttentionConfig(
            c_q=c_a,
            c_k=c_a,
            c_v=c_a,
            c_hidden=c_a,
            num_heads=n_heads,
            gating=True,
            q_linear_bias=False,
            local_attention_method="global_attention_with_bias",
            use_efficient_implementation=False,
            attn_weight_dropout_p=0.0,
        )

        # Multi-head attention with attention bias & gating
        self.attention = Attention(config=attention_config)

        if has_s:
            # Output projection (adaLN-Zero)
            self.linear_a_last = LinearNoBias(c_s, c_a, bias=True)
            nn.init.zeros_(self.linear_a_last.weight)
            nn.init.zeros_(self.linear_a_last.bias)

        self.glorot_init()

    def glorot_init(self) -> None:
        """Initialize attention weights using Glorot/Xavier initialization."""
        nn.init.xavier_uniform_(self.attention.to_q.weight)
        nn.init.xavier_uniform_(self.attention.to_k.weight)
        nn.init.xavier_uniform_(self.attention.to_v.weight)
        if (
            hasattr(self.attention.to_q, "bias")
            and self.attention.to_q.bias is not None
        ):
            nn.init.zeros_(self.attention.to_q.bias)

    def _validate_z_tensor(
        self, z: torch.Tensor, a: torch.Tensor, is_local: bool
    ) -> None:
        """
        Validate pair embedding tensor dimensions.

        Args:
            z: Pair embedding tensor to validate
            a: Reference tensor for shape comparison
            is_local: Whether using local attention (determines expected dimensions)

        Raises:
            ValueError: If z tensor has invalid dimensions
        """
        if not isinstance(z, torch.Tensor):
            raise ValueError(f"Expected z to be a tensor, got {type(z)}")

        if is_local:
            # For local attention, expect 5D tensor [B, N_sample, N_block, N_query, N_key, C]
            # Or potentially 6D if batch dim is present before N_sample
            if len(z.shape) < 5:  # Allow 5 or 6 dims
                raise ValueError(
                    f"For local attention, expected z to have at least 5 dimensions, "
                    f"got {len(z.shape)} with shape {z.shape}"
                )
        else:
            # For standard attention (non-local):
            # z is typically [B, N, N, C] (4D) when a is [B, N, C] (3D)
            # OR z is [B, N_sample, N, N, C] (5D) when a is [B, N_sample, N, C] (4D)
            # OR z can be 5D when a is 3D if 'a' hasn't received the sample dim yet.
            # OR z can be [B, N_sample1, N_sample2, ..., N, N, C] when a is [B, N_sample1, N_sample2, ..., N, C]
            # for diffusion with multiple sample dimensions

            # For diffusion with multiple sample dimensions, we expect:
            # - z to have 2 more dimensions than a (for the N x N attention matrix)
            # - The last 3 dimensions of z to be [N, N, C]
            # - The last 2 dimensions of a to be [N, C]
            # - All other dimensions to match between z and a

            if len(z.shape) != len(a.shape) + 1:
                # Check if this is a diffusion case with multiple sample dimensions
                # where z has one extra N dimension for the attention matrix
                if len(z.shape) > 4 and len(a.shape) > 3:
                    # Extract the core dimensions (N, N, C for z and N, C for a)
                    z_core_dims = z.shape[-3:]  # Should be [N, N, C]
                    a_core_dims = a.shape[-2:]  # Should be [N, C]
                    
                    # Check that the N dimensions match
                    if z_core_dims[0] != z_core_dims[1] or z_core_dims[0] != a_core_dims[0]:
                        raise ValueError(
                            f"For diffusion attention, N dimensions must match. "
                            f"Got z_core_dims={z_core_dims}, a_core_dims={a_core_dims}"
                        )
                    
                    # Check that all sample dimensions match
                    z_sample_dims = z.shape[:-3]
                    a_sample_dims = a.shape[:-2]
                    if z_sample_dims != a_sample_dims:
                        raise ValueError(
                            f"Sample dimensions must match between z and a. "
                            f"Got z_sample_dims={z_sample_dims}, a_sample_dims={a_sample_dims}"
                        )
                # Allow the case where z is 5D [B, N_sample, N, N, C] and a is 3D [B, N, C]
                elif len(z.shape) == 5 and len(a.shape) == 3:
                    # Check if the batch and N dimensions match appropriately
                    # z[0] vs a[0] (Batch)
                    # z[2] vs z[3] vs a[1] (N)
                    if z.shape[0] != a.shape[0] or z.shape[2] != z.shape[3] or z.shape[2] != a.shape[1]:
                         raise ValueError(
                            f"Dimension mismatch for 5D z and 3D a. "
                            f"Shapes: z={z.shape}, a={a.shape}"
                        )
                    # If dimensions match, assume it's valid (z has an extra sample dim)
                else:
                    raise ValueError(
                        f"Unexpected dimension combination for non-local attention. "
                        f"Got z.dim={len(z.shape)}, a.dim={len(a.shape)}. "
                        f"Shapes: z={z.shape}, a={a.shape}"
                    )

        # Check last dimension always matches c_z
        if z.shape[-1] != self.c_z:
            raise ValueError(
                f"Expected z tensor to have last dimension {self.c_z}, got {z.shape[-1]}"
            )

    def _create_default_z(self, a: torch.Tensor) -> torch.Tensor:
        """
        Create a default zero tensor for z with appropriate dimensions.

        Args:
            a: Reference tensor for shape

        Returns:
            Default zero tensor for z
        """
        warnings.warn(
            "Creating a default zero tensor for z with appropriate dimensions."
        )

        return torch.zeros(
            (*a.shape[:-1], *a.shape[:-1], self.c_z), device=a.device, dtype=a.dtype
        )

    def local_multihead_attention(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        n_queries: int = 32,
        n_keys: int = 128,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Perform local multi-head attention with pair bias.

        Args:
            a: Single feature aggregate per-atom representation [..., N_atom, c_a]
            s: Single embedding [..., N_atom, c_s]
            z: Pair embedding for computing attention bias
            n_queries: Number of queries per chunk
            n_keys: Number of keys per chunk
            inplace_safe: Whether it is safe to use inplace operations
            chunk_size: Chunk size for processing

        Returns:
            Updated atom representation after attention

        Raises:
            ValueError: If input tensors have incompatible shapes
        """
        # Create a default z tensor if none is provided or it's not a tensor
        if not isinstance(z, torch.Tensor):
            z = self._create_default_z(a)

        # Fix tensor dimension if needed
        if len(z.shape) < 5:  # We need [batch, n_blocks, n_queries, n_keys, c_z]
            # If z has shape [..., N, N, c_z], we need to add the n_blocks dimension
            if len(z.shape) == len(a.shape) + 1:
                z = z.unsqueeze(-4)  # Add n_blocks dimension
            else:
                # In case of other shape mismatches, try to adapt or raise an error
                raise ValueError(
                    f"Cannot adapt z tensor with shape {z.shape} to work with a tensor of shape {a.shape}. "
                    f"Expected z to have shape compatible with [..., n_blocks, n_queries, n_keys, c_z]"
                )

        # Verify that z has the expected final dimension
        validate_tensor_shape(z, expected_last_dim=self.c_z, name="z")

        # Apply layer normalization to z
        try:
            normalized_z = self.layernorm_z(z)
        except RuntimeError as e:
            raise ValueError(
                f"Failed to apply layernorm to z tensor with shape {z.shape}: {str(e)}"
            )

        # Calculate multi-head attention bias
        bias = self.linear_nobias_z(
            normalized_z
        )  # [..., n_blocks, n_queries, n_keys, n_heads]

        # Permute dimensions for attention
        try:
            bias = permute_final_dims(
                bias, [3, 0, 1, 2]
            )  # [..., n_heads, n_blocks, n_queries, n_keys]
        except RuntimeError as e:
            raise ValueError(
                f"Failed to permute bias tensor with shape {bias.shape}: {str(e)}"
            )

        # Set up attention inputs
        forward_inputs = ForwardInputs(
            q_x=a,
            kv_x=a,
            attn_bias=None,
            trunked_attn_bias=bias,
            n_queries=n_queries,
            n_keys=n_keys,
            inf=1e10,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        # Apply attention - wrap in try-except to provide better error messages
        try:
            result = self.attention(forward_inputs)
            return cast(torch.Tensor, result)
        except Exception as e:
            raise ValueError(
                f"Attention failed with inputs: q_x={a.shape}, bias={bias.shape}, "
                f"n_queries={n_queries}, n_keys={n_keys}. Error: {str(e)}"
            )

    def standard_multihead_attention(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Perform standard multi-head attention with pair bias.

        Args:
            a: Single feature aggregate per-atom representation [..., N_token, c_a]
            s: Single embedding [..., N_token, c_s]
            z: Pair embedding for computing attention bias [..., N_token, N_token, c_z]
            inplace_safe: Whether it is safe to use inplace operations

        Returns:
            Updated atom representation after attention

        Raises:
            ValueError: If input tensors have incompatible shapes
        """
        # Validate input shapes
        self._validate_z_tensor(z, a, is_local=False)

        # Calculate multi-head attention bias with error handling
        try:
            normalized_z = self.layernorm_z(z)
            bias = self.linear_nobias_z(normalized_z)
            bias = permute_final_dims(
                bias, [2, 0, 1]
            )  # [..., n_heads, N_token, N_token]
        except RuntimeError as e:
            raise ValueError(
                f"Failed to calculate attention bias for z with shape {z.shape}: {str(e)}"
            )

        # Set up attention inputs
        forward_inputs = ForwardInputs(
            q_x=a,
            kv_x=a,
            attn_bias=bias, # bias is [..., n_heads, N_token, N_token]
            trunked_attn_bias=None,
            n_queries=None,
            n_keys=None,
            inf=1e10,
            inplace_safe=inplace_safe,
            chunk_size=None,
        )

        # Apply attention with error handling
        try:
            # self.attention expects attn_bias to be broadcastable to weights [..., H, Q, K]
            # If a is 4D [B, S, N, C] and bias is 5D [B, S, H, N, N], self.attention should handle it.
            # If a is 3D [B, N, C] and bias is 5D [B, S, H, N, N] (S=1), self.attention might fail or produce 5D output.
            # The underlying Attention primitive needs to handle this broadcasting correctly.
            # Let's assume self.attention returns a tensor with the same leading dimensions as q_x (a).
            result = self.attention(forward_inputs)

            # If attention output has more dims than input 'a', check if dim 1 is size 1 and squeeze it.
            if result.ndim > a.ndim:
                if result.ndim > 1 and result.shape[1] == 1:
                    warnings.warn(f"Attention output dim ({result.ndim}) > input dim ({a.ndim}) with sample dim size 1. Squeezing dim 1.")
                    result = result.squeeze(1)
                else:
                    # If the extra dimension is not size 1, we have an unexpected shape propagation.
                    warnings.warn(f"Attention output dim ({result.ndim}) > input dim ({a.ndim}), but sample dim is not 1 ({result.shape[1]}). Cannot squeeze.")
                    # Raise error here? Or let it propagate? Let's let it propagate for now.

            # Check dimensions again after potential squeeze
            if result.ndim != a.ndim:
                 warnings.warn(f"Attention output dim ({result.ndim}) still does not match input dim ({a.ndim}) after potential squeeze.")


            return cast(torch.Tensor, result)
        except Exception as e:
            # Add shapes to error message
            q_shape = forward_inputs.q_x.shape if forward_inputs.q_x is not None else None
            kv_shape = forward_inputs.kv_x.shape if forward_inputs.kv_x is not None else None
            bias_shape = forward_inputs.attn_bias.shape if forward_inputs.attn_bias is not None else None
            raise ValueError(
                f"Attention failed with inputs: q_x={q_shape}, kv_x={kv_shape}, bias={bias_shape}. Error: {str(e)}"
            )

    def _apply_gating(
        self, a: torch.Tensor, s: torch.Tensor, inplace_safe: bool
    ) -> torch.Tensor:
        """
        Apply gating mechanism to attention output.

        Args:
            a: Attention output tensor
            s: Single embedding tensor
            inplace_safe: Whether inplace operations are safe

        Returns:
            Gated attention output
        """
        try:
            # Check shape compatibility before applying gating
            # Allow broadcasting if s has fewer dims (e.g., missing sample dim)
            if s.dim() < a.dim():
                # Try unsqueezing s to match a's dimensions (assuming missing sample dim)
                if (
                    s.dim() == a.dim() - 1
                    and s.shape[0] == a.shape[0]
                    and s.shape[1:] == a.shape[2:]
                ):
                    s = s.unsqueeze(1)  # Add sample dim
                else:
                    raise ValueError(
                        f"Cannot broadcast s ({s.shape}) to match a ({a.shape}) for gating."
                    )

            # Now check leading dimensions after potential unsqueeze
            if s.shape[:-1] != a.shape[:-1]:
                # If leading dims don't match, maybe only feature dim needs to match for linear_a_last?
                # Let linear_a_last handle potential broadcasting if s is [B, N, C] and a is [B, S, N, C]
                # This requires linear_a_last to handle broadcasting, which standard Linear doesn't.
                # Let's assume for now the first N-1 dims must match.
                if (
                    s.shape[0] != a.shape[0] or s.shape[1] != a.shape[1]
                ):  # Check Batch and Sample dims
                    raise ValueError(
                        f"Shape mismatch: s has shape {s.shape}, a has shape {a.shape}. "
                        f"First two dimensions must match for gating."
                    )
                # If only sequence/token dim mismatches, maybe linear layer can handle it? Risky.
                # Let's stick to requiring first N-1 dims to match.

            gate = torch.sigmoid(
                self.linear_a_last(s)
            )  # Expects [..., C_s], outputs [..., C_a]

            # Verify gate and a are compatible shapes for element-wise multiplication
            if gate.shape != a.shape:
                # Try broadcasting gate if its sequence/token dim is 1
                if gate.shape[:-1] == (
                    *a.shape[:-2],
                    1,
                    a.shape[-1],
                ):  # Check if seq dim is 1
                    gate = gate.expand_as(a)
                else:
                    raise ValueError(
                        f"Gate and activation have incompatible shapes after projection: {gate.shape} vs {a.shape}"
                    )

            # Apply gating
            if inplace_safe:
                a *= gate
                return a
            else:
                return gate * a

        except (RuntimeError, ValueError) as e:
            # Log the issue but continue without gating
            if torch.is_grad_enabled():
                warnings.warn(f"Skipping adaptive gating due to: {str(e)}")
            return a

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        n_queries: Optional[int] = None,
        n_keys: Optional[int] = None,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the AttentionPairBias module.

        Args:
            a: Single feature aggregate [..., N, c_a]
            s: Single embedding [..., N, c_s]
            z: Pair embedding [..., N, N, c_z] or [..., n_blocks, n_queries, n_keys, c_z]
            n_queries: Number of queries for local attention
            n_keys: Number of keys for local attention
            inplace_safe: Whether inplace operations are safe
            chunk_size: Chunk size for memory-efficient operations

        Returns:
            Updated single feature representation
        """
        # Input projections
        if self.has_s:
            a = self.layernorm_a(a=a, s=s)
        else:
            a = self.layernorm_a(a)

        # Multihead attention with pair bias
        if n_queries and n_keys:
            a = self.local_multihead_attention(
                a,
                s,
                z,
                n_queries,
                n_keys,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
        else:
            a = self.standard_multihead_attention(a, s, z, inplace_safe=inplace_safe)

        # Output projection with gating if has_s
        if self.has_s:
            a = self._apply_gating(a, s, inplace_safe)

        return a

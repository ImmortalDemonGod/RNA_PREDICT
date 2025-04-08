"""
Atom attention decoder module for RNA structure prediction.
"""

import warnings
from dataclasses import dataclass
from typing import Optional, cast

import torch
import torch.nn as nn

from rna_predict.pipeline.stageA.input_embedding.current.primitives import (
    LayerNorm,
    LinearNoBias,
)
from rna_predict.pipeline.stageA.input_embedding.current.transformer.atom_attention_encoder import (
    AtomAttentionConfig,
)
from rna_predict.pipeline.stageA.input_embedding.current.transformer.atom_transformer import (
    AtomTransformer,
)
from rna_predict.pipeline.stageA.input_embedding.current.utils import (
    broadcast_token_to_atom,
)


@dataclass
class DecoderForwardParams:
    """Parameters for AtomAttentionDecoder forward method."""

    a: torch.Tensor  # Token-level features from DiffusionTransformer
    r_l: torch.Tensor  # Noisy atom coordinates
    extra_feats: Optional[torch.Tensor] = (
        None  # Atom-level features (e.g., from encoder q_skip) to be used as 's' conditioning
    )
    p_lm: Optional[torch.Tensor] = None  # Pair embedding from encoder (used as 'p')
    mask: Optional[torch.Tensor] = None  # Token mask
    atom_mask: Optional[torch.Tensor] = None  # Atom mask
    atom_to_token_idx: Optional[torch.Tensor] = None
    chunk_size: Optional[int] = None


class AtomAttentionDecoder(nn.Module):
    """
    Decoder that processes token-level embeddings and produces atom-level coordinates.
    Implements Algorithm 6 in AlphaFold3.
    """

    def __init__(self, config: AtomAttentionConfig) -> None:
        """
        Initialize the AtomAttentionDecoder with a configuration object.

        Args:
            config: Configuration parameters for the decoder
        """
        super(AtomAttentionDecoder, self).__init__()
        self.n_blocks = config.n_blocks
        self.n_heads = config.n_heads
        self.c_token = config.c_token
        self.c_atom = config.c_atom
        self.c_atompair = config.c_atompair
        self.n_queries = config.n_queries
        self.n_keys = config.n_keys
        # Override non-positive c_s with a default value (384)
        self.c_s = config.c_s if config.c_s > 0 else 384

        # Project token features to atom dimension
        self.linear_no_bias_a = LinearNoBias(
            in_features=config.c_token, out_features=config.c_atom
        )

        # Project token features to atom pair dimension
        self.linear_no_bias_p = LinearNoBias(
            in_features=config.c_token, out_features=config.c_atompair
        )

        # Layer normalization and output projection
        self.layernorm_q = LayerNorm(config.c_atom)
        self.linear_no_bias_out = LinearNoBias(
            in_features=config.c_atom, out_features=3
        )

        # Initialize the internal atom transformer
        self.atom_transformer = AtomTransformer(
            n_blocks=config.n_blocks,
            n_heads=config.n_heads,
            c_atom=config.c_atom,
            c_s=self.c_s,  # Use the effective c_s value
            c_atompair=config.c_atompair,
            n_queries=config.n_queries,
            n_keys=config.n_keys,
            blocks_per_ckpt=config.blocks_per_ckpt,
        )

    def forward(
        self,
        params: DecoderForwardParams,
    ) -> torch.Tensor:
        """
        Forward pass for the AtomAttentionDecoder.

        Args:
            params: Parameters for the forward pass

        Returns:
            Predicted atom coordinates, shape [..., N_atom, 3]
        """
        # Project token features to atom dimension
        q = self.linear_no_bias_a(params.a)

        # Broadcast from token to atom if needed
        if params.atom_to_token_idx is not None:
            # Broadcast token features to atoms
            q = broadcast_token_to_atom(q, params.atom_to_token_idx)

        # Include extra features if provided (these become the input 'q' to atom_transformer)
        if params.extra_feats is not None:
            # Ensure extra_feats has the same feature dimension as q
            if q.shape[-1] != params.extra_feats.shape[-1]:
                # Adjust extra_feats dimension if necessary (e.g., padding/truncating)
                # This part might need refinement based on expected behavior
                target_dim = q.shape[-1]
                current_dim = params.extra_feats.shape[-1]
                if current_dim < target_dim:
                    padding = torch.zeros(
                        *params.extra_feats.shape[:-1],
                        target_dim - current_dim,
                        device=q.device,
                        dtype=q.dtype,
                    )
                    params.extra_feats = torch.cat(
                        [params.extra_feats, padding], dim=-1
                    )
                else:
                    params.extra_feats = params.extra_feats[..., :target_dim]

            q = (
                q + params.extra_feats
            )  # Add extra features to the projected token features

        # Apply atom mask if provided
        # Ensure atom_mask has a trailing singleton dimension for broadcasting
        if params.atom_mask is not None and params.atom_mask.shape[-1] != 1:
            params.atom_mask = params.atom_mask.unsqueeze(-1)
        if params.atom_mask is not None:
            # The mask should have shape [B, N_atom, 1] to broadcast correctly
            q = q * params.atom_mask # Removed .unsqueeze(-1)

        # Process through atom transformer
        # Pass q (projected token features + extra_feats),
        # s (conditioning signal - should be derived appropriately, maybe from params.a or a dedicated input?),
        # p (pair embedding from encoder)
        p_input = params.p_lm
        if p_input is not None and p_input.dim() == 4:
            # Unsqueeze to add block dimension: [B, N_queries, N_keys, C] -> [B, 1, N_queries, N_keys, C]
            p_input = p_input.unsqueeze(1)

        # *** Correction: Pass params.extra_feats as the conditioning signal 's' ***
        # Assuming extra_feats holds the appropriate conditioning information for the decoder's transformer
        # If extra_feats is None, we might need a default or raise an error depending on design.
        s_conditioning = params.extra_feats
        if s_conditioning is None:
            # Handle missing conditioning signal - create zeros or raise error
            # For now, create zeros matching q's batch/atom dims but with c_s features
            warnings.warn(
                "Conditioning signal 's' (extra_feats) is None in AtomAttentionDecoder. Creating zeros."
            )
            s_conditioning = torch.zeros(
                *q.shape[:-1], self.c_s, device=q.device, dtype=q.dtype
            )
        elif s_conditioning.shape[-1] != self.c_s:
            # Adapt s_conditioning feature dimension if it doesn't match self.c_s
            warnings.warn(
                f"Conditioning signal 's' has incorrect feature dim {s_conditioning.shape[-1]}, expected {self.c_s}. Adapting."
            )
            target_dim = self.c_s
            current_dim = s_conditioning.shape[-1]
            if current_dim < target_dim:
                padding = torch.zeros(
                    *s_conditioning.shape[:-1],
                    target_dim - current_dim,
                    device=q.device,
                    dtype=q.dtype,
                )
                s_conditioning = torch.cat([s_conditioning, padding], dim=-1)
            else:
                s_conditioning = s_conditioning[..., :target_dim]

        q = self.atom_transformer(
            q=q, s=s_conditioning, p=p_input, chunk_size=params.chunk_size
        )

        # Apply token mask if provided (Note: q is now atom-level, mask should be atom_mask)
        if params.atom_mask is not None:  # Changed from params.mask
            # Mask shape [B, N_atom, 1] should broadcast correctly with q [B, N_atom, C]
            q = q * params.atom_mask # Removed .unsqueeze(-1)

        # Project to 3D coordinates
        r = self.linear_no_bias_out(self.layernorm_q(q))
        return cast(torch.Tensor, r)

    # For backward compatibility - uses the old parameter style
    def forward_legacy(
        self,
        a: torch.Tensor,
        r_l: torch.Tensor,
        extra_feats: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        atom_mask: Optional[torch.Tensor] = None,
        atom_to_token_idx: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        p_lm: Optional[torch.Tensor] = None,  # Added p_lm for compatibility
    ) -> torch.Tensor:
        """
        Legacy forward pass with individual parameters.

        Args:
            a: Token embedding, shape [..., N_token, c_token]
            r_l: Initial atom positions, shape [..., N_atom, 3]
            extra_feats: Additional atom features, shape [..., N_atom, c_atom]
            mask: Token mask, shape [..., N_token]
            atom_mask: Atom mask, shape [..., N_atom]
            atom_to_token_idx: Mapping from atoms to tokens, shape [..., N_atom]
            chunk_size: If set, break computations into smaller chunks to reduce memory usage
            p_lm: Pair embedding from encoder

        Returns:
            Predicted atom coordinates, shape [..., N_atom, 3]
        """
        params = DecoderForwardParams(
            a=a,
            r_l=r_l,
            extra_feats=extra_feats,
            p_lm=p_lm,  # Pass p_lm
            mask=mask,
            atom_mask=atom_mask,
            atom_to_token_idx=atom_to_token_idx,
            chunk_size=chunk_size,
        )
        return self.forward(params)

    # For backward compatibility - creates a config from args
    @classmethod
    def from_args(
        cls,
        n_blocks: int = 3,
        n_heads: int = 4,
        c_token: int = 384,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_s: int = 384,  # Added c_s
        n_queries: int = 32,
        n_keys: int = 128,
        blocks_per_ckpt: Optional[int] = None,
    ) -> "AtomAttentionDecoder":
        """
        Create AtomAttentionDecoder from individual arguments.

        Args:
            n_blocks: Number of blocks in the atom transformer
            n_heads: Number of attention heads
            c_token: Token embedding dimension
            c_atom: Atom embedding dimension
            c_atompair: Atom pair embedding dimension
            c_s: Single (style/conditioning) embedding dimension
            n_queries: Number of queries for local attention
            n_keys: Number of keys for local attention
            blocks_per_ckpt: Number of blocks per checkpoint for memory efficiency

        Returns:
            Initialized AtomAttentionDecoder
        """
        config = AtomAttentionConfig(
            has_coords=True,  # Decoder always deals with coordinates
            c_token=c_token,
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_s=c_s,  # Use provided c_s
            c_z=0,  # Not used directly in decoder config, but AtomTransformer needs it via c_s
            n_blocks=n_blocks,
            n_heads=n_heads,
            n_queries=n_queries,
            n_keys=n_keys,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        return cls(config)

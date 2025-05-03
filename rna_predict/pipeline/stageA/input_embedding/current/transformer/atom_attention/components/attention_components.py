"""
Attention components for atom attention.
"""

from typing import Optional

import torch
import torch.nn as nn

from rna_predict.pipeline.stageA.input_embedding.current.primitives import LinearNoBias
from rna_predict.pipeline.stageA.input_embedding.current.transformer.atom_transformer import (
    AtomTransformer,
)


class AttentionComponents:
    """Handles attention-related components and operations."""

    def __init__(
        self,
        c_atom: int,
        c_atompair: int,
        n_blocks: int,
        n_heads: int,
        n_queries: int,
        n_keys: int,
        blocks_per_ckpt: Optional[int] = None,
        debug_logging: bool = False,
    ):
        """
        Initialize the attention components.

        Args:
            c_atom: Atom embedding dimension
            c_atompair: Atom pair embedding dimension
            n_blocks: Number of transformer blocks
            n_heads: Number of attention heads
            n_queries: Number of queries
            n_keys: Number of keys
            blocks_per_ckpt: Number of blocks per checkpoint
            debug_logging: Whether to print debug logs
        """
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.debug_logging = debug_logging

        # Set up pair projections
        self._setup_pair_projections()

        # Set up small MLP
        self._setup_small_mlp()

        # Create atom transformer
        self.atom_transformer = self._create_atom_transformer(
            n_blocks, n_heads, blocks_per_ckpt
        )

    def _setup_pair_projections(self) -> None:
        """Set up linear projections for atom features to pair dimension."""
        self.linear_no_bias_cl = LinearNoBias(
            in_features=self.c_atom, out_features=self.c_atompair
        )
        self.linear_no_bias_cm = LinearNoBias(
            in_features=self.c_atom, out_features=self.c_atompair
        )

    def _setup_small_mlp(self) -> None:
        """Set up small MLP for pair feature processing."""
        self.small_mlp = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(in_features=self.c_atompair, out_features=self.c_atompair),
            nn.ReLU(),
            LinearNoBias(in_features=self.c_atompair, out_features=self.c_atompair),
            nn.ReLU(),
            LinearNoBias(in_features=self.c_atompair, out_features=self.c_atompair),
        )

    def _create_atom_transformer(
        self, n_blocks: int, n_heads: int, blocks_per_ckpt: Optional[int] = None
    ) -> AtomTransformer:
        """
        Create the AtomTransformer instance.

        Args:
            n_blocks: Number of blocks in transformer
            n_heads: Number of attention heads
            blocks_per_ckpt: Number of blocks per checkpoint

        Returns:
            Configured AtomTransformer instance
        """
        return AtomTransformer(
            n_blocks=n_blocks,
            n_heads=n_heads,
            c_atom=self.c_atom,
            c_atompair=self.c_atompair,
            n_queries=self.n_queries,
            n_keys=self.n_keys,
            blocks_per_ckpt=blocks_per_ckpt,
        )

    def process_pair_features(
        self,
        c_l: torch.Tensor,
        m_l: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process pair features through MLP.

        Args:
            c_l: Left pair tensor
            m_l: Middle pair tensor

        Returns:
            Processed pair features of shape [num_atoms, num_atoms, c_atompair]
        """
        # Project to pair dimension
        p_l = self.linear_no_bias_cl(c_l) + self.linear_no_bias_cm(
            m_l
        )  # [N, c_atompair]

        # Process through MLP
        p_l = self.small_mlp(p_l)  # [N, c_atompair]

        # Create outer product to get pair features
        p_i = p_l.unsqueeze(1)  # [N, 1, c_atompair]
        p_j = p_l.unsqueeze(0)  # [1, N, c_atompair]
        p_ij = p_i + p_j  # [N, N, c_atompair]

        # Ensure the tensor has the correct dtype and shape
        p_ij = p_ij.to(dtype=torch.float32)

        # Ensure the tensor has the correct last dimension
        if p_ij.shape[-1] != self.c_atompair:
            # This should not happen normally, but handle it just in case
            p_ij = p_ij.expand(*p_ij.shape[:-1], self.c_atompair)

        return p_ij

    def apply_transformer(
        self,
        a: torch.Tensor,
        p: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        debug_logging: bool = False,
    ) -> torch.Tensor:
        """
        Apply atom transformer to features.

        Args:
            a: Atom features
            p: Pair features
            mask: Attention mask
            chunk_size: Size of chunks for processing
            debug_logging: Whether to print debug logs

        Returns:
            Transformed features
        """
        # Create default mask if not provided
        if mask is None:
            mask = torch.ones_like(a[..., 0], dtype=torch.bool)

        # Ensure mask has correct shape
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)

        # Debug prints for tensor shapes
        if debug_logging:
            print(f"DEBUG: In apply_transformer: a.shape={a.shape}, p.shape={p.shape}, mask.shape={mask.shape}")
            print(f"DEBUG: c_atompair={self.c_atompair}, n_queries={self.n_queries}, n_keys={self.n_keys}")

        # Ensure pair features have correct dimensions
        if p.shape[-1] != self.c_atompair:
            # If p is a 3D tensor with last dim of 1, expand it to c_atompair
            if p.dim() == 3 and p.shape[-1] == 1:
                p = p.expand(*p.shape[:-1], self.c_atompair)
            # If p doesn't have a last dimension, add one and expand
            elif p.dim() < 3 or p.shape[-1] == 0:
                p = p.unsqueeze(-1).expand(*p.shape, self.c_atompair)
            # Otherwise just expand the last dimension
            else:
                p = p.unsqueeze(-1).expand(*p.shape[:-1], self.c_atompair)

        if debug_logging:
            print(f"DEBUG: After fixing dimensions: p.shape={p.shape}")

        # Convert mask to float32 if it's a boolean tensor
        if mask.dtype == torch.bool:
            mask = mask.to(dtype=torch.float32)

        # Create a token-level style/conditioning tensor with zeros
        # This is needed for the AtomTransformer.forward method
        batch_dims = a.shape[:-2]
        n_tokens = a.shape[-2]  # Use the same number of tokens as in 'a'
        s = torch.zeros(*batch_dims, n_tokens, self.atom_transformer.c_s,
                        device=a.device, dtype=a.dtype)

        # Use local attention with n_queries and n_keys
        # Pass arguments as keyword arguments to match the expected signature:
        # def forward(self, q: torch.Tensor, s: torch.Tensor, p: torch.Tensor, **kwargs)
        return self.atom_transformer(
            q=a,                # First positional arg: atom features
            s=s,                # Second positional arg: token/style features
            p=p,                # Third positional arg: pair features
            mask=mask,          # Keyword arg: attention mask
            chunk_size=chunk_size if chunk_size is not None else 0,
            n_queries=self.n_queries,  # Pass n_queries explicitly
            n_keys=self.n_keys,        # Pass n_keys explicitly
        )

# TODO: Refactor this file to improve code quality score - needs work on complexity and argument count

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
        """
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.n_queries = n_queries
        self.n_keys = n_keys

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
        self, c_l: torch.Tensor, m_l: torch.Tensor
    ) -> torch.Tensor:
        """
        Process pair features through MLP.

        Args:
            c_l: Left pair tensor
            m_l: Middle pair tensor

        Returns:
            Processed pair features
        """
        # Project to pair dimension
        p_l = self.linear_no_bias_cl(c_l) + self.linear_no_bias_cm(m_l)

        # Process through MLP
        return self.small_mlp(p_l)

    def apply_transformer(
        self,
        a: torch.Tensor,
        p: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Apply atom transformer to features.

        Args:
            a: Atom features
            p: Pair features
            mask: Attention mask
            chunk_size: Size of chunks for processing

        Returns:
            Transformed features
        """
        return self.atom_transformer(
            a,
            p,
            mask if mask is not None else torch.ones_like(a[..., 0], dtype=torch.bool),
            chunk_size if chunk_size is not None else 0,
        ) 
"""
Feature processor for atom attention.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from rna_predict.pipeline.stageA.input_embedding.current.primitives import LinearNoBias


class FeatureProcessor:
    """Processes atom and pair features for attention."""

    def __init__(
        self,
        c_atom: int,
        c_atompair: int,
        n_queries: int,
        n_keys: int,
    ):
        """
        Initialize the feature processor.

        Args:
            c_atom: Atom embedding dimension
            c_atompair: Atom pair embedding dimension
            n_queries: Number of queries
            n_keys: Number of keys
        """
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.n_queries = n_queries
        self.n_keys = n_keys

        # Set up linear projections
        self._setup_linear_projections()

    def _setup_linear_projections(self) -> None:
        """Set up linear projections for atom features."""
        self.linear_no_bias_q = LinearNoBias(
            in_features=self.c_atom, out_features=self.c_atom
        )
        self.linear_no_bias_k = LinearNoBias(
            in_features=self.c_atom, out_features=self.c_atom
        )
        self.linear_no_bias_v = LinearNoBias(
            in_features=self.c_atom, out_features=self.c_atom
        )

    def create_pair_embedding(
        self,
        a: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create pair embeddings from atom features.

        Args:
            a: Atom features tensor
            mask: Optional attention mask

        Returns:
            Tuple of (pair embeddings, attention mask)
        """
        # Project atom features
        q = self.linear_no_bias_q(a)
        k = self.linear_no_bias_k(a)
        v = self.linear_no_bias_v(a)

        # Create attention mask if not provided
        if mask is None:
            mask = torch.ones_like(a[..., 0], dtype=torch.bool)

        # Reshape for attention
        q = q.view(-1, self.n_queries, self.c_atom)
        k = k.view(-1, self.n_keys, self.c_atom)
        v = v.view(-1, self.n_keys, self.c_atom)

        # Create pair embeddings
        p = torch.matmul(q, k.transpose(-2, -1))
        p = p.view(-1, self.n_queries * self.n_keys, self.c_atompair)

        return p, mask 
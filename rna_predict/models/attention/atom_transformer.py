import warnings

import torch
import torch.nn as nn

from rna_predict.models.attention.block_sparse import (
    BlockSparseAttentionOptimized,
    LocalBlockSparseAttentionNaive,
    LocalSparseInput,
)
from rna_predict.utils.scatter_utils import layernorm

###############################################################################
# Atom Transformer Blocks
###############################################################################


class AtomTransformerBlock(nn.Module):
    """
    One transformer block consisting of local multi-head self-attention and an MLP.

    There are two possible local-attention paths:
      (1) A naive loop-based approach (LocalBlockSparseAttentionNaive)
      (2) An optimized approach using block_sparse_attn.

    You can toggle use_optimized = True/False.
    """

    def __init__(
        self, c_atom: int = 128, num_heads: int = 4, use_optimized: bool = False
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.c_atom = c_atom
        self.use_optimized = use_optimized

        # Linear projections for Q, K, and V.
        self.W_q = nn.Linear(c_atom, c_atom, bias=False)
        self.W_k = nn.Linear(c_atom, c_atom, bias=False)
        self.W_v = nn.Linear(c_atom, c_atom, bias=False)
        # To compute a per-head pairwise bias from pair embeddings.
        #
        # NOTE: We are feeding pair_emb.mean(dim=-1, keepdim=True),
        # so the input dimension is 1. This is the shape fix.
        self.W_b = nn.Linear(1, num_heads, bias=False)

        # Final projection after multi-head attention.
        self.W_out = nn.Linear(c_atom, c_atom, bias=False)

        # MLP transition (feed-forward network)
        self.mlp = nn.Sequential(
            nn.Linear(c_atom, 4 * c_atom),
            nn.SiLU(),
            nn.Linear(4 * c_atom, c_atom),
        )

        # If we use the optimized approach:
        if self.use_optimized:
            self.bs_attn = BlockSparseAttentionOptimized(
                num_heads, block_size=128, local_window=32, causal=False
            )

    def forward(
        self, x: torch.Tensor, pair_emb: torch.Tensor, block_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
          x: [N_atom, c_atom] – per-atom embeddings
          pair_emb: [N_atom, N_atom, c_pair] – pairwise embeddings
          block_index: [N_atom, block_size] – indices for local attention

        Returns:
          A Tensor of shape [N_atom, c_atom], updated after attention + MLP.
        """
        # (1) Apply LayerNorm.
        normed_embeddings = layernorm(x)

        # (2) Compute query, key, and value projections.
        q = self.W_q(normed_embeddings)
        k = self.W_k(normed_embeddings)
        v = self.W_v(normed_embeddings)

        # (3) Reshape for multi-head attention.
        c_per_head = self.c_atom // self.num_heads
        N_atom = x.size(0)
        q = q.view(N_atom, self.num_heads, c_per_head)
        k = k.view(N_atom, self.num_heads, c_per_head)
        v = v.view(N_atom, self.num_heads, c_per_head)

        # (4) Compute a pairwise bias:
        # We simply take the mean over the last dimension (c_pair) -> shape [N_atom, N_atom],
        # then unsqueeze for a single channel -> shape [N_atom, N_atom, 1]
        pair_mean = pair_emb.mean(dim=-1, keepdim=True)  # [N_atom, N_atom, 1]
        # Now we project from 1 -> num_heads:
        pair_bias_heads = self.W_b(pair_mean)  # [N_atom, N_atom, num_heads]

        # (5) local block-sparse attention
        if self.use_optimized:
            try:
                # Attempt to use the optimized block-sparse kernel.
                attention_output = self.bs_attn(q, k, v, pair_bias_heads)
            except RuntimeError as e:
                if not hasattr(self, "_optimized_warning_printed"):
                    warnings.warn(
                        f"Optimized block-sparse attention failed: {e} Falling back to naive attention."
                    )
                    self._optimized_warning_printed = True
                lsi = LocalSparseInput(q, k, v, pair_bias_heads, block_index)
                attention_output = LocalBlockSparseAttentionNaive.apply(
                    lsi.q, lsi.k, lsi.v, lsi.pair_bias, lsi.block_index
                )
        else:
            lsi = LocalSparseInput(q, k, v, pair_bias_heads, block_index)
            attention_output = LocalBlockSparseAttentionNaive.apply(
                lsi.q, lsi.k, lsi.v, lsi.pair_bias, lsi.block_index
            )
        attention_output = attention_output.reshape(N_atom, self.c_atom)

        # (7) Add the residual connection.
        x = x + self.W_out(attention_output)

        # (8) Apply a second residual branch with an MLP.
        normed_post_residual = layernorm(x)
        mlp_out = self.mlp(normed_post_residual)
        x = x + mlp_out

        return x


class AtomTransformer(nn.Module):
    """
    A stack of AtomTransformerBlock layers.
    """

    def __init__(
        self,
        c_atom: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        use_optimized: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                AtomTransformerBlock(c_atom, num_heads, use_optimized=use_optimized)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, x: torch.Tensor, pair_emb: torch.Tensor, block_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies a sequence of AtomTransformerBlock modules.

        Args:
          x: [N_atom, c_atom]
          pair_emb: [N_atom, N_atom, c_pair]
          block_index: [N_atom, block_size]

        Returns:
          A Tensor of shape [N_atom, c_atom] after the final block.
        """
        for block in self.blocks:
            x = block(x, pair_emb, block_index)
        return x

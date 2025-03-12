import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from rna_predict.utils import layernorm
from rna_predict.block_sparse_attention import BlockSparseAttentionOptimized, LocalBlockSparseAttentionNaive

###############################################################################
# Atom Transformer Blocks
###############################################################################

class AtomTransformerBlock(nn.Module):
    """
    One transformer block consisting of local multi-head self-attention and an MLP.

    Now we have two possible local-attention paths:
      (1) A naive loop-based approach (LocalBlockSparseAttentionNaive)
      (2) An optimized approach using block_sparse_attn.

    You can toggle use_optimized = True/False.
    """
    def __init__(self, c_atom=128, num_heads=4, use_optimized=False):
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
            nn.Linear(4 * c_atom, c_atom)
        )

        # If we use the optimized approach:
        if self.use_optimized:
            self.bs_attn = BlockSparseAttentionOptimized(num_heads, block_size=128,
                                                         local_window=32, causal=False)

    def forward(self, x, pair_emb, block_index):
        """
        Args:
          x: [N_atom, c_atom] – per-atom embeddings
          pair_emb: [N_atom, N_atom, c_pair] – pair embeddings (can be a lower dim than c_atom)
          block_index: [N_atom, block_size] – indices for local attention
        """
        # (1) Apply LayerNorm.
        x_ln = layernorm(x)
        # (2) Compute query, key, and value projections.
        q = self.W_q(x_ln)
        k = self.W_k(x_ln)
        v = self.W_v(x_ln)

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
            # Use the block-sparse kernel
            attn_out = self.bs_attn(q, k, v, pair_bias_heads)
        else:
            # Use naive version
            attn_out = LocalBlockSparseAttentionNaive.apply(q, k, v, pair_bias_heads, block_index)

        # (6) Merge heads.
        attn_out = attn_out.reshape(N_atom, self.c_atom)

        # (7) Add the residual connection.
        x = x + self.W_out(attn_out)

        # (8) Apply a second residual branch with an MLP.
        x_ln2 = layernorm(x)
        mlp_out = self.mlp(x_ln2)
        x = x + mlp_out

        return x

class AtomTransformer(nn.Module):
    """
    A stack of AtomTransformerBlock layers.
    """
    def __init__(self, c_atom=128, num_heads=4, num_layers=3, use_optimized=False):
        super().__init__()
        self.blocks = nn.ModuleList(
            [AtomTransformerBlock(c_atom, num_heads, use_optimized=use_optimized)
             for _ in range(num_layers)]
        )

    def forward(self, x, pair_emb, block_index):
        for block in self.blocks:
            x = block(x, pair_emb, block_index)
        return x

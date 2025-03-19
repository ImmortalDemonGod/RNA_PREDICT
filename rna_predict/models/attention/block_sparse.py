import math

import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
# NAIVE Local Block-Sparse Attention (with Naive Backprop)
###############################################################################


class LocalBlockSparseAttentionNaive(torch.autograd.Function):
    """
    (Old) naive local block-sparse attention. Each atom attends to a fixed set
    of neighbors specified by block_index. We keep this for demonstration;
    a new optimized version is introduced below.
    """

    @staticmethod
    def forward(ctx, q, k, v, pair_bias, block_index):
        """
        Args:
          q, k, v: [N_atom, n_heads, c_per_head]
          pair_bias: [N_atom, N_atom, n_heads]
          block_index: [N_atom, block_size]
        Returns:
          out: [N_atom, n_heads, c_per_head]
        """
        N_atom, n_heads, c_per_head = q.shape
        block_index.shape[1]
        out = torch.zeros_like(q)

        # We'll store intermediate results (and neighbors) for naive backward.
        saved_neighbors = []
        saved_attn_weights = []

        # Loop over each atom and perform local attention over its neighbors.
        for i in range(N_atom):
            neighbor_idxs = block_index[i]  # shape: [block_size]
            k_neighbors = k[neighbor_idxs]  # [block_size, n_heads, c_per_head]
            v_neighbors = v[neighbor_idxs]  # [block_size, n_heads, c_per_head]
            bias_neighbors = pair_bias[i, neighbor_idxs]  # [block_size, n_heads]

            q_i = q[i].unsqueeze(0)  # [1, n_heads, c_per_head]

            # (q_i · k_neighbors) scaled by sqrt(c_per_head)
            logits = (q_i * k_neighbors).sum(dim=-1) * (1.0 / math.sqrt(c_per_head))
            logits = logits + bias_neighbors  # add pairwise bias

            attn_weights = F.softmax(logits, dim=0)  # [block_size, n_heads]
            out[i] = (v_neighbors * attn_weights.unsqueeze(-1)).sum(dim=0)

            saved_neighbors.append(neighbor_idxs)
            saved_attn_weights.append(attn_weights)

        # Save for backward
        ctx.save_for_backward(q, k, v, pair_bias)
        ctx.block_index = block_index
        ctx.saved_neighbors = saved_neighbors
        ctx.saved_attn = saved_attn_weights
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        Naive backward pass for demonstration.
        """
        q, k, v, pair_bias = ctx.saved_tensors
        saved_neighbors = ctx.saved_neighbors
        saved_attn = ctx.saved_attn

        N_atom, n_heads, c_per_head = q.shape

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        dpbias = torch.zeros_like(pair_bias)

        # We'll redo partial derivatives in a naive loop over each atom
        for i in range(N_atom):
            neighbor_idxs = saved_neighbors[i]
            attn_weights = saved_attn[i]  # shape: [block_size, n_heads]
            grad_i = grad_out[i]  # shape: [n_heads, c_per_head]

            # =========================
            # gradient wrt v_neighbors
            dv_neighbors = grad_i.unsqueeze(0) * attn_weights.unsqueeze(-1)
            # [block_size, n_heads, c_per_head]
            dv[neighbor_idxs] += dv_neighbors

            # =========================
            # gradient wrt attn_weights
            v_neighbors = v[neighbor_idxs]
            dlogits = torch.sum(v_neighbors * grad_i.unsqueeze(0), dim=-1)
            # [block_size, n_heads]
            sum_d = torch.sum(attn_weights * dlogits, dim=0, keepdim=True)
            dlogits_ = attn_weights * (dlogits - sum_d)

            # =========================
            # gradient wrt pair_bias
            dpbias[i, neighbor_idxs] += dlogits_

            # =========================
            # gradient wrt q[i] and k_neighbors
            alpha = 1.0 / math.sqrt(c_per_head)
            dq_i = torch.sum(dlogits_.unsqueeze(-1) * (alpha * k[neighbor_idxs]), dim=0)
            dq[i] += dq_i

            dk_neighbors = dlogits_.unsqueeze(-1) * (alpha * q[i].unsqueeze(0))
            dk[neighbor_idxs] += dk_neighbors

        return dq, dk, dv, dpbias, None


###############################################################################
# Optimized Local Block-Sparse w/ Block Sparse Attention
###############################################################################

_HAS_BSA = False
try:
    from block_sparse_attn import block_sparse_attn_func

    _HAS_BSA = True
except ImportError:
    print("block_sparse_attn not found; using only naive LocalBlockSparseAttention.")


def build_local_blockmask(
    N_atom, block_size=128, local_window=32, nheads=4, causal=False
):
    """
    Build a 2D blockmask for local attention. We'll produce shape:
      [batch=1, nheads, nrow, ncol]
    where nrow = ceil(N_atom / block_size), ncol = ceil(N_atom / block_size).
    local_window is the band in #atoms to each side of the diagonal block.

    If block_sparse_attn is unavailable, we return None or do something minimal
    to avoid errors (the fallback path doesn't need this mask).
    """
    if not _HAS_BSA:
        return None

    from math import ceil

    nrow = ceil(N_atom / block_size)
    ncol = ceil(N_atom / block_size)
    mask = torch.zeros((1, nheads, nrow, ncol), dtype=torch.bool)

    blocks_per_side = max(1, local_window // block_size)
    for row_idx in range(nrow):
        c_min = max(0, row_idx - blocks_per_side)
        c_max = min(ncol, row_idx + blocks_per_side + 1)
        mask[:, :, row_idx, c_min:c_max] = True

    # If causal => zero out any block col > row
    if causal:
        for row_idx in range(nrow):
            mask[:, :, row_idx, row_idx + 1 :] = False

    return mask


class BlockSparseAttentionOptimized(nn.Module):
    """
    A module that calls block_sparse_attn_func from the block_sparse_attn library
    to do local block-sparse attention in a single fused kernel.

    If the library is not installed, we raise an error in forward().
    """

    def __init__(self, nheads, block_size=128, local_window=32, causal=False):
        super().__init__()
        self.nheads = nheads
        self.block_size = block_size
        self.local_window = local_window
        self.causal = causal

    def forward(self, q, k, v, pair_bias):
        """
        q, k, v: [N_atom, nheads, c_per_head]
        pair_bias: [N_atom, N_atom, nheads]
        Returns: [N_atom, nheads, c_per_head]
        """
        if not _HAS_BSA:
            raise RuntimeError(
                "block_sparse_attn not installed. No optimized path available."
            )

        device = q.device
        N_atom = q.shape[0]
        _, nheads, c_per_head = q.shape
        assert nheads == self.nheads, "Mismatch in #heads"

        # Flatten so shape = [N_atom, nheads, c_per_head]
        q_reshape = q.reshape(N_atom, nheads, c_per_head)
        k_reshape = k.reshape(N_atom, nheads, c_per_head)
        v_reshape = v.reshape(N_atom, nheads, c_per_head)

        # Typically you'd incorporate pair_bias into QK^T or use a custom kernel.
        # We skip it or zero it out for demonstration.

        blockmask = build_local_blockmask(
            N_atom, self.block_size, self.local_window, self.nheads, self.causal
        )
        if blockmask is not None:
            blockmask = blockmask.to(device)

        head_mask_type = torch.ones((nheads,), dtype=torch.int32, device=device)
        streaming_info = None
        cu_seqlens = torch.tensor([0, N_atom], dtype=torch.int32, device=device)
        max_seqlen = N_atom

        out_reshape = block_sparse_attn_func(
            q_reshape,
            k_reshape,
            v_reshape,
            cu_seqlens,
            cu_seqlens,
            head_mask_type,
            streaming_info,
            blockmask,
            max_seqlen,
            max_seqlen,
            p_dropout=0.0,
            deterministic=True,
            softmax_scale=None,
            is_causal=self.causal,
            exact_streaming=False,
            return_attn_probs=False,
        )

        return out_reshape

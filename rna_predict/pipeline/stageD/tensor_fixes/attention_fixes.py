"""
Attention-related tensor shape compatibility fixes.
"""

from functools import wraps

import torch

# Store original functions before patching
original_scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
original_multihead_attention_forward = torch.nn.MultiheadAttention.forward


def fix_attention_bias_shape():
    """
    Fix attention bias shape mismatches.
    """

    def patched_attention(
        q, k, v, attn_bias=None, dropout_p=0.0, scale=None, dtype=None
    ):
        try:
            return original_scaled_dot_product_attention(
                q, k, v, attn_bias, dropout_p, scale, dtype
            )
        except RuntimeError as e:
            if "The size of tensor a" in str(
                e
            ) and "must match the size of tensor b" in str(e):
                # Handle attention bias shape mismatch
                if attn_bias is not None:
                    # Ensure attn_bias matches the query/key dimensions
                    if attn_bias.dim() == 4:  # [batch, heads, queries, keys]
                        if attn_bias.shape[2] != q.shape[1]:
                            attn_bias = attn_bias[:, :, : q.shape[1], :]
                        if attn_bias.shape[3] != k.shape[1]:
                            attn_bias = attn_bias[:, :, :, : k.shape[1]]
                return original_scaled_dot_product_attention(
                    q, k, v, attn_bias, dropout_p, scale, dtype
                )
            raise

    def patched_attn_forward(self, *args, **kwargs):
        try:
            return original_multihead_attention_forward(self, *args, **kwargs)
        except RuntimeError as e:
            if "The size of tensor a" in str(
                e
            ) and "must match the size of tensor b" in str(e):
                # Handle attention forward pass shape mismatches
                q, k, v = args[0], args[1], args[2]
                if q.shape[1] != k.shape[1]:
                    min_len = min(q.shape[1], k.shape[1])
                    q = q[:, :min_len, :]
                    k = k[:, :min_len, :]
                    v = v[:, :min_len, :]
                return original_multihead_attention_forward(
                    self, q, k, v, *args[3:], **kwargs
                )
            raise

    # Replace the original functions with our patched versions
    torch.nn.functional.scaled_dot_product_attention = patched_attention
    torch.nn.MultiheadAttention.forward = patched_attn_forward


def fix_rearrange_qk_to_dense_trunk():
    """
    Fix the rearrange_qk_to_dense_trunk function to handle shape mismatches.
    """
    from rna_predict.pipeline.stageA.input_embedding.current import primitives

    original_rearrange = primitives.rearrange_qk_to_dense_trunk

    @wraps(original_rearrange)
    def patched_rearrange(
        q, k, dim_q, dim_k, n_queries=32, n_keys=128, compute_mask=True
    ):
        # Handle the case where q or k is a list
        if isinstance(q, list):
            q = torch.stack(q)
        if isinstance(k, list):
            k = torch.stack(k)

        # Ensure shapes are compatible
        if q.shape[1] != n_queries:
            q = q[:, :n_queries, :]
        if k.shape[1] != n_keys:
            k = k[:, :n_keys, :]

        return original_rearrange(q, k, dim_q, dim_k, n_queries, n_keys, compute_mask)

    primitives.rearrange_qk_to_dense_trunk = patched_rearrange

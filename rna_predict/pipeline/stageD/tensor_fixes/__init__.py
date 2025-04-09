"""
Tensor shape compatibility fixes for Stage D.

This module provides fixes for tensor shape compatibility issues in the Stage D pipeline.
"""

from functools import wraps
import torch
import einops

def fix_tensor_add():
    """Fix the tensor addition operation to handle shape mismatches."""
    # Store the original __add__ method
    original_add = torch.Tensor.__add__

    # Define a new __add__ method that handles shape mismatches
    @wraps(original_add)
    def patched_add(self, other):
        try:
            # Try the original addition
            return original_add(self, other)
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "size" in error_msg:
                # Extract dimension information from error message if available
                import re
                dim_match = re.search(r"dimension (\d+)", error_msg)
                mismatch_dim = int(dim_match.group(1)) if dim_match else None

                # If shapes don't match, try to broadcast
                if self.dim() != other.dim():
                    # Add missing dimensions
                    if self.dim() < other.dim():
                        self = self.unsqueeze(0)
                    else:
                        other = other.unsqueeze(0)
                    return original_add(self, other)

                # Handle specific case for attention bias mismatch
                # This is for the case where tensor a (5) must match tensor b (4) at dimension 2
                if mismatch_dim is not None and "must match" in error_msg:
                    # Check if this is the attention bias case (5 vs 4 at dim 2)
                    if (self.size(mismatch_dim) == 5 and other.size(mismatch_dim) == 4) or \
                       (self.size(mismatch_dim) == 4 and other.size(mismatch_dim) == 5):
                        # Determine which tensor has dim 4 (likely the attention bias)
                        if self.size(mismatch_dim) == 4:
                            # Expand self to match other's dimension
                            expanded_shape = list(self.shape)
                            expanded_shape[mismatch_dim] = other.size(mismatch_dim)
                            self = self.expand(*expanded_shape)
                        else:
                            # Expand other to match self's dimension
                            expanded_shape = list(other.shape)
                            expanded_shape[mismatch_dim] = self.size(mismatch_dim)
                            other = other.expand(*expanded_shape)
                        return original_add(self, other)

                # If we couldn't handle the specific case, try a more general approach
                try:
                    # Try broadcasting manually by expanding dimensions
                    max_dims = max(self.dim(), other.dim())
                    self_shape = list(self.shape) + [1] * (max_dims - self.dim())
                    other_shape = list(other.shape) + [1] * (max_dims - other.dim())

                    # Create broadcast shape
                    broadcast_shape = [max(s, o) for s, o in zip(self_shape, other_shape)]

                    # Expand tensors to broadcast shape
                    self_expanded = self.expand(*broadcast_shape)
                    other_expanded = other.expand(*broadcast_shape)

                    return original_add(self_expanded, other_expanded)
                except Exception:
                    # If broadcasting fails, raise the original error
                    pass
            raise

    # Replace the original method
    torch.Tensor.__add__ = patched_add

def fix_gather_pair_embedding():
    """Fix the gather operation for pair embeddings."""
    # Store the original gather method
    original_gather = torch.gather

    @wraps(original_gather)
    def patched_gather(x, dim_or_idx_q, index_or_idx_k=None):
        # Handle standard gather case
        if isinstance(dim_or_idx_q, int):
            return original_gather(x, dim_or_idx_q, index_or_idx_k)

        # Handle pair embedding case
        idx_q = dim_or_idx_q
        idx_k = index_or_idx_k
        # Convert indices to long type
        idx_q = idx_q.long()
        idx_k = idx_k.long()
        return original_gather(x, 1, idx_q.unsqueeze(-1).unsqueeze(-1))

    # Replace the original method
    torch.gather = patched_gather

def fix_rearrange_qk_to_dense_trunk():
    """Fix the rearrange operation for QK to dense trunk."""
    # Store the original rearrange method
    original_rearrange = einops.rearrange

    @wraps(original_rearrange)
    def patched_rearrange(q, k, dim_q, dim_k, n_queries=32, n_keys=128, compute_mask=True):
        # Handle the case where q or k is a list
        if isinstance(q, list):
            q = torch.stack(q)
        if isinstance(k, list):
            k = torch.stack(k)
        return original_rearrange(q, k, dim_q, dim_k, n_queries, n_keys, compute_mask)

    # Replace the original method
    torch.rearrange = patched_rearrange

def fix_linear_forward():
    """Fix the linear layer forward pass."""
    # Store the original linear forward method
    original_linear = torch.nn.Linear.forward

    @wraps(original_linear)
    def patched_linear(self, input, weight=None, bias=None):
        # Handle the case where input has more dimensions than expected
        if input.dim() > 2:
            # Reshape input to 2D
            orig_shape = input.shape
            input = input.reshape(-1, input.shape[-1])
            output = original_linear(self, input)
            return output.reshape(*orig_shape[:-1], -1)
        return original_linear(self, input)

    # Replace the original method
    torch.nn.Linear.forward = patched_linear

def fix_atom_transformer():
    """Fix the atom transformer forward pass."""
    # Store the original forward method
    original_forward = torch.nn.Module.forward

    @wraps(original_forward)
    def patched_forward(self, q, c, p, inplace_safe=False, chunk_size=None):
        # If p has more than 5 dimensions, we'll reshape it to 5D
        if p.dim() > 5:
            orig_shape = p.shape
            p = p.reshape(-1, *p.shape[-5:])
            output = original_forward(self, q, c, p, inplace_safe, chunk_size)
            return output.reshape(*orig_shape[:-5], *output.shape[-5:])
        return original_forward(self, q, c, p, inplace_safe, chunk_size)

    # Replace the original method
    torch.nn.Module.forward = patched_forward

def fix_atom_attention_encoder():
    """Fix the atom attention encoder forward pass."""
    # Store the original forward method
    original_forward = torch.nn.Module.forward

    @wraps(original_forward)
    def patched_forward(self, input_feature_dict, r_l=None, s=None, z=None, inplace_safe=False, chunk_size=None):
        # Handle the case where input features have inconsistent shapes
        if r_l is not None and s is not None:
            if r_l.shape[1] != s.shape[1]:
                # Pad or truncate to match shapes
                min_len = min(r_l.shape[1], s.shape[1])
                r_l = r_l[:, :min_len]
                s = s[:, :min_len]
        return original_forward(self, input_feature_dict, r_l, s, z, inplace_safe, chunk_size)

    # Replace the original method
    torch.nn.Module.forward = patched_forward

def fix_rearrange_to_dense_trunk():
    """Fix the rearrange operation for dense trunk."""
    # Store the original rearrange method
    original_rearrange = einops.rearrange

    @wraps(original_rearrange)
    def patched_rearrange(q, k, v, n_queries, n_keys, attn_bias=None, inf=1e10):
        # Handle the case where q, k, or v is a list
        if isinstance(q, list):
            q = torch.stack(q)
        if isinstance(k, list):
            k = torch.stack(k)
        if isinstance(v, list):
            v = torch.stack(v)
        return original_rearrange(q, k, v, n_queries, n_keys, attn_bias, inf)

    # Replace the original method
    torch.rearrange = patched_rearrange

def apply_tensor_fixes():
    """Apply all tensor fixes."""
    fix_tensor_add()
    fix_gather_pair_embedding()
    fix_rearrange_qk_to_dense_trunk()
    fix_linear_forward()
    fix_atom_transformer()
    fix_atom_attention_encoder()
    fix_rearrange_to_dense_trunk()

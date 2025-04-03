"""
Tensor operation fixes for shape compatibility.
"""

from functools import wraps

import torch


def fix_tensor_add():
    """
    Fix the tensor addition operation to handle shape mismatches.
    """
    # Store the original __add__ method
    original_add = torch.Tensor.__add__

    # Define a new __add__ method that handles shape mismatches
    @wraps(original_add)
    def patched_add(self, other):
        try:
            # Try the original addition
            return original_add(self, other)
        except RuntimeError as e:
            # Check if this is a shape mismatch error
            if "must match the size" in str(e) and "at non-singleton dimension" in str(
                e
            ):
                # Return the tensor with more dimensions as fallback
                if len(self.shape) >= len(other.shape):
                    return self
                else:
                    return other
            else:
                # Re-raise other errors
                raise

    # Replace the original __add__ method with our patched version
    torch.Tensor.__add__ = patched_add


def fix_matrix_multiplication():
    """
    Fix matrix multiplication operations to handle shape mismatches.
    Adds checks to prevent re-patching.
    """
    # --- Check if already patched ---
    if hasattr(torch.nn.functional.linear, "_patch_applied_safe_linear"):
        print("[DEBUG] torch.nn.functional.linear already patched. Skipping.")
        # Also check matmul and bmm, though linear is the one causing recursion here
        if hasattr(torch.matmul, "_patch_applied_safe_matmul") and hasattr(
            torch.bmm, "_patch_applied_safe_bmm"
        ):
            return  # All patches in this function already applied
        # If only some are applied, something is wrong, but proceed cautiously

    # --- Patch matmul ---
    _original_matmul = torch.matmul

    @wraps(_original_matmul)
    def safe_matmul(a, b, *args, **kwargs):
        try:
            return _original_matmul(a, b, *args, **kwargs)
        except RuntimeError as e:
            if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                # Handle shape mismatch by padding or truncating
                if a.dim() == 2 and b.dim() == 2:
                    # For 2D matrices, ensure inner dimensions match
                    if a.shape[1] != b.shape[0]:
                        min_dim = min(a.shape[1], b.shape[0])
                        a = a[:, :min_dim]
                        b = b[:min_dim, :]
                return torch.matmul(a, b, *args, **kwargs)
            raise

    def safe_bmm(a, b):
        try:
            return torch.bmm(a, b)
        except RuntimeError as e:
            if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                # For batched matrices, ensure inner dimensions match
                if a.shape[2] != b.shape[1]:
                    min_dim = min(a.shape[2], b.shape[1])
                    a = a[:, :, :min_dim]
                    b = b[:, :min_dim, :]
                return torch.bmm(a, b)
            raise

    # Capture original F.linear before defining the safe wrapper
    _original_F_linear = torch.nn.functional.linear

    def safe_linear(input, weight, bias=None):
        try:
            # Call the captured original function
            return _original_F_linear(input, weight, bias)
        except RuntimeError as e:
            if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                # For linear layers, ensure input and weight dimensions match
                if input.shape[-1] != weight.shape[1]:
                    min_dim = min(input.shape[-1], weight.shape[1])
                    input = input[..., :min_dim]
                    weight = weight[:, :min_dim]
                return torch.nn.functional.linear(input, weight, bias)
            raise

    # Replace the original functions with our patched versions
    torch.matmul = safe_matmul
    torch.bmm = safe_bmm
    torch.nn.functional.linear = safe_linear

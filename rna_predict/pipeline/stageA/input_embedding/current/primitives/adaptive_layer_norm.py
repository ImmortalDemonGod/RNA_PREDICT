"""
Adaptive Layer Normalization implementation.

This module contains the AdaptiveLayerNorm class which implements Algorithm 26 in AF3.
"""

import warnings
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import Linear
from .adaptive_layer_norm_utils import (
    adjust_tensor_shapes,
    check_and_adjust_dimensions,
    needs_singleton_dimension,
    restore_original_shape,
)
from .linear_primitives import LinearNoBias


class AdaptiveLayerNorm(nn.Module):
    """
    Implements Algorithm 26 in AF3
    """

    def __init__(self, c_a: int = 768, c_s: int = 384, c_s_layernorm: Optional[int] = None) -> None:
        """
        Args:
            c_a (int, optional): the embedding dim of a(single feature aggregated atom info). Defaults to 768.
            c_s (int, optional): hidden dim [for single embedding]. Defaults to 384.
            c_s_layernorm (Optional[int], optional): layer norm dimension for single embedding. Defaults to None.
        """
        super().__init__()
        self.c_a = c_a  # Store c_a for reference
        self.c_s = c_s  # Store c_s for reference
        if c_s_layernorm is None:
            c_s_layernorm = c_s
        self.c_s_layernorm = c_s_layernorm
        self.layernorm_a = nn.LayerNorm(c_a, elementwise_affine=True)
        self.layernorm_s = nn.LayerNorm(c_s_layernorm, elementwise_affine=True)
        self.linear_s = Linear(in_features=c_s, out_features=c_a)
        self.linear_nobias_s = LinearNoBias(in_features=c_s, out_features=c_a)

    def zero_init(self) -> None:
        """Initialize the weights and biases to zero."""
        nn.init.zeros_(self.linear_s.weight)
        nn.init.zeros_(self.linear_s.bias)
        nn.init.zeros_(self.linear_nobias_s.weight)

    def _prepare_scale_and_shift(
        self, s: torch.Tensor, a: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare scale and shift tensors from the conditioning tensor s.

        Args:
            s: Conditioning tensor (already normalized)
            a: Target tensor for shape reference

        Returns:
            Tuple of (scale tensor, shift tensor)
        """
        from rna_predict.utils.shape_utils import adjust_tensor_feature_dim

        # Ensure s has the correct feature dimension
        s = adjust_tensor_feature_dim(s, self.c_s, tensor_name="AdaLN conditioning 's'")

        # Generate scale and shift tensors
        scale = torch.sigmoid(self.linear_s(s))
        shift = self.linear_nobias_s(s)

        # Add singleton dimension if needed for broadcasting
        if needs_singleton_dimension(a, scale):
            scale = scale.unsqueeze(1)  # Shape [B, 1, N, C_a]
            shift = shift.unsqueeze(1)  # Shape [B, 1, N, C_a]

        return scale, shift

    def _try_broadcasting(
        self, a: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
    ) -> torch.Tensor:
        """
        Try to use PyTorch's broadcasting to align tensor shapes.

        Args:
            a: Input tensor
            scale: Scale tensor
            shift: Shift tensor

        Returns:
            Conditioned tensor if broadcasting succeeds

        Raises:
            RuntimeError: If broadcasting fails
        """
        a_b, scale_b, shift_b = torch.broadcast_tensors(a, scale, shift)
        return scale_b * a_b + shift_b

    
    def _apply_conditioning(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        print(f"[DEBUG][AdaLN][_apply_conditioning] ENTRY: a.requires_grad={a.requires_grad}, s.requires_grad={s.requires_grad}, a.shape={a.shape}, s.shape={s.shape}")
        """
        Apply conditioning from s to a.

        Args:
            a (torch.Tensor): normalized representation
            s (torch.Tensor): normalized single embedding

        Returns:
            torch.Tensor: conditioned tensor with proper shape adjustment
        """
        # Store original shape for later restoration
        a_original_shape = a.shape

        # Step 1: Check and adjust dimensions
        a, a_was_unsqueezed = check_and_adjust_dimensions(a, s)

        # Step 2: Prepare scale and shift tensors
        scale, shift = self._prepare_scale_and_shift(s, a)
        print(f"[DEBUG][AdaLN][_apply_conditioning] after _prepare_scale_and_shift: scale.shape={scale.shape}, shift.shape={shift.shape}, scale.requires_grad={scale.requires_grad}, shift.requires_grad={shift.requires_grad}")

        # Step 3: Try broadcasting approach first
        try:
            conditioned_a = self._try_broadcasting(a, scale, shift)
            print(f"[DEBUG][AdaLN][_apply_conditioning] after _try_broadcasting: conditioned_a.requires_grad={conditioned_a.requires_grad}, conditioned_a.shape={conditioned_a.shape}")
            return restore_original_shape(
                conditioned_a, a_original_shape, a_was_unsqueezed
            )

        except RuntimeError as e:
            # Step 4: If broadcasting fails, use direct shape adjustment
            warnings.warn(
                f"Broadcasting failed in AdaptiveLayerNorm: {e}. Attempting direct shape adjustment."
            )

            # Adjust tensor shapes to be compatible
            scale, shift = adjust_tensor_shapes(scale, shift, a)
            print(f"[DEBUG][AdaLN][_apply_conditioning] after adjust_tensor_shapes: scale.shape={scale.shape}, shift.shape={shift.shape}, scale.requires_grad={scale.requires_grad}, shift.requires_grad={shift.requires_grad}")

            # Apply conditioning directly
            conditioned_a = scale * a + shift
            print(f"[DEBUG][AdaLN][_apply_conditioning] after direct conditioning: conditioned_a.requires_grad={conditioned_a.requires_grad}, conditioned_a.shape={conditioned_a.shape}")

            # Restore original shape if needed
            return restore_original_shape(
                conditioned_a, a_original_shape, a_was_unsqueezed
            )

    #@snoop
    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        print(f"[DEBUG][AdaLN] ENTRY: a.requires_grad={a.requires_grad}, s.requires_grad={s.requires_grad}, a.shape={a.shape}, s.shape={s.shape}")
        print(f"[DEBUG][AdaLN] ENTRY: a.device={a.device}, a.dtype={a.dtype}, a.is_leaf={a.is_leaf}, a.grad_fn={a.grad_fn}")
        print(f"[DEBUG][AdaLN] ENTRY: s.device={s.device}, s.dtype={s.dtype}, s.is_leaf={s.is_leaf}, s.grad_fn={s.grad_fn}")
        print(f"[DEBUG][AdaLN] layernorm_a.weight.device={self.layernorm_a.weight.device if hasattr(self.layernorm_a, 'weight') and self.layernorm_a.weight is not None else 'N/A'}, layernorm_a.weight.dtype={self.layernorm_a.weight.dtype if hasattr(self.layernorm_a, 'weight') and self.layernorm_a.weight is not None else 'N/A'}")
        print(f"[DEBUG][AdaLN] layernorm_s.weight.device={self.layernorm_s.weight.device if hasattr(self.layernorm_s, 'weight') and self.layernorm_s.weight is not None else 'N/A'}, layernorm_s.weight.dtype={self.layernorm_s.weight.dtype if hasattr(self.layernorm_s, 'weight') and self.layernorm_s.weight is not None else 'N/A'}")
        """
        Args:
            a (torch.Tensor): the single feature aggregate per-atom representation
                [..., N_token, c_a]
            s (torch.Tensor): single embedding
                [..., N_token, c_s]

        Returns:
            torch.Tensor: the updated a from AdaLN
                [..., N_token, c_a]
        """
        from rna_predict.utils.shape_utils import adjust_tensor_feature_dim

        # Ensure a has the correct feature dimension (self.c_a)
        a = adjust_tensor_feature_dim(a, self.c_a, tensor_name="AdaLN input 'a'")
        print(f"[DEBUG][AdaLN] after adjust_tensor_feature_dim: a.device={a.device}, a.dtype={a.dtype}, a.is_leaf={a.is_leaf}, a.grad_fn={a.grad_fn}")

        # Normalize inputs
        a_norm = self.layernorm_a(a)
        print(f"[DEBUG][AdaLN] after layernorm_a: a_norm.requires_grad={a_norm.requires_grad}, a_norm.shape={a_norm.shape}")
        print(f"[DEBUG][AdaLN] after layernorm_a: a_norm.device={a_norm.device}, a_norm.dtype={a_norm.dtype}, a_norm.is_leaf={a_norm.is_leaf}, a_norm.grad_fn={a_norm.grad_fn}")

        # Ensure s matches the expected feature dimension for layernorm_s; adjust if not
        s_dim = s.shape[-1]
        ln_dim = self.layernorm_s.normalized_shape[0]
        if s_dim != ln_dim:
            warnings.warn(f"[AdaLN] s.shape[-1] ({s_dim}) does not match layernorm_s.normalized_shape ({ln_dim}). Recreating layernorm_s.")
            import torch.nn as nn
            # Recreate layernorm_s to match new s_dim
            self.layernorm_s = nn.LayerNorm(s_dim, elementwise_affine=True).to(s.device)
            # Recreate linear mappings to match new s dimension
            self.linear_s = nn.Linear(in_features=s_dim, out_features=self.c_a).to(s.device)
            self.linear_nobias_s = LinearNoBias(in_features=s_dim, out_features=self.c_a).to(s.device)
            # Update stored s dimensions to prevent future mismatch
            self.c_s = s_dim
            self.c_s_layernorm = s_dim
        s_norm = self.layernorm_s(s)
        print(f"[DEBUG][AdaLN] after layernorm_s: s_norm.requires_grad={s_norm.requires_grad}, s_norm.shape={s_norm.shape}")
        print(f"[DEBUG][AdaLN] after layernorm_s: s_norm.device={s_norm.device}, s_norm.dtype={s_norm.dtype}, s_norm.is_leaf={s_norm.is_leaf}, s_norm.grad_fn={s_norm.grad_fn}")

        # Apply conditioning
        out = self._apply_conditioning(a_norm, s_norm)
        print(f"[DEBUG][AdaLN] RETURN: out.requires_grad={out.requires_grad}, out.shape={out.shape}")
        return out

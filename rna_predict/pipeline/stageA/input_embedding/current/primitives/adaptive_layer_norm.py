"""
Adaptive Layer Normalization implementation.

This module contains the AdaptiveLayerNorm class which implements Algorithm 26 in AF3.
"""

import warnings

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

    def __init__(self, c_a: int = 768, c_s: int = 384) -> None:
        """
        Args:
            c_a (int, optional): the embedding dim of a(single feature aggregated atom info). Defaults to 768.
            c_s (int, optional): hidden dim [for single embedding]. Defaults to 384.
        """
        super().__init__()
        self.c_a = c_a  # Store c_a for reference
        self.c_s = c_s  # Store c_s for reference
        self.layernorm_a = nn.LayerNorm(c_a, elementwise_affine=False, bias=False)
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

        # Step 3: Try broadcasting approach first
        try:
            conditioned_a = self._try_broadcasting(a, scale, shift)
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

            # Apply conditioning directly
            conditioned_a = scale * a + shift

            # Restore original shape if needed
            return restore_original_shape(
                conditioned_a, a_original_shape, a_was_unsqueezed
            )

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
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

        # Normalize inputs
        a_norm = self.layernorm_a(a)

        # Create a new layer norm for s with the correct dimension
        s_last_dim = s.size(-1)  # Use size() instead of shape
        layernorm_s = nn.LayerNorm(s_last_dim, bias=False).to(s.device)
        s_norm = layernorm_s(s)

        # Apply conditioning
        return self._apply_conditioning(a_norm, s_norm)

"""
Adaptive Layer Normalization implementation.

This module contains the AdaptiveLayerNorm class which implements Algorithm 26 in AF3.
"""

import warnings
import torch
import torch.nn as nn
from torch.nn import Linear

from .linear_primitives import LinearNoBias
from .adaptive_layer_norm_utils import (
    check_and_adjust_dimensions,
    needs_singleton_dimension,
    adjust_tensor_shapes,
    restore_original_shape
)


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
        super(AdaptiveLayerNorm, self).__init__()
        self.c_a = c_a  # Store c_a for reference
        self.c_s = c_s  # Store c_s for reference
        self.layernorm_a = nn.LayerNorm(c_a, elementwise_affine=False, bias=False)
        self.linear_s = Linear(in_features=c_s, out_features=c_a)
        self.linear_nobias_s = LinearNoBias(in_features=c_s, out_features=c_a)

    def zero_init(self) -> None:
        """Zero-initialize conditioning layer parameters.
        
        Zeroes out the weights and biases of the linear conditioning layer `linear_s`
        and the weights of the bias-less layer `linear_nobias_s` to ensure that the
        conditioning mechanism starts with no initial influence.
        """
        nn.init.zeros_(self.linear_s.weight)
        nn.init.zeros_(self.linear_s.bias)
        nn.init.zeros_(self.linear_nobias_s.weight)



    def _prepare_scale_and_shift(
        self, s: torch.Tensor, a: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepares scale and shift factors for adaptive layer normalization.
        
        Adjusts the feature dimension of the normalized conditioning tensor s to match the
        expected hidden size and computes the corresponding scale and shift tensors. The
        scale tensor is generated using a sigmoid-activated linear transformation, and the
        shift tensor is produced using a linear transformation without bias. If the target
        tensor a requires an additional singleton dimension for proper broadcasting, this
        function unsqueezes the scale and shift tensors accordingly.
        
        Args:
            s: Normalized conditioning tensor whose feature dimension is adjusted to match the hidden size.
            a: Target tensor used to infer the required shape for broadcasting.
        
        Returns:
            A tuple (scale, shift) containing the tensors used for adaptive layer normalization.
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
        Attempts to broadcast the input, scale, and shift tensors and apply conditioning.
        
        This function aligns the shapes of the given tensors using PyTorch's broadcasting and
        computes the conditioned tensor as (scale * a) + shift.
        
        Args:
            a: The tensor to be conditioned.
            scale: The tensor of scaling factors.
            shift: The tensor of shifting values.
        
        Returns:
            The conditioned tensor computed using the broadcasted values of a, scale, and shift.
        
        Raises:
            RuntimeError: If broadcasting the tensors to a common shape is not possible.
        """
        a_b, scale_b, shift_b = torch.broadcast_tensors(a, scale, shift)
        return scale_b * a_b + shift_b



    def _apply_conditioning(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Condition a normalized tensor using a conditioning embedding.
        
        Adjusts the input tensor's dimensions and computes scaling and shifting factors from
        the conditioning tensor. The method first attempts to apply the conditioning via
        broadcasting; if that fails, it falls back to direct shape adjustment after issuing a
        warning. The final output is restored to the original input shape.
          
        Args:
            a (torch.Tensor): Normalized representation tensor.
            s (torch.Tensor): Normalized conditioning embedding.
          
        Returns:
            torch.Tensor: The conditioned tensor with the original shape.
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

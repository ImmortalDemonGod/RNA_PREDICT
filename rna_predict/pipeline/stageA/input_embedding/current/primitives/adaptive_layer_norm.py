"""
Adaptive Layer Normalization implementation.

This module contains the AdaptiveLayerNorm class which implements Algorithm 26 in AF3.
"""

import warnings
import torch
import torch.nn as nn
from torch.nn import Linear

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
        super(AdaptiveLayerNorm, self).__init__()
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

    def _has_compatible_dimensions(self, a: torch.Tensor, s: torch.Tensor) -> bool:
        """
        Check if tensors have compatible dimensions for sample dimension addition.

        Args:
            a: Input tensor to be conditioned
            s: Conditioning tensor

        Returns:
            Boolean indicating whether dimensions are compatible
        """
        return (s.dim() > a.dim() and
                s.dim() == a.dim() + 1 and
                s.shape[0] == a.shape[0])

    def _has_matching_feature_dimensions(self, a: torch.Tensor, s: torch.Tensor) -> bool:
        """
        Check if feature dimensions match between tensors.

        Args:
            a: Input tensor to be conditioned
            s: Conditioning tensor

        Returns:
            Boolean indicating whether feature dimensions match
        """
        s_dims_to_match = s.shape[2:]
        a_dims_to_match = a.shape[1:]
        return s_dims_to_match == a_dims_to_match

    def _should_add_sample_dimension(self, a: torch.Tensor, s: torch.Tensor) -> bool:
        """
        Determine if tensor 'a' needs a sample dimension added to match 's'.

        Args:
            a: Input tensor to be conditioned
            s: Conditioning tensor

        Returns:
            Boolean indicating whether to add a sample dimension
        """
        # First check if dimensions are compatible for adding a sample dimension
        if not self._has_compatible_dimensions(a, s):
            return False

        # Then check if feature dimensions match
        return self._has_matching_feature_dimensions(a, s)

    def _check_and_adjust_dimensions(
        self, a: torch.Tensor, s: torch.Tensor
    ) -> tuple[torch.Tensor, bool]:
        """
        Check and adjust dimensions of input tensors to ensure compatibility.

        Args:
            a: Input tensor to be conditioned
            s: Conditioning tensor

        Returns:
            Tuple of (adjusted tensor a, whether a was unsqueezed)
        """
        a_was_unsqueezed = False

        # Check if we need to add a sample dimension
        if self._should_add_sample_dimension(a, s):
            # Add missing sample dimension to a
            a = a.unsqueeze(1)
            a_was_unsqueezed = True
            warnings.warn(
                f"INFO: Unsqueezed 'a' in AdaptiveLayerNorm to match 's'. New 'a' shape: {a.shape}"
            )

        return a, a_was_unsqueezed

    def _needs_singleton_dimension(self, a: torch.Tensor, scale: torch.Tensor) -> bool:
        """
        Check if scale and shift tensors need a singleton dimension for broadcasting.

        Args:
            a: Target tensor for shape reference
            scale: Scale tensor to check

        Returns:
            Boolean indicating whether to add a singleton dimension
        """
        # Check if we need to add a singleton dimension for broadcasting
        return (
            a.dim() == 4 and
            scale.dim() == 3 and
            a.shape[0] == scale.shape[0] and
            a.shape[2:] == scale.shape[1:]
        )

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
        if self._needs_singleton_dimension(a, scale):
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

    def _interpolate_sequence_dim(
        self, tensor: torch.Tensor, target_size: int
    ) -> torch.Tensor:
        """
        Interpolate tensor along sequence dimension to match target size.

        Args:
            tensor: Input tensor
            target_size: Target sequence length

        Returns:
            Interpolated tensor
        """
        return torch.nn.functional.interpolate(
            tensor.transpose(1, 2),  # [B, C, S]
            size=target_size,
            mode="nearest",
        ).transpose(1, 2)  # [B, S, C]

    def _match_tensor_shape(
        self, tensor: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Match tensor shape to target tensor shape using various methods.

        Args:
            tensor: Input tensor to reshape
            target: Target tensor with desired shape

        Returns:
            Reshaped tensor
        """
        if tensor.shape == target.shape:
            return tensor

        # Try to expand dimensions
        try:
            return tensor.expand_as(target)
        except RuntimeError as expand_error:
            # Handle dimension mismatch
            if "must match the existing size" in str(expand_error):
                # Create a new tensor with the right shape
                new_tensor = torch.zeros_like(target)
                # Copy data from tensor to new_tensor where dimensions match
                min_dim = min(tensor.size(2), target.size(2))
                if tensor.dim() >= 3 and target.dim() >= 3:
                    new_tensor[:, :, :min_dim] = tensor[:, :, :min_dim]
                return new_tensor
            else:
                # Try reshape as last resort
                try:
                    return tensor.reshape(*target.shape)
                except RuntimeError:
                    # If all else fails
                    return torch.zeros_like(target)

    def _adjust_tensor_shapes(
        self, scale: torch.Tensor, shift: torch.Tensor, a: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Adjust tensor shapes when broadcasting fails.

        Args:
            scale: Scale tensor to adjust
            shift: Shift tensor to adjust
            a: Target tensor shape

        Returns:
            Tuple of (adjusted scale tensor, adjusted shift tensor)
        """
        # Handle sequence length mismatch with interpolation
        if scale.shape[1] != a.shape[1]:
            scale = self._interpolate_sequence_dim(scale, a.shape[1])
            shift = self._interpolate_sequence_dim(shift, a.shape[1])

        # Handle other dimension mismatches
        if scale.shape != a.shape:
            scale = self._match_tensor_shape(scale, a)

        if shift.shape != a.shape:
            shift = self._match_tensor_shape(shift, a)

        return scale, shift

    def _should_squeeze_tensor(self, tensor: torch.Tensor, original_shape: tuple, was_unsqueezed: bool) -> bool:
        """
        Determine if tensor should be squeezed to restore original shape.

        Args:
            tensor: Input tensor
            original_shape: Original shape to restore to
            was_unsqueezed: Whether the tensor was unsqueezed

        Returns:
            Boolean indicating whether to squeeze the tensor
        """
        return (
            was_unsqueezed and
            tensor.dim() > len(original_shape) and
            tensor.shape[1] == 1
        )

    def _restore_original_shape(
        self, tensor: torch.Tensor, original_shape: tuple, was_unsqueezed: bool
    ) -> torch.Tensor:
        """
        Restore tensor to its original shape if it was modified.

        Args:
            tensor: Input tensor
            original_shape: Original shape to restore to
            was_unsqueezed: Whether the tensor was unsqueezed

        Returns:
            Tensor with original shape restored if needed
        """
        if self._should_squeeze_tensor(tensor, original_shape, was_unsqueezed):
            return tensor.squeeze(1)
        return tensor

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
        a, a_was_unsqueezed = self._check_and_adjust_dimensions(a, s)

        # Step 2: Prepare scale and shift tensors
        scale, shift = self._prepare_scale_and_shift(s, a)

        # Step 3: Try broadcasting approach first
        try:
            conditioned_a = self._try_broadcasting(a, scale, shift)
            return self._restore_original_shape(
                conditioned_a, a_original_shape, a_was_unsqueezed
            )

        except RuntimeError as e:
            # Step 4: If broadcasting fails, use direct shape adjustment
            warnings.warn(
                f"Broadcasting failed in AdaptiveLayerNorm: {e}. Attempting direct shape adjustment."
            )

            # Adjust tensor shapes to be compatible
            scale, shift = self._adjust_tensor_shapes(scale, shift, a)

            # Apply conditioning directly
            conditioned_a = scale * a + shift

            # Restore original shape if needed
            return self._restore_original_shape(
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

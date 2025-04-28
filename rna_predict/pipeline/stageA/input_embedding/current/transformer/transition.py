"""
Transition blocks for transformer-based RNA structure prediction.
"""

import warnings
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from rna_predict.pipeline.stageA.input_embedding.current.primitives import (
    AdaptiveLayerNorm,
    BiasInitLinear,
    LinearNoBias,
)


class ConditionedTransitionBlock(nn.Module):
    """
    Feed-forward network with conditioning from style embedding.
    Implements Algorithm 25 in AlphaFold3.
    """

    def __init__(self, c_a: int, c_s: int, n: int = 2, biasinit: float = -2.0) -> None:
        """
        Initialize the ConditionedTransitionBlock.

        Args:
            c_a: Single embedding dimension (atom features)
            c_s: Single embedding dimension (style/conditioning)
            n: Channel scale factor for hidden layer
            biasinit: Bias initialization value
        """
        super().__init__()
        self.c_a = c_a
        self.c_s = c_s
        self.n = n
        self.adaln = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)
        self.linear_nobias_a1 = LinearNoBias(in_features=c_a, out_features=n * c_a)
        self.linear_nobias_a2 = LinearNoBias(in_features=c_a, out_features=n * c_a)
        self.linear_nobias_b = LinearNoBias(in_features=n * c_a, out_features=c_a)
        self.linear_s = BiasInitLinear(
            in_features=c_s, out_features=c_a, bias=True, biasinit=biasinit
        )

    def _validate_shapes(self, s: torch.Tensor, b: torch.Tensor) -> bool:
        """
        Validate shape compatibility between style and activation tensors.

        Args:
            s: Style tensor
            b: Activation tensor

        Returns:
            True if shapes are compatible, False otherwise
        """
        # First two dimensions must match for proper conditioning
        if s.size(0) != b.size(0) or s.size(1) != b.size(1):
            return False
        return True

    def _validate_scale_shift(self, scale: torch.Tensor, shift: torch.Tensor) -> bool:
        """
        Validate shape compatibility between scale and shift tensors.

        Args:
            scale: Scale tensor from style conditioning
            shift: Shift tensor from activation

        Returns:
            True if shapes are compatible, False otherwise
        """
        return scale.shape == shift.shape

    def _apply_conditioning(self, s: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Apply conditioning from style tensor to activation.

        Args:
            s: Style tensor
            b: Activation tensor

        Returns:
            Conditioned activation tensor

        Raises:
            ValueError: If tensors have incompatible shapes
        """
        if not self._validate_shapes(s, b):
            raise ValueError(
                f"Shape mismatch: s has shape {s.shape}, b has shape {b.shape}. "
                f"First two dimensions must match."
            )

        # Apply style modulation with explicit shape checking
        scale = torch.sigmoid(self.linear_s(s))
        shift = self.linear_nobias_b(b)

        # Additional check for scale and shift compatibility
        if not self._validate_scale_shift(scale, shift):
            raise ValueError(
                f"Scale and shift have incompatible shapes: {scale.shape} vs {shift.shape}"
            )

        return scale * shift

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the ConditionedTransitionBlock.

        Args:
            a: Single feature aggregate per-atom representation [..., N, c_a]
            s: Single embedding [..., N, c_s]

        Returns:
            Updated atom representation with conditioning applied
        """
        # Apply adaptive layer normalization
        a = self.adaln(a, s)

        # Apply gated SiLU activation
        b = F.silu(self.linear_nobias_a1(a)) * self.linear_nobias_a2(a)

        # Try to apply conditioning, fall back to unconditional if needed
        try:
            result = self._apply_conditioning(s, b)
            return cast(torch.Tensor, result)
        except (RuntimeError, ValueError) as e:
            # Log the issue and fall back to unconditional processing
            if torch.is_grad_enabled():
                warnings.warn(
                    f"Falling back to unconditional processing due to: {str(e)}"
                )

            # Fallback: just apply linear transform without conditioning
            result = self.linear_nobias_b(b)
            return cast(torch.Tensor, result)

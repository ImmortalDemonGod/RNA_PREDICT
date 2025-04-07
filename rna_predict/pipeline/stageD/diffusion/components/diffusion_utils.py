from typing import Optional, Protocol, TypedDict, Union

import torch


class InputFeatureDict(TypedDict):
    """Type definition for input feature dictionary."""

    ref_charge: torch.Tensor
    ref_pos: torch.Tensor
    expected_n_tokens: int


class DiffusionError(Exception):
    """Base class for diffusion-related errors."""

    pass


class ShapeMismatchError(DiffusionError):
    """Raised when tensor shapes don't match expected dimensions."""

    pass


def validate_tensor_shapes(
    s_trunk: torch.Tensor, s_inputs: torch.Tensor, c_s: int, c_s_inputs: int
) -> None:
    """
    Validate tensor shapes and raise informative errors.

    Args:
        s_trunk: Single feature embedding from PairFormer
        s_inputs: Single embedding from InputFeatureEmbedder
        c_s: Expected hidden dimension for single embedding
        c_s_inputs: Expected input embedding dimension
    """
    if s_trunk.shape[-1] != c_s:
        raise ShapeMismatchError(
            f"Expected last dimension {c_s} for s_trunk, got {s_trunk.shape[-1]}"
        )
    if s_inputs.shape[-1] != c_s_inputs:
        raise ShapeMismatchError(
            f"Expected last dimension {c_s_inputs} for s_inputs, got {s_inputs.shape[-1]}"
        )
    if s_trunk.shape[-2] != s_inputs.shape[-2]:
        raise ShapeMismatchError(
            f"Token dimension mismatch: {s_trunk.shape[-2]} vs {s_inputs.shape[-2]}"
        )


def create_zero_tensor_like(
    reference: torch.Tensor,
    shape: tuple[int, ...],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Create a zero tensor with specified shape, matching device and dtype of reference.

    Args:
        reference: Tensor to match device and dtype from
        shape: Desired shape for the new tensor
        device: Optional override for device
        dtype: Optional override for dtype

    Returns:
        Zero tensor with specified shape and matching device/dtype
    """
    device = device or reference.device
    dtype = dtype or reference.dtype
    return torch.zeros(shape, device=device, dtype=dtype)


class DiffusionProtocol(Protocol):
    """Protocol defining the interface for diffusion components."""

    def forward(
        self,
        x_noisy: torch.Tensor,
        t_hat_noise_level: torch.Tensor,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward pass of the diffusion model."""
        ...

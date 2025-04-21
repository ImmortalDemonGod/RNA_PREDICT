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
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Validate tensor shapes and adapt them if needed.

    Args:
        s_trunk: Single feature embedding from PairFormer
        s_inputs: Single embedding from InputFeatureEmbedder
        c_s: Expected hidden dimension for single embedding
        c_s_inputs: Expected input embedding dimension

    Returns:
        Tuple of (adapted_s_trunk, adapted_s_inputs)
    """
    adapted_s_trunk = s_trunk
    adapted_s_inputs = s_inputs

    # Adapt s_trunk to match expected dimension if needed
    if s_trunk.shape[-1] != c_s:
        print(f"Adapting s_trunk from dimension {s_trunk.shape[-1]} to {c_s}")
        if s_trunk.shape[-1] < c_s:
            # Pad with zeros
            padding = torch.zeros(
                *s_trunk.shape[:-1], c_s - s_trunk.shape[-1],
                device=s_trunk.device, dtype=s_trunk.dtype
            )
            adapted_s_trunk = torch.cat([s_trunk, padding], dim=-1)
        else:
            # Truncate
            adapted_s_trunk = s_trunk[..., :c_s]

    # Adapt s_inputs to match expected dimension if needed
    if s_inputs.shape[-1] != c_s_inputs:
        print(f"Adapting s_inputs from dimension {s_inputs.shape[-1]} to {c_s_inputs}")
        if s_inputs.shape[-1] < c_s_inputs:
            # Pad with zeros
            padding = torch.zeros(
                *s_inputs.shape[:-1], c_s_inputs - s_inputs.shape[-1],
                device=s_inputs.device, dtype=s_inputs.dtype
            )
            adapted_s_inputs = torch.cat([s_inputs, padding], dim=-1)
        else:
            # Truncate
            adapted_s_inputs = s_inputs[..., :c_s_inputs]

    # Check token dimension match
    if adapted_s_trunk.shape[-2] != adapted_s_inputs.shape[-2]:
        raise ShapeMismatchError(
            f"Token dimension mismatch: {adapted_s_trunk.shape[-2]} vs {adapted_s_inputs.shape[-2]}"
        )

    return adapted_s_trunk, adapted_s_inputs


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

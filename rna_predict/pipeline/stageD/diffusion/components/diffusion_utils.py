from typing import Optional, Protocol, TypedDict, Union, Tuple
import torch
import warnings


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


# --- Utility Functions ---

def _ensure_tensor_shape(
    tensor: torch.Tensor,
    target_ndim: int,
    ref_tensor: Optional[torch.Tensor] = None,
    target_shape: Optional[Tuple[Optional[int], ...]] = None,
    warn_prefix: str = "",
) -> torch.Tensor:
    """
    Ensures a tensor has the target number of dimensions and optionally matches
    parts of a reference tensor's shape or a specific target shape.

    Args:
        tensor: The tensor to reshape.
        target_ndim: The desired number of dimensions.
        ref_tensor: Optional reference tensor to match shape dimensions (except the last).
        target_shape: Optional specific shape tuple (use None for dimensions to ignore).
        warn_prefix: Prefix for warning messages.

    Returns:
        The potentially reshaped tensor.
    """
    current_ndim = tensor.ndim
    if current_ndim == target_ndim:
        # Already has the correct number of dimensions, check specific shape if needed
        if target_shape:
            # Check if shape matches target_shape pattern (None acts as wildcard)
            shape_matches = True
            if len(target_shape) > tensor.ndim: # Target shape has more dims than tensor
                shape_matches = False
            else:
                for i, s in enumerate(target_shape):
                    if s is not None and tensor.shape[i] != s:
                        shape_matches = False
                        break
            if not shape_matches:
                 warnings.warn(
                    f"{warn_prefix} Shape mismatch: Got {tensor.shape}, expected pattern {target_shape}. Attempting to proceed."
                )
        return tensor

    # Add or remove dimensions
    if current_ndim < target_ndim:
        # Unsqueeze dimensions at the beginning
        diff = target_ndim - current_ndim
        tensor = tensor.view((1,) * diff + tensor.shape) # More robust unsqueezing
        # warnings.warn( # Potentially verbose
        #     f"{warn_prefix} Unsqueezed tensor from {current_ndim}D to {target_ndim}D. New shape: {tensor.shape}"
        # )
    else: # current_ndim > target_ndim
        # Squeeze leading dimensions of size 1
        original_shape = tensor.shape # Keep original shape for warning
        dims_to_squeeze = []
        for i in range(current_ndim - target_ndim):
            if tensor.shape[i] == 1:
                dims_to_squeeze.append(i)
            else:
                # Cannot squeeze non-singleton dimension to reduce ndim
                 warnings.warn(
                    f"{warn_prefix} Cannot squeeze tensor from {current_ndim}D to {target_ndim}D (shape: {original_shape}, non-squeezable leading dims). Proceeding with original."
                )
                 return tensor # Cannot safely squeeze
        if dims_to_squeeze: # Only squeeze if there are dimensions to squeeze
            tensor = tensor.squeeze(tuple(dims_to_squeeze))
            # warnings.warn( # Potentially verbose
            #     f"{warn_prefix} Squeezed tensor from {current_ndim}D to {target_ndim}D. New shape: {tensor.shape}"
            # )
        # If no leading dims were size 1, tensor remains unchanged, ndim mismatch persists
        if tensor.ndim != target_ndim:
             warnings.warn(
                f"{warn_prefix} Could not change tensor ndim from {current_ndim} to {target_ndim} (shape: {original_shape}). Proceeding with original."
            )
             return tensor


    # Expand dimensions to match reference tensor if provided
    if ref_tensor is not None and tensor.ndim == ref_tensor.ndim:
        try:
            # Expand all dimensions except the last feature dimension
            expand_shape = list(ref_tensor.shape)
            if tensor.ndim > 1: # Avoid expanding scalar feature dim
                expand_shape[-1] = -1 # Keep original feature dim size
            elif tensor.ndim == 1 and ref_tensor.ndim == 1:
                 pass # Don't set -1 if both are 1D
            tensor = tensor.expand(expand_shape)
        except RuntimeError as e:
            warnings.warn(
                f"{warn_prefix} Could not broadcast tensor {tensor.shape} to ref_tensor {ref_tensor.shape}: {e}"
            )
    elif target_shape:
         # Expand dimensions to match target_shape if provided
        try:
            # Build expand shape from target_shape, using -1 for None or original size
            final_expand_shape = list(tensor.shape) # Start with current shape
            if len(target_shape) > len(final_expand_shape):
                 warnings.warn(f"{warn_prefix} Target shape {target_shape} has more dimensions than tensor {tensor.shape}. Cannot expand.")
            else:
                for i, s in enumerate(target_shape):
                    if s is not None:
                        if final_expand_shape[i] == 1:
                            final_expand_shape[i] = s # Expand singleton dim
                        elif final_expand_shape[i] != s:
                             warnings.warn(f"{warn_prefix} Cannot expand dim {i} from {final_expand_shape[i]} to {s}. Keeping original.")
                             # Keep original dim size if mismatch and not singleton

                if tuple(final_expand_shape) != tensor.shape:
                    tensor = tensor.expand(final_expand_shape)

        except RuntimeError as e:
            warnings.warn(
                f"{warn_prefix} Could not broadcast tensor {tensor.shape} to target_shape {target_shape}: {e}"
            )

    return tensor


def _calculate_edm_scaling_factors(
    sigma: torch.Tensor, sigma_data: float, ref_tensor: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the Elucidating Diffusion Models (EDM) scaling factors c_in, c_skip, c_out.

    Args:
        sigma: Noise level tensor. Should be broadcastable to ref_tensor shape.
        sigma_data: Standard deviation of the data.
        ref_tensor: Reference tensor (e.g., x_noisy) to match the shape of the factors.

    Returns:
        Tuple containing (c_in, c_skip, c_out) tensors, broadcasted to match ref_tensor shape.
    """
    dtype = ref_tensor.dtype
    # Ensure sigma has shape compatible for broadcasting with ref_tensor
    # We expect sigma to potentially have fewer dims (e.g., [B, N_sample] vs ref [B, N_sample, N, 3])
    # Add trailing dims to sigma to match ref_tensor's ndim for calculation
    sigma_expanded = sigma.clone()
    while sigma_expanded.ndim < ref_tensor.ndim:
        sigma_expanded = sigma_expanded.unsqueeze(-1)

    # Ensure calculations happen in float32 for stability, then cast back
    sigma_float = sigma_expanded.float()
    sigma_data_float = float(sigma_data)

    # Add small epsilon to prevent division by zero if sigma_float is zero
    epsilon = torch.finfo(sigma_float.dtype).eps
    # Ensure denominator is positive before sqrt
    denom_in_sq = sigma_data_float**2 + sigma_float**2
    denom_skip_out = denom_in_sq # Same denominator for c_skip and c_out base

    # Clamp to avoid issues with very small sigmas leading to large c_in/c_out
    # Or handle potential NaNs if sigma_float can be negative (shouldn't be for noise levels)
    denom_in = torch.sqrt(torch.clamp(denom_in_sq, min=epsilon))
    denom_skip_out_safe = torch.clamp(denom_skip_out, min=epsilon)


    c_in = 1.0 / denom_in
    c_skip = sigma_data_float**2 / denom_skip_out_safe
    # Use sigma_float directly for c_out numerator, sqrt(denom) for denominator
    c_out = (sigma_data_float * sigma_float) / denom_in

    # Cast back to original dtype
    c_in = c_in.to(dtype)
    c_skip = c_skip.to(dtype)
    c_out = c_out.to(dtype)

    # Final check: Ensure factors can broadcast to ref_tensor shape
    # This relies on PyTorch broadcasting rules rather than explicit expansion here
    try:
        # Test broadcast compatibility by adding zero
        _ = c_in + torch.zeros_like(ref_tensor)
        _ = c_skip + torch.zeros_like(ref_tensor)
        _ = c_out + torch.zeros_like(ref_tensor)
    except RuntimeError as e:
         warnings.warn(f"EDM scaling factors c_in={c_in.shape}, c_skip={c_skip.shape}, c_out={c_out.shape} "
                       f"may not broadcast correctly to ref_tensor {ref_tensor.shape}: {e}")


    return c_in, c_skip, c_out

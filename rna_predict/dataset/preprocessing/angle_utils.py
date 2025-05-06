import torch

def angles_rad_to_sin_cos(angles_rad: torch.Tensor) -> torch.Tensor:
    """
    Convert angles in radians to interleaved sin/cos pairs.
    Args:
        angles_rad: Tensor of shape [..., num_angles]
    Returns:
        Tensor of shape [..., num_angles * 2] with [sin_0, cos_0, sin_1, cos_1, ...] order
    """
    sin_angles = torch.sin(angles_rad)
    cos_angles = torch.cos(angles_rad)
    # Interleave sin and cos: [sin_0, cos_0, sin_1, cos_1, ...]
    stacked = torch.stack((sin_angles, cos_angles), dim=-1)  # [..., num_angles, 2]
    sincos_pairs = stacked.reshape(*angles_rad.shape[:-1], angles_rad.shape[-1] * 2)
    return sincos_pairs

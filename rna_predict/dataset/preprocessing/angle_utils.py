import torch

def angles_rad_to_sin_cos(angles_rad: torch.Tensor) -> torch.Tensor:
    """
    Converts a tensor of angles in radians to an interleaved tensor of sine and cosine values.
    
    Given a tensor of shape [..., num_angles], returns a tensor of shape [..., num_angles * 2]
    where the last dimension contains values in the order [sin_0, cos_0, sin_1, cos_1, ...].
    """
    sin_angles = torch.sin(angles_rad)
    cos_angles = torch.cos(angles_rad)
    # Interleave sin and cos: [sin_0, cos_0, sin_1, cos_1, ...]
    stacked = torch.stack((sin_angles, cos_angles), dim=-1)  # [..., num_angles, 2]
    sincos_pairs = stacked.reshape(*angles_rad.shape[:-1], angles_rad.shape[-1] * 2)
    return sincos_pairs

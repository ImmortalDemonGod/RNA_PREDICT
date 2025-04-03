"""
Instrumented version of massive_pnerf.py for debugging NaN issues.

This version adds detailed logging and safety checks to identify where NaNs
are first appearing in the MP-NeRF calculations.
"""

import logging
import os
import sys

import numpy as np
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("instrumented_mpnerf")


def check_tensor(tensor, name, location):
    """Check if a tensor contains NaN or Inf values and log details."""
    if tensor is None:
        logger.warning(f"{location}: {name} is None")
        return True

    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()

    if has_nan or has_inf:
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        logger.error(
            f"{location}: {name} contains {nan_count} NaNs and {inf_count} Infs"
        )

        # If it's a multi-dimensional tensor, find where the NaNs are
        if tensor.dim() > 1:
            nan_indices = torch.where(torch.isnan(tensor))
            if len(nan_indices[0]) > 0:
                logger.error(
                    f"First few NaN positions: {[(nan_indices[i][j].item() for i in range(len(nan_indices))) for j in range(min(5, len(nan_indices[0])))]}"
                )
        return True
    return False


def mp_nerf_torch_safe(a, b, c, l, theta, chi):
    """
    Safe version of mp_nerf_torch with detailed logging and safety checks.
    This is a drop-in replacement for the original function.
    """
    logger.debug(
        f"mp_nerf_torch_safe inputs: a={a}, b={b}, c={c}, l={l}, theta={theta}, chi={chi}"
    )

    # Check inputs
    check_tensor(a, "a", "mp_nerf_torch input")
    check_tensor(b, "b", "mp_nerf_torch input")
    check_tensor(c, "c", "mp_nerf_torch input")
    check_tensor(l, "l", "mp_nerf_torch input")
    check_tensor(theta, "theta", "mp_nerf_torch input")
    check_tensor(chi, "chi", "mp_nerf_torch input")

    # Safety check for theta
    if not ((-np.pi <= theta) * (theta <= np.pi)).all().item():
        logger.warning(
            f"theta(s) must be in radians and in [-pi, pi]. theta(s) = {theta}"
        )
        # Clamp theta to valid range instead of raising error
        theta = torch.clamp(theta, -np.pi, np.pi)

    # Calculate vectors
    ba = b - a
    check_tensor(ba, "ba", "mp_nerf_torch ba")

    cb = c - b
    check_tensor(cb, "cb", "mp_nerf_torch cb")

    # Check for zero magnitude vectors and add small perturbation if needed
    ba_norm = torch.norm(ba, dim=-1)
    cb_norm = torch.norm(cb, dim=-1)

    if (ba_norm < 1e-10).any():
        logger.warning("mp_nerf_torch: ba has near-zero magnitude")
        # Add small perturbation to avoid zero vectors
        perturb = torch.tensor([1e-10, 1e-10, 1e-10], device=ba.device)
        ba = ba + perturb

    if (cb_norm < 1e-10).any():
        logger.warning("mp_nerf_torch: cb has near-zero magnitude")
        # Add small perturbation to avoid zero vectors
        perturb = torch.tensor([1e-10, 1e-10, 1e-10], device=cb.device)
        cb = cb + perturb

    # Calculate rotation matrix
    n_plane = torch.cross(ba, cb, dim=-1)
    check_tensor(n_plane, "n_plane", "mp_nerf_torch n_plane")

    # Check for zero magnitude vectors
    n_plane_norm = torch.norm(n_plane, dim=-1)

    if (n_plane_norm < 1e-10).any():
        logger.warning("mp_nerf_torch: n_plane has near-zero magnitude")
        # Add small perturbation to avoid collinearity
        perturb = torch.tensor([0.0, 1e-10, 1e-10], device=ba.device)
        ba = ba + perturb
        # Recalculate cross product
        n_plane = torch.cross(ba, cb, dim=-1)

    n_plane_ = torch.cross(n_plane, cb, dim=-1)
    check_tensor(n_plane_, "n_plane_", "mp_nerf_torch n_plane_")

    rotate = torch.stack([cb, n_plane_, n_plane], dim=-1)
    check_tensor(rotate, "rotate", "mp_nerf_torch rotate")

    # Safe normalization with epsilon to avoid division by zero
    norm = torch.norm(rotate, dim=-2, keepdim=True)
    check_tensor(norm, "norm", "mp_nerf_torch norm")

    # Add epsilon to avoid division by zero
    norm = torch.clamp(norm, min=1e-10)
    rotate = rotate / norm
    check_tensor(rotate, "rotate after normalization", "mp_nerf_torch rotate_norm")

    # Calculate proto point
    d = torch.stack(
        [
            -torch.cos(theta),
            torch.sin(theta) * torch.cos(chi),
            torch.sin(theta) * torch.sin(chi),
        ],
        dim=-1,
    ).unsqueeze(-1)
    check_tensor(d, "d", "mp_nerf_torch d")

    # Matrix multiplication
    rotated_d = torch.matmul(rotate, d).squeeze()
    check_tensor(rotated_d, "rotated_d", "mp_nerf_torch rotated_d")

    # Final result
    result = c + l.unsqueeze(-1) * rotated_d
    check_tensor(result, "result", "mp_nerf_torch result")

    return result


def patch_mp_nerf():
    """
    Patch the original mp_nerf_torch function with our safe version.
    Call this before running any tests.
    """
    # Add the project root to the Python path if needed
    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    )

    from rna_predict.pipeline.stageC.mp_nerf import massive_pnerf as mp_nerf_module

    # Save the original function for reference
    if not hasattr(mp_nerf_module, "mp_nerf_torch_original"):
        mp_nerf_module.mp_nerf_torch_original = mp_nerf_module.mp_nerf_torch

    # Replace with our safe version
    mp_nerf_module.mp_nerf_torch = mp_nerf_torch_safe

    logger.info("Patched mp_nerf_torch with instrumented version")


def restore_mp_nerf():
    """
    Restore the original mp_nerf_torch function.
    Call this after tests are complete.
    """
    # Add the project root to the Python path if needed
    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    )

    from rna_predict.pipeline.stageC.mp_nerf import massive_pnerf as mp_nerf_module

    if hasattr(mp_nerf_module, "mp_nerf_torch_original"):
        mp_nerf_module.mp_nerf_torch = mp_nerf_module.mp_nerf_torch_original
        logger.info("Restored original mp_nerf_torch function")


if __name__ == "__main__":
    # Example usage
    a = torch.tensor([0.0, 0.0, 0.0])
    b = torch.tensor([1.0, 0.0, 0.0])
    c = torch.tensor([2.0, 0.0, 0.0])
    l = torch.tensor(1.5)
    theta = torch.tensor(1.0)
    chi = torch.tensor(0.5)

    # Test the safe version
    result = mp_nerf_torch_safe(a, b, c, l, theta, chi)
    print("Result:", result)

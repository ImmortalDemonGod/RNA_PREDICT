"""
RNA atom positioning functions for MP-NeRF implementation.
"""

import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_atom_position(a, b, c, bond_length_cd, theta, chi, device):
    """
    Calculate the position of a new atom 'd' based on three previous atoms (a, b, c)
    and geometric parameters (bond length c-d, bond angle b-c-d, torsion angle a-b-c-d).
    Adapted from EleutherAI's mp_nerf_torch.

    Args:
        a: Position of atom 'a' (N-3 wrt new atom d at N)
        b: Position of atom 'b' (N-2 wrt new atom d at N)
        c: Position of atom 'c' (N-1 wrt new atom d at N, new atom d is bonded to c)
        bond_length_cd: Bond length between c and new atom d.
        theta: Bond angle b-c-d (in radians).
        chi: Torsion angle a-b-c-d (in radians).
        device: Device to place tensors on.

    Returns:
        Position of the new atom 'd'.
    """
    # Ensure inputs are on the correct device if they are tensors
    # This should ideally be handled by the caller, but as a safeguard:
    if torch.is_tensor(a) and a.device != device:
        a = a.to(device)
    if torch.is_tensor(b) and b.device != device:
        b = b.to(device)
    if torch.is_tensor(c) and c.device != device:
        c = c.to(device)
    if torch.is_tensor(bond_length_cd) and bond_length_cd.device != device:
        bond_length_cd = bond_length_cd.to(device)
    if torch.is_tensor(theta) and theta.device != device:
        theta = theta.to(device)
    if torch.is_tensor(chi) and chi.device != device:
        chi = chi.to(device)

    # Safety check for theta (bond angle)
    # Ensure theta is a tensor for the comparison
    theta_tensor = theta if torch.is_tensor(theta) else torch.tensor(theta, device=device)
    if not ((-np.pi <= theta_tensor) & (theta_tensor <= np.pi)).all().item():
        logger.warning(f"[WARN-RNAPREDICT-ANGLE-RANGE-001] Bond angle theta ({theta_tensor.item()}) is outside [-pi, pi]. Ensure it's in radians.")
        # Depending on strictness, could raise ValueError here:
        # raise ValueError(f"theta(s) must be in radians and in [-pi, pi]. theta(s) = {theta}")

    # Ensure l, theta, chi are scalar tensors if they are not already batched
    # This implementation assumes a single atom placement, not batched.
    # If inputs are already scalar tensors, this does nothing harmful.
    bond_length_cd = bond_length_cd.squeeze() if torch.is_tensor(bond_length_cd) and bond_length_cd.numel() == 1 else bond_length_cd
    theta = theta.squeeze() if torch.is_tensor(theta) and theta.numel() == 1 else theta
    chi = chi.squeeze() if torch.is_tensor(chi) and chi.numel() == 1 else chi

    # Calculate vectors for the local coordinate system
    ba = b - a
    cb = c - b

    # Normalize cb to ensure it's a unit vector for the x-axis of the local frame
    cb_unit = cb / (torch.norm(cb) + 1e-8) # Add epsilon for numerical stability

    # Calculate normal to the a-b-c plane (local z-axis)
    n_plane = torch.cross(ba, cb_unit, dim=-1)
    n_plane_unit = n_plane / (torch.norm(n_plane) + 1e-8)

    # Calculate vector orthogonal to cb_unit and n_plane_unit (local y-axis)
    n_plane_orthogonal = torch.cross(n_plane_unit, cb_unit, dim=-1)
    # This vector should already be unit length if cb_unit and n_plane_unit are orthogonal unit vectors.

    # Construct the rotation matrix [cb_unit, n_plane_orthogonal, n_plane_unit] (columns)
    # This matrix transforms coordinates from the local frame to the global frame.
    # Transpose is used because torch.stack creates rows, and we need columns for the rotation matrix M where M * v_local = v_global
    rotate = torch.stack([cb_unit, n_plane_orthogonal, n_plane_unit], dim=-1)

    # Coordinates of the new atom 'd' in the local frame defined by c, b, a.
    # c is the origin, cb_unit is the x-axis.
    # The angle theta (b-c-d) is between vector c->b and c->d.
    # Standard NeRF equations place d relative to c:
    # x_local = l * cos(angle_between_cb_and_cd)
    # y_local = l * sin(angle_between_cb_and_cd) * cos(torsion_abcd)
    # z_local = l * sin(angle_between_cb_and_cd) * sin(torsion_abcd)
    # Here, theta is angle b-c-d. The angle between vector c->b and c->d is (pi - theta_bcd).
    # So, cos(pi - theta) = -cos(theta), and sin(pi - theta) = sin(theta).
    d_local_coords = torch.stack([
        -torch.cos(theta),  # Along cb_unit (local x-axis), scaled by l later
         torch.sin(theta) * torch.cos(chi), # Along n_plane_orthogonal (local y-axis)
         torch.sin(theta) * torch.sin(chi)  # Along n_plane_unit (local z-axis)
    ], dim=-1).to(device) # Ensure d_local_coords is on the correct device
    
    # Transform d_local_coords to global coordinates and scale by bond length
    # Unsqueeze d_local_coords to make it a column vector for matrix multiplication
    d_global_offset = torch.matmul(rotate, d_local_coords.unsqueeze(-1)).squeeze(-1)
    
    # Add the offset to atom c's position
    new_position = c + bond_length_cd * d_global_offset

    # Debug: Log inputs and outputs if needed
    logger.debug(f"[DEBUG-CALCATOM-NEW] Inputs: a={a}, b={b}, c={c}, l={bond_length_cd}, theta={theta}, chi={chi}")
    logger.debug(f"[DEBUG-CALCATOM-NEW] cb_unit={cb_unit}, n_plane_orthogonal={n_plane_orthogonal}, n_plane_unit={n_plane_unit}")
    logger.debug(f"[DEBUG-CALCATOM-NEW] rotate_matrix:\n{rotate}")
    logger.debug(f"[DEBUG-CALCATOM-NEW] d_local_coords={d_local_coords}")
    logger.debug(f"[DEBUG-CALCATOM-NEW] d_global_offset={d_global_offset}")
    logger.debug(f"[DEBUG-CALCATOM-NEW] new_position={new_position}")

    if torch.isnan(new_position).any():
        logger.error(f"[ERR-RNAPREDICT-NAN-PLACEMENT-002] NaN detected in calculate_atom_position (new NeRF).\n"
                     f"  a: {a}\n  b: {b}\n  c: {c}\n  l: {bond_length_cd}\n  theta: {theta}\n  chi: {chi}\n"
                     f"  cb_unit: {cb_unit}\n  n_plane_orthogonal: {n_plane_orthogonal}\n  n_plane_unit: {n_plane_unit}\n"
                     f"  rotate: {rotate}\n  d_local_coords: {d_local_coords}\n  d_global_offset: {d_global_offset}")
        # Potentially raise an error or return a sentinel value

    return new_position

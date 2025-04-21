"""
RNA atom positioning functions for MP-NeRF implementation.
"""

import torch

def calculate_atom_position(
    prev_prev_atom, prev_atom, bond_length, bond_angle, torsion_angle, device
):
    """
    Calculate the position of a new atom based on previous atoms and geometric parameters.

    Args:
        prev_prev_atom: Position of the atom before the previous atom
        prev_atom: Position of the previous atom
        bond_length: Length of the bond to the new atom
        bond_angle: Angle between prev_prev_atom, prev_atom, and new atom
        torsion_angle: Dihedral angle for rotation around the bond
        device: Device to place tensors on

    Returns:
        Position of the new atom
    """
    # Convert inputs to tensors if they aren't already
    prev_prev_atom = torch.as_tensor(prev_prev_atom, device=device)
    prev_atom = torch.as_tensor(prev_atom, device=device)
    bond_length = torch.as_tensor(bond_length, device=device)
    bond_angle = torch.as_tensor(bond_angle, device=device)
    torsion_angle = torch.as_tensor(torsion_angle, device=device)

    # Calculate bond vector
    bond_vector = prev_atom - prev_prev_atom
    bond_vector = bond_vector / (
        torch.norm(bond_vector) + 1e-8
    )  # Normalize with epsilon to avoid division by zero

    # Calculate perpendicular vector
    perpendicular = torch.linalg.cross(
        input=torch.linalg.cross(input=bond_vector, other=torch.tensor([0.0, 0.0, 1.0], device=device), dim=-1),
        other=bond_vector,
        dim=-1
    )
    if torch.norm(perpendicular) < 1e-8:  # If parallel to z-axis
        perpendicular = torch.tensor([1.0, 0.0, 0.0], device=device)
    perpendicular = perpendicular / (torch.norm(perpendicular) + 1e-8)

    # Calculate rotation matrix for bond angle
    cos_theta = torch.cos(bond_angle)
    sin_theta = torch.sin(bond_angle)
    rotation_bond = torch.tensor(
        [[cos_theta, -sin_theta, 0.0], [sin_theta, cos_theta, 0.0], [0.0, 0.0, 1.0]],
        device=device,
    )

    # Calculate rotation matrix for torsion angle
    cos_phi = torch.cos(torsion_angle)
    sin_phi = torch.sin(torsion_angle)
    rotation_torsion = torch.tensor(
        [[cos_phi, -sin_phi, 0.0], [sin_phi, cos_phi, 0.0], [0.0, 0.0, 1.0]],
        device=device,
    )

    # Combine rotations
    rotation = torch.matmul(rotation_torsion, rotation_bond)

    # Calculate new position
    new_vector = torch.matmul(rotation, bond_vector.unsqueeze(-1)).squeeze(-1)
    new_position = prev_atom + bond_length * new_vector

    # --- DEBUG: Check for NaN in output ---
    if torch.isnan(new_position).any():
        print(f"[ERR-RNAPREDICT-NAN-PLACEMENT-001] NaN detected in calculate_atom_position.\n  prev_prev_atom: {prev_prev_atom}\n  prev_atom: {prev_atom}\n  bond_length: {bond_length}\n  bond_angle: {bond_angle}\n  torsion_angle: {torsion_angle}\n  device: {device}")

    return new_position

"""
RNA atom positioning functions for MP-NeRF implementation.
"""

import torch
import logging

logger = logging.getLogger("rna_predict.pipeline.stageC.mp_nerf.rna_atom_positioning")

##@snoop  
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
    # DEBUG: Log requires_grad and grad_fn for all inputs
    logger.debug(f"[DEBUG-CALCATOM] prev_prev_atom requires_grad: {getattr(prev_prev_atom, 'requires_grad', None)}, grad_fn: {getattr(prev_prev_atom, 'grad_fn', None)}")
    logger.debug(f"[DEBUG-CALCATOM] prev_atom requires_grad: {getattr(prev_atom, 'requires_grad', None)}, grad_fn: {getattr(prev_atom, 'grad_fn', None)}")
    logger.debug(f"[DEBUG-CALCATOM] bond_length requires_grad: {getattr(bond_length, 'requires_grad', None)}, grad_fn: {getattr(bond_length, 'grad_fn', None)}")
    logger.debug(f"[DEBUG-CALCATOM] bond_angle requires_grad: {getattr(bond_angle, 'requires_grad', None)}, grad_fn: {getattr(bond_angle, 'grad_fn', None)}")
    logger.debug(f"[DEBUG-CALCATOM] torsion_angle requires_grad: {getattr(torsion_angle, 'requires_grad', None)}, grad_fn: {getattr(torsion_angle, 'grad_fn', None)}")

    # Calculate bond vector
    bond_vector = prev_atom - prev_prev_atom
    if bond_vector.device != torch.device(device):
        bond_vector = bond_vector.to(device)
    bond_vector = bond_vector / (torch.norm(bond_vector) + 1e-8)

    # Calculate perpendicular vector
    z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=prev_atom.dtype, device=prev_atom.device)
    perp = torch.cross(torch.cross(bond_vector, z_axis, dim=-1), bond_vector, dim=-1)
    if torch.norm(perp) < 1e-8:
        perp = torch.tensor([1.0, 0.0, 0.0], dtype=prev_atom.dtype, device=prev_atom.device)
    perp = perp / (torch.norm(perp) + 1e-8)

    # Rotation matrices using tensor ops
    cos_theta = torch.cos(bond_angle)
    sin_theta = torch.sin(bond_angle)
    rotation_bond = torch.stack([
        torch.stack([cos_theta, -sin_theta, torch.zeros_like(cos_theta, device=device)]),
        torch.stack([sin_theta, cos_theta, torch.zeros_like(sin_theta, device=device)]),
        torch.stack([torch.zeros_like(cos_theta, device=device), torch.zeros_like(cos_theta, device=device), torch.ones_like(cos_theta, device=device)])
    ])

    cos_phi = torch.cos(torsion_angle)
    sin_phi = torch.sin(torsion_angle)
    rotation_torsion = torch.stack([
        torch.stack([cos_phi, -sin_phi, torch.zeros_like(cos_phi, device=device)]),
        torch.stack([sin_phi, cos_phi, torch.zeros_like(sin_phi, device=device)]),
        torch.stack([torch.zeros_like(cos_phi, device=device), torch.zeros_like(cos_phi, device=device), torch.ones_like(cos_phi, device=device)])
    ])

    rotation = torch.matmul(rotation_torsion, rotation_bond)
    new_vector = torch.matmul(rotation, bond_vector.unsqueeze(-1)).squeeze(-1)
    new_position = prev_atom + bond_length * new_vector
    logger.debug(f"[DEBUG-CALCATOM] new_position requires_grad: {getattr(new_position, 'requires_grad', None)}, grad_fn: {getattr(new_position, 'grad_fn', None)}")
    if torch.isnan(new_position).any():
        logger.error(f"[ERR-RNAPREDICT-NAN-PLACEMENT-001] NaN detected in calculate_atom_position.\n  prev_prev_atom: {prev_prev_atom}\n  prev_atom: {prev_atom}\n  bond_length: {bond_length}\n  bond_angle: {bond_angle}\n  torsion_angle: {torsion_angle}\n  device: {prev_atom.device}")
    return new_position

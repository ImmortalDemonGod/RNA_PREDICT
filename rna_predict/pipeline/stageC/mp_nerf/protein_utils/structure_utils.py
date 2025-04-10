"""
Protein structure manipulation utilities.
Functions for handling protein backbone and sidechain structures.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np
import torch

from rna_predict.pipeline.stageC.mp_nerf.massive_pnerf import (
    MpNerfParams,
    mp_nerf_torch,
)

from .scaffold_builders import build_scaffolds_from_scn_angles
from .sidechain_data import BB_BUILD_INFO


class AtomCheckType(Enum):
    """Enum for atom validity check types."""

    EXISTS = auto()  # Check if atom exists in the mask
    COORDS = auto()  # Check if coordinates are valid (not zeros)
    BOTH = auto()  # Check both existence and coordinates


@dataclass
class ProteinFoldConfig:
    """Configuration for protein folding operations.

    This class centralizes parameters used across multiple functions to reduce
    function argument counts and improve code organization.
    """

    seq: str
    seq_len: int
    output_coords: torch.Tensor
    cloud_mask: torch.Tensor
    point_ref_mask: torch.Tensor
    std_angles_mask: torch.Tensor
    std_bond_mask: torch.Tensor
    device: torch.device
    phi: torch.Tensor
    psi: torch.Tensor
    omega: torch.Tensor


def _check_atom_property(
    config: ProteinFoldConfig, i: int, idx: int, property_type: str
) -> bool:
    """Check a property of an atom (existence or coordinate validity).

    Args:
        config: Protein fold configuration
        i: Residue index
        idx: Atom index
        property_type: Type of property to check ('exists' or 'coords')

    Returns:
        True if the atom passes the specified check, False otherwise
    """
    if property_type == "exists":
        return bool(config.cloud_mask[i, int(idx)])
    elif property_type == "coords":
        return not torch.all(config.output_coords[i, int(idx)] == 0)
    else:
        raise ValueError(f"Unsupported property type: {property_type}")


def _check_atom_validity(
    config: ProteinFoldConfig, i: int, idx: int, check_type: AtomCheckType
) -> bool:
    """Check if an atom exists and/or has valid coordinates.

    Args:
        config: Protein fold configuration
        i: Residue index
        idx: Atom index
        check_type: Type of check to perform (EXISTS, COORDS, or BOTH)

    Returns:
        True if the atom passes the specified check, False otherwise
    """
    if check_type == AtomCheckType.EXISTS:
        return _check_atom_property(config, i, idx, "exists")
    elif check_type == AtomCheckType.COORDS:
        return _check_atom_property(config, i, idx, "coords")
    else:  # AtomCheckType.BOTH
        return _check_atom_property(config, i, idx, "exists") and _check_atom_property(
            config, i, idx, "coords"
        )


@dataclass
class AtomPlacementParams:
    """Parameters for placing an atom using NeRF.

    This class encapsulates the parameters needed to place an atom using
    Natural Extension Reference Frame (NeRF).
    """

    bond_length: float
    bond_angle_deg: float
    dihedral_angle: torch.Tensor
    ref_atoms: List[torch.Tensor]

    def to_mp_nerf_params(self, device: torch.device) -> MpNerfParams:
        """Convert to MpNerfParams for use with mp_nerf_torch.

        Args:
            device: Device to place tensors on

        Returns:
            MpNerfParams object for use with mp_nerf_torch
        """
        return MpNerfParams(
            a=self.ref_atoms[0],
            b=self.ref_atoms[1],
            c=self.ref_atoms[2],
            bond_length=torch.tensor(
                self.bond_length, device=device, dtype=torch.float32
            ),
            theta=torch.tensor(
                np.radians(self.bond_angle_deg),
                device=device,
                dtype=torch.float32,
            ),
            chi=self.dihedral_angle,
        )


def _get_atom_placement_params(
    atom_type: str, ref_atoms: List[torch.Tensor], dihedral_angle: torch.Tensor
) -> AtomPlacementParams:
    """Get parameters for placing an atom.

    Args:
        atom_type: Type of atom to place (N, CA, C, O, CB)
        ref_atoms: Reference atoms for placement
        dihedral_angle: Dihedral angle for placement

    Returns:
        AtomPlacementParams object with placement parameters
    """
    if atom_type == "N":
        bond_length = BB_BUILD_INFO.get("BONDLENS", {}).get("c-n", 1.329)
        bond_angle_deg = BB_BUILD_INFO.get("BONDANGS", {}).get("ca-c-n", 116.2)
    elif atom_type == "CA":
        bond_length = BB_BUILD_INFO.get("BONDLENS", {}).get("n-ca", 1.458)
        bond_angle_deg = BB_BUILD_INFO.get("BONDANGS", {}).get("c-n-ca", 121.7)
    elif atom_type == "C":
        bond_length = BB_BUILD_INFO.get("BONDLENS", {}).get("ca-c", 1.525)
        bond_angle_deg = BB_BUILD_INFO.get("BONDANGS", {}).get("n-ca-c", 111.0)
    elif atom_type == "O":
        bond_length = BB_BUILD_INFO.get("BONDLENS", {}).get("c-o", 1.229)
        bond_angle_deg = BB_BUILD_INFO.get("BONDANGS", {}).get("ca-c-o", 120.8)
    elif atom_type == "CB":
        bond_length = BB_BUILD_INFO.get("BONDLENS", {}).get("ca-cb", 1.52)
        bond_angle_deg = BB_BUILD_INFO.get("BONDANGS", {}).get("n-ca-cb", 110.5)
    else:
        raise ValueError(f"Unsupported atom type: {atom_type}")

    return AtomPlacementParams(
        bond_length=bond_length,
        bond_angle_deg=bond_angle_deg,
        dihedral_angle=dihedral_angle,
        ref_atoms=ref_atoms,
    )


def _place_atom(params: AtomPlacementParams, device: torch.device) -> torch.Tensor:
    """Place an atom using NeRF.

    Args:
        params: Parameters for atom placement
        device: Device to place tensors on

    Returns:
        Coordinates of the placed atom
    """
    mp_params = params.to_mp_nerf_params(device)
    return mp_nerf_torch(mp_params)


def _place_backbone_atom(
    config: ProteinFoldConfig,
    i: int,
    atom_type: str,
) -> None:
    """Place a backbone atom for the next residue.

    Args:
        config: Protein fold configuration
        i: Current residue index
        atom_type: Type of atom to place (N, CA, C)
    """
    if atom_type == "N":
        # Reference atoms for N(i+1): N(i), CA(i), C(i)
        ref_atoms = [
            config.output_coords[i, 0].unsqueeze(0),  # N(i)
            config.output_coords[i, 1].unsqueeze(0),  # CA(i)
            config.output_coords[i, 2].unsqueeze(0),  # C(i)
        ]
        dihedral_angle = config.psi[i]  # Use psi(i)
        target_idx = (i + 1, 0)  # N(i+1)
    elif atom_type == "CA":
        # Reference atoms for CA(i+1): CA(i), C(i), N(i+1)
        ref_atoms = [
            config.output_coords[i, 1].unsqueeze(0),  # CA(i)
            config.output_coords[i, 2].unsqueeze(0),  # C(i)
            config.output_coords[i + 1, 0].unsqueeze(0),  # N(i+1)
        ]
        dihedral_angle = config.omega[i]  # Use omega(i)
        target_idx = (i + 1, 1)  # CA(i+1)
    elif atom_type == "C":
        # Reference atoms for C(i+1): C(i), N(i+1), CA(i+1)
        ref_atoms = [
            config.output_coords[i, 2].unsqueeze(0),  # C(i)
            config.output_coords[i + 1, 0].unsqueeze(0),  # N(i+1)
            config.output_coords[i + 1, 1].unsqueeze(0),  # CA(i+1)
        ]
        dihedral_angle = config.phi[i + 1]  # Use phi(i+1)
        target_idx = (i + 1, 2)  # C(i+1)
    else:
        raise ValueError(f"Unsupported backbone atom type: {atom_type}")

    params = _get_atom_placement_params(atom_type, ref_atoms, dihedral_angle)
    config.output_coords[target_idx[0], target_idx[1]] = _place_atom(
        params, config.device
    ).squeeze(0)


def _build_backbone_chain(config: ProteinFoldConfig) -> None:
    """Builds the protein backbone chain residue by residue using NeRF.

    Args:
        config: Protein fold configuration
    """
    for i in range(config.seq_len - 1):
        # Place N(i+1) using NeRF
        _place_backbone_atom(config, i, "N")

        # Place CA(i+1) using NeRF
        _place_backbone_atom(config, i, "CA")

        # Place C(i+1) using NeRF
        _place_backbone_atom(config, i, "C")


def _place_oxygen_atom(config: ProteinFoldConfig, i: int) -> None:
    """Places the oxygen atom for a residue.

    Args:
        config: Protein fold configuration
        i: Residue index
    """
    # Reference atoms for O: N, CA, C
    ref_atoms = [
        config.output_coords[i, 0].unsqueeze(0),  # N
        config.output_coords[i, 1].unsqueeze(0),  # CA
        config.output_coords[i, 2].unsqueeze(0),  # C
    ]

    # Standard dihedral for N-CA-C-O
    dihedral_angle = torch.tensor(
        np.radians(BB_BUILD_INFO.get("DIHEDRS", {}).get("n-ca-c-o", 180.0)),
        device=config.device,
        dtype=torch.float32,
    )

    params = _get_atom_placement_params("O", ref_atoms, dihedral_angle)
    config.output_coords[i, 3] = _place_atom(params, config.device).squeeze(0)


def _are_reference_atoms_valid(
    config: ProteinFoldConfig, i: int, ref_indices: List[int]
) -> bool:
    """Check if reference atoms exist and have valid coordinates.

    Args:
        config: Protein fold configuration
        i: Residue index
        ref_indices: Indices of reference atoms

    Returns:
        True if all reference atoms are valid, False otherwise
    """
    # Check if all atoms exist and have valid coordinates
    for idx in ref_indices:
        if not _check_atom_validity(config, i, idx, AtomCheckType.BOTH):
            return False

    return True


def _are_geometry_parameters_valid(
    thetas: torch.Tensor, dihedrals: torch.Tensor, bond_lengths: torch.Tensor
) -> bool:
    """Check if geometry parameters are valid (not NaN).

    Args:
        thetas: Bond angles
        dihedrals: Dihedral angles
        bond_lengths: Bond lengths

    Returns:
        True if all parameters are valid, False otherwise
    """
    return not (
        torch.isnan(thetas) or torch.isnan(dihedrals) or torch.isnan(bond_lengths)
    )


def _place_beta_carbon(config: ProteinFoldConfig, i: int) -> None:
    """Places the beta carbon atom for a residue if applicable.

    Args:
        config: Protein fold configuration
        i: Residue index
    """
    # Skip if residue is Glycine (G) which has no CB
    if config.seq[i] == "G":
        return

    # Standard dihedral for C-N-CA-CB (L-amino acids)
    dihedral_angle = torch.tensor(
        np.radians(-122.0),
        device=config.device,
        dtype=torch.float32,
    )

    # For first residue, we don't have a previous C, so use a different reference
    if i == 0:
        # Use the current C and O atoms as references instead
        ref_atoms = [
            config.output_coords[i, 2].unsqueeze(0),  # C
            config.output_coords[i, 0].unsqueeze(0),  # N
            config.output_coords[i, 1].unsqueeze(0),  # CA
        ]
    else:
        # Use the previous C atom as reference
        ref_atoms = [
            config.output_coords[i - 1, 2].unsqueeze(0),  # Previous C
            config.output_coords[i, 0].unsqueeze(0),  # N
            config.output_coords[i, 1].unsqueeze(0),  # CA
        ]

    params = _get_atom_placement_params("CB", ref_atoms, dihedral_angle)
    config.output_coords[i, 4] = _place_atom(params, config.device).squeeze(0)


def _place_sidechain_atom(config: ProteinFoldConfig, i: int, level: int) -> bool:
    """Places a sidechain atom for a residue.

    Args:
        config: Protein fold configuration
        i: Residue index
        level: Atom level (5-13)

    Returns:
        bool: True if atom was placed, False otherwise
    """
    # Get intra-residue references for NeRF
    ref_mask_level_idx = level - 3
    idx_a = int(config.point_ref_mask[0, i, ref_mask_level_idx].item())
    idx_b = int(config.point_ref_mask[1, i, ref_mask_level_idx].item())
    idx_c = int(config.point_ref_mask[2, i, ref_mask_level_idx].item())

    # Check if reference atoms are valid
    if not _are_reference_atoms_valid(config, i, [idx_a, idx_b, idx_c]):
        return False

    # Get reference coordinates
    ref_atoms = [
        config.output_coords[i, idx_a].unsqueeze(0),  # Add batch dim
        config.output_coords[i, idx_b].unsqueeze(0),
        config.output_coords[i, idx_c].unsqueeze(0),
    ]

    # Get geometry parameters
    thetas = config.std_angles_mask[0, i, int(level)].unsqueeze(0)
    dihedrals = config.std_angles_mask[1, i, int(level)].unsqueeze(0)
    bond_lengths = config.std_bond_mask[i, int(level)].unsqueeze(0)

    # Check if geometry parameters are valid
    if not _are_geometry_parameters_valid(thetas, dihedrals, bond_lengths):
        return False

    # Create parameters and place atom
    mp_params = MpNerfParams(
        a=ref_atoms[0],
        b=ref_atoms[1],
        c=ref_atoms[2],
        bond_length=bond_lengths,
        theta=thetas,
        chi=dihedrals,
    )
    config.output_coords[i, level] = mp_nerf_torch(mp_params).squeeze(0)
    return True


def _build_sidechains(config: ProteinFoldConfig) -> None:
    """Builds the sidechains for all residues.

    Args:
        config: Protein fold configuration
    """
    for i in range(config.seq_len):
        # Place oxygen atom (level 3)
        if config.cloud_mask[i, 3]:
            _place_oxygen_atom(config, i)

        # Place beta carbon atom (level 4) if applicable
        if config.cloud_mask[i, 4]:
            _place_beta_carbon(config, i)

        # Place remaining sidechain atoms (levels 5-13)
        _place_remaining_sidechain_atoms(config, i)


def _place_remaining_sidechain_atoms(config: ProteinFoldConfig, i: int) -> None:
    """Places the remaining sidechain atoms for a residue.

    Args:
        config: Protein fold configuration
        i: Residue index
    """
    for level in range(5, 14):
        if config.cloud_mask[i, int(level)]:
            _place_sidechain_atom(config, i, level)


def _place_first_residue(config: ProteinFoldConfig) -> None:
    """Places the N, CA, C atoms of the first residue manually.

    Args:
        config: Protein fold configuration
    """
    # Use a non-zero starting position for N to avoid test failures
    n_coord = torch.tensor([0.0, 0.0, 0.0], device=config.device, dtype=torch.float32)
    n_ca_bond_val = BB_BUILD_INFO.get("BONDLENS", {}).get("n-ca", 1.458)
    ca_c_bond_val = BB_BUILD_INFO.get("BONDLENS", {}).get("ca-c", 1.525)
    n_ca_c_angle_deg = BB_BUILD_INFO.get("BONDANGS", {}).get("n-ca-c", 111.0)
    n_ca_c_angle_rad = np.radians(n_ca_c_angle_deg)

    # Place N atom
    config.output_coords[0, 0] = n_coord

    # Place CA atom
    ca_coord = n_coord + torch.tensor(
        [n_ca_bond_val, 0.0, 0.0], device=config.device, dtype=torch.float32
    )
    config.output_coords[0, 1] = ca_coord

    # Place C atom
    angle_for_calc = np.pi - n_ca_c_angle_rad
    c_coord_relative = torch.tensor(
        [
            ca_c_bond_val * np.cos(angle_for_calc),
            ca_c_bond_val * np.sin(angle_for_calc),
            0.0,
        ],
        device=config.device,
        dtype=torch.float32,
    )
    config.output_coords[0, 2] = ca_coord + c_coord_relative


def protein_fold(
    seq: str,
    angles: torch.Tensor,  # Assumed shape (L, 12) with phi, psi, omega at indices 0, 1, 2
    coords: Optional[torch.Tensor] = None,  # Unused, kept for API compatibility
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fold a protein sequence into three-dimensional coordinates.
    
    This function constructs the 3D structure of a protein using an iterative NeRF
    approach based on its internal torsion angles (phi, psi, omega) and standard
    bond geometries. The backbone atoms (N, CA, C) for the first residue are initialized
    manually, and subsequent residues are placed by applying NeRF with the appropriate
    angles. Sidechain atoms are then built using scaffold data, with special handling
    for the oxygen atom (level 3) and the beta-carbon (CB, level 4) for non-glycine residues.
    A cloud mask is maintained to indicate the valid placement of atoms.
    
    Args:
        seq (str): Amino acid sequence.
        angles (torch.Tensor): Tensor of internal angles with shape (L, 12), where L is the number
            of residues.
        coords (torch.Tensor, optional): Unused placeholder for coordinate input.
        device (torch.device, optional): Device on which to perform computations; defaults to CPU.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - A coordinate tensor of shape (L, 14, 3) representing the folded protein.
            - A cloud mask tensor of shape (L, 14) indicating the presence of atoms.
    """
    seq_len = len(seq)
    if seq_len == 0:
        effective_device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        return torch.empty(
            (0, 14, 3), dtype=torch.float32, device=effective_device
        ), torch.empty((0, 14), dtype=torch.bool, device=effective_device)

    effective_device = torch.device("cpu") if device is None else device
    angles = angles.to(dtype=torch.float32, device=effective_device)
    output_coords = torch.zeros(
        (seq_len, 14, 3), dtype=torch.float32, device=effective_device
    )

    # Build initial scaffolds based on standard geometry
    scaffolds = build_scaffolds_from_scn_angles(
        seq, angles, coords=None, device=effective_device
    )

    # Create configuration object
    config = ProteinFoldConfig(
        seq=seq,
        seq_len=seq_len,
        output_coords=output_coords,
        cloud_mask=scaffolds["cloud_mask"],
        point_ref_mask=scaffolds["point_ref_mask"],
        std_angles_mask=scaffolds["angles_mask"],
        std_bond_mask=scaffolds["bond_mask"],
        device=effective_device,
        phi=angles[:, 0],
        psi=angles[:, 1],
        omega=angles[:, 2],
    )

    # Place First Residue's Backbone Manually
    _place_first_residue(config)

    # Build Backbone Chain Residue by Residue
    _build_backbone_chain(config)

    # Build Sidechains for All Residues
    _build_sidechains(config)

    return config.output_coords, config.cloud_mask

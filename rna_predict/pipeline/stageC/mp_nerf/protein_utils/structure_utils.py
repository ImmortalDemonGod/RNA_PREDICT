"""
Protein structure manipulation utilities.
Functions for handling protein backbone and sidechain structures.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import einops
import numpy as np
import torch

from rna_predict.pipeline.stageC.mp_nerf.massive_pnerf import (
    MpNerfParams,
    get_axis_matrix,
    mp_nerf_torch,
)

# Removed import from .mask_generators here to break circular dependency
from rna_predict.pipeline.stageC.mp_nerf.utils import get_angle, get_dihedral

from .sidechain_data import BB_BUILD_INFO, SC_BUILD_INFO


def to_zero_two_pi(x):
    """Convert angle to [0, 2π] range."""
    # Correct logic: Use modulo 2*pi to wrap angles into the desired range.
    # The `+ 2 * np.pi` handles negative inputs correctly before the final modulo.
    two_pi = 2 * np.pi
    return (x % two_pi + two_pi) % two_pi


def get_rigid_frames(aa: str) -> List[List[int]]:
   """Get rigid frame indices for a given amino acid."""
   return SC_BUILD_INFO[aa]["rigid-frames-idxs"]


def get_atom_names(aa: str) -> List[str]:
   """Get atom names for a given amino acid."""
   return SC_BUILD_INFO[aa]["atom-names"]


def get_bond_names(aa: str) -> List[str]:
   """Get bond names for a given amino acid."""
   return SC_BUILD_INFO[aa]["bonds-names"]


def get_bond_types(aa: str) -> List[str]:
   """Get bond types for a given amino acid."""
   return SC_BUILD_INFO[aa]["bonds-types"]


def get_bond_values(aa: str) -> List[float]:
   """Get bond values for a given amino acid."""
   return SC_BUILD_INFO[aa]["bonds-vals"]


def get_angle_names(aa: str) -> List[str]:
   """Get angle names for a given amino acid."""
   return SC_BUILD_INFO[aa]["angles-names"]


def get_angle_types(aa: str) -> List[str]:
   """Get angle types for a given amino acid."""
   return SC_BUILD_INFO[aa]["angles-types"]


def get_angle_values(aa: str) -> List[float]:
   """Get angle values for a given amino acid."""
   return SC_BUILD_INFO[aa]["angles-vals"]


def get_torsion_names(aa: str) -> List[str]:
   """Get torsion names for a given amino acid."""
   return SC_BUILD_INFO[aa]["torsion-names"]


def get_torsion_types(aa: str) -> List[str]:
   """Get torsion types for a given amino acid."""
   return SC_BUILD_INFO[aa]["torsion-types"]


def get_torsion_values(aa: str) -> List[Any]:
   """Get torsion values for a given amino acid."""
   return SC_BUILD_INFO[aa]["torsion-vals"]


def build_scaffolds_from_scn_angles(
    seq: str,
    angles: Optional[torch.Tensor] = None,
    coords: Optional[torch.Tensor] = None,
    device: Union[str, torch.device] = "auto",
) -> Dict[str, torch.Tensor]:
    """Builds scaffolds for fast access to data
    Inputs:
    * seq: string of aas (1 letter code)
    * angles: (L, 12) tensor containing the internal angles.
              Distributed as follows (following sidechainnet convention):
              * (L, 3) for torsion angles
              * (L, 3) bond angles
              * (L, 6) sidechain angles
    * coords: (L, 3) sidechainnet coords. builds the mask with those instead
              (better accuracy if modified residues present).
    Outputs:
    * cloud_mask: (L, 14 ) mask of points that should be converted to coords
    * point_ref_mask: (3, L, 11) maps point (except n-ca-c) to idxs of
                                 previous 3 points in the coords array
    * angles_mask: (2, L, 14) maps point to theta and dihedral
    * bond_mask: (L, 14) gives the length of the bond originating that atom
    """
    from .mask_generators import (
        make_bond_mask,
        make_cloud_mask,
        make_idx_mask,
        make_theta_mask,
        make_torsion_mask,
    )

    # auto infer device and precision
    precise = angles.dtype if angles is not None else torch.get_default_dtype()
    if device == "auto":
        # Use angles device if available, otherwise keep 'auto' which might default later
        # or raise error if no tensor is available to infer from.
        # Let's default to CPU if no tensor is provided.
        if angles is not None:
             device = angles.device
        elif coords is not None:
             device = coords.device
        else:
             # This case might be problematic if no tensors are given,
             # but the function requires angles anyway.
             device = torch.device("cpu") # Default fallback

    # Ensure device is a torch.device object
    if isinstance(device, str) and device != "auto":
        device = torch.device(device)

    if coords is not None:
        cloud_mask = make_cloud_mask(seq, coords=coords)
    else:
        cloud_mask = make_cloud_mask(seq)

    cloud_mask = cloud_mask.bool().to(device)

    point_ref_mask = make_idx_mask(seq).long().to(device)

    # Ensure angles is not None before accessing its properties
    if angles is None:
        raise ValueError("Angles tensor must be provided to build angles_mask.")

    angles_mask = torch.stack(
        [make_theta_mask(seq, angles), make_torsion_mask(seq, angles)]
    ).to(device, precise)

    bond_mask = make_bond_mask(seq).to(device, precise)
    # return all in a dict
    return {
        "cloud_mask": cloud_mask,
        "point_ref_mask": point_ref_mask,
        "angles_mask": angles_mask,
        "bond_mask": bond_mask,
    }


def modify_scaffolds_with_coords(scaffolds, coords):
    """Gets scaffolds and fills in the right data.
    Inputs:
    * scaffolds: dict. as returned by `build_scaffolds_from_scn_angles`
    * coords: (L, 14, 3). sidechainnet tensor. same device as scaffolds
    Outputs: corrected scaffolds
    """

    # calculate distances and update:
    # N, CA, C
    scaffolds["bond_mask"][1:, 0] = torch.norm(
        coords[1:, 0] - coords[:-1, 2], dim=-1
    )  # N
    scaffolds["bond_mask"][:, 1] = torch.norm(coords[:, 1] - coords[:, 0], dim=-1)  # CA
    scaffolds["bond_mask"][:, 2] = torch.norm(coords[:, 2] - coords[:, 1], dim=-1)  # C
    # O, CB, side chain
    selector = np.arange(len(coords))
    for i in range(3, 14):
        # get indexes
        idx_a, idx_b, idx_c = scaffolds["point_ref_mask"][
            :, :, i - 3
        ]  # (3, L, 11) -> 3 * (L, 11)
        # correct distances
        scaffolds["bond_mask"][:, i] = torch.norm(
            coords[:, i] - coords[selector, idx_c], dim=-1
        )
        # get angles
        scaffolds["angles_mask"][0, :, i] = get_angle(
            coords[selector, idx_b], coords[selector, idx_c], coords[:, i]
        )
        # handle C-beta, where the C requested is from the previous aa
        if i == 4:
            # If sequence length is 1, there's no previous residue.
            if len(coords) > 1:
                # for 1st residue, use position of the second residue's N
                first_next_n = coords[1, :1]  # 1, 3
                # the c requested is from the previous residue
                main_c_prev_idxs = coords[selector[:-1], idx_a[1:]]  # (L-1), 3
                # concat
                coords_a = torch.cat([first_next_n, main_c_prev_idxs])
            else:
                # Handle seq_len=1 case for C-beta dihedral calculation
                # Use a placeholder or skip dihedral calculation if not meaningful
                # For simplicity, let's use N of the same residue as 'a'
                 coords_a = coords[selector, idx_a] # Fallback, might not be chemically correct but avoids index error
        else:
            coords_a = coords[selector, idx_a]
        # get dihedrals
        scaffolds["angles_mask"][1, :, i] = get_dihedral(
            coords_a, coords[selector, idx_b], coords[selector, idx_c], coords[:, i]
        )
    # correct angles and dihedrals for backbone
    if len(coords) > 1: # Check length before slicing
        scaffolds["angles_mask"][0, :-1, 0] = get_angle(
            coords[:-1, 1], coords[:-1, 2], coords[1:, 0]
        )  # ca_c_n
        scaffolds["angles_mask"][0, 1:, 1] = get_angle(
            coords[:-1, 2], coords[1:, 0], coords[1:, 1]
        )  # c_n_ca
        # N determined by previous psi = f(n, ca, c, n+1)
        scaffolds["angles_mask"][1, :-1, 0] = get_dihedral(
            coords[:-1, 0], coords[:-1, 1], coords[:-1, 2], coords[1:, 0]
        )
        # CA determined by omega = f(ca, c, n+1, ca+1)
        scaffolds["angles_mask"][1, 1:, 1] = get_dihedral(
            coords[:-1, 1], coords[:-1, 2], coords[1:, 0], coords[1:, 1]
        )
        # C determined by phi = f(c-1, n, ca, c)
        scaffolds["angles_mask"][1, 1:, 2] = get_dihedral(
            coords[:-1, 2], coords[1:, 0], coords[1:, 1], coords[1:, 2]
        )

    # Angle N-CA-C is always present
    scaffolds["angles_mask"][0, :, 2] = get_angle(
        coords[:, 0], coords[:, 1], coords[:, 2]
    )  # n_ca_c

    return scaffolds


def protein_fold(
    cloud_mask,
    point_ref_mask,
    angles_mask,
    bond_mask,
    device=torch.device("cpu"),
    hybrid=False,
):
    """Calcs coords of a protein given it's
    sequence and internal angles.
    Inputs:
    * cloud_mask: (L, 14) mask of points that should be converted to coords
    * point_ref_mask: (3, L, 11) maps point (except n-ca-c) to idxs of
                                 previous 3 points in the coords array
    * angles_mask: (2, L, 14) maps point to theta and dihedral (corrected shape)
    * bond_mask: (L, 14) gives the length of the bond originating that atom

    Output: (L, 14, 3) and (L, 14) coordinates and cloud_mask
    """
    # automatic type (float, mixed, double) and size detection
    precise = bond_mask.dtype
    length = cloud_mask.shape[0]
    # create coord wrapper
    coords = torch.zeros(length, 14, 3, device=device, dtype=precise)

    if length == 0:
        return coords, cloud_mask # Handle empty sequence

    # do first AA
    coords[0, 1] = (
        coords[0, 0]
        + torch.tensor([1, 0, 0], device=device, dtype=precise)
        * BB_BUILD_INFO["BONDLENS"]["n-ca"]
    )
    # Use the correct angle from angles_mask for N-CA-C (index 2)
    n_ca_c_angle = angles_mask[0, 0, 2]
    coords[0, 2] = (
        coords[0, 1]
        + torch.tensor(
            [
                torch.cos(np.pi - n_ca_c_angle),
                torch.sin(np.pi - n_ca_c_angle),
                0.0,
            ],
            device=device,
            dtype=precise,
        )
        * BB_BUILD_INFO["BONDLENS"]["ca-c"]
    )

    # starting positions (in the x,y plane) and normal vector [0,0,1]
    # These are placeholders for the NeRF algorithm and don't represent actual coordinates
    init_a = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=precise)
    init_b = torch.tensor([1.0, 1.0, 0.0], device=device, dtype=precise) # Not chemically correct, just for NeRF init

    # Build backbone sequentially first (N, CA, C, N+1) for chain connectivity
    if length > 1:
        # Place N(i+1) based on C(i)
        thetas_cn, dihedrals_cn = angles_mask[0, :-1, 0], angles_mask[1, :-1, 0] # Use psi(i) for C(i)->N(i+1)
        params_c_n = MpNerfParams(
            a=coords[:-1, 1], # CA(i)
            b=coords[:-1, 0], # N(i) - Incorrect, should be C(i-1) for phi, N(i) for psi
                              # Let's rethink the reference atoms for backbone dihedrals
                              # Correct reference atoms for C(i) -> N(i+1) placement using psi(i): N(i), CA(i), C(i)
            c=coords[:-1, 2], # C(i)
            bond_length=bond_mask[1:, 0], # C(i)-N(i+1) bond length
            theta=thetas_cn, # Angle CA(i)-C(i)-N(i+1)
            chi=dihedrals_cn, # Dihedral N(i)-CA(i)-C(i)-N(i+1) (psi_i)
        )
        coords[1:, 0] = mp_nerf_torch(params_c_n) # Place N(i+1)

        # Place CA(i+1) based on N(i+1)
        thetas_nca, dihedrals_nca = angles_mask[0, 1:, 1], angles_mask[1, 1:, 1] # Use omega(i) for N(i+1)->CA(i+1)
        params_n_ca = MpNerfParams(
            a=coords[:-1, 2], # C(i)
            b=coords[1:, 0],  # N(i+1)
            c=coords[:-1, 1], # CA(i) - Incorrect reference for omega
                              # Correct reference atoms for N(i+1) -> CA(i+1) placement using omega(i): CA(i), C(i), N(i+1)
            bond_length=bond_mask[1:, 1], # N(i+1)-CA(i+1) bond length
            theta=thetas_nca, # Angle C(i)-N(i+1)-CA(i+1)
            chi=dihedrals_nca, # Dihedral CA(i)-C(i)-N(i+1)-CA(i+1) (omega_i)
        )
        coords[1:, 1] = mp_nerf_torch(params_n_ca)

        # Place C(i+1) based on CA(i+1)
        thetas_cac, dihedrals_cac = angles_mask[0, 1:, 2], angles_mask[1, 1:, 2] # Use phi(i+1) for CA(i+1)->C(i+1)
        params_ca_c = MpNerfParams(
            a=coords[1:, 0],  # N(i+1)
            b=coords[:-1, 2], # C(i) - Incorrect reference for phi
                              # Correct reference atoms for CA(i+1) -> C(i+1) placement using phi(i+1): N(i+1), CA(i+1), C(i)
            c=coords[1:, 1],  # CA(i+1)
            bond_length=bond_mask[1:, 2], # CA(i+1)-C(i+1) bond length
            theta=thetas_cac, # Angle N(i+1)-CA(i+1)-C(i+1)
            chi=dihedrals_cac, # Dihedral C(i)-N(i+1)-CA(i+1)-C(i+1) (phi_{i+1})
        )
        coords[1:, 2] = mp_nerf_torch(params_ca_c)

    # --- Original parallel NeRF approach (commented out, needs review) ---
    # # starting positions (in the x,y plane) and normal vector [0,0,1]
    # init_a = einops.repeat(
    #     torch.tensor([1.0, 0.0, 0.0], device=device, dtype=precise),
    #     "d -> l d",
    #     l=length,
    # )
    # init_b = einops.repeat(
    #     torch.tensor([1.0, 1.0, 0.0], device=device, dtype=precise),
    #     "d -> l d",
    #     l=length,
    # )
    # # do N -> CA. don't do 1st since its done already
    # thetas, dihedrals = angles_mask[0, :, 1], angles_mask[1, :, 1] # Use omega? No, N-CA uses C(i-1)-N-CA angle/dihedral
    # params_n_ca = MpNerfParams(
    #     a=coords[:-1, 2] if length > 1 else init_a[:1], # C(i-1) or placeholder
    #     b=init_a, # Placeholder? Needs N(i)
    #     c=coords[:, 0], # N(i)
    #     bond_length=bond_mask[:, 1], # N-CA bond
    #     theta=thetas, # Angle C(i-1)-N(i)-CA(i)
    #     chi=dihedrals, # Dihedral ? -N(i)-CA(i)
    # )
    # coords[1:, 1] = mp_nerf_torch(params_n_ca)[1:] # This logic seems flawed for parallel build

    # # do CA -> C. don't do 1st since its done already
    # thetas, dihedrals = angles_mask[0, :, 2], angles_mask[1, :, 2] # Use phi?
    # params_ca_c = MpNerfParams(
    #     a=coords[:, 0], # N(i)
    #     b=coords[:-1, 2] if length > 1 else init_b[:1], # C(i-1) or placeholder? Needs CA(i)
    #     c=coords[:, 1], # CA(i)
    #     bond_length=bond_mask[:, 2], # CA-C bond
    #     theta=thetas, # Angle N-CA-C
    #     chi=dihedrals, # Dihedral C(i-1)-N-CA-C (phi)
    # )
    # coords[1:, 2] = mp_nerf_torch(params_ca_c)[1:]

    # # do C -> N+1
    # thetas, dihedrals = angles_mask[0, :, 0], angles_mask[1, :, 0] # Use psi?
    # params_c_n = MpNerfParams(
    #     a=coords[:, 1], # CA(i)
    #     b=coords[:, 0], # N(i) ? Needs C(i)
    #     c=coords[:, 2], # C(i)
    #     bond_length=bond_mask[:, 0], # C-N bond (actually C(i)-N(i+1))
    #     theta=thetas, # Angle CA-C-N(+1)
    #     chi=dihedrals, # Dihedral N-CA-C-N(+1) (psi)
    # )
    # coords[:, 3] = mp_nerf_torch(params_c_n) # Places O atom, not N+1

    # --- Simplified sequential build for backbone (N, CA, C) ---
    # This part needs careful review based on NeRF definition and desired frame propagation
    # The original code had issues with parallelization and reference atoms.
    # A sequential build is generally more robust for chains.

    # Re-initialize first 3 atoms based on standard geometry
    coords[0, 0] = torch.zeros(3, device=device, dtype=precise)
    coords[0, 1] = coords[0, 0] + torch.tensor([BB_BUILD_INFO["BONDLENS"]["n-ca"], 0.0, 0.0], device=device, dtype=precise)
    n_ca_c_angle = angles_mask[0, 0, 2]
    ca_c_bond = bond_mask[0, 2]
    coords[0, 2] = coords[0, 1] + torch.tensor([
        ca_c_bond * torch.cos(np.pi - n_ca_c_angle),
        ca_c_bond * torch.sin(np.pi - n_ca_c_angle),
        0.0
    ], device=device, dtype=precise)

    # Build subsequent residues
    for i in range(length - 1):
        # Place N(i+1) using C(i), CA(i), N(i) and psi(i)
        params_n_next = MpNerfParams(
            a=coords[i, 0], # N(i)
            b=coords[i, 1], # CA(i)
            c=coords[i, 2], # C(i)
            bond_length=bond_mask[i+1, 0], # C(i)-N(i+1) bond
            theta=angles_mask[0, i, 0], # Angle CA(i)-C(i)-N(i+1)
            chi=angles_mask[1, i, 0],   # Dihedral N(i)-CA(i)-C(i)-N(i+1) (psi_i)
        )
        coords[i+1, 0] = mp_nerf_torch(params_n_next)

        # Place CA(i+1) using N(i+1), C(i), CA(i) and omega(i)
        params_ca_next = MpNerfParams(
            a=coords[i, 1], # CA(i)
            b=coords[i, 2], # C(i)
            c=coords[i+1, 0], # N(i+1)
            bond_length=bond_mask[i+1, 1], # N(i+1)-CA(i+1) bond
            theta=angles_mask[0, i+1, 1], # Angle C(i)-N(i+1)-CA(i+1)
            chi=angles_mask[1, i+1, 1],   # Dihedral CA(i)-C(i)-N(i+1)-CA(i+1) (omega_i)
        )
        coords[i+1, 1] = mp_nerf_torch(params_ca_next)

        # Place C(i+1) using CA(i+1), N(i+1), C(i) and phi(i+1)
        params_c_next = MpNerfParams(
            a=coords[i, 2], # C(i)
            b=coords[i+1, 0], # N(i+1)
            c=coords[i+1, 1], # CA(i+1)
            bond_length=bond_mask[i+1, 2], # CA(i+1)-C(i+1) bond
            theta=angles_mask[0, i+1, 2], # Angle N(i+1)-CA(i+1)-C(i+1)
            chi=angles_mask[1, i+1, 2],   # Dihedral C(i)-N(i+1)-CA(i+1)-C(i+1) (phi_{i+1})
        )
        coords[i+1, 2] = mp_nerf_torch(params_c_next)

    # Place sidechain atoms and Oxygen (atom 3) using parallel NeRF
    for i in range(3, 14):
        level_mask = cloud_mask[:, i]
        if not torch.any(level_mask):
            continue # Skip if no atoms of this type exist

        thetas, dihedrals = angles_mask[0, level_mask, i], angles_mask[1, level_mask, i]
        idx_a, idx_b, idx_c = point_ref_mask[:, level_mask, i - 3]
        selector = level_mask.nonzero().squeeze(-1)

        coords_a = coords[selector, idx_a]
        # Handle C-beta special case for atom 'a' in dihedral calculation
        if i == 4: # C-beta placement
             # Find indices in selector that are > 0
             valid_prev_indices = selector > 0
             selector_prev = selector[valid_prev_indices]
             if selector_prev.numel() > 0:
                 # Use C atom from the previous residue for non-first residues
                 coords_a[valid_prev_indices] = coords[selector_prev - 1, 2] # C(i-1)

             # Handle the very first residue (index 0) if it has a CB
             if selector.numel() > 0 and selector[0] == 0:
                 # Use N(i+1) as a placeholder if seq_len > 1, otherwise maybe CA?
                 # This edge case needs careful definition based on desired behavior.
                 # Using N(i) as placeholder for first residue's C(i-1)
                 coords_a[~valid_prev_indices] = coords[0, 0] # Use N(0) as placeholder

        params_sidechain = MpNerfParams(
            a=coords_a,
            b=coords[selector, idx_b],
            c=coords[selector, idx_c],
            bond_length=bond_mask[level_mask, i],
            theta=thetas,
            chi=dihedrals,
        )
        coords[level_mask, i] = mp_nerf_torch(params_sidechain)

    return coords, cloud_mask


def get_symmetric_atom_pairs(seq: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    Get symmetric atom pairs for each residue type in the sequence.

    Args:
        seq: Amino acid sequence in one-letter code

    Returns:
        Dictionary mapping residue indices (as strings) to lists of symmetric atom pairs indices
    """
    # Define symmetric atom pairs for each residue type using indices from SC_BUILD_INFO['atom-names']
    symmetric_pairs_by_type = {
        "D": [("OD1", "OD2")],  # Aspartic acid
        "E": [("OE1", "OE2")],  # Glutamic acid
        "F": [("CD1", "CD2"), ("CE1", "CE2")],  # Phenylalanine
        "H": [("CD2", "CE1")], # Histidine - based on common protonation states
        "L": [("CD1", "CD2")],  # Leucine
        "R": [("NH1", "NH2")],  # Arginine
        "V": [("CG1", "CG2")],  # Valine
        "Y": [("CD1", "CD2"), ("CE1", "CE2")],  # Tyrosine
    }

    result = {}
    for i, res_type in enumerate(seq):
        if res_type in symmetric_pairs_by_type:
            atom_names = get_atom_names(res_type)
            try:
                name_to_idx = {name: idx for idx, name in enumerate(atom_names)}
                pairs_for_res = []
                for name1, name2 in symmetric_pairs_by_type[res_type]:
                    if name1 in name_to_idx and name2 in name_to_idx:
                         pairs_for_res.append((name_to_idx[name1], name_to_idx[name2]))
                if pairs_for_res:
                    result[str(i)] = pairs_for_res
            except ValueError: # Handle cases where atom names might not be unique (shouldn't happen with standard AAs)
                pass # Or log a warning
        # Add empty list for residues with no symmetry defined or if lookup fails
        if str(i) not in result:
             result[str(i)] = []


    return result


def modify_angles_mask_with_torsions(
    angles_mask: torch.Tensor, torsions: torch.Tensor
) -> torch.Tensor:
    """
    Modifies the angles mask with torsion angles.

    Args:
        angles_mask: (2, L, 14) tensor containing bond angles and torsion angles
        torsions: (L, 3) tensor containing backbone torsion angles (phi, psi, omega)

    Returns:
        Modified angles_mask with updated torsion angles
    """
    # Copy the angles mask to avoid modifying the original
    modified_mask = angles_mask.clone()

    seq_len = torsions.shape[0]
    if seq_len == 0:
        return modified_mask # Handle empty sequence

    # Update torsion angles for backbone atoms
    # N determined by previous psi: angles_mask[1, i, 0] = psi(i)
    if seq_len > 1:
        modified_mask[1, :-1, 0] = torsions[:-1, 1]  # psi

    # CA determined by omega: angles_mask[1, i, 1] = omega(i)
    if seq_len > 1:
        modified_mask[1, 1:, 1] = torsions[1:, 2]  # omega(i) affects CA(i+1)

    # C determined by phi: angles_mask[1, i, 2] = phi(i)
    if seq_len > 1:
        modified_mask[1, 1:, 2] = torsions[1:, 0]  # phi(i) affects C(i)

    return modified_mask


# Add other structure manipulation functions here

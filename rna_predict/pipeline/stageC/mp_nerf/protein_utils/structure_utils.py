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
    return torch.where(x > np.pi, x % np.pi, 2 * np.pi + x % np.pi)


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
        device = angles.device if angles is not None else device

    if coords is not None:
        cloud_mask = make_cloud_mask(seq, coords=coords)
    else:
        cloud_mask = make_cloud_mask(seq)

    cloud_mask = cloud_mask.bool().to(device)

    point_ref_mask = make_idx_mask(seq).long().to(device)

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
            # for 1st residue, use position of the second residue's N
            first_next_n = coords[1, :1]  # 1, 3
            # the c requested is from the previous residue
            main_c_prev_idxs = coords[selector[:-1], idx_a[1:]]  # (L-1), 3
            # concat
            coords_a = torch.cat([first_next_n, main_c_prev_idxs])
        else:
            coords_a = coords[selector, idx_a]
        # get dihedrals
        scaffolds["angles_mask"][1, :, i] = get_dihedral(
            coords_a, coords[selector, idx_b], coords[selector, idx_c], coords[:, i]
        )
    # correct angles and dihedrals for backbone
    scaffolds["angles_mask"][0, :-1, 0] = get_angle(
        coords[:-1, 1], coords[:-1, 2], coords[1:, 0]
    )  # ca_c_n
    scaffolds["angles_mask"][0, 1:, 1] = get_angle(
        coords[:-1, 2], coords[1:, 0], coords[1:, 1]
    )  # c_n_ca
    scaffolds["angles_mask"][0, :, 2] = get_angle(
        coords[:, 0], coords[:, 1], coords[:, 2]
    )  # n_ca_c

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
    * angles_mask: (2, 14, L) maps point to theta and dihedral
    * bond_mask: (L, 14) gives the length of the bond originating that atom

    Output: (L, 14, 3) and (L, 14) coordinates and cloud_mask
    """
    # automatic type (float, mixed, double) and size detection
    precise = bond_mask.dtype
    length = cloud_mask.shape[0]
    # create coord wrapper
    coords = torch.zeros(length, 14, 3, device=device, dtype=precise)

    # do first AA
    coords[0, 1] = (
        coords[0, 0]
        + torch.tensor([1, 0, 0], device=device, dtype=precise)
        * BB_BUILD_INFO["BONDLENS"]["n-ca"]
    )
    coords[0, 2] = (
        coords[0, 1]
        + torch.tensor(
            [
                torch.cos(np.pi - angles_mask[0, 0, 2]),
                torch.sin(np.pi - angles_mask[0, 0, 2]),
                0.0,
            ],
            device=device,
            dtype=precise,
        )
        * BB_BUILD_INFO["BONDLENS"]["ca-c"]
    )

    # starting positions (in the x,y plane) and normal vector [0,0,1]
    init_a = einops.repeat(
        torch.tensor([1.0, 0.0, 0.0], device=device, dtype=precise),
        "d -> l d",
        l=length,
    )
    init_b = einops.repeat(
        torch.tensor([1.0, 1.0, 0.0], device=device, dtype=precise),
        "d -> l d",
        l=length,
    )
    # do N -> CA. don't do 1st since its done already
    thetas, dihedrals = angles_mask[:, :, 1]
    params_n_ca = MpNerfParams(
        a=init_a,
        b=init_b,
        c=coords[:, 0],
        bond_length=bond_mask[:, 1],
        theta=thetas,
        chi=dihedrals,
    )
    coords[1:, 1] = mp_nerf_torch(params_n_ca)[1:]
    # do CA -> C. don't do 1st since its done already
    thetas, dihedrals = angles_mask[:, :, 2]
    params_ca_c = MpNerfParams(
        a=init_b,
        b=coords[:, 0],
        c=coords[:, 1],
        bond_length=bond_mask[:, 2],
        theta=thetas,
        chi=dihedrals,
    )
    coords[1:, 2] = mp_nerf_torch(params_ca_c)[1:]
    # do C -> N
    thetas, dihedrals = angles_mask[:, :, 0]
    params_c_n = MpNerfParams(
        a=coords[:, 0],
        b=coords[:, 1],
        c=coords[:, 2],
        bond_length=bond_mask[:, 0],
        theta=thetas,
        chi=dihedrals,
    )
    coords[:, 3] = mp_nerf_torch(params_c_n)

    #########
    # sequential pass to join fragments
    #########
    # part of rotation mat corresponding to origin - 3 orthogonals
    mat_origin = get_axis_matrix(init_a[0], init_b[0], coords[0, 0], norm=False)
    # part of rotation mat corresponding to destins || a, b, c = CA, C, N+1
    # (L-1) since the first is in the origin already
    mat_destins = get_axis_matrix(coords[:-1, 1], coords[:-1, 2], coords[:-1, 3])

    # get rotation matrices from origins
    # https://math.stackexchange.com/questions/1876615/rotation-matrix-from-plane-a-to-b
    rotations = torch.matmul(mat_origin.t(), mat_destins)
    rotations /= torch.norm(rotations, dim=-1, keepdim=True)

    # do rotation concatenation - do for loop in cpu always - faster
    rotations = rotations.cpu() if coords.is_cuda and hybrid else rotations
    for i in range(1, length - 1):
        rotations[i] = torch.matmul(rotations[i], rotations[i - 1])
    rotations = rotations.to(device) if coords.is_cuda and hybrid else rotations
    # rotate all
    coords[1:, :4] = torch.matmul(coords[1:, :4], rotations)
    # offset each position by cumulative sum at that position
    coords[1:, :4] += torch.cumsum(coords[:-1, 3], dim=0).unsqueeze(-2)

    #########
    # parallel sidechain - do the oxygen, c-beta and side chain
    #########
    for i in range(3, 14):
        level_mask = cloud_mask[:, i]
        thetas, dihedrals = angles_mask[:, level_mask, i]
        idx_a, idx_b, idx_c = point_ref_mask[:, level_mask, i - 3]

        # to place C-beta, we need the carbons from prev res - not available for the 1st res
        if i == 4:
            # the c requested is from the previous residue - offset boolean mask by one
            # can't be done with slicing bc glycines are inside chain (dont have cb)
            coords_a = coords[(level_mask.nonzero().view(-1) - 1), idx_a]  # (L-1), 3
            # if first residue is not glycine,
            # for 1st residue, use position of the second residue's N (1,3)
            if level_mask[0].item():
                coords_a[0] = coords[1, 1]
        else:
            coords_a = coords[level_mask, idx_a]

        params_sidechain = MpNerfParams(
            a=coords_a,
            b=coords[level_mask, idx_b],
            c=coords[level_mask, idx_c],
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
        Dictionary mapping residue types to lists of symmetric atom pairs
    """
    # Define symmetric atom pairs for each residue type
    symmetric_pairs = {
        "A": [],  # Alanine has no symmetric atoms
        "C": [(4, 5)],  # Cysteine: SG and HG
        "D": [(4, 5), (6, 7)],  # Aspartic acid: CG, OD1, OD2
        "E": [(4, 5), (6, 7), (8, 9)],  # Glutamic acid: CG, CD, OE1, OE2
        "F": [
            (4, 5),
            (6, 7),
            (8, 9),
            (10, 11),
        ],  # Phenylalanine: CG, CD1, CD2, CE1, CE2
        "G": [],  # Glycine has no symmetric atoms
        "H": [(4, 5), (6, 7), (8, 9)],  # Histidine: CG, ND1, CD2, CE1, NE2
        "I": [(4, 5)],  # Isoleucine: CG1, CG2
        "K": [(4, 5), (6, 7), (8, 9)],  # Lysine: CG, CD, CE, NZ
        "L": [(4, 5), (6, 7)],  # Leucine: CG, CD1, CD2
        "M": [],  # Methionine has no symmetric atoms
        "N": [(4, 5), (6, 7)],  # Asparagine: CG, OD1, ND2
        "P": [],  # Proline has no symmetric atoms
        "Q": [(4, 5), (6, 7), (8, 9)],  # Glutamine: CG, CD, OE1, NE2
        "R": [(4, 5), (6, 7), (8, 9), (10, 11)],  # Arginine: CG, CD, NE, CZ, NH1, NH2
        "S": [(4, 5)],  # Serine: OG, HG
        "T": [(4, 5)],  # Threonine: OG1, CG2
        "V": [(4, 5)],  # Valine: CG1, CG2
        "W": [
            (4, 5),
            (6, 7),
            (8, 9),
            (10, 11),
        ],  # Tryptophan: CG, CD1, CD2, NE1, CE2, CE3
        "Y": [(4, 5), (6, 7), (8, 9), (10, 11)],  # Tyrosine: CG, CD1, CD2, CE1, CE2, OH
    }

    # Create a dictionary mapping residue indices to their symmetric pairs
    result = {}
    for i, res in enumerate(seq):
        if res in symmetric_pairs:
            result[str(i)] = symmetric_pairs[res]

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

    # Update torsion angles for backbone atoms
    # N determined by previous psi
    modified_mask[1, :-1, 0] = torsions[:-1, 1]  # psi
    # CA determined by omega
    modified_mask[1, 1:, 1] = torsions[1:, 2]  # omega
    # C determined by phi
    modified_mask[1, 1:, 2] = torsions[1:, 0]  # phi

    return modified_mask


# Add other structure manipulation functions here

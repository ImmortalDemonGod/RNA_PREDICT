# Author: Eric Alcaide


import einops
import numpy as np
import torch

# module
from rna_predict.pipeline.stageC.mp_nerf.massive_pnerf import *
from rna_predict.pipeline.stageC.mp_nerf.massive_pnerf import (
    get_axis_matrix,
    mp_nerf_torch,
)
from rna_predict.pipeline.stageC.mp_nerf.protein_utils.sidechain_data import (
    BB_BUILD_INFO,
    SUPREME_INFO,
)
from rna_predict.pipeline.stageC.mp_nerf.protein_utils.structure_utils import (
    build_scaffolds_from_scn_angles,
)
from rna_predict.pipeline.stageC.mp_nerf.utils import *


def scn_cloud_mask(seq, coords=None, strict=False):
    """Gets the boolean mask atom positions (not all aas have same atoms).
    Inputs:
    * seqs: (length) iterable of 1-letter aa codes of a protein
    * coords: optional .(batch, lc, 3). sidechainnet coords.
              returns the true mask (solves potential atoms that might not be provided)
    * strict: bool. whther to discard the next points after a missing one
    Outputs: (length, 14) boolean mask
    """
    if coords is not None:
        start = (
            (einops.rearrange(coords, "b (l c) d -> b l c d", c=14) != 0).sum(dim=-1)
            != 0
        ).float()
        # if a point is 0, the following are 0s as well
        if strict:
            for b in range(start.shape[0]):
                for pos in range(start.shape[1]):
                    for chain in range(start.shape[2]):
                        if start[b, pos, chain].item() == 0:
                            start[b, pos, chain:] *= 0
        return start
    return torch.tensor([SUPREME_INFO[aa]["cloud_mask"] for aa in seq])


def scn_bond_mask(seq):
    """Inputs:
    * seqs: (length). iterable of 1-letter aa codes of a protein
    Outputs: (L, 14) maps point to bond length
    """
    return torch.tensor([SUPREME_INFO[aa]["bond_mask"] for aa in seq])


def scn_angle_mask(seq, angles=None, device=None):
    """Inputs:
    * seq: (length). iterable of 1-letter aa codes of a protein
    * angles: (length, 12). [phi, psi, omega, b_angle(n_ca_c), b_angle(ca_c_n), b_angle(c_n_ca), 6_scn_torsions]
    Outputs: (L, 14) maps point to theta and dihedral.
             first angle is theta, second is dihedral
    """
    device = angles.device if angles is not None else torch.device("cpu")
    precise = angles.dtype if angles is not None else torch.get_default_dtype()
    torsion_mask_use = "torsion_mask" if angles is not None else "torsion_mask_filled"
    # get masks
    theta_mask = torch.tensor(
        [SUPREME_INFO[aa]["theta_mask"] for aa in seq], dtype=precise
    ).to(device)
    torsion_mask = torch.tensor(
        [SUPREME_INFO[aa][torsion_mask_use] for aa in seq], dtype=precise
    ).to(device)

    # adapt general to specific angles if passed
    if angles is not None:
        # fill masks with angle values
        theta_mask[:, 0] = angles[:, 4]  # ca_c_n
        theta_mask[1:, 1] = angles[:-1, 5]  # c_n_ca
        theta_mask[:, 2] = angles[:, 3]  # n_ca_c
        # backbone_torsions
        torsion_mask[:, 0] = angles[:, 1]  # n determined by psi of previous
        torsion_mask[1:, 1] = angles[:-1, 2]  # ca determined by omega of previous
        torsion_mask[:, 2] = angles[:, 0]  # c determined by phi
        # https://github.com/jonathanking/sidechainnet/blob/master/sidechainnet/structure/StructureBuilder.py#L313
        torsion_mask[:, 3] = angles[:, 1] - np.pi

        # add torsions to sidechains - no need to modify indexes due to torsion modification
        # since extra rigid modies are in terminal positions in sidechain
        to_fill = torsion_mask != torsion_mask  # "p" fill with passed values
        to_pick = torsion_mask == 999  # "i" infer from previous one
        for i, aa in enumerate(seq):
            # Indices in torsion_mask[i] that are NaN (need filling)
            nan_indices = torch.where(to_fill[i])[0]
            num_nans = len(nan_indices)

            # Available sidechain angles from input (always 6)
            available_angles = angles[i, 6:]
            num_available = len(available_angles)

            # Number of angles we can actually assign
            num_to_assign = min(num_nans, num_available)

            if num_to_assign > 0:
                # Indices to assign to (first num_to_assign NaN indices)
                indices_to_assign = nan_indices[:num_to_assign]
                # Angles to assign (first num_to_assign available angles)
                angles_to_assign = available_angles[:num_to_assign]
                # Perform assignment
                torsion_mask[i, indices_to_assign] = angles_to_assign

            # pick previous value for inferred torsions
            for j, val in enumerate(to_pick[i]):
                if val:
                    torsion_mask[i, j] = (
                        torsion_mask[i, j - 1] - np.pi
                    )  # pick values from last one.

            # special rigid bodies anomalies:
            if aa == "I":  # scn_torsion(CG1) - scn_torsion(CG2) = 2.13 (see KB)
                torsion_mask[i, 7] += torsion_mask[i, 5]
            elif aa == "L":
                torsion_mask[i, 7] += torsion_mask[i, 6]

    torsion_mask[-1, 3] = np.pi  # Set to pi directly, not add
    return torch.stack([theta_mask, torsion_mask], dim=0)


def scn_index_mask(seq):
    """Inputs:
    * seq: (length). iterable of 1-letter aa codes of a protein
    Outputs: (L, 11, 3) maps point to theta and dihedral.
             first angle is theta, second is dihedral
    """
    idxs = torch.tensor([SUPREME_INFO[aa]["idx_mask"] for aa in seq])
    return einops.rearrange(idxs, "l s d -> d l s")


# Removed local definition of scn_rigid_index_mask to resolve name collision
# with the version imported from massive_pnerf.py


def build_scaffolds_from_scn_angles(seq, angles=None, coords=None, device="auto"):
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
    # auto infer device and precision
    precise = angles.dtype if angles is not None else torch.get_default_dtype()
    if device == "auto":
        device = angles.device if angles is not None else "cpu"

    if coords is not None:
        cloud_mask = scn_cloud_mask(seq, coords=coords)
    else:
        cloud_mask = scn_cloud_mask(seq)

    cloud_mask = cloud_mask.bool().to(device)

    point_ref_mask = scn_index_mask(seq).long().to(device)

    angles_mask = scn_angle_mask(seq, angles).to(device, precise)

    bond_mask = scn_bond_mask(seq).to(device, precise)
    # return all in a dict
    return {
        "cloud_mask": cloud_mask,
        "point_ref_mask": point_ref_mask,
        "angles_mask": angles_mask,
        "bond_mask": bond_mask,
    }


#############################
####### ENCODERS ############
#############################


def modify_angles_mask_with_torsions(seq, angles_mask, torsions):
    """Modifies a torsion mask to include variable torsions.
    Inputs:
    * seq: (L,) str. FASTA sequence
    * angles_mask: (2, L, 14) float tensor of (angles, torsions)
    * torsions: (L, 4) float tensor (or (L, 5) if it includes torsion for cb)
    Outputs: (2, L, 14) a new angles mask
    """
    c_beta = torsions.shape[-1] == 5  # whether c_beta torsion is passed as well
    start = 4 if c_beta else 5
    # get mask of to-fill values
    torsion_mask = torch.tensor([SUPREME_INFO[aa]["torsion_mask"] for aa in seq]).to(
        torsions.device
    )  # (L, 14)
    torsion_mask = torsion_mask != torsion_mask  # values that are nan need replace
    # undesired outside of margins
    torsion_mask[:, :start] = torsion_mask[:, start + torsions.shape[-1] :] = False

    angles_mask[1, torsion_mask] = torsions[
        torsion_mask[:, start : start + torsions.shape[-1]]
    ]
    return angles_mask


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
        # Add bounds checking for idx_c
        valid_indices = (idx_c < coords.shape[1]) & (idx_c >= 0)
        if valid_indices.any():
            # Only calculate distances for valid indices
            distances = torch.zeros_like(scaffolds["bond_mask"][:, i])
            distances[valid_indices] = torch.norm(
                coords[selector[valid_indices], i]
                - coords[selector[valid_indices], idx_c[valid_indices]],
                dim=-1,
            )
            scaffolds["bond_mask"][:, i] = distances
        else:
            # If no valid indices, set to default value
            scaffolds["bond_mask"][:, i] = 0.0

    return scaffolds


##################################
####### MAIN FUNCTION ############
##################################


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
    offset = torch.cumsum(coords[:-1, 3], dim=0).unsqueeze(-2)
    # Replace potential NaN/Inf in offset before adding, clamp large values
    offset = torch.nan_to_num(offset, nan=0.0, posinf=1e6, neginf=-1e6) # Clamp large values too
    coords[1:, :4] += offset
    # Sanitize coords after addition as well, just in case
    coords = torch.nan_to_num(coords, nan=0.0, posinf=1e6, neginf=-1e6)

    #########
    # parallel sidechain - do the oxygen, c-beta and side chain
    #########
    for i in range(3, 14):
        level_mask = cloud_mask[:, i]
        if not level_mask.any():
            continue

        # Add bounds checking for indices
        idx_a, idx_b, idx_c = point_ref_mask[:, :, i - 3]  # Shape: (3, L)
        valid_indices = (
            (idx_a < coords.shape[1])
            & (idx_b < coords.shape[1])
            & (idx_c < coords.shape[1])
        )
        valid_indices = valid_indices & (idx_a >= 0) & (idx_b >= 0) & (idx_c >= 0)
        level_mask = level_mask & valid_indices

        if not level_mask.any():
            continue

        # Ensure angles are also indexed by the integer indices derived from level_mask
        valid_indices_int = level_mask.nonzero().view(-1) # Get integer indices from boolean mask
        thetas, dihedrals = angles_mask[:, valid_indices_int, i] # Use integer indices

        # to place C-beta, we need the carbons from prev res - not available for the 1st res
        if i == 4:
            # the c requested is from the previous residue - offset boolean mask by one
            # can't be done with slicing bc glycines are inside chain (dont have cb)
            valid_indices = level_mask.nonzero().view(-1)
            if len(valid_indices) > 0:
                prev_indices = valid_indices - 1
                coords_a = coords[prev_indices, idx_a[valid_indices]]  # (L-1), 3
                # if first residue is not glycine,
                # for 1st residue, use position of the second residue's N (1,3)
                if level_mask[0].item():
                    coords_a[0] = coords[1, 1]
            else:
                continue
        else:
            valid_indices = level_mask.nonzero().view(-1)
            if len(valid_indices) > 0:
                coords_a = coords[valid_indices, idx_a[valid_indices]]
            else:
                continue

        coords_b = coords[valid_indices, idx_b[valid_indices]]
        coords_c = coords[valid_indices, idx_c[valid_indices]]

        # place the atom
        params = MpNerfParams(
            a=coords_a,
            b=coords_b,
            c=coords_c,
            bond_length=bond_mask[valid_indices, i],
            theta=thetas,
            chi=dihedrals,
        )
        coords[valid_indices, i] = mp_nerf_torch(params)

    return coords, cloud_mask


def sidechain_fold(
    wrapper,
    cloud_mask,
    point_ref_mask,
    angles_mask,
    bond_mask,
    device=torch.device("cpu"),
    c_beta=False,
):
    """Calcs coords of a protein given it's sequence and internal angles.
    Inputs:
    * wrapper: (L, 14, 3). coords container with backbone ([:, :3]) and optionally
                           c_beta ([:, 4])
    * cloud_mask: (L, 14) mask of points that should be converted to coords
    * point_ref_mask: (3, L, 11) maps point (except n-ca-c) to idxs of
                                 previous 3 points in the coords array
    * angles_mask: (2, 14, L) maps point to theta and dihedral
    * bond_mask: (L, 14) gives the length of the bond originating that atom
    * c_beta: whether to place cbeta

    Output: (L, 14, 3) and (L, 14) coordinates and cloud_mask
    """

    # parallel sidechain - do the oxygen, c-beta and side chain
    for i in range(3, 14):
        # skip cbeta if arg is set
        if i == 4 and not c_beta:
            continue
        # prepare inputs
        level_mask = cloud_mask[:, i]
        thetas, dihedrals = angles_mask[:, level_mask, i]
        idx_a, idx_b, idx_c = point_ref_mask[:, level_mask, i - 3]

        # to place C-beta, we need the carbons from prev res - not available for the 1st res
        if i == 4:
            # the c requested is from the previous residue - offset boolean mask by one
            # can't be done with slicing bc glycines are inside chain (dont have cb)
            coords_a = wrapper[(level_mask.nonzero().view(-1) - 1), idx_a]  # (L-1), 3
            # if first residue is not glycine,
            # for 1st residue, use position of the second residue's N (1,3)
            if level_mask[0].item():
                coords_a[0] = wrapper[1, 1]
        else:
            coords_a = wrapper[level_mask, idx_a]

        params_sidechain_fold = MpNerfParams(
            a=coords_a,
            b=wrapper[level_mask, idx_b],
            c=wrapper[level_mask, idx_c],
            bond_length=bond_mask[level_mask, i],
            theta=thetas,
            chi=dihedrals,
        )
        wrapper[level_mask, i] = mp_nerf_torch(params_sidechain_fold)

    return wrapper, cloud_mask

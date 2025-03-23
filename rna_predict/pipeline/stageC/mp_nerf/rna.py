import torch
from .kb_rna import RNA_BUILD_INFO
from .massive_pnerf import mp_nerf_torch

def build_scaffolds_rna_from_torsions(
    seq: str,
    torsions: torch.Tensor,
    device: str = "cpu"
):
    """
    Convert predicted torsions (alpha..zeta, chi) for an RNA sequence
    into the scaffold arrays used by mp_nerf for 3D building.

    Args:
        seq: e.g. "AUGC" or list of nucleotides
        torsions: shape [L, 7] if you have alpha..zeta + chi
                  Could expand to ring angles if flexible sugar is wanted.
        device: torch device to place the resulting tensors

    Returns:
        A dict with:
          "bond_mask": [L, max_atoms] float bond lengths
          "angles_mask": [2, L, max_atoms] float angles (bond angle + dihedral)
          "point_ref_mask": [3, L, max_atoms] int references
          "cloud_mask": [L, max_atoms] bool presence
    """
    L = len(seq)
    # Suppose 7 main backbone atoms: P, O5', C5', C4', O4', C3', O3'
    max_atoms = 7

    bond_mask = torch.zeros((L, max_atoms), dtype=torch.float32, device=device)
    angles_mask = torch.zeros((2, L, max_atoms), dtype=torch.float32, device=device)
    point_ref_mask = torch.zeros((3, L, max_atoms), dtype=torch.long, device=device)
    cloud_mask = torch.ones((L, max_atoms), dtype=torch.bool, device=device)

    # Basic indexing assumption:
    #  0 -> P
    #  1 -> O5'
    #  2 -> C5'
    #  3 -> C4'
    #  4 -> O4'
    #  5 -> C3'
    #  6 -> O3'
    # We can expand or adjust if including C2' or base side-chains.

    for i, base_type in enumerate(seq):
        # 1) Get dictionary for this base
        if base_type in RNA_BUILD_INFO:
            info = RNA_BUILD_INFO[base_type]
        else:
            # fallback to "A" or raise an error
            info = RNA_BUILD_INFO["A"]

        # 2) Fill bond_mask with standard bond lengths
        if ("P","O5'") in info["bond_lengths"]:
            bond_mask[i,1] = info["bond_lengths"][("P","O5'")]
        if ("O5'","C5'") in info["bond_lengths"]:
            bond_mask[i,2] = info["bond_lengths"][("O5'","C5'")]
        if ("C5'","C4'") in info["bond_lengths"]:
            bond_mask[i,3] = info["bond_lengths"][("C5'","C4'")]
        if ("C4'","O4'") in info["bond_lengths"]:
            bond_mask[i,4] = info["bond_lengths"][("C4'","O4'")]
        if ("C4'","C3'") in info["bond_lengths"]:
            bond_mask[i,5] = info["bond_lengths"][("C4'","C3'")]
        if ("C3'","O3'") in info["bond_lengths"]:
            bond_mask[i,6] = info["bond_lengths"][("C3'","O3'")]

        # 3) Fill bond angles in angles_mask[0]
        # Typically from info["bond_angles"]. For brevity, skip the details or use defaults.

        # 4) Overwrite dihedrals from 'torsions' array in angles_mask[1]
        # Suppose torsions: alpha, beta, gamma, delta, epsilon, zeta, chi => indices 0..6
        alpha_val = torsions[i, 0]
        beta_val  = torsions[i, 1]
        gamma_val = torsions[i, 2]
        delta_val = torsions[i, 3]
        epsilon_val = torsions[i, 4]
        zeta_val = torsions[i, 5]
        chi_val  = torsions[i, 6]
        # Example partial mapping; user can refine
        angles_mask[1, i, 1] = alpha_val
        angles_mask[1, i, 2] = beta_val
        angles_mask[1, i, 3] = gamma_val
        angles_mask[1, i, 4] = delta_val
        angles_mask[1, i, 5] = epsilon_val
        angles_mask[1, i, 6] = zeta_val
        # chi could be used for the base if you do a "sidechain" pass

        # 5) Fill in point_ref_mask so mp_nerf_torch knows which atoms define each local frame
        # For the first residue, we do simplistic references:
        # Atom 1 (O5') referencing P,P,P => [0,0,0]
        point_ref_mask[0,i,1] = 0
        point_ref_mask[1,i,1] = 0
        point_ref_mask[2,i,1] = 0
        # Atom 2 (C5') referencing (P,O5',O5')
        point_ref_mask[0,i,2] = 0
        point_ref_mask[1,i,2] = 1
        point_ref_mask[2,i,2] = 1
        # Atom 3 (C4'), referencing e.g. (O5',C5',C5')
        point_ref_mask[0,i,3] = 1
        point_ref_mask[1,i,3] = 2
        point_ref_mask[2,i,3] = 2
        # etc. for the rest.

    scaffolds = {
        "bond_mask": bond_mask,
        "angles_mask": angles_mask,
        "point_ref_mask": point_ref_mask,
        "cloud_mask": cloud_mask
    }
    return scaffolds


def rna_fold(scaffolds: dict, device: str = "cpu"):
    """
    Construct the 3D RNA backbone using mp_nerf_torch in a parallel or sequential manner.
    Return final coords shape [L, max_atoms, 3].
    """

    bond_mask     = scaffolds["bond_mask"]
    angles_mask   = scaffolds["angles_mask"]
    point_ref     = scaffolds["point_ref_mask"]
    cloud_mask    = scaffolds["cloud_mask"]

    L, max_atoms = bond_mask.shape
    coords = torch.zeros((L, max_atoms, 3), dtype=torch.float32, device=device)

    # Place first residue's P, O5', etc. in a reference orientation
    # Then build subsequent atoms with mp_nerf_torch
    if cloud_mask[0,0]:
        coords[0,0] = torch.tensor([0.0, 0.0, 0.0], device=device)
    if cloud_mask[0,1]:
        length_PO5 = bond_mask[0,1]
        coords[0,1] = torch.tensor([length_PO5, 0.0, 0.0], device=device)

    # Build the rest for residue 0
    for j in range(2, max_atoms):
        if cloud_mask[0,j]:
            refs = point_ref[:,0,j]
            a = coords[0, refs[0]]
            b = coords[0, refs[1]]
            c = coords[0, refs[2]]
            l_val = bond_mask[0,j]
            theta = angles_mask[0,0,j]
            chi   = angles_mask[1,0,j]
            coords[0,j] = mp_nerf_torch(a, b, c, l_val, theta, chi)

    # Build subsequent residues
    for i in range(1, L):
        for j in range(max_atoms):
            if not cloud_mask[i,j]:
                continue
            refs = point_ref[:,i,j]
            a = coords[i, refs[0]]
            b = coords[i, refs[1]]
            c = coords[i, refs[2]]
            l_val = bond_mask[i,j]
            theta = angles_mask[0,i,j]
            chi   = angles_mask[1,i,j]
            coords[i,j] = mp_nerf_torch(a, b, c, l_val, theta, chi)

    return coords
import torch
from .kb_rna import RNA_BUILD_INFO
from .massive_pnerf import mp_nerf_torch

def build_scaffolds_rna_from_torsions(
    seq: str,
    torsions: torch.Tensor,
    device: str = "cpu"
):
    """
    Convert predicted torsions (alpha..zeta, chi, and optionally ring angles) for an RNA sequence
    into the scaffold arrays used by mp_nerf for 3D building.

    Now we have 10 backbone atoms per residue:
      0->P, 1->O5', 2->C5', 3->C4', 4->O4', 5->C3', 6->O3', 7->C2', 8->O2', 9->C1'.

    We'll reference the previous residue's O3' (index 6) to place the new residue's P (index 0).
    """

    L = len(seq)
    max_atoms = 10  # expanded from 7 to 10

    bond_mask = torch.zeros((L, max_atoms), dtype=torch.float32, device=device)
    angles_mask = torch.zeros((2, L, max_atoms), dtype=torch.float32, device=device)
    point_ref_mask = torch.zeros((3, L, max_atoms), dtype=torch.long, device=device)
    cloud_mask = torch.ones((L, max_atoms), dtype=torch.bool, device=device)

    for i, base_type in enumerate(seq):
        # 1) Attempt to retrieve geometry dictionary
        if base_type in RNA_BUILD_INFO:
            info = RNA_BUILD_INFO[base_type]
        else:
            # fallback to "A"
            info = RNA_BUILD_INFO["A"]

        # 2) Fill bond_mask from "bond_lengths"
        # We'll create an index_map to help find numeric indices for each backbone atom
        index_map = {}
        for idx, atom_name in enumerate(info["backbone_atoms"]):
            index_map[atom_name] = idx

        for (a1, a2), val in info["bond_lengths"].items():
            if a1 in index_map and a2 in index_map:
                idx1 = index_map[a1]
                idx2 = index_map[a2]
                # We store the bond length at the "child" index (idx2)
                bond_mask[i, idx2] = val

        # 3) Fill angles_mask for bond angles (angles_mask[0])
        # Convert to radians
        for triple, ang_deg in info["bond_angles"].items():
            x1, x2, x3 = triple
            if x1 in index_map and x2 in index_map and x3 in index_map:
                i1 = index_map[x1]
                i2 = index_map[x2]
                i3 = index_map[x3]
                # We'll place it at the child index i3 for convenience
                angles_mask[0, i, i3] = ang_deg * (3.14159 / 180.0)

        # 4) Overwrite dihedrals from 'torsions' array in angles_mask[1]
        # Suppose torsions shape [L, 7] => alpha..zeta + chi
        if torsions.shape[1] >= 7:
            alpha_val = torsions[i, 0]
            beta_val  = torsions[i, 1]
            gamma_val = torsions[i, 2]
            delta_val = torsions[i, 3]
            epsilon_val = torsions[i, 4]
            zeta_val = torsions[i, 5]
            chi_val  = torsions[i, 6]

            # We assign dihedrals to these atom indices, as an example:
            # alpha -> index 1 (O5')
            # beta -> index 2 (C5')
            # gamma -> index 3 (C4')
            # delta -> index 4 (O4')
            # epsilon -> index 5 (C3')
            # zeta -> index 6 (O3')
            # chi -> index 9 (C1')
            angles_mask[1, i, 1] = alpha_val
            angles_mask[1, i, 2] = beta_val
            angles_mask[1, i, 3] = gamma_val
            angles_mask[1, i, 4] = delta_val
            angles_mask[1, i, 5] = epsilon_val
            angles_mask[1, i, 6] = zeta_val
            angles_mask[1, i, 9] = chi_val

        # 5) Bridging references for the P atom (index 0)
        if i == 0:
            # For the very first residue, reference is dummy for P
            point_ref_mask[0,i,0] = 0
            point_ref_mask[1,i,0] = 0
            point_ref_mask[2,i,0] = 0
        else:
            # Link residue i's P to residue (i-1)'s O3' (index 6)
            # We'll flatten the index as (i-1)*max_atoms + 6
            prev_O3_global = (i-1)*max_atoms + 6
            point_ref_mask[0,i,0] = prev_O3_global
            point_ref_mask[1,i,0] = prev_O3_global
            point_ref_mask[2,i,0] = prev_O3_global

        # 6) For the other backbone atoms, define references relative to the same residue
        # We'll do a simple scheme: each child's references are the previous one repeated
        # In practice, you might want more precise local frames referencing the prior 3 atoms
        for j in range(1, max_atoms):
            if j == 1 and i == 0:
                # O5' referencing P,P,P for residue 0
                point_ref_mask[0,i,1] = 0
                point_ref_mask[1,i,1] = 0
                point_ref_mask[2,i,1] = 0
            else:
                # For subsequent atoms, reference the previous local index
                ref_global = i*max_atoms + (j-1)
                point_ref_mask[0,i,j] = ref_global
                point_ref_mask[1,i,j] = ref_global
                point_ref_mask[2,i,j] = ref_global

    scaffolds = {
        "bond_mask": bond_mask,
        "angles_mask": angles_mask,
        "point_ref_mask": point_ref_mask,
        "cloud_mask": cloud_mask
    }
    return scaffolds


def rna_fold(scaffolds: dict, device: str = "cpu"):
    """
    Construct the 3D RNA backbone + sugar ring in a single pass using mp_nerf_torch.
    We have L residues, max_atoms=10 each. We'll place them carefully so that
    for residue i>0, the P references residue (i-1)'s O3'.
    """

    bond_mask     = scaffolds["bond_mask"]
    angles_mask   = scaffolds["angles_mask"]
    point_ref     = scaffolds["point_ref_mask"]
    cloud_mask    = scaffolds["cloud_mask"]

    L, max_atoms = bond_mask.shape
    coords = torch.zeros((L, max_atoms, 3), dtype=torch.float32, device=device)

    # We'll do a nested loop; for each residue & each atom, place them via mp_nerf
    for i in range(L):
        for j in range(max_atoms):
            if not cloud_mask[i,j]:
                continue

            # gather references
            refA = point_ref[0,i,j].item()
            refB = point_ref[1,i,j].item()
            refC = point_ref[2,i,j].item()

            # flatten to handle bridging indices
            a_coords = coords.view(-1,3)[refA] if refA < L*max_atoms else torch.zeros(3, device=device)
            b_coords = coords.view(-1,3)[refB] if refB < L*max_atoms else torch.zeros(3, device=device)
            c_coords = coords.view(-1,3)[refC] if refC < L*max_atoms else torch.zeros(3, device=device)

            l_val = bond_mask[i,j]
            theta = angles_mask[0,i,j]
            phi   = angles_mask[1,i,j]

            coords[i,j] = mp_nerf_torch(a_coords, b_coords, c_coords, l_val, theta, phi)

    return coords


def build_and_fold_rna(seq, torsions, device="cpu"):
    """
    Convenience function to do both steps in one call.
    """
    scaffolds = build_scaffolds_rna_from_torsions(seq, torsions, device=device)
    coords = rna_fold(scaffolds, device=device)
    return coords
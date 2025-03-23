import torch
import math
from .kb_rna import RNA_BUILD_INFO
from .massive_pnerf import mp_nerf_torch

def compute_max_rna_atoms():
    """
    Compute the maximum total atoms for a single RNA residue
    (backbone + base). We assume 'RNA_BUILD_INFO' has keys for
    'A','U','G','C' etc., each with 'backbone_atoms' and 'base_atoms'.
    """
    from .kb_rna import RNA_BUILD_INFO

    max_count = 0
    for base in ["A","U","G","C"]:
        info = RNA_BUILD_INFO.get(base, RNA_BUILD_INFO["A"])
        backbone_count = len(info["backbone_atoms"])
        base_count = len(info.get("base_atoms", []))
        count = backbone_count + base_count
        if count > max_count:
            max_count = count
    return max_count

def build_scaffolds_rna_from_torsions(seq: str,
                                      torsions: torch.Tensor,
                                      device: str = "cpu"):
    """
    Convert predicted torsions (alpha..zeta, chi) into scaffolding arrays:
    bond_mask, angles_mask, point_ref_mask, cloud_mask for each residue.
    Also sets bridging references from O3'(i-1) to P(i).
    """
    L = len(seq)
    max_atoms = 10  # backbone only, P..C1'

    bond_mask = torch.zeros((L, max_atoms), dtype=torch.float32, device=device)
    angles_mask = torch.zeros((2, L, max_atoms), dtype=torch.float32, device=device)
    point_ref_mask = torch.zeros((3, L, max_atoms), dtype=torch.long, device=device)
    cloud_mask = torch.ones((L, max_atoms), dtype=torch.bool, device=device)

    for i, base_type in enumerate(seq):
        info = RNA_BUILD_INFO.get(base_type, RNA_BUILD_INFO["A"])
        index_map = { name: idx for idx, name in enumerate(info["backbone_atoms"]) }

        # 1) Fill bond lengths in bond_mask
        for (a1, a2), val in info["bond_lengths"].items():
            if a1 in index_map and a2 in index_map:
                c_idx = index_map[a2]
                bond_mask[i, c_idx] = val

        # 2) Fill bond angles in angles_mask[0], converting deg->rad
        for (x1, x2, x3), ang_deg in info["bond_angles"].items():
            if x1 in index_map and x2 in index_map and x3 in index_map:
                c_idx = index_map[x3]
                angles_mask[0, i, c_idx] = ang_deg * (math.pi / 180.0)

        # 3) Fill dihedral angles from 'torsions' input
        # Expect shape [L, 7], each row is alpha..zeta, chi
        if torsions.size(1) >= 7:
            alpha, beta, gamma, delta, eps, zeta, chi = torsions[i]
            # Map to indices: O5'(1)->alpha, C5'(2)->beta, C4'(3)->gamma,
            # O4'(4)->delta, C3'(5)->epsilon, O3'(6)->zeta, C1'(9)->chi
            angles_mask[1, i, 1] = alpha
            angles_mask[1, i, 2] = beta
            angles_mask[1, i, 3] = gamma
            angles_mask[1, i, 4] = delta
            angles_mask[1, i, 5] = eps
            angles_mask[1, i, 6] = zeta
            angles_mask[1, i, 9] = chi

        # 4) Bridging reference for P(0)
        if i == 0:
            # For the very first residue, reference itself
            point_ref_mask[0, i, 0] = i*max_atoms
            point_ref_mask[1, i, 0] = i*max_atoms
            point_ref_mask[2, i, 0] = i*max_atoms
        else:
            # previous residue's O3' is index 6
            prev_o3_global = (i-1)*max_atoms + 6
            point_ref_mask[0, i, 0] = prev_o3_global
            point_ref_mask[1, i, 0] = prev_o3_global
            point_ref_mask[2, i, 0] = prev_o3_global

        # 5) For local references, each child references the previous local index
        for j in range(1, max_atoms):
            point_ref_mask[0, i, j] = i*max_atoms + (j-1)
            point_ref_mask[1, i, j] = i*max_atoms + (j-1)
            point_ref_mask[2, i, j] = i*max_atoms + (j-1)

    return {
        "bond_mask": bond_mask,
        "angles_mask": angles_mask,
        "point_ref_mask": point_ref_mask,
        "cloud_mask": cloud_mask
    }

def rna_fold(scaffolds: dict, device="cpu", do_ring_closure=False):
    """
    Build backbone coords from scaffolds using mp_nerf_torch. Optionally do ring closure.
    Returns shape [L, max_atoms, 3].
    """
    bond_mask = scaffolds["bond_mask"]
    angles_mask = scaffolds["angles_mask"]
    point_ref = scaffolds["point_ref_mask"]
    cloud_mask = scaffolds["cloud_mask"]

    L, max_atoms = bond_mask.shape
    coords = torch.zeros((L, max_atoms, 3), dtype=torch.float32, device=device)
    coords_flat = coords.view(-1, 3)
    total = L*max_atoms

    for i in range(L):
        for j in range(max_atoms):
            if not cloud_mask[i, j]:
                continue
            refA = point_ref[0, i, j].item()
            refB = point_ref[1, i, j].item()
            refC = point_ref[2, i, j].item()

            a_xyz = coords_flat[refA] if (0 <= refA < total) else torch.zeros(3, device=device)
            b_xyz = coords_flat[refB] if (0 <= refB < total) else torch.zeros(3, device=device)
            c_xyz = coords_flat[refC] if (0 <= refC < total) else torch.zeros(3, device=device)

            l_val = bond_mask[i, j]
            theta = angles_mask[0, i, j]
            phi   = angles_mask[1, i, j]

            coords[i, j] = mp_nerf_torch(a_xyz, b_xyz, c_xyz, l_val, theta, phi)

    if do_ring_closure:
        coords = ring_closure_refinement(coords)
    return coords

def ring_closure_refinement(coords: torch.Tensor) -> torch.Tensor:
    """
    Placeholder function to fix ring closure if flexible pucker is used.
    For now, return coords as-is, or add a small iterative correction.
    """
    return coords

def skip_missing_atoms(seq: str, scaffolds: dict) -> dict:
    """
    Potentially skip certain atoms if your pipeline knows they're missing.
    This function can manipulate scaffolds["cloud_mask"] accordingly.
    """
    # Example usage:
    # if some condition: scaffolds["cloud_mask"][i,8] = False
    return scaffolds

def handle_mods(seq: str, scaffolds: dict) -> dict:
    """
    If you detect modified bases (m5C, PSU, etc.), either update bond lengths/angles or skip them.
    """
    return scaffolds

def place_rna_bases(backbone_coords: torch.Tensor,
                    seq: str,
                    angles_mask: torch.Tensor,
                    device="cpu"):
    """
    For each residue i, we have the backbone coords shape [L, backbone_count, 3].
    We attach base atoms, which may vary (purines vs. pyrimidines).
    To avoid shape mismatch when concatenating, we zero-pad each residue 
    to [max_atoms_per_residue, 3]. Then stack them into [L, max_atoms_per_residue, 3].
    """
    from .kb_rna import RNA_BUILD_INFO
    L, backbone_count, _ = backbone_coords.shape

    # 1) compute max
    max_atoms = compute_max_rna_atoms()

    new_coords_list = []
    for i in range(L):
        base_type = seq[i]
        info = RNA_BUILD_INFO.get(base_type, RNA_BUILD_INFO["A"])
        base_atoms = info.get("base_atoms", [])

        # backbone slice => shape [backbone_count, 3]
        bcoords = backbone_coords[i]

        # 2) Optionally build base coords
        # Right now, we just create a dummy zero for demonstration
        # or you can do an actual mini-NeRF approach using angles_mask if desired.
        base_coords = torch.zeros((len(base_atoms), 3), dtype=torch.float32, device=device)
        # e.g. placeholder; in production, place the base with a mini nerf or so.

        combined = torch.cat([bcoords, base_coords], dim=0)  # shape [(Bk+Bs), 3]
        # 3) zero-pad to max_atoms
        padded = torch.zeros((max_atoms, 3), dtype=torch.float32, device=device)
        count = combined.size(0)
        padded[:count] = combined

        new_coords_list.append(padded.unsqueeze(0))  # shape [1, max_atoms, 3]

    # 4) cat along residues dimension => [L, max_atoms, 3]
    final_coords = torch.cat(new_coords_list, dim=0)
    return final_coords

def validate_rna_geometry(coords: torch.Tensor):
    """
    Optionally measure bond lengths/angles and compare with references,
    raising warnings for major deviations.
    """
    pass

def mini_refinement(coords: torch.Tensor, method="none"):
    """
    Stub for local MD or gradient-based refinement if needed.
    """
    return coords
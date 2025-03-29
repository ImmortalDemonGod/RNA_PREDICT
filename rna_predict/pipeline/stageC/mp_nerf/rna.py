import math

import torch

# Replace older references to kb_rna with final_kb_rna via absolute import
from rna_predict.pipeline.stageC.mp_nerf.final_kb_rna import (
    get_bond_angle,
    get_bond_length,
)
from rna_predict.pipeline.stageC.mp_nerf.massive_pnerf import mp_nerf_torch
import snoop

###############################################################################
# 1) DEFINE A CANONICAL BACKBONE ORDER
###############################################################################
# We standardize the 10 backbone atoms for typical RNA (P..C1'):
BACKBONE_ATOMS = [
    "P",  # 0
    "O5'",  # 1
    "C5'",  # 2
    "C4'",  # 3
    "O4'",  # 4
    "C3'",  # 5
    "O3'",  # 6
    "C2'",  # 7
    "O2'",  # 8
    "C1'",  # 9
]
BACKBONE_INDEX_MAP = {name: i for i, name in enumerate(BACKBONE_ATOMS)}


###############################################################################
# 2) BASE-ATOM HELPER
###############################################################################
def get_base_atoms(base_type: str) -> list:
    """
    Return a canonical list of base-atom names for the given residue (A, G, C, U).
    If in the future we have expansions (m5C, PSU, etc.), we could adapt here.
    """
    base_map = {
        "A": ["N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
        "G": ["N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
        "C": ["N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
        "U": ["N1", "C2", "N3", "C4", "C5", "C6", "O2", "O4"],
    }
    return base_map.get(base_type, [])


###############################################################################
# 3) COMPUTE MAX ATOMS (BACKBONE + BASE)
###############################################################################
def compute_max_rna_atoms():
    """
    For zero-padding or a single [L, max_atoms, 3] shape, we find the largest
    residue's total (10 backbone atoms + base).
    """
    # typical standard bases
    base_types = ["A", "G", "C", "U"]
    max_count = 0
    for btype in base_types:
        base_list = get_base_atoms(btype)
        count = len(BACKBONE_ATOMS) + len(base_list)
        if count > max_count:
            max_count = count
    return max_count


###############################################################################
# 4) PRIMARY FUNCTION: build_scaffolds_rna_from_torsions
###############################################################################

def build_scaffolds_rna_from_torsions(
    seq: str,
    torsions: torch.Tensor,
    device: str = "cpu",
    sugar_pucker: str = "C3'-endo",
) -> dict:
    """
    Convert predicted backbone torsions (alpha..zeta, chi) into scaffolding arrays
    for an MP-NeRF approach. We rely on final_kb_rna.py for bond lengths & angles.

    Args:
      seq: A string of nucleotides (e.g. "ACGU") of length L
      torsions: shape [L, 7] => alpha..zeta, chi (in DEGREES)
      sugar_pucker: Typically "C3'-endo" for A-form, or "C2'-endo" for B-like
      device: "cpu" or "cuda"

    Returns a dictionary with:
      - "bond_mask": Tensor of shape [L, B=10], each entry is the bond length
      - "angles_mask": Tensor of shape [2, L, B] => row=0 are bond angles in radians,
                                              row=1 are dihedral angles in radians
      - "point_ref_mask": Tensor of shape [3, L, B], for the mp_nerf references
      - "cloud_mask": (L, B) bool, all True here, but you can toggle if you skip some atoms
    """
    L = len(seq)
    B = len(BACKBONE_ATOMS)

    bond_mask = torch.zeros((L, B), dtype=torch.float32, device=device)
    angles_mask = torch.zeros((2, L, B), dtype=torch.float32, device=device)
    point_ref_mask = torch.zeros((3, L, B), dtype=torch.long, device=device)
    cloud_mask = torch.ones((L, B), dtype=torch.bool, device=device)

    # We'll define a small table for backbone bonds & angles. We rely on final_kb_rna.py to fetch numeric values.
    # For example, P->O5', O5'->C5', etc.
    backbone_bonds = [
        ("P", "O5'"),
        ("O5'", "C5'"),
        ("C5'", "C4'"),
        ("C4'", "O4'"),
        ("C4'", "C3'"),
        ("C3'", "O3'"),
        ("C3'", "C2'"),
        ("C2'", "O2'"),
        ("O4'", "C1'"),
    ]
    # For angles, we define the triplets in string form, e.g. "P-O5'-C5'" => we fetch from final_kb_rna
    backbone_triplets = [
        ("P", "O5'", "C5'"),
        ("O5'", "C5'", "C4'"),
        ("C5'", "C4'", "O4'"),
        ("C5'", "C4'", "C3'"),
        ("C4'", "C3'", "O3'"),
        ("C4'", "C3'", "C2'"),
        ("C3'", "C2'", "O2'"),
        ("C1'", "C2'", "C3'"),
        ("C1'", "C2'", "O2'"),
        ("O3'", "P", "O5'"),  # bridging angle
    ]

    def deg2rad(x):
        return x * (math.pi / 180.0)

    for i, base_nt in enumerate(seq):
        # 1) Fill bond lengths by calling get_bond_length("C4'-C3'", sugar_pucker=...).
        for atomA, atomB in backbone_bonds:
            # convert to index for the second atom
            idxB = BACKBONE_INDEX_MAP[atomB]
            # build a "A-B" string
            pair_str = f"{atomA}-{atomB}"
            length_val = get_bond_length(pair_str, sugar_pucker=sugar_pucker)
            if length_val is not None:
                bond_mask[i, idxB] = length_val

        # 2) Fill bond angles => angles_mask[0, i, indexOfAtom]
        for a1, a2, a3 in backbone_triplets:
            idx3 = BACKBONE_INDEX_MAP[a3]
            angle_deg = get_bond_angle(
                f"{a1}-{a2}-{a3}", sugar_pucker=sugar_pucker, degrees=True
            )
            if angle_deg is not None:
                angles_mask[0, i, idx3] = deg2rad(angle_deg)

        # 3) Fill dihedral angles from predicted (alpha..zeta, chi) in degrees => convert to rad
        if torsions.size(1) >= 7:
            alpha_deg, beta_deg, gamma_deg, delta_deg, eps_deg, zeta_deg, chi_deg = (
                torsions[i]
            )
            alpha_rad = deg2rad(alpha_deg)
            beta_rad = deg2rad(beta_deg)
            gamma_rad = deg2rad(gamma_deg)
            delta_rad = deg2rad(delta_deg)
            eps_rad = deg2rad(eps_deg)
            zeta_rad = deg2rad(zeta_deg)
            chi_rad = deg2rad(chi_deg)
            # Map them to backbone indexes. We'll do alpha->1, beta->2, gamma->3, delta->4, eps->5, zeta->6, chi->9
            angles_mask[1, i, 1] = alpha_rad
            angles_mask[1, i, 2] = beta_rad
            angles_mask[1, i, 3] = gamma_rad
            angles_mask[1, i, 4] = delta_rad
            angles_mask[1, i, 5] = eps_rad
            angles_mask[1, i, 6] = zeta_rad
            angles_mask[1, i, 9] = chi_rad

        # 4) bridging references in point_ref_mask. For j=0 => 'P'.
        for j in range(B):
            if j == 0:
                # For the first residue i=0, reference itself. For subsequent residues, reference previous O3'
                if i == 0:
                    point_ref_mask[:, i, j] = i * B
                else:
                    prev_o3_global = (i - 1) * B + BACKBONE_INDEX_MAP["O3'"]
                    point_ref_mask[0, i, j] = prev_o3_global
                    point_ref_mask[1, i, j] = prev_o3_global
                    point_ref_mask[2, i, j] = prev_o3_global
            else:
                # local references from j-1
                point_ref_mask[0, i, j] = i * B + (j - 1)
                point_ref_mask[1, i, j] = i * B + (j - 1)
                point_ref_mask[2, i, j] = i * B + (j - 1)

    return {
        "bond_mask": bond_mask,
        "angles_mask": angles_mask,
        "point_ref_mask": point_ref_mask,
        "cloud_mask": cloud_mask,
    }


###############################################################################
# 5) FOLDING: rna_fold
###############################################################################
def rna_fold(
    scaffolds: dict, device: str = "cpu", do_ring_closure: bool = False
) -> torch.Tensor:
    """
    Convert the scaffolds into 3D backbone coordinates using an mp_nerf approach.
    If do_ring_closure=True, optionally do a ring_closure_refinement.

    Returns shape [L, B, 3], where B=10 for the backbone.
    """
    bond_mask = scaffolds["bond_mask"]
    angles_mask = scaffolds["angles_mask"]
    point_ref = scaffolds["point_ref_mask"]
    cloud_mask = scaffolds["cloud_mask"]

    L, B = bond_mask.shape
    coords = torch.zeros((L, B, 3), dtype=torch.float32, device=device)
    coords_flat = coords.view(-1, 3)
    total = L * B

    for i in range(L):
        for j in range(B):
            if not cloud_mask[i, j]:
                continue
            refA = point_ref[0, i, j].item()
            refB = point_ref[1, i, j].item()
            refC = point_ref[2, i, j].item()

            a_xyz = (
                coords_flat[refA]
                if 0 <= refA < total
                else torch.zeros(3, device=device)
            )
            b_xyz = (
                coords_flat[refB]
                if 0 <= refB < total
                else torch.zeros(3, device=device)
            )
            c_xyz = (
                coords_flat[refC]
                if 0 <= refC < total
                else torch.zeros(3, device=device)
            )

            l_val = bond_mask[i, j]
            theta = angles_mask[0, i, j]
            phi = angles_mask[1, i, j]

            coords[i, j] = mp_nerf_torch(a_xyz, b_xyz, c_xyz, l_val, theta, phi)

    if do_ring_closure:
        coords = ring_closure_refinement(coords)
    return coords


def ring_closure_refinement(coords: torch.Tensor) -> torch.Tensor:
    """
    Placeholder. We could do a small iterative approach to ensure the
    ribose ring closes properly for the sugar pucker.
    Currently returns coords as-is.
    """
    return coords


###############################################################################
# 6) place_rna_bases
###############################################################################
def place_rna_bases(
    backbone_coords: torch.Tensor,
    seq: str,
    angles_mask: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Attach base atoms for each residue. We do a zero-padded approach to
    produce [L, max_atoms, 3]. For a real pipeline, you'd do a mini mp_nerf
    for the base ring referencing get_base_geometry() from final_kb_rna.
    """
    L, B, _ = backbone_coords.shape
    max_atoms = compute_max_rna_atoms()  # e.g., up to 21 for G

    final_coords = torch.zeros((L, max_atoms, 3), dtype=torch.float32, device=device)

    for i, base_nt in enumerate(seq):
        # copy backbone
        final_coords[i, :B] = backbone_coords[i]
        # get base atoms
        base_list = get_base_atoms(base_nt)
        base_count = len(base_list)
        # For now, we place them as zeros. Expand later with miniNeRF if needed.
        # e.g. final_coords[i, B : B+base_count] = some build

    return final_coords


###############################################################################
# 7) OPTIONAL SKIP & HANDLE REFS
###############################################################################
def skip_missing_atoms(seq: str, scaffolds: dict) -> dict:
    """
    If some RNA modifications or partial data is missing, we can set cloud_mask to False
    for certain atoms. For now, we do nothing.
    """
    return scaffolds


def handle_mods(seq: str, scaffolds: dict) -> dict:
    """
    If we detect special modifications or pseudouridines, we might override bond angles, etc.
    Currently a placeholder that returns scaffolds unmodified.
    """
    return scaffolds


###############################################################################
# 7.a) OPTIONAL VALIDATION & REFINEMENT
###############################################################################
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


###############################################################################
# 8) DEMO
###############################################################################
if __name__ == "__main__":
    # Example usage
    sample_seq = "ACGU"
    L = len(sample_seq)
    # alpha..zeta, chi in degrees => shape [L,7]
    # e.g. alpha=300, etc. We'll just do zeros for demonstration
    dummy_torsions = torch.zeros((L, 7))

    # build scaffolds, pass sugar_pucker="C3'-endo" or "C2'-endo"
    scaff = build_scaffolds_rna_from_torsions(
        sample_seq, dummy_torsions, sugar_pucker="C3'-endo"
    )

    # fold
    coords_bb = rna_fold(scaff, do_ring_closure=False)

    # place bases
    coords_full = place_rna_bases(coords_bb, sample_seq, scaff["angles_mask"])
    print("Backbone coords shape:", coords_bb.shape)
    print("Full coords shape (with base placeholders):", coords_full.shape)

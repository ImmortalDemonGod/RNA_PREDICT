
You are given one or more automatically generated Python test files that test various classes and functions. These tests may have issues such as poor naming conventions, inconsistent usage of self, lack of setUp methods, minimal docstrings, redundant or duplicate tests, and limited assertion coverage. They may also fail to leverage hypothesis and unittest.mock effectively, and might not be logically grouped.

Your task is to produce a single, consolidated, high-quality test file from the given input files. The refactored test file should incorporate the following improvements:
	1.	Consolidation and Organization
	•	Combine all tests from the provided files into one coherent Python test file.
	•	Group tests into classes that correspond logically to the functionality they are testing (e.g., separate test classes by the class or function under test).
	•	Within each class, order test methods logically (e.g., basic functionality first, edge cases, error handling, round-trip tests afterward).
	2.	Clean, Readable Code
	•	Use descriptive, PEP 8-compliant class and method names.
	•	Add docstrings to each test class and test method, explaining their purpose and what they verify.
	•	Remove redundant, duplicate, or meaningless tests. Combine or refactor tests that cover the same functionality into a single, comprehensive test method when appropriate.
	3.	Proper Test Fixtures
	•	Utilize setUp methods to instantiate commonly used objects before each test method, reducing redundancy.
	•	Ensure that instance methods of classes under test are called on properly instantiated objects rather than passing self incorrectly as an argument.
	4.	Robust Assertions and Coverage
	•	Include multiple assertions in each test to thoroughly verify behavior and correctness.
	•	Use unittest’s assertRaises for expected exceptions to validate error handling.
	•	Implement at least one round-trip test (e.g., encode then decode a data structure, or transform an object multiple times to ensure idempotency).
	5.	Effective Use of Hypothesis
	•	Employ hypothesis to generate a wide range of input data, ensuring better coverage and exposing edge cases.
	•	Use strategies like st.builds to create complex objects (e.g., custom dataclasses) with varied attribute values.
	•	Enforce constraints (e.g., allow_nan=False) to avoid nonsensical test inputs.
	6.	Mocking External Dependencies
	•	Use unittest.mock where appropriate to simulate external dependencies or environments, ensuring tests are reliable and isolated from external conditions.

⸻

Additional Context: Getting Started with Hypothesis

Below is a practical guide that outlines common use cases and best practices for leveraging hypothesis:
	1.	Basic Usage
	•	Decorate test functions with @given and specify a strategy (e.g., @given(st.text())).
	•	Let hypothesis generate diverse test cases automatically.
	2.	Common Strategies
	•	Use built-in strategies like st.integers(), st.floats(), st.text(), etc.
	•	Combine strategies with st.lists, st.builds, or st.composite to generate complex objects.
	3.	Composing Tests
	•	Employ assume() to filter out unwanted test cases.
	•	Compose or build custom objects to test domain-specific logic.
	4.	Advanced Features
	•	Fine-tune test runs with @settings (e.g., max_examples=1000).
	•	Create reusable strategies via @composite.
	5.	Best Practices
	•	Keep tests focused on one property at a time.
	•	Use explicit examples with @example() for edge cases.
	•	Manage performance by choosing realistic strategy bounds.
	6.	Debugging Failed Tests
	•	Hypothesis shows minimal failing examples and seeds to help reproduce and fix issues.

⸻

Input Format

TEST CODE: 
# ----- test_modify_angles_mask_with_torsions_basic.py -----
import proteins
import unittest
from hypothesis import given, strategies as st

class TestFuzzModify_Angles_Mask_With_Torsions(unittest.TestCase):

    @given(seq=st.nothing(), angles_mask=st.nothing(), torsions=st.nothing())
    def test_fuzz_modify_angles_mask_with_torsions(self, seq, angles_mask, torsions) -> None:
        proteins.modify_angles_mask_with_torsions(seq=seq, angles_mask=angles_mask, torsions=torsions)

# ----- test_protein_fold_basic.py -----
import proteins
import unittest
from hypothesis import given, strategies as st

class TestFuzzProtein_Fold(unittest.TestCase):

    @given(cloud_mask=st.nothing(), point_ref_mask=st.nothing(), angles_mask=st.nothing(), bond_mask=st.nothing(), device=st.just(device(type='cpu')), hybrid=st.booleans())
    def test_fuzz_protein_fold(self, cloud_mask, point_ref_mask, angles_mask, bond_mask, device, hybrid) -> None:
        proteins.protein_fold(cloud_mask=cloud_mask, point_ref_mask=point_ref_mask, angles_mask=angles_mask, bond_mask=bond_mask, device=device, hybrid=hybrid)

# ----- test_scn_angle_mask_basic.py -----
import proteins
import unittest
from hypothesis import given, strategies as st

class TestFuzzScn_Angle_Mask(unittest.TestCase):

    @given(seq=st.nothing(), angles=st.none(), device=st.none())
    def test_fuzz_scn_angle_mask(self, seq, angles, device) -> None:
        proteins.scn_angle_mask(seq=seq, angles=angles, device=device)

# ----- test_sidechain_fold_basic.py -----
import proteins
import unittest
from hypothesis import given, strategies as st

class TestFuzzSidechain_Fold(unittest.TestCase):

    @given(wrapper=st.nothing(), cloud_mask=st.nothing(), point_ref_mask=st.nothing(), angles_mask=st.nothing(), bond_mask=st.nothing(), device=st.just(device(type='cpu')), c_beta=st.booleans())
    def test_fuzz_sidechain_fold(self, wrapper, cloud_mask, point_ref_mask, angles_mask, bond_mask, device, c_beta) -> None:
        proteins.sidechain_fold(wrapper=wrapper, cloud_mask=cloud_mask, point_ref_mask=point_ref_mask, angles_mask=angles_mask, bond_mask=bond_mask, device=device, c_beta=c_beta)

# ----- test_build_scaffolds_from_scn_angles_basic.py -----
import proteins
import unittest
from hypothesis import given, strategies as st

class TestFuzzBuild_Scaffolds_From_Scn_Angles(unittest.TestCase):

    @given(seq=st.nothing(), angles=st.none(), coords=st.none(), device=st.just('auto'))
    def test_fuzz_build_scaffolds_from_scn_angles(self, seq, angles, coords, device) -> None:
        proteins.build_scaffolds_from_scn_angles(seq=seq, angles=angles, coords=coords, device=device)

# ----- test_modify_scaffolds_with_coords_basic.py -----
import proteins
import unittest
from hypothesis import given, strategies as st

class TestFuzzModify_Scaffolds_With_Coords(unittest.TestCase):

    @given(scaffolds=st.nothing(), coords=st.nothing())
    def test_fuzz_modify_scaffolds_with_coords(self, scaffolds, coords) -> None:
        proteins.modify_scaffolds_with_coords(scaffolds=scaffolds, coords=coords)

# ----- test_scn_bond_mask_basic.py -----
import proteins
import unittest
from hypothesis import given, strategies as st

class TestFuzzScn_Bond_Mask(unittest.TestCase):

    @given(seq=st.nothing())
    def test_fuzz_scn_bond_mask(self, seq) -> None:
        proteins.scn_bond_mask(seq=seq)

# ----- test_scn_index_mask_basic.py -----
import proteins
import unittest
from hypothesis import given, strategies as st

class TestFuzzScn_Index_Mask(unittest.TestCase):

    @given(seq=st.nothing())
    def test_fuzz_scn_index_mask(self, seq) -> None:
        proteins.scn_index_mask(seq=seq)

# ----- test_scn_cloud_mask_basic.py -----
import proteins
import unittest
from hypothesis import given, strategies as st

class TestFuzzScn_Cloud_Mask(unittest.TestCase):

    @given(seq=st.nothing(), coords=st.none(), strict=st.booleans())
    def test_fuzz_scn_cloud_mask(self, seq, coords, strict) -> None:
        proteins.scn_cloud_mask(seq=seq, coords=coords, strict=strict)

# ----- test_scn_rigid_index_mask_basic.py -----
import proteins
import unittest
from hypothesis import given, strategies as st

class TestFuzzScn_Rigid_Index_Mask(unittest.TestCase):

    @given(seq=st.nothing(), c_alpha=st.none())
    def test_fuzz_scn_rigid_index_mask(self, seq, c_alpha) -> None:
        proteins.scn_rigid_index_mask(seq=seq, c_alpha=c_alpha)

FULL SRC CODE: # science
import numpy as np

# diff / ml
import torch
from einops import repeat

from rna_predict.pipeline.stageC.mp_nerf.kb_proteins import *

# module
from rna_predict.pipeline.stageC.mp_nerf.massive_pnerf import *
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
            (rearrange(coords, "b (l c) d -> b l c d", c=14) != 0).sum(dim=-1) != 0
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
            # check if any is nan -> fill the holes
            number = to_fill[i].long().sum()
            torsion_mask[i, to_fill[i]] = angles[i, 6 : 6 + number]

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

    torsion_mask[-1, 3] += np.pi
    return torch.stack([theta_mask, torsion_mask], dim=0)


def scn_index_mask(seq):
    """Inputs:
    * seq: (length). iterable of 1-letter aa codes of a protein
    Outputs: (L, 11, 3) maps point to theta and dihedral.
             first angle is theta, second is dihedral
    """
    idxs = torch.tensor([SUPREME_INFO[aa]["idx_mask"] for aa in seq])
    return rearrange(idxs, "l s d -> d l s")


def scn_rigid_index_mask(seq, c_alpha=None):
    """Inputs:
    * seq: (length). iterable of 1-letter aa codes of a protein
    * c_alpha: bool. whether to return only the c_alpha rigid group
    Outputs: (3, Length * Groups). indexes for 1st, 2nd and 3rd point
              to construct frames for each group.
    """
    if c_alpha:
        return torch.cat(
            [
                torch.tensor(SUPREME_INFO[aa]["rigid_idx_mask"])[:1] + 14 * i
                for i, aa in enumerate(seq)
            ],
            dim=0,
        ).t()
    return torch.cat(
        [
            torch.tensor(SUPREME_INFO[aa]["rigid_idx_mask"]) + 14 * i
            for i, aa in enumerate(seq)
        ],
        dim=0,
    ).t()


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
        device = angles.device if angles is not None else device

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
    init_a = repeat(
        torch.tensor([1.0, 0.0, 0.0], device=device, dtype=precise),
        "d -> l d",
        l=length,
    )
    init_b = repeat(
        torch.tensor([1.0, 1.0, 0.0], device=device, dtype=precise),
        "d -> l d",
        l=length,
    )
    # do N -> CA. don't do 1st since its done already
    thetas, dihedrals = angles_mask[:, :, 1]
    coords[1:, 1] = mp_nerf_torch(
        init_a, init_b, coords[:, 0], bond_mask[:, 1], thetas, dihedrals
    )[1:]
    # do CA -> C. don't do 1st since its done already
    thetas, dihedrals = angles_mask[:, :, 2]
    coords[1:, 2] = mp_nerf_torch(
        init_b, coords[:, 0], coords[:, 1], bond_mask[:, 2], thetas, dihedrals
    )[1:]
    # do C -> N
    thetas, dihedrals = angles_mask[:, :, 0]
    coords[:, 3] = mp_nerf_torch(
        coords[:, 0], coords[:, 1], coords[:, 2], bond_mask[:, 0], thetas, dihedrals
    )

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

        coords[level_mask, i] = mp_nerf_torch(
            coords_a,
            coords[level_mask, idx_b],
            coords[level_mask, idx_c],
            bond_mask[level_mask, i],
            thetas,
            dihedrals,
        )

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
    precise = wrapper.dtype

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

        wrapper[level_mask, i] = mp_nerf_torch(
            coords_a,
            wrapper[level_mask, idx_b],
            wrapper[level_mask, idx_c],
            bond_mask[level_mask, i],
            thetas,
            dihedrals,
        )

    return wrapper, cloud_mask


Where:
	•	
# ----- test_modify_angles_mask_with_torsions_basic.py -----
import proteins
import unittest
from hypothesis import given, strategies as st

class TestFuzzModify_Angles_Mask_With_Torsions(unittest.TestCase):

    @given(seq=st.nothing(), angles_mask=st.nothing(), torsions=st.nothing())
    def test_fuzz_modify_angles_mask_with_torsions(self, seq, angles_mask, torsions) -> None:
        proteins.modify_angles_mask_with_torsions(seq=seq, angles_mask=angles_mask, torsions=torsions)

# ----- test_protein_fold_basic.py -----
import proteins
import unittest
from hypothesis import given, strategies as st

class TestFuzzProtein_Fold(unittest.TestCase):

    @given(cloud_mask=st.nothing(), point_ref_mask=st.nothing(), angles_mask=st.nothing(), bond_mask=st.nothing(), device=st.just(device(type='cpu')), hybrid=st.booleans())
    def test_fuzz_protein_fold(self, cloud_mask, point_ref_mask, angles_mask, bond_mask, device, hybrid) -> None:
        proteins.protein_fold(cloud_mask=cloud_mask, point_ref_mask=point_ref_mask, angles_mask=angles_mask, bond_mask=bond_mask, device=device, hybrid=hybrid)

# ----- test_scn_angle_mask_basic.py -----
import proteins
import unittest
from hypothesis import given, strategies as st

class TestFuzzScn_Angle_Mask(unittest.TestCase):

    @given(seq=st.nothing(), angles=st.none(), device=st.none())
    def test_fuzz_scn_angle_mask(self, seq, angles, device) -> None:
        proteins.scn_angle_mask(seq=seq, angles=angles, device=device)

# ----- test_sidechain_fold_basic.py -----
import proteins
import unittest
from hypothesis import given, strategies as st

class TestFuzzSidechain_Fold(unittest.TestCase):

    @given(wrapper=st.nothing(), cloud_mask=st.nothing(), point_ref_mask=st.nothing(), angles_mask=st.nothing(), bond_mask=st.nothing(), device=st.just(device(type='cpu')), c_beta=st.booleans())
    def test_fuzz_sidechain_fold(self, wrapper, cloud_mask, point_ref_mask, angles_mask, bond_mask, device, c_beta) -> None:
        proteins.sidechain_fold(wrapper=wrapper, cloud_mask=cloud_mask, point_ref_mask=point_ref_mask, angles_mask=angles_mask, bond_mask=bond_mask, device=device, c_beta=c_beta)

# ----- test_build_scaffolds_from_scn_angles_basic.py -----
import proteins
import unittest
from hypothesis import given, strategies as st

class TestFuzzBuild_Scaffolds_From_Scn_Angles(unittest.TestCase):

    @given(seq=st.nothing(), angles=st.none(), coords=st.none(), device=st.just('auto'))
    def test_fuzz_build_scaffolds_from_scn_angles(self, seq, angles, coords, device) -> None:
        proteins.build_scaffolds_from_scn_angles(seq=seq, angles=angles, coords=coords, device=device)

# ----- test_modify_scaffolds_with_coords_basic.py -----
import proteins
import unittest
from hypothesis import given, strategies as st

class TestFuzzModify_Scaffolds_With_Coords(unittest.TestCase):

    @given(scaffolds=st.nothing(), coords=st.nothing())
    def test_fuzz_modify_scaffolds_with_coords(self, scaffolds, coords) -> None:
        proteins.modify_scaffolds_with_coords(scaffolds=scaffolds, coords=coords)

# ----- test_scn_bond_mask_basic.py -----
import proteins
import unittest
from hypothesis import given, strategies as st

class TestFuzzScn_Bond_Mask(unittest.TestCase):

    @given(seq=st.nothing())
    def test_fuzz_scn_bond_mask(self, seq) -> None:
        proteins.scn_bond_mask(seq=seq)

# ----- test_scn_index_mask_basic.py -----
import proteins
import unittest
from hypothesis import given, strategies as st

class TestFuzzScn_Index_Mask(unittest.TestCase):

    @given(seq=st.nothing())
    def test_fuzz_scn_index_mask(self, seq) -> None:
        proteins.scn_index_mask(seq=seq)

# ----- test_scn_cloud_mask_basic.py -----
import proteins
import unittest
from hypothesis import given, strategies as st

class TestFuzzScn_Cloud_Mask(unittest.TestCase):

    @given(seq=st.nothing(), coords=st.none(), strict=st.booleans())
    def test_fuzz_scn_cloud_mask(self, seq, coords, strict) -> None:
        proteins.scn_cloud_mask(seq=seq, coords=coords, strict=strict)

# ----- test_scn_rigid_index_mask_basic.py -----
import proteins
import unittest
from hypothesis import given, strategies as st

class TestFuzzScn_Rigid_Index_Mask(unittest.TestCase):

    @given(seq=st.nothing(), c_alpha=st.none())
    def test_fuzz_scn_rigid_index_mask(self, seq, c_alpha) -> None:
        proteins.scn_rigid_index_mask(seq=seq, c_alpha=c_alpha)
 is the content of your automatically generated Python test files (potentially multiple files’ content combined or listed).
	•	# science
import numpy as np

# diff / ml
import torch
from einops import repeat

from rna_predict.pipeline.stageC.mp_nerf.kb_proteins import *

# module
from rna_predict.pipeline.stageC.mp_nerf.massive_pnerf import *
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
            (rearrange(coords, "b (l c) d -> b l c d", c=14) != 0).sum(dim=-1) != 0
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
            # check if any is nan -> fill the holes
            number = to_fill[i].long().sum()
            torsion_mask[i, to_fill[i]] = angles[i, 6 : 6 + number]

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

    torsion_mask[-1, 3] += np.pi
    return torch.stack([theta_mask, torsion_mask], dim=0)


def scn_index_mask(seq):
    """Inputs:
    * seq: (length). iterable of 1-letter aa codes of a protein
    Outputs: (L, 11, 3) maps point to theta and dihedral.
             first angle is theta, second is dihedral
    """
    idxs = torch.tensor([SUPREME_INFO[aa]["idx_mask"] for aa in seq])
    return rearrange(idxs, "l s d -> d l s")


def scn_rigid_index_mask(seq, c_alpha=None):
    """Inputs:
    * seq: (length). iterable of 1-letter aa codes of a protein
    * c_alpha: bool. whether to return only the c_alpha rigid group
    Outputs: (3, Length * Groups). indexes for 1st, 2nd and 3rd point
              to construct frames for each group.
    """
    if c_alpha:
        return torch.cat(
            [
                torch.tensor(SUPREME_INFO[aa]["rigid_idx_mask"])[:1] + 14 * i
                for i, aa in enumerate(seq)
            ],
            dim=0,
        ).t()
    return torch.cat(
        [
            torch.tensor(SUPREME_INFO[aa]["rigid_idx_mask"]) + 14 * i
            for i, aa in enumerate(seq)
        ],
        dim=0,
    ).t()


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
        device = angles.device if angles is not None else device

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
    init_a = repeat(
        torch.tensor([1.0, 0.0, 0.0], device=device, dtype=precise),
        "d -> l d",
        l=length,
    )
    init_b = repeat(
        torch.tensor([1.0, 1.0, 0.0], device=device, dtype=precise),
        "d -> l d",
        l=length,
    )
    # do N -> CA. don't do 1st since its done already
    thetas, dihedrals = angles_mask[:, :, 1]
    coords[1:, 1] = mp_nerf_torch(
        init_a, init_b, coords[:, 0], bond_mask[:, 1], thetas, dihedrals
    )[1:]
    # do CA -> C. don't do 1st since its done already
    thetas, dihedrals = angles_mask[:, :, 2]
    coords[1:, 2] = mp_nerf_torch(
        init_b, coords[:, 0], coords[:, 1], bond_mask[:, 2], thetas, dihedrals
    )[1:]
    # do C -> N
    thetas, dihedrals = angles_mask[:, :, 0]
    coords[:, 3] = mp_nerf_torch(
        coords[:, 0], coords[:, 1], coords[:, 2], bond_mask[:, 0], thetas, dihedrals
    )

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

        coords[level_mask, i] = mp_nerf_torch(
            coords_a,
            coords[level_mask, idx_b],
            coords[level_mask, idx_c],
            bond_mask[level_mask, i],
            thetas,
            dihedrals,
        )

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
    precise = wrapper.dtype

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

        wrapper[level_mask, i] = mp_nerf_torch(
            coords_a,
            wrapper[level_mask, idx_b],
            wrapper[level_mask, idx_c],
            bond_mask[level_mask, i],
            thetas,
            dihedrals,
        )

    return wrapper, cloud_mask
 is the content of the source code under test (if needed for context).

Output Format

Provide a single Python code block containing the fully refactored, consolidated test file. The output should be ready-to-run with:

python -m unittest

It must exhibit all of the improvements listed above, including:
	•	Logical grouping of tests,
	•	Clear and correct usage of setUp,
	•	Docstrings for test classes and methods,
	•	Consolidated and refactored tests (no duplicates),
	•	Robust assertions and coverage,
	•	Use of hypothesis with one or more examples,
	•	Use of mock where appropriate.

⸻
============
EXTRA USEFUL CONTEXT TO AID YOU IN YOUR TASK:
Hypothesis: A Comprehensive Best-Practice and Reference Guide

Hypothesis is a powerful property-based testing library for Python, designed to help you find subtle bugs by generating large numbers of test inputs and minimizing failing examples. This document combines the strengths and core ideas of three earlier guides. It serves as a broad, in-depth resource: covering Hypothesis usage from the basics to advanced methods, including background on its internal mechanisms (Conjecture) and integration with complex workflows.

⸻

Table of Contents
	1.	Introduction to Property-Based Testing
1.1 What Is Property-Based Testing?
1.2 Why Use Property-Based Testing?
1.3 Installing Hypothesis
	2.	First Steps with Hypothesis
2.1 A Simple Example
2.2 Basic Workflows and Key Concepts
2.3 Troubleshooting the First Failures
	3.	Core Hypothesis Concepts
3.1 The @given Decorator
3.2 Strategies: Building and Composing Data Generators
3.3 Shrinking and Minimizing Failing Examples
3.4 Example Database and Replay
	4.	Advanced Data Generation
4.1 Understanding Strategies vs. Types
4.2 Composing Strategies (map, filter, flatmap)
4.3 Working with Complex or Recursive Data
4.4 Using @composite Functions
4.5 Integration and Edge Cases
	5.	Practical Usage Patterns
5.1 Testing Numeric Code (Floating-Point, Bounds)
5.2 Text and String Generation (Character Sets, Regex)
5.3 Dates, Times, and Time Zones
5.4 Combining Hypothesis with Fixtures and Other Test Tools
	6.	Stateful/Model-Based Testing
6.1 The RuleBasedStateMachine and @rule Decorators
6.2 Designing Operations and Invariants
6.3 Managing Complex State and Multiple Bundles
6.4 Example: Testing a CRUD System or Other Stateful API
	7.	Performance and Health Checks
7.1 Diagnosing Slow Tests with Deadlines
7.2 Common Health Check Warnings and Their Meanings
7.3 Filtering Pitfalls (assume / Over-Filters)
7.4 Tuning Hypothesis Settings (max_examples, phases, etc.)
7.5 Speed vs. Thoroughness
	8.	Multiple Failures and Multi-Bug Discovery
8.1 How Hypothesis Detects and Distinguishes Bugs
8.2 Typical Bug Slippage and the “Threshold Problem”
8.3 Strategies for Handling Multiple Distinct Failures
	9.	Internals: The Conjecture Engine
9.1 Overview of Bytestream-Based Generation
9.2 Integrated Shrinking vs. Type-Based Shrinking
9.3 How Conjecture Tracks and Minimizes Examples
9.4 The Example Database in Depth
	10.	Hypothesis in Real-World Scenarios
10.1 Using Hypothesis in CI/CD
10.2 Collaborative Testing in Teams
10.3 Integrating with Other Tools (pytest, coverage, etc.)
10.4 Best Practices for Large Projects
	11.	Extensibility and Advanced Topics
11.1 Third-Party Extensions (e.g., Hypothesis-Bio, Hypothesis-NetworkX)
11.2 Targeted Property-Based Testing (Scoring)
11.3 Hybrid Approaches (Combining Examples with Generation)
11.4 Glass-Box Testing and Potential Future Work
	12.	Troubleshooting and FAQs
12.1 Common Error Messages
12.2 Reproduce Failures with @reproduce_failure and Seeds
12.3 Overcoming Flaky or Non-Deterministic Tests
12.4 Interpreting Statistics
	13.	Summary and Further Reading
13.1 Key Takeaways and Next Steps
13.2 Recommended Resources and Papers
13.3 Contributing to Hypothesis

⸻

1. Introduction to Property-Based Testing

1.1 What Is Property-Based Testing?

Property-based testing (PBT) shifts your focus from manually enumerating test inputs to describing the properties your code should fulfill for all valid inputs. Instead of hardcoding specific examples (like assert f(2) == 4), you define requirements: e.g., “Sorting a list is idempotent.” Then the library (Hypothesis) generates test inputs to find edge cases or scenarios violating those properties.

Example

from hypothesis import given, strategies as st

@given(st.lists(st.integers()))
def test_sort_idempotent(xs):
    once = sorted(xs)
    twice = sorted(once)
    assert once == twice

Hypothesis tries diverse lists (including empty lists, duplicates, large sizes, negative or positive numbers). If something fails, it shrinks the input to a minimal failing example.

1.2 Why Use Property-Based Testing?
	•	Coverage of Edge Cases: Automatically covers many corner cases—empty inputs, large values, special floats, etc.
	•	Reduced Manual Labor: You specify broad properties, and the tool handles enumerations.
	•	Debugging Aid: Found a failing input? Hypothesis shrinks it to a simpler version, making debug cycles shorter.
	•	Less Test Boilerplate: Fewer individual test cases to write while achieving higher coverage.

1.3 Installing Hypothesis

You can install the base library with pip install hypothesis. For specialized extras (e.g., date/time, Django), consult Hypothesis extras docs.

⸻

2. First Steps with Hypothesis

2.1 A Simple Example

from hypothesis import given
from hypothesis.strategies import integers

@given(integers())
def test_square_is_nonnegative(x):
    assert x*x >= 0

Run with pytest, unittest, or another runner. Hypothesis calls test_square_is_nonnegative multiple times with varied integers (positive, negative, zero).

2.2 Basic Workflows and Key Concepts
	1.	Test Functions: Decorate with @given(<strategies>).
	2.	Generation and Execution: Hypothesis runs tests many times with random values, tries to find failures.
	3.	Shrinking: If a failure occurs, Hypothesis narrows down (shrinks) the input to a minimal failing example.

2.3 Troubleshooting the First Failures
	•	Assertion Errors: If you see Falsifying example: ..., Hypothesis found a failing scenario. Use that scenario to fix your code or refine your property.
	•	Health Check Warnings: If you see warnings like “filter_too_much” or “too_slow,” see the Health Checks section.

⸻

3. Core Hypothesis Concepts

3.1 The @given Decorator

@given ties strategies to a test function’s parameters:

from hypothesis import given
from hypothesis.strategies import text, emails

@given(email=emails(), note=text())
def test_process_email(email, note):
    ...

Hypothesis calls test_process_email() repeatedly with random emails and text. If everything passes, the test is green. Otherwise, you get a shrunk failing example.

3.2 Strategies: Building and Composing Data Generators

Hypothesis’s data generation revolves around “strategies.” Basic ones:
	•	integers(), floats(), text(), booleans(), etc.
	•	Containers: lists(elements, ...), dictionaries(keys=..., values=...)
	•	Map/Filter: Transform or constrain existing strategies.
	•	Composite: Build custom strategies for domain objects.

3.3 Shrinking and Minimizing Failing Examples

If a test fails on a complicated input, Hypothesis tries simpler versions: removing elements from lists, changing large ints to smaller ints, etc. The final reported failing input is minimal by lex ordering.

Falsifying example: test_sort_idempotent(xs=[2, 1, 1])

Hypothesis might have started with [random, complicated list] but ended with [2,1,1].

3.4 Example Database and Replay

Failures are saved in a local .hypothesis/ directory. On subsequent runs, Hypothesis replays known failing inputs before generating fresh ones. This ensures consistent reporting once a failing case is discovered.

⸻

4. Advanced Data Generation

4.1 Understanding Strategies vs. Types

Hypothesis does not rely solely on type information. You can define custom constraints to ensure the data you generate matches your domain. E.g., generating only non-empty lists or restricting floats to finite values:

import math

@given(st.lists(st.floats(allow_infinity=False, allow_nan=False), min_size=1))
def test_mean_in_bounds(xs):
    avg = sum(xs)/len(xs)
    assert min(xs) <= avg <= max(xs)

4.2 Composing Strategies (map, filter, flatmap)
	•	map(f) transforms data after generation:

even_integers = st.integers().map(lambda x: x * 2)


	•	filter(pred) discards values that fail pred; be mindful of over-filtering performance.
	•	flatmap(...) draws a value, then uses it to define a new strategy:

# Draw an int n, then a list of length n
st.integers(min_value=0, max_value=10).flatmap(lambda n: st.lists(st.text(), min_size=n, max_size=n))



4.3 Working with Complex or Recursive Data

For tree-like or nested data, use st.recursive(base_strategy, extend_strategy, max_leaves=...) to limit growth. Also consider the @composite decorator to build logic step by step.

from hypothesis import strategies as st, composite

@composite
def user_records(draw):
    name = draw(st.text(min_size=1))
    age = draw(st.integers(min_value=0))
    return "name": name, "age": age

4.4 Using @composite Functions

@composite is a more explicit style than map/flatmap. It helps define multi-step draws within one function. It’s usually simpler for highly interdependent data.

4.5 Integration and Edge Cases
	•	Ensuring Valid Domain Data: Use composites or partial filtering. Overuse of filter(...) can cause slow tests and health-check failures.
	•	Large/Complex Structures: Limit sizes or use constraints (max_size, bounding integers, etc.) to avoid timeouts.

⸻

5. Practical Usage Patterns

5.1 Testing Numeric Code (Floating-Point, Bounds)

Floating point nuances:

@given(st.floats(allow_nan=False, allow_infinity=False))
def test_floats(x):
    ...

Constrain or skip NaNs/infinities if your domain doesn’t handle them. Keep an eye on overflows if sums get large.

5.2 Text and String Generation (Character Sets, Regex)

Hypothesis can generate ASCII, Unicode, or custom sets:

from hypothesis.strategies import text

@given(text(alphabet="ABCDE", min_size=1))
def test_some_text(s):
    assert s[0] in "ABCDE"

Or use from_regex(r"MyPattern") for more specialized scenarios.

5.3 Dates, Times, and Time Zones

Install hypothesis[datetime] for strategies like dates(), datetimes(), timezones(). These handle cross-timezone issues or restricted intervals.

5.4 Combining Hypothesis with Fixtures and Other Test Tools

With pytest, you can pass both fixture arguments and Hypothesis strategy arguments:

import pytest

@pytest.fixture
def db():
    return init_db()

@given(x=st.integers())
def test_db_invariant(db, x):
    assert my_query(db, x) == ...

Function-scoped fixtures are invoked once per test function, not per example, so plan accordingly or do manual setup for each iteration.

⸻

6. Stateful/Model-Based Testing

6.1 The RuleBasedStateMachine and @rule Decorators

For testing stateful systems, Hypothesis uses a rule-based approach:

from hypothesis.stateful import RuleBasedStateMachine, rule

class SimpleCounter(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.counter = 0

    @rule(increment=st.integers(min_value=1, max_value=100))
    def inc(self, increment):
        self.counter += increment
        assert self.counter >= 0

TestCounter = SimpleCounter.TestCase

Hypothesis runs random sequences of operations, checking for invariant violations.

6.2 Designing Operations and Invariants
	•	Each @rule modifies the system under test.
	•	Use @precondition to ensure certain rules only fire in valid states.
	•	Use @invariant to check conditions after each rule.

6.3 Managing Complex State and Multiple Bundles
	•	Bundle(...) helps track created objects and pass them between rules.
	•	Perfect for simulating CRUD or multi-object interactions.

6.4 Example: Testing a CRUD System or Other Stateful API

class CRUDSystem(RuleBasedStateMachine):
    Records = Bundle('records')

    @rule(target=Records, data=st.text())
    def create(self, data):
        record_id = my_create_fn(data)
        return record_id

    @rule(record=Records)
    def delete(self, record):
        my_delete_fn(record)

Hypothesis will produce sequences of create/delete calls. If a bug arises, it provides a minimal sequence reproducing it.

⸻

7. Performance and Health Checks

7.1 Diagnosing Slow Tests with Deadlines

Hypothesis can treat slow examples as errors:

from hypothesis import settings, HealthCheck

@settings(deadline=100)  # 100ms deadline
@given(st.lists(st.integers()))
def test_something(xs):
    ...

If a single test run exceeds 100 ms, it raises DeadlineExceeded. This helps identify performance bottlenecks quickly.

7.2 Common Health Check Warnings and Their Meanings
	•	filter_too_much: A large proportion of generated data is being thrown away. Fix by refining your strategy or combining strategies (instead of heavy use of filter).
	•	too_slow: The test or generation logic is slow. Lower max_examples or investigate your code’s performance.
	•	data_too_large: Possibly generating very large structures. Restrict sizes.

7.3 Filtering Pitfalls (assume / Over-Filters)

Using assume(condition) forcibly discards any example that doesn’t meet condition. Overdoing it can degrade performance drastically. Instead, refine your data strategies:

# Instead of:
@given(st.lists(st.integers()).filter(lambda xs: sum(xs) < 100))

# Use a better approach:
@given(st.lists(st.integers(max_value=100), max_size=10))

7.4 Tuning Hypothesis Settings (max_examples, phases, etc.)
	•	max_examples: Controls how many examples are generated per test (default ~200).
	•	phases: Choose which parts of the test lifecycle (e.g. “shrink”, “reuse”) run.
	•	suppress_health_check: Silence known but acceptable warnings.

7.5 Speed vs. Thoroughness

Balance thorough coverage with test suite runtime. Trim unhelpful extra complexity in data generation. Use deadline or lower max_examples for large test suites.

⸻

8. Multiple Failures and Multi-Bug Discovery

8.1 How Hypothesis Detects and Distinguishes Bugs

Hypothesis typically shrinks until it finds the smallest failing example. But if a test can fail in multiple ways, Hypothesis 3.29+ tries to keep track of each distinct bug (by exception type and line number).

8.2 Typical Bug Slippage and the “Threshold Problem”
	•	Bug Slippage: Starting with one bug scenario but shrinking to a different scenario. Hypothesis tries to keep track and track distinct failures.
	•	Threshold Problem: When tests fail due to crossing a numeric threshold, shrunk examples tend to be just barely beyond that threshold, potentially obscuring the severity of the issue. Techniques to mitigate this can involve “targeting” or custom test logic.

8.3 Strategies for Handling Multiple Distinct Failures

Hypothesis’s multi-failure mode ensures it shrinks each failing scenario independently. You may see multiple minimal failures reported. This can be turned on automatically if distinct bug states are detected.

⸻

9. Internals: The Conjecture Engine

9.1 Overview of Bytestream-Based Generation

Conjecture is the underlying fuzzing engine. It treats every generated example as a lazily consumed byte stream. Strategies interpret segments of bytes as integers, floats, text, etc. This uniform approach:
	•	Simplifies storing known failures to replay them.
	•	Allows integrated shrinking by reducing or rewriting parts of the byte stream.

9.2 Integrated Shrinking vs. Type-Based Shrinking

Old or simpler property-based systems often rely on “type-based” shrinking. Conjecture’s approach integrates shrinking with data generation. This ensures that if you build data by composition (e.g. mapping or flattening strategies), Hypothesis can still shrink effectively.

9.3 How Conjecture Tracks and Minimizes Examples
	•	Each test run has a “buffer” of bytes.
	•	On failure, Conjecture tries different transformations (removing or reducing bytes).
	•	The result is simpler failing input but consistent with the constraints of your strategy.

9.4 The Example Database in Depth

All interesting examples get stored in .hypothesis/examples by default. On re-run, Hypothesis tries these before generating new data. This yields repeatable failures for regression tests—especially helpful in CI setups.

⸻

10. Hypothesis in Real-World Scenarios

10.1 Using Hypothesis in CI/CD
	•	Run Hypothesis-based tests as part of your continuous integration.
	•	The example database can be committed to share known failures across devs.
	•	Set a deadline or use smaller max_examples to keep test times predictable.

10.2 Collaborative Testing in Teams
	•	Consistent Strategy Definitions: Keep your custom strategies in a shared “strategies.py.”
	•	Version Control: The .hypothesis directory can be versioned to share known failing examples, though watch out for merge conflicts.

10.3 Integrating with Other Tools (pytest, coverage, etc.)
	•	Pytest integration is seamless—just write @given tests, run pytest.
	•	Coverage tools measure tested code as usual, but remember Hypothesis can deeply cover corner cases.

10.4 Best Practices for Large Projects
	•	Modular Strategies: Break them down for maintainability.
	•	Tackle Invariants Early: Short-circuit with assume() or well-structured strategies.
	•	Monitor Performance: Use health checks, deadlines, and max_examples config to scale.

⸻

11. Extensibility and Advanced Topics

11.1 Third-Party Extensions
	•	hypothesis-bio: Specialized for bioinformatics data formats.
	•	hypothesis-networkx: Generate networkx graphs, test graph algorithms.
	•	Many more unofficial or domain-specific libraries exist. Creating your own extension is easy.

11.2 Targeted Property-Based Testing (Scoring)

You can “guide” test generation by calling target(score) in your code. Hypothesis tries to evolve test cases with higher scores, focusing on “interesting” or extreme behaviors (like maximizing error metrics).

from hypothesis import given, target
from hypothesis.strategies import floats

@given(x=floats(-1e6, 1e6))
def test_numerical_stability(x):
    err = some_error_metric(x)
    target(err)
    assert err < 9999

11.3 Hybrid Approaches (Combining Examples with Generation)

You can add “example-based tests” to complement property-based ones. Also, you can incorporate real-world test data as seeds or partial strategies.

11.4 Glass-Box Testing and Potential Future Work

Hypothesis largely treats tests as a black box but can be extended with coverage data or other instrumentation for more advanced test generation. This is an open area of R&D.

⸻

12. Troubleshooting and FAQs

12.1 Common Error Messages
	•	Unsatisfiable: Hypothesis can’t find enough valid examples. Possibly an over-filter or an unrealistic requirement.
	•	DeadlineExceeded: Your test or code is too slow for the set deadline ms.
	•	FailedHealthCheck: Usually means you’re doing too much filtering or the example is too large.

12.2 Reproduce Failures with @reproduce_failure and Seeds

If Hypothesis can’t express your failing data via a standard repr, it shows a snippet like:

@reproduce_failure('3.62.0', b'...')
def test_something():
    ...

Adding that snippet ensures the bug is replayed exactly. Alternatively, you can do:

from hypothesis import seed

@seed(12345)
@given(st.integers())
def test_x(x):
    ...

But seeds alone are insufficient if your .hypothesis database is relevant or if your test uses inline data.

12.3 Overcoming Flaky or Non-Deterministic Tests

If code is time-sensitive or concurrency-based, you may see spurious failures. Try limiting concurrency, raising deadlines, or disabling shrinking for certain tests. Alternatively, fix the non-determinism in the tested code.

12.4 Interpreting Statistics

Running pytest --hypothesis-show-statistics yields info on distribution of generated examples, data-generation time vs. test time, etc. This helps find bottlenecks, excessive filtering, or unexpectedly large inputs.

⸻

13. Summary and Further Reading

13.1 Key Takeaways and Next Steps
	•	Write Clear Properties: A crisp property is simpler for Hypothesis to exploit.
	•	Refine Strategies: Good strategy design yields fewer discards and faster tests.
	•	Use Health Checks: They highlight anti-patterns early.
	•	Explore Stateful Testing: Perfect for integration tests or persistent-state bugs.

13.2 Recommended Resources and Papers
	•	Official Hypothesis Documentation
	•	QuickCheck papers: Claessen and Hughes, 2000
	•	Testing–reduction synergy: Regehr et al. “Test-case Reduction via Delta Debugging” (PLDI 2012)
	•	“Hypothesis: A New Approach to Property-Based Testing” (HypothesisWorks website)

13.3 Contributing to Hypothesis

Hypothesis is open source. If you have ideas or find issues:
	•	Check our GitHub repo
	•	Read the Contributing Guide
	•	Every improvement is welcomed—documentation, bug reports, or code!

⸻

Final Thoughts

We hope this unified, comprehensive guide helps you unlock the power of Hypothesis. From quick introductions to advanced stateful testing, from performance pitfalls to internal design details, you now have a toolkit for robust property-based testing in Python.

Happy testing! If you run into any questions, re-check the relevant sections here or visit the community resources. Once you incorporate Hypothesis into your testing workflow, you might find hidden bugs you never anticipated—and that’s the point!
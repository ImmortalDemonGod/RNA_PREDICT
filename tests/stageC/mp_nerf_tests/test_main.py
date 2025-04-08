import numpy as np
import torch

from rna_predict.pipeline.stageC.mp_nerf.massive_pnerf import (
    MpNerfParams,
    mp_nerf_torch,
)
from rna_predict.pipeline.stageC.mp_nerf.proteins import (
    modify_angles_mask_with_torsions,
)
from rna_predict.pipeline.stageC.mp_nerf.utils import (
    get_dihedral,
)


def test_nerf_and_dihedral():
    # create points
    a = torch.tensor([1, 2, 3]).float()
    b = torch.tensor([1, 4, 5]).float()
    c = torch.tensor([1, 4, 7]).float()
    d = torch.tensor([1, 8, 8]).float()
    # calculate internal references
    (b - a).numpy()
    v2 = (c - b).numpy()
    v3 = (d - c).numpy()
    # get angles
    theta = np.arccos(np.dot(v2, v3) / (np.linalg.norm(v2) * np.linalg.norm(v3)))

    # Calculate bond length and bond angle (theta)
    bond_length = torch.tensor(np.linalg.norm(v3))
    theta = torch.tensor(theta)  # Already calculated using np.arccos

    # Calculate dihedral angle (chi) using the imported function for correct sign
    chi = get_dihedral(a, b, c, d)

    # reconstruct
    # The comment about scn angle might be outdated or related to a different issue.
    # We now use standard geometric calculations for bond_length, theta and a signed chi.
    params = MpNerfParams(
        a=a,
        b=b,
        c=c,
        bond_length=bond_length,
        theta=torch.tensor(np.pi) - theta,  # Pass pi - theta
        chi=chi,  # Use chi from get_dihedral
    )
    reconstructed_d = mp_nerf_torch(params)

    # Assert that the NeRF reconstruction matches the original point d
    # A tolerance is needed due to floating point precision.
    assert torch.allclose(
        reconstructed_d, d, atol=1e-5
    ), f"Reconstructed {reconstructed_d} != Original {d}"


def test_modify_angles_mask_with_torsions():
    # create inputs
    seq = "AGHHKLHRTVNMSTIL"
    angles_mask = torch.randn(2, 16, 14)
    torsions = torch.ones(16, 4)
    # ensure shape
    assert (
        modify_angles_mask_with_torsions(seq, angles_mask, torsions).shape
        == angles_mask.shape
    ), "Shapes don't match"

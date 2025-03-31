import torch

from rna_predict.pipeline.stageC.mp_nerf.ml_utils import (
    chain2atoms,
    fape_torch,
    rename_symmetric_atoms,
    scn_atom_embedd,
    torsion_angle_loss,
)
from rna_predict.pipeline.stageC.mp_nerf.proteins import scn_cloud_mask


# test ML utils
def test_scn_atom_embedd():
    seq_list = ["AGHHKLHRTVNMSTIL", "WERTQLITANMWTCSD"]
    embedds = scn_atom_embedd(seq_list)
    assert embedds.shape == torch.Size([2, 16, 14]), "Shapes don't match"


def test_chain_to_atoms():
    chain = torch.randn(100, 3)
    atoms = chain2atoms(chain, c=14)
    assert atoms.shape == torch.Size([100, 14, 3]), "Shapes don't match"


def test_rename_symmetric_atoms():
    seq_list = ["AGHHKLHRTVNMSTIL"]
    pred_coors = torch.randn(1, 16, 14, 3)
    true_coors = torch.randn(1, 16, 14, 3)
    cloud_mask = scn_cloud_mask(seq_list[0]).unsqueeze(0)
    pred_feats = torch.randn(1, 16, 14, 16)

    renamed = rename_symmetric_atoms(
        pred_coors, true_coors, seq_list, cloud_mask, pred_feats=pred_feats
    )
    assert (
        renamed[0].shape == pred_coors.shape and renamed[1].shape == pred_feats.shape
    ), "Shapes don't match"


def test_torsion_angle_loss():
    pred_torsions = torch.randn(1, 100, 7)
    true_torsions = torch.randn(1, 100, 7)

    loss = torsion_angle_loss(pred_torsions, true_torsions, coeff=2.0, angle_mask=None)
    assert loss.shape == pred_torsions.shape, "Shapes don't match"


def test_fape_loss_torch():
    seq_list = ["AGHHKLHRTVNMSTIL"]
    pred_coords = torch.randn(1, 16, 14, 3)
    true_coords = torch.randn(1, 16, 14, 3)

    fape_torch(pred_coords, true_coords, c_alpha=True, seq_list=seq_list)
    fape_torch(pred_coords, true_coords, c_alpha=False, seq_list=seq_list)

    assert True

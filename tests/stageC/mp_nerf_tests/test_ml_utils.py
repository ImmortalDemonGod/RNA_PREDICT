import torch

from rna_predict.pipeline.stageC.mp_nerf.ml_utils import (
    chain2atoms,
    fape_torch,
    rename_symmetric_atoms,
    scn_atom_embedd,
    torsion_angle_loss,
)


# test ML utils
def test_scn_atom_embedd():
    seq_list = ["AGCDEFGIKLMNPQRSTVWY", "WERTQLITANMWTCSDAAA_"]
    embedds = scn_atom_embedd(seq_list)
    assert embedds.shape == torch.Size([2, 20, 14]), "Shapes don't match"


def test_chain_to_atoms():
    chain = torch.randn(100, 3)
    atoms = chain2atoms(chain, c=14)
    assert atoms.shape == torch.Size([100, 14, 3]), "Shapes don't match"


def test_rename_symmetric_atoms():
    seq_list = ["AGCDEFGIKLMNPQRSTV"]
    # Adjust shapes to match expected input format (batch, num_atoms, 3/features)
    # Assuming num_atoms = seq_len * 14 for SCN format
    seq_len = len(seq_list[0])
    num_atoms = seq_len * 14
    pred_coors = torch.randn(1, num_atoms, 3)  # Example: Batch size 1
    pred_feats = torch.randn(1, num_atoms, 16)  # Example: Batch size 1, 16 features
    # true_coors and cloud_mask are no longer needed for the refactored function call

    # Call with the updated signature
    renamed_coors, renamed_feats = rename_symmetric_atoms(
        pred_coors=pred_coors[0],  # Pass the first batch element
        pred_feats=pred_feats[0],  # Pass the first batch element
        seq=seq_list[0],
    )

    # Check output shapes (adjusting for single batch element processing)
    assert (
        renamed_coors.shape == pred_coors[0].shape
    ), f"Coordinate shapes don't match: Expected {pred_coors[0].shape}, Got {renamed_coors.shape}"
    assert (
        renamed_feats.shape == pred_feats[0].shape
    ), f"Feature shapes don't match: Expected {pred_feats[0].shape}, Got {renamed_feats.shape}"


def test_torsion_angle_loss():
    pred_torsions = torch.randn(1, 100, 7)
    true_torsions = torch.randn(1, 100, 7)

    loss = torsion_angle_loss(pred_torsions, true_torsions, coeff=2.0, angle_mask=None)
    assert loss.shape == pred_torsions.shape, "Shapes don't match"


def test_fape_loss_torch():
    seq_list = ["AGCDEFGIKLMNPQRSTV"]
    pred_coords = torch.randn(1, 18, 14, 3)
    true_coords = torch.randn(1, 18, 14, 3)

    fape_torch(pred_coords, true_coords, c_alpha=True, seq_list=seq_list)
    fape_torch(pred_coords, true_coords, c_alpha=False, seq_list=seq_list)

    assert True

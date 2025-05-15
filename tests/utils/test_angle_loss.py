import torch
import pytest
from rna_predict.utils.angle_loss import angle_loss

@ pytest.mark.parametrize("pred,target,mask,expected", [
    # identical predictions → zero loss
    (
        torch.tensor([[[0.0, 1.0], [2.0, 3.0]]]),
        torch.tensor([[[0.0, 1.0], [2.0, 3.0]]]),
        None,
        0.0
    ),
    # known differences without mask
    (
        torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
        torch.tensor([[[2.0, 4.0], [3.0, 0.0]]]),
        None,
        ((1.0**2 + 2.0**2 + 0.0 + 16.0) / 4.0)
    ),
])

def test_angle_loss_no_mask(pred, target, mask, expected):
    loss = angle_loss(pred, target, mask)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() == pytest.approx(expected, rel=1e-6)


def test_angle_loss_with_mask():
    pred = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    target = torch.tensor([[[2.0, 4.0], [3.0, 0.0]]])
    # mask only first residue
    mask = torch.tensor([[True, False]], dtype=torch.bool)
    # differences: first row: [-1, -2] squares 1+4=5; only that row counted (mask.sum = 1)
    expected = 5.0
    loss = angle_loss(pred, target, mask)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() == pytest.approx(expected, abs=1e-6)


def test_angle_loss_all_mask_zero_returns_zero():
    pred = torch.tensor([[[1.0]]])
    target = torch.tensor([[[0.0]]])
    mask = torch.tensor([[False]], dtype=torch.bool)
    loss = angle_loss(pred, target, mask)
    # mask sum=0 → denominator tiny → returns 0
    assert loss.item() == pytest.approx(0.0, abs=1e-6)

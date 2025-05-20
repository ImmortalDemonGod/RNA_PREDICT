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
    """
    Tests angle_loss without a mask, verifying output type and numerical correctness.
    
    Asserts that the loss is a scalar tensor and matches the expected value within a relative tolerance of 1e-6.
    """
    loss = angle_loss(pred, target, mask)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() == pytest.approx(expected, rel=1e-6)


def test_angle_loss_with_mask():
    """
    Tests that angle_loss computes the loss only over masked elements.
    
    Verifies that the loss is calculated using only the residues selected by the boolean mask and matches the expected sum of squared differences.
    """
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
    """
    Tests that angle_loss returns zero when the mask excludes all elements.
    
    Verifies that the loss is zero if no elements are selected by the mask, ensuring correct handling of the zero denominator case.
    """
    pred = torch.tensor([[[1.0]]])
    target = torch.tensor([[[0.0]]])
    mask = torch.tensor([[False]], dtype=torch.bool)
    loss = angle_loss(pred, target, mask)
    # mask sum=0 → denominator tiny → returns 0
    assert loss.item() == pytest.approx(0.0, abs=1e-6)

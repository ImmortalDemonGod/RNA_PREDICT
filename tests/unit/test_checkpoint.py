"""
Unit tests for partial_load_state_dict utility in rna_predict.utils.checkpoint
Covers full, partial, and mismatched state dict scenarios.
"""
import torch
import torch.nn as nn
import pytest
from rna_predict.utils.checkpoint import partial_load_state_dict

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(4, 4)
        self.adapter = nn.Linear(4, 2)

    def forward(self, x):
        x = self.base(x)
        return self.adapter(x)


def test_full_state_dict_strict():
    model = DummyModel()
    sd = model.state_dict()
    # Should load fully, no errors
    missing, unexpected = partial_load_state_dict(model, sd, strict=True)
    assert missing == []
    assert unexpected == []


def test_partial_state_dict_non_strict():
    model = DummyModel()
    # Only adapter weights
    partial_sd = {k: v for k, v in model.adapter.state_dict().items()}
    # Key names must be prefixed as in full state dict
    partial_sd = {f"adapter.{k}": v for k, v in partial_sd.items()}
    missing, unexpected = partial_load_state_dict(model, partial_sd, strict=False)
    # Should report base keys as missing, but not error
    assert any("base" in k for k in missing)
    assert unexpected == []


def test_partial_state_dict_strict_raises():
    model = DummyModel()
    partial_sd = {k: v for k, v in model.adapter.state_dict().items()}
    partial_sd = {f"adapter.{k}": v for k, v in partial_sd.items()}
    with pytest.raises(RuntimeError):
        partial_load_state_dict(model, partial_sd, strict=True)


def test_extra_keys_in_state_dict():
    model = DummyModel()
    sd = model.state_dict()
    # Add an extra key
    sd_extra = dict(sd)
    sd_extra["extra.weight"] = torch.randn(2, 2)
    missing, unexpected = partial_load_state_dict(model, sd_extra, strict=False)
    assert "extra.weight" in unexpected


def test_shape_mismatch_raises():
    model = DummyModel()
    sd = model.state_dict()
    # Change shape of a parameter
    broken_sd = dict(sd)
    broken_sd["base.weight"] = torch.randn(2, 2)  # Wrong shape
    with pytest.raises(RuntimeError):
        partial_load_state_dict(model, broken_sd, strict=True)

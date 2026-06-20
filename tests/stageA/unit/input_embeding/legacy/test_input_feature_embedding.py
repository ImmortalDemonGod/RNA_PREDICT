"""
Tests for InputFeatureEmbedder.

Each test names the bug from input_feature_embedding.bug-catalog.md it is designed to catch.

These tests are intentionally RED until the fix for finding s2c3l0-020 lands:
  rna_predict/pipeline/stageA/input_embedding/legacy/encoder/input_feature_embedding.py
  must replace the two `from rna_predict.models.encoder.atom_encoder import ...`
  lines with the correct path in the legacy sub-package.
"""

import importlib

import pytest
import torch


# ---------------------------------------------------------------------------
# BUG-1 — top-level import of nonexistent rna_predict.models (line 4)
# ---------------------------------------------------------------------------


def test_module_is_importable__guards_against_missing_rna_predict_models_package():
    """
    BUG-1: input_feature_embedding.py:4 does
        from rna_predict.models.encoder.atom_encoder import AtomAttentionEncoder
    rna_predict.models does not exist; the correct path is
        rna_predict.pipeline.stageA.input_embedding.legacy.encoder.atom_encoder
    This test fails (ModuleNotFoundError) until that line is corrected.
    """
    mod = importlib.import_module(
        "rna_predict.pipeline.stageA.input_embedding.legacy.encoder.input_feature_embedding"
    )
    assert hasattr(mod, "InputFeatureEmbedder"), (
        "Module imported but InputFeatureEmbedder not found — class may have been renamed"
    )


# ---------------------------------------------------------------------------
# BUG-1 follow-through — class must be a proper nn.Module subclass
# ---------------------------------------------------------------------------


def test_class_is_nn_module_subclass__guards_against_phantom_class_from_broken_import():
    """
    BUG-1: If the module is importable, InputFeatureEmbedder must be a real
    nn.Module subclass, not a stub or re-export of a wrong symbol.
    Fails until BUG-1 is fixed (import raises before this assertion runs).
    """
    mod = importlib.import_module(
        "rna_predict.pipeline.stageA.input_embedding.legacy.encoder.input_feature_embedding"
    )
    cls = mod.InputFeatureEmbedder
    import torch.nn as nn
    assert issubclass(cls, nn.Module), (
        f"InputFeatureEmbedder must subclass nn.Module, got bases: {cls.__bases__}"
    )


# ---------------------------------------------------------------------------
# BUG-2 — lazy import inside __init__ also references rna_predict.models (line 36)
# ---------------------------------------------------------------------------


def test_instantiation_succeeds__guards_against_init_lazy_import_of_missing_models_package():
    """
    BUG-2: input_feature_embedding.py:36 (inside __init__) repeats
        from rna_predict.models.encoder.atom_encoder import AtomEncoderConfig
    Even if line 4 were patched, instantiation would still raise ModuleNotFoundError.
    This test verifies that a minimal InputFeatureEmbedder() can be constructed —
    it fails until both line 4 AND line 36 are corrected to the legacy path.
    Uses smallest valid hyperparams to minimise test weight.
    """
    mod = importlib.import_module(
        "rna_predict.pipeline.stageA.input_embedding.legacy.encoder.input_feature_embedding"
    )
    cls = mod.InputFeatureEmbedder
    # Smallest possible config that satisfies internal constraints.
    instance = cls(
        c_token=64,
        restype_dim=4,
        profile_dim=4,
        c_atom=16,
        c_pair=8,
        num_heads=2,
        num_layers=1,
        use_optimized=False,
        pairformer_blocks=1,
    )
    assert instance is not None


# ---------------------------------------------------------------------------
# Contract pin — goal-verification command from the finding description
# ---------------------------------------------------------------------------


def test_goal_verification_import_exits_clean__primary_deliverable_of_finding_s2c3l0_020():
    """
    Mirrors the exact goal-verification stated in finding s2c3l0-020:
        uv run python -c 'from rna_predict.pipeline.stageA.input_embedding.legacy.encoder.input_feature_embedding import InputFeatureEmbedder'
    This test is the in-process equivalent: import succeeds and the class is
    accessible. Fails until BUG-1 is fixed.
    """
    from rna_predict.pipeline.stageA.input_embedding.legacy.encoder.input_feature_embedding import (  # noqa: E501
        InputFeatureEmbedder,
    )
    assert InputFeatureEmbedder is not None

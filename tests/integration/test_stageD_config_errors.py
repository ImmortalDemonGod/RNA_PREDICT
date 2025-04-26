"""
Test for regression: Stage D config errors (missing feature_dimensions, dummy config, etc)
This test ensures that if Stage D is called with an incomplete config (as in the previous bug),
we get a ValueError or AttributeError, and that the error is caught and reported.
"""
import pytest
from omegaconf import OmegaConf
import torch
from rna_predict.pipeline.stageD.diffusion.bridging.residue_atom_bridge import bridge_residue_to_atom, BridgingInput
import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf
import os
import sys

# Enforce running from project root for Hydra config correctness
assert os.path.basename(os.getcwd()) == "RNA_PREDICT", (
    f"Hydra-based tests must be run from the project root. Current CWD: {os.getcwd()}"
)

def make_dummy_bridging_input():
    # Minimal dummy input for bridge_residue_to_atom
    sequence = "ACGUACGU"
    partial_coords = torch.zeros((8, 44, 3))
    s_inputs = torch.zeros((8, 384))
    input_features = {"sequence": sequence, "atom_metadata": None}
    return BridgingInput(
        partial_coords=partial_coords,
        trunk_embeddings={"s_inputs": s_inputs},
        input_features=input_features,
        sequence=sequence,
    )

def make_dummy_config_missing_feature_dimensions():
    # This mimics the old bug: config missing feature_dimensions
    return OmegaConf.create({"mode": "inference", "device": "cpu"})

def make_dummy_config_none_trunk_embeddings():
    # This mimics the bug where trunk_embeddings is None
    return OmegaConf.create({"feature_dimensions": {"c_s": 384}, "mode": "inference", "device": "cpu"})


def test_stageD_missing_feature_dimensions_raises():
    bridging_input = make_dummy_bridging_input()
    dummy_config = make_dummy_config_missing_feature_dimensions()
    # Accept any message mentioning 'feature_dimensions' (robust to rewording)
    with pytest.raises(ValueError, match=r"feature_dimensions"):
        bridge_residue_to_atom(bridging_input, dummy_config, debug_logging=True)

def test_stageD_none_trunk_embeddings_raises():
    bridging_input = make_dummy_bridging_input()
    dummy_config = make_dummy_config_none_trunk_embeddings()
    # Simulate None trunk_embeddings in input
    bridging_input.trunk_embeddings = None
    # Accept any message mentioning 's_inputs' (robust to error location)
    with pytest.raises(ValueError, match=r"s_inputs"):  # robust to error message wording
        bridge_residue_to_atom(bridging_input, dummy_config, debug_logging=True)

def test_stageD_missing_s_inputs_in_real_config_raises():
    config_path = "../../rna_predict/conf"  # Per project rule: must be relative to CWD (tests/integration)
    with initialize(config_path=config_path, job_name="test"):
        cfg = compose(config_name="default")
        # Temporarily disable struct mode to allow deletion
        OmegaConf.set_struct(cfg, False)

        # Ensure feature_dimensions exists in both places
        if not hasattr(cfg.model.stageD, "feature_dimensions"):
            cfg.model.stageD.feature_dimensions = OmegaConf.create({})
        if not hasattr(cfg.model.stageD.diffusion, "feature_dimensions"):
            cfg.model.stageD.diffusion.feature_dimensions = OmegaConf.create({})

        # Delete both s_inputs and c_s_inputs to ensure the test fails
        # Delete from both top-level and diffusion section to be thorough
        if "s_inputs" in cfg.model.stageD.feature_dimensions:
            del cfg.model.stageD.feature_dimensions["s_inputs"]
        if "s_inputs" in cfg.model.stageD.diffusion.feature_dimensions:
            del cfg.model.stageD.diffusion.feature_dimensions["s_inputs"]

        if "c_s_inputs" in cfg.model.stageD.feature_dimensions:
            del cfg.model.stageD.feature_dimensions["c_s_inputs"]
        if "c_s_inputs" in cfg.model.stageD.diffusion.feature_dimensions:
            del cfg.model.stageD.diffusion.feature_dimensions["c_s_inputs"]

        OmegaConf.set_struct(cfg, True)
        bridging_input = make_dummy_bridging_input()

        # Should raise ValueError about missing s_inputs
        # The error might be raised during validation or during bridging
        with pytest.raises((ValueError, AttributeError), match=r"(s_inputs|feature_dimensions)"):  # robust to error message wording
            bridge_residue_to_atom(bridging_input, cfg.model.stageD.diffusion, debug_logging=True)

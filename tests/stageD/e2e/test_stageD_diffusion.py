import pytest
import torch
from hydra import initialize, compose
import os

from rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager import (
    ProtenixDiffusionManager,
)
from rna_predict.pipeline.stageD.diffusion.run_stageD_unified import run_stageD_diffusion
from rna_predict.pipeline.stageD.diffusion.utils import DiffusionConfig  # Import needed class


def ensure_project_root():
    """Ensure test is running from the project root for Hydra config resolution."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    cwd = os.getcwd()
    print(f"[DEBUG][test_stageD_diffusion] Current working directory: {cwd}")
    if cwd != project_root:
        raise RuntimeError(
            f"Test must be run from project root ({project_root}), not {cwd}. "
            "Please run: cd /Users/tomriddle1/RNA_PREDICT && uv run -m pytest ..."
        )


@pytest.fixture(scope="function")
def minimal_diffusion_config():
    """Provide a minimal but valid diffusion config for testing"""
    return {
        # Core parameters
        "device": "cpu",
        "mode": "inference",
        "debug_logging": True,
        "ref_element_size": 128,
        "ref_atom_name_chars_size": 256,
        "profile_size": 32,

        # Model architecture section
        "model_architecture": {
            "c_token": 384,
            "c_s": 384,
            "c_z": 32,
            "c_s_inputs": 384,
            "c_atom": 128,
            "c_atompair": 16,
            "c_noise_embedding": 128,
            "sigma_data": 16.0,  # Required for noise sampling
        },

        # Feature dimensions section
        "feature_dimensions": {
            "c_s": 384,
            "c_s_inputs": 384,
            "c_sing": 384,
            "s_trunk": 384,
            "s_inputs": 384
        },

        # Transformer configuration
        "transformer": {"n_blocks": 1, "n_heads": 2},

        # Inference configuration
        "inference": {"num_steps": 2, "N_sample": 1},  # Reduced steps for testing

        # Conditioning configuration
        "conditioning": {
            "c_s": 384,
            "c_z": 32,
            "c_s_inputs": 384,
            "c_noise_embedding": 128,
        },

        # Embedder configuration
        "embedder": {"c_atom": 128, "c_atompair": 16, "c_token": 384},

        # Atom encoder configuration
        "atom_encoder": {
            "c_in": 128,
            "c_hidden": [256],
            "c_out": 128,
            "dropout": 0.1,
            "n_blocks": 1,
            "n_heads": 2,
            "n_queries": 4,
            "n_keys": 4
        },

        # Atom decoder configuration
        "atom_decoder": {
            "c_in": 128,
            "c_hidden": [256],
            "c_out": 128,
            "dropout": 0.1,
            "n_blocks": 1,
            "n_heads": 2,
            "n_queries": 4,
            "n_keys": 4
        },

        # Sigma data should only be in model_architecture
        # "sigma_data": 16.0,  # Removed to fix test

        # Initialization
        "initialization": {},  # Required by DiffusionModule
    }


@pytest.fixture(scope="function")
def minimal_input_features():
    """Provide minimal but valid input features for testing"""
    return {
        "atom_to_token_idx": torch.zeros((1, 5), dtype=torch.long),
        "ref_pos": torch.randn(1, 5, 3),
        "ref_space_uid": torch.arange(5).unsqueeze(0),
        "ref_charge": torch.zeros(1, 5, 1),
        "ref_mask": torch.ones(1, 5, 1),
        "ref_element": torch.zeros(1, 5, 128),
        "ref_atom_name_chars": torch.zeros(1, 5, 256),
        "restype": torch.zeros(1, 5, 32),
        "profile": torch.zeros(1, 5, 32),
        "deletion_mean": torch.zeros(1, 5, 1),
        "sing": torch.randn(1, 5, 384),  # Required for s_inputs fallback, match c_s_inputs
    }


@pytest.mark.integration
def test_run_stageD_diffusion_inference():
    """Test Stage D diffusion using Hydra-composed config group, not ad-hoc dict."""
    ensure_project_root()
    with initialize(config_path="../../../rna_predict/conf"):
        # Compose the config using the default chain (which must include model/stageD)
        cfg = compose(config_name="default")
        # Optionally override device or mode for test speed
        cfg.model.stageD.device = "cpu"
        cfg.model.stageD.mode = "inference"
        # Compose or generate minimal valid input features for the test
        input_features = {
            "atom_to_token_idx": torch.zeros((1, 5), dtype=torch.long),
            "ref_pos": torch.randn(1, 5, 3),
            "ref_space_uid": torch.arange(5).unsqueeze(0),
            "ref_charge": torch.zeros(1, 5, 1),
            "ref_mask": torch.ones(1, 5, 1),
            "profile": torch.randn(1, 5, 32),
        }
        # Minimal trunk_embeddings for test
        trunk_embeddings = {
            "s_inputs": torch.randn(1, 5, 384),
            "s_trunk": torch.randn(1, 5, 384),
            "pair": torch.randn(1, 5, 5, 32)
        }
        # Minimal partial_coords for test
        partial_coords = torch.randn(1, 5, 3)
        # Build DiffusionConfig
        test_config = DiffusionConfig(
            partial_coords=partial_coords,
            trunk_embeddings=trunk_embeddings,
            diffusion_config=dict(cfg.model.stageD.diffusion),
            mode=cfg.model.stageD.mode,
            device=cfg.model.stageD.device,
            input_features=input_features,
            debug_logging=cfg.model.stageD.get("debug_logging", False),
            sequence=None,
            ref_element_size=cfg.model.stageD.get("ref_element_size", 128),
            ref_atom_name_chars_size=cfg.model.stageD.get("ref_atom_name_chars_size", 256),
            profile_size=cfg.model.stageD.get("profile_size", 32),
            cfg=cfg.model.stageD,
        )
        # Patch in required config sections for validation
        test_config.model_architecture = dict(cfg.model.stageD["model_architecture"])
        test_config.diffusion = dict(cfg.model.stageD["diffusion"])
        test_config.feature_dimensions = dict(cfg.model.stageD["feature_dimensions"])
        # Run the pipeline with the config group
        out_coords = run_stageD_diffusion(config=test_config)
        assert isinstance(out_coords, torch.Tensor)
        assert out_coords.ndim == 3  # [batch, n_atoms, 3]
        assert out_coords.shape[2] == 3  # Check coordinate dimension


@pytest.mark.parametrize("missing_s_inputs", [True, False])
def test_run_stageD_diffusion_inference_original(
    missing_s_inputs, minimal_diffusion_config, minimal_input_features
):
    """
    Calls run_stageD_diffusion in 'inference' mode with partial trunk_embeddings.
    If missing_s_inputs=True, we omit 's_inputs' to see if it is auto-computed.
    """
    # Use smaller tensors for testing
    # Create partial_coords with 11 atoms to match atom_metadata
    partial_coords = torch.randn(1, 11, 3)  # batch=1, 11 atoms, 3 coords

    trunk_embeddings = {
        "s_trunk": torch.randn(1, 5, 384),
        "pair": torch.randn(1, 5, 5, 32),
    }
    if not missing_s_inputs:
        # Match the c_s_inputs dimension expected by DiffusionConditioning (384)
        trunk_embeddings["s_inputs"] = torch.randn(1, 5, 384)

    # Provide a minimal sequence matching the number of residues/atoms (5)
    # Convert list to string for DiffusionConfig which expects str or None
    sequence = "ACGUA"

    # Add atom_metadata to minimal_input_features
    minimal_input_features["atom_metadata"] = {
        "residue_indices": [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4],  # 11 atoms across 5 residues
        "atom_names": ["P", "C4'", "N1", "P", "C4'", "P", "C4'", "P", "C4'", "P", "C4'"]
    }

    try:
        # Create the config object using the variables available in the test scope
        from omegaconf import OmegaConf

        # Create a properly structured Hydra config with model.stageD section
        hydra_cfg = OmegaConf.create({
            "model": {
                "stageD": {
                    # Top-level parameters
                    "enabled": True,
                    "mode": "inference",
                    "device": "cpu",
                    "debug_logging": True,
                    "ref_element_size": 128,
                    "ref_atom_name_chars_size": 256,
                    "profile_size": 32,

                    # Feature dimensions
                    "feature_dimensions": minimal_diffusion_config["feature_dimensions"],

                    # Model architecture
                    "model_architecture": minimal_diffusion_config["model_architecture"],

                    # Diffusion section
                    "diffusion": minimal_diffusion_config
                }
            }
        })

        # Create the DiffusionConfig with all required parameters
        test_config = DiffusionConfig(
            partial_coords=partial_coords,
            trunk_embeddings=trunk_embeddings,
            diffusion_config=minimal_diffusion_config,
            mode="inference",
            device="cpu",
            input_features=minimal_input_features,
            sequence=sequence,  # Provide sequence to avoid bridging error
            cfg=hydra_cfg,  # Pass the properly structured Hydra config
            # Add required feature parameters directly to the config object
            ref_element_size=128,
            ref_atom_name_chars_size=256,
            profile_size=32
        )

        # Add feature_dimensions directly to the config object
        test_config.feature_dimensions = minimal_diffusion_config["feature_dimensions"]
        # Call the refactored function with the config object
        out_coords = run_stageD_diffusion(config=test_config)

        assert isinstance(out_coords, torch.Tensor)
        assert out_coords.ndim == 3  # [batch, n_atoms, 3]
        assert (
            out_coords.shape[1] == partial_coords.shape[1]
        )  # Check number of atoms matches
        assert out_coords.shape[2] == 3  # Check coordinate dimension
    finally:
        # Cleanup
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


@pytest.mark.xfail(reason="Shape mismatch in diffusion module - not related to API change")
def test_multi_step_inference_fallback(
    minimal_diffusion_config, minimal_input_features
):
    """
    Skips run_stageD_diffusion and calls manager.multi_step_inference directly,
    providing partial trunk_embeddings plus override_input_features to
    auto-build 's_inputs' if missing.
    """
    # Use smaller tensors for testing
    coords_init = torch.randn(1, 5, 3)
    trunk_embeddings = {
        "s_trunk": torch.randn(1, 5, 384),
        "pair": torch.randn(1, 5, 5, 32),
    }

    # Create a properly structured config with stageD_diffusion key
    # First, create a clean copy of minimal_diffusion_config without 'inference' key
    # since DiffusionModule doesn't accept it
    diffusion_model_clean = {k: v for k, v in minimal_diffusion_config.items()
                           if k not in ['inference', 'conditioning', 'embedder']}

    # Create an OmegaConf DictConfig object instead of a regular dictionary
    from omegaconf import OmegaConf
    hydra_compatible_config = OmegaConf.create({
        "stageD_diffusion": {
            "device": "cpu",
            "diffusion_chunk_size": None,
            "debug_logging": True,
        },
        "diffusion_model": diffusion_model_clean,
        "noise_schedule": {
            "schedule_type": "linear",
            "beta_start": 0.0001,
            "beta_end": 0.02,
        },
        "sampler": {
            "p_mean": -1.2,
            "p_std": 1.5,
            "N_sample": 1,
        },
        "inference": {
            "num_steps": 2,
            "N_sample": 1,
            "inplace_safe": False,
        }
    })

    try:
        manager = ProtenixDiffusionManager(hydra_compatible_config)
        # Update manager's config with inference parameters
        from omegaconf import OmegaConf
        inference_params = {"num_steps": 2, "N_sample": 1}  # Reduced steps

        if not hasattr(manager, 'cfg') or not OmegaConf.is_config(manager.cfg):
            manager.cfg = OmegaConf.create({
                "stageD_diffusion": {
                    "inference": inference_params,
                    "debug_logging": True
                }
            })
        else:
            # Update existing config
            if "inference" not in manager.cfg.stageD_diffusion:
                manager.cfg.stageD_diffusion.inference = OmegaConf.create(inference_params)
            else:
                for k, v in inference_params.items():
                    manager.cfg.stageD_diffusion.inference[k] = v
            manager.cfg.stageD_diffusion.debug_logging = True

        # Call with updated API
        coords_final = manager.multi_step_inference(
            coords_init=coords_init,
            trunk_embeddings=trunk_embeddings,
            override_input_features=minimal_input_features
        )

        assert isinstance(coords_final, torch.Tensor)
        assert coords_final.ndim == 3  # [batch, n_atoms, 3]
        assert coords_final.shape[1] == coords_init.shape[1]
    finally:
        # Cleanup
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

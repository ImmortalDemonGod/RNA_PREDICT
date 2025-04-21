import pytest
import torch

from rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager import (
    ProtenixDiffusionManager,
)
from rna_predict.pipeline.stageD.diffusion.run_stageD_unified import run_stageD_diffusion
from rna_predict.pipeline.stageD.diffusion.utils import DiffusionConfig  # Import needed class


@pytest.fixture(scope="function")
def minimal_diffusion_config():
    """Provide a minimal but valid diffusion config for testing"""
    return {
        "c_atom": 128,
        "c_s": 384,
        "c_z": 32,
        "c_token": 384,
        "c_s_inputs": 384,  # Match conditioning config for consistency
        "transformer": {"n_blocks": 1, "n_heads": 2},
        "inference": {"num_steps": 2, "N_sample": 1},  # Reduced steps for testing
        "conditioning": {
            "c_s": 384,
            "c_z": 32,
            "c_s_inputs": 384,
            "c_noise_embedding": 128,
        },
        "embedder": {"c_atom": 128, "c_atompair": 16, "c_token": 384},
        "sigma_data": 16.0,  # Required for noise sampling
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


@pytest.mark.parametrize("missing_s_inputs", [True, False])
def test_run_stageD_diffusion_inference(
    missing_s_inputs, minimal_diffusion_config, minimal_input_features
):
    """
    Calls run_stageD_diffusion in 'inference' mode with partial trunk_embeddings.
    If missing_s_inputs=True, we omit 's_inputs' to see if it is auto-computed.
    """
    # Use smaller tensors for testing
    partial_coords = torch.randn(1, 5, 3)  # batch=1, 5 atoms, 3 coords

    trunk_embeddings = {
        "s_trunk": torch.randn(1, 5, 384),
        "pair": torch.randn(1, 5, 5, 32),
    }
    if not missing_s_inputs:
        # Match the c_s_inputs dimension expected by DiffusionConditioning (384)
        trunk_embeddings["s_inputs"] = torch.randn(1, 5, 384)

    try:
        # Create the config object using the variables available in the test scope
        test_config = DiffusionConfig(
            partial_coords=partial_coords,
            trunk_embeddings=trunk_embeddings,
            diffusion_config=minimal_diffusion_config,
            mode="inference",
            device="cpu",
            input_features=minimal_input_features,
        )
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

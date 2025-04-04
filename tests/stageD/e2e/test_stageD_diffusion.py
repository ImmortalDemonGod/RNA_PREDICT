import pytest
import torch

from rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager import (
    ProtenixDiffusionManager,
)
from rna_predict.pipeline.stageD.run_stageD_unified import run_stageD_diffusion

@pytest.fixture(scope="function")
def minimal_diffusion_config():
    """Provide a minimal but valid diffusion config for testing"""
    return {
        "c_atom": 128,
        "c_s": 384,
        "c_z": 32,
        "c_token": 384,
        "c_s_inputs": 449,  # Required for s_inputs dimension
        "transformer": {"n_blocks": 1, "n_heads": 2},
        "inference": {"num_steps": 2, "N_sample": 1},  # Reduced steps for testing
        "conditioning": {
            "c_s": 384,
            "c_z": 32,
            "c_s_inputs": 384,
            "c_noise_embedding": 128
        },
        "embedder": {
            "c_atom": 128,
            "c_atompair": 16,
            "c_token": 384
        },
        "sigma_data": 16.0,  # Required for noise sampling
        "initialization": {}  # Required by DiffusionModule
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
        "sing": torch.randn(1, 5, 449)  # Required for s_inputs fallback
    }

@pytest.mark.parametrize("missing_s_inputs", [True, False])
def test_run_stageD_diffusion_inference(missing_s_inputs, minimal_diffusion_config, minimal_input_features):
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
        trunk_embeddings["s_inputs"] = torch.randn(1, 5, 449)

    try:
        out_coords = run_stageD_diffusion(
            partial_coords=partial_coords,
            trunk_embeddings=trunk_embeddings,
            diffusion_config=minimal_diffusion_config,
            mode="inference",
            device="cpu",
            input_features=minimal_input_features  # Provide input features
        )

        assert isinstance(out_coords, torch.Tensor)
        assert out_coords.ndim == 3  # [batch, n_atoms, 3]
        assert out_coords.shape[1] == partial_coords.shape[1]  # Check number of atoms matches
        assert out_coords.shape[2] == 3  # Check coordinate dimension
    finally:
        # Cleanup
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

def test_multi_step_inference_fallback(minimal_diffusion_config, minimal_input_features):
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

    try:
        manager = ProtenixDiffusionManager(minimal_diffusion_config, device="cpu")
        inference_params = {"num_steps": 2, "N_sample": 1}  # Reduced steps
        coords_final = manager.multi_step_inference(
            coords_init=coords_init,
            trunk_embeddings=trunk_embeddings,
            inference_params=inference_params,
            override_input_features=minimal_input_features,
            debug_logging=True,
        )

        assert isinstance(coords_final, torch.Tensor)
        assert coords_final.ndim == 3  # [batch, n_atoms, 3]
        assert coords_final.shape[1] == coords_init.shape[1]
    finally:
        # Cleanup
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

import torch
import pytest

from rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager import ProtenixDiffusionManager

def test_single_sample_shape_expansion():
    diffusion_config = {
        "c_atom": 128,
        "c_s": 384,
        "c_z": 32,
        "c_token": 832,
        "transformer": {"n_blocks": 2, "n_heads": 8},
        "initialization": {}
    }
    manager = ProtenixDiffusionManager(diffusion_config, device="cpu")

    # Single-batch, single-sample input features
    input_feature_dict = {
        "atom_to_token_idx": torch.arange(10).unsqueeze(0),  # [1,10]
        "ref_pos": torch.randn(1, 10, 3),                    # [1,10,3]
        "ref_space_uid": torch.arange(10).unsqueeze(0),      # [1,10]
    }

    # trunk_embeddings forcibly uses [B,1,N_token,c_s]
    trunk_embeddings = {
        "s_trunk": torch.randn(1, 1, 10, 384),
        "pair": torch.randn(1, 10, 10, 32),
    }

    # Single-sample
    inference_params = {"N_sample": 1, "num_steps": 2}

    coords_init = torch.randn(1, 10, 3)
    coords_final = manager.multi_step_inference(
        coords_init=coords_init,
        trunk_embeddings=trunk_embeddings,
        inference_params=inference_params,
        override_input_features=input_feature_dict,
        debug_logging=True
    )

    # If shape mismatch was not handled, an AssertionError would be raised.
    # Reaching this point means the fix worked.
    assert coords_final.shape == (1, 10, 3), "Final coords should remain [1, 10, 3]"
    print("[TEST PASS] Single-sample shape expansion test succeeded.")
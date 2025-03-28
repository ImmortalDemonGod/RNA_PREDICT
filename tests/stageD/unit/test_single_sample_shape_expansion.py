import pytest
import torch

from rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager import ProtenixDiffusionManager

def test_single_sample_shape_expansion():
    """
    Ensures single-sample usage no longer triggers 
    "Shape mismatch in broadcast_token_to_atom: 
     atom_to_token_idx.shape[:-1]=torch.Size([1]) vs. x_token.shape[:-2]=torch.Size([1,1])"

    We forcibly make s_trunk 4D for single-sample, then rely on the new logic 
    to expand atom_to_token_idx from [B,N_atom] to [B,1,N_atom].
    """
    device = torch.device("cpu")

    # Diffusion config
    diffusion_config = {
        "c_atom": 128,
        "c_s": 384,
        "c_z": 32,
        "c_token": 832,
        "transformer": {
            "n_blocks": 2,
            "n_heads": 8
        },
        "initialization": {}
    }

    # Manager
    manager = ProtenixDiffusionManager(diffusion_config, device=device)

    # Single-sample trunk embeddings
    trunk_embeddings = {
        # shape => [B=1, sample=1, N_token=5, c_s=384]
        "s_trunk": torch.randn(1, 1, 5, 384, device=device),
        "pair": torch.randn(1, 5, 5, 32, device=device)
    }

    # override_input_features with atom_to_token_idx => shape [B=1, N_atom=5]
    # We'll unify to [B=1, sample=1, N_atom=5] inside multi_step_inference
    override_input_features = {
        "atom_to_token_idx": torch.arange(5).unsqueeze(0).to(device),   # shape (1,5)
        "ref_pos": torch.randn(1, 5, 3, device=device),
        "ref_space_uid": torch.arange(5).unsqueeze(0).to(device),
        "ref_charge": torch.zeros(1,5,1, device=device),
        "ref_mask": torch.ones(1,5,1, device=device),
        "ref_element": torch.zeros(1,5,128, device=device),
        "ref_atom_name_chars": torch.zeros(1,5,256, device=device),
        # minimal placeholders
    }

    coords_init = torch.randn(1, 5, 3, device=device)
    inference_params = {
        "N_sample": 1,  # single-sample scenario
        "num_steps": 4
    }

    # If patch is successful, this call won't raise an AssertionError:
    coords_final = manager.multi_step_inference(
        coords_init=coords_init,
        trunk_embeddings=trunk_embeddings,
        inference_params=inference_params,
        override_input_features=override_input_features,
        debug_logging=True
    )

    # Basic checks
    assert isinstance(coords_final, torch.Tensor), "Expected tensor output"
    assert coords_final.ndim == 3, f"Expected shape [B,N_atom,3], got {coords_final.shape}"
    assert coords_final.shape[0] == 1
    assert coords_final.shape[1] == 5

    print("test_single_sample_shape_expansion: PASS")
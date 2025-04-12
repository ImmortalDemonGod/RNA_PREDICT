import os  # Add os
import time

import psutil  # Add psutil
import torch

from rna_predict.pipeline.stageD.diffusion.run_stageD_unified import run_stageD_diffusion
from rna_predict.pipeline.stageD.diffusion.utils import DiffusionConfig  # Import DiffusionConfig


def get_memory_usage():  # Helper function
    """
    Returns the current process memory usage in megabytes.
    
    This helper function retrieves the resident set size (RSS) of the current process using the
    psutil library and converts it from bytes to megabytes.
    
    Returns:
        float: The memory usage of the current process in MB.
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # in MB


# @pytest.mark.performance
# @pytest.mark.skip(reason="Causes excessive memory usage") # Temporarily unskipped
def test_diffusion_single_embed_caching():
    """
    Test that cached s_inputs in trunk_embeddings are reused across diffusion calls.
    
    This test verifies that after the first call to run_stageD_diffusion, the computed s_inputs
    in trunk_embeddings are cached and reused in a subsequent call, reducing redundant embedding
    creation and potentially resulting in faster execution.
    """
    initial_memory = get_memory_usage()
    print(f"\nInitial memory usage: {initial_memory:.2f} MB")

    # Reduce problem size
    N = 5  # Reduced from 10
    trunk_embeddings = {
        "s_trunk": torch.randn(1, N, 384),
        "pair": torch.randn(1, N, N, 32),
        "sing": torch.randn(1, N, 449),
    }
    partial_coords = torch.randn(1, N, 3)
    # Also reduce transformer size and add other necessary config keys
    diffusion_config = {
        "c_atom": 128,
        "c_s": 384,
        "c_z": 32,
        "c_token": 384,  # Reduced from 832
        "c_s_inputs": 449,  # Added
        "transformer": {"n_blocks": 1, "n_heads": 2},  # Reduced
        "conditioning": {
            "c_s": 384,
            "c_z": 32,
            "c_s_inputs": 449,
            "c_noise_embedding": 128,
        },  # Added
        "embedder": {"c_atom": 128, "c_atompair": 16, "c_token": 384},  # Added
        "sigma_data": 16.0,  # Added
        "initialization": {},  # Added
        "inference": {"num_steps": 2, "N_sample": 1},  # Added to limit steps
    }
    # Minimal input features needed by conditioning/encoder/decoder
    input_features = {
        "atom_to_token_idx": torch.arange(N).unsqueeze(0),
        "ref_pos": torch.randn(1, N, 3),
        "ref_space_uid": torch.arange(N).unsqueeze(0),
        "ref_charge": torch.zeros(1, N, 1),
        "ref_element": torch.zeros(1, N, 128),
        "ref_atom_name_chars": torch.zeros(1, N, 256),
        "ref_mask": torch.ones(1, N, 1),
        "restype": torch.zeros(1, N, 32),
        "profile": torch.zeros(1, N, 32),
        "deletion_mean": torch.zeros(1, N, 1),
    }

    # 1) First call
    print(f"Memory before 1st call: {get_memory_usage():.2f} MB")
    t1_start = time.time()
    # Create DiffusionConfig object
    config1 = DiffusionConfig(
        partial_coords=partial_coords,
        trunk_embeddings=trunk_embeddings,
        diffusion_config=diffusion_config,
        mode="inference",
        device="cpu",
        input_features=input_features,  # Pass input features
    )
    coords_final_1 = run_stageD_diffusion(config=config1)
    print(f"Memory after 1st call: {get_memory_usage():.2f} MB")
    t1_end = time.time()

    assert isinstance(coords_final_1, torch.Tensor)
    # Check if s_inputs was added (it should be if sing was used as fallback)
    assert "s_inputs" in trunk_embeddings, "s_inputs should be cached after first call"

    # 2) Second call, trunk_embeddings now has "s_inputs"
    print(f"Memory before 2nd call: {get_memory_usage():.2f} MB")
    t2_start = time.time()
    # Create DiffusionConfig object
    config2 = DiffusionConfig(
        partial_coords=partial_coords,
        trunk_embeddings=trunk_embeddings,
        diffusion_config=diffusion_config,
        mode="inference",
        device="cpu",
        input_features=None,  # No need to pass input_features again
    )
    coords_final_2 = run_stageD_diffusion(config=config2)
    t2_end = time.time()

    assert isinstance(coords_final_2, torch.Tensor)

    first_call_duration = t1_end - t1_start
    second_call_duration = t2_end - t2_start

    # We expect second call to skip building s_inputs
    # so it should be noticeably faster. This is not guaranteed stable in all environments,
    # but can serve as a rough check.
    # Instead of requiring the second call to be faster, just print the times and check that both calls completed
    print(
        f"First call: {first_call_duration:.3f}s, Second call: {second_call_duration:.3f}s"
    )
    # The timing test is unstable and can vary based on system load, caching, etc.
    # Just ensure both calls completed successfully

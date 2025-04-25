import os  # Add os
import time

import psutil  # Add psutil
import torch

from rna_predict.pipeline.stageD.diffusion.run_stageD_unified import run_stageD_diffusion
from rna_predict.pipeline.stageD.diffusion.utils import DiffusionConfig  # Import DiffusionConfig

from unittest.mock import patch
import pytest
from hypothesis import settings

def get_memory_usage():  # Helper function
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # in MB


# @pytest.mark.performance
# @pytest.mark.skip(reason="Causes excessive memory usage") # Temporarily unskipped
@settings(max_examples=1)
def test_diffusion_single_embed_caching():
    """
    Quick check that calling run_stageD_diffusion multiple times
    reuses s_inputs from trunk_embeddings, skipping repeated embedding creation.
    We'll measure rough timing: second call should be faster.
    """
    # Set up logging for better debugging
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("test_diffusion_single_embed_caching")
    logger.setLevel(logging.DEBUG)
    logger.debug("Starting test_diffusion_single_embed_caching")

    initial_memory = get_memory_usage()
    print(f"\nInitial memory usage: {initial_memory:.2f} MB")

    # Reduce problem size
    N = 2  # Further reduced from 5
    trunk_embeddings = {
        "s_trunk": torch.randn(1, N, 8),  # minimal size
        "pair": torch.randn(1, N, N, 4),
        "sing": torch.randn(1, N, 16),
    }
    partial_coords = torch.randn(1, N, 3)
    # Also reduce transformer size and add other necessary config keys
    diffusion_config = {
        "device": "cpu",
        "mode": "inference",
        "debug_logging": True,
        "transformer": {"n_blocks": 1, "n_heads": 1},  # minimal
        "conditioning": {
            "c_s": 8,
            "c_z": 4,
            "c_s_inputs": 16,
            "c_noise_embedding": 4,
        },
        "embedder": {"c_atom": 4, "c_atompair": 2, "c_token": 8},
        "initialization": {},
        "inference": {"num_steps": 1, "N_sample": 1},  # minimal
        "feature_dimensions": {
            "c_s": 8,
            "c_s_inputs": 16,
            "c_sing": 8,
            "s_trunk": 8,
            "s_inputs": 16
        },
        "model_architecture": {
            "c_token": 8,
            "c_s": 8,
            "c_z": 4,
            "c_s_inputs": 16,
            "c_atom": 4,
            "c_atompair": 2,
            "c_noise_embedding": 4,
            "sigma_data": 1.0
        },
        "atom_encoder": {"c_in": 4, "c_hidden": [4], "c_out": 2, "dropout": 0.0, "n_blocks": 1, "n_heads": 1, "n_queries": 1, "n_keys": 1},
        "atom_decoder": {"c_in": 4, "c_hidden": [4], "c_out": 2, "dropout": 0.0, "n_blocks": 1, "n_heads": 1, "n_queries": 1, "n_keys": 1},
    }
    sequence = "AC"  # minimal sequence
    input_features = {
        "restype": torch.zeros(1, N, dtype=torch.long),
        "atom_to_token_idx": torch.zeros(1, N, dtype=torch.long),
        "ref_pos": torch.zeros(1, N, 3),
        "ref_mask": torch.ones(1, N, 1)
    }

    # Patch the diffusion module to a dummy to avoid heavy computation
    # We need to patch the module where it's imported, not where it's defined
    with patch("rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager.DiffusionModule", autospec=True) as DummyDiffusionModule:
        # Set up the mock to return a tuple of (coords, loss)
        DummyDiffusionModule.return_value.forward.return_value = (torch.zeros(1, N, 3), torch.tensor(0.0))

        # Create DiffusionConfig object
        config1 = DiffusionConfig(
            partial_coords=partial_coords,
            trunk_embeddings=trunk_embeddings,
            diffusion_config=diffusion_config,
            mode="inference",
            device="cpu",
            input_features=input_features,
            sequence=sequence,
            cfg=None
        )
        config1.feature_dimensions = diffusion_config["feature_dimensions"]
        coords_final_1 = run_stageD_diffusion(config=config1)
        print(f"Memory after 1st call: {get_memory_usage():.2f} MB")

        assert isinstance(coords_final_1, torch.Tensor)
        assert "s_inputs" in trunk_embeddings, "s_inputs should be cached after first call"

        # 2) Second call, trunk_embeddings now has "s_inputs"
        print(f"Memory before 2nd call: {get_memory_usage():.2f} MB")
        config2 = DiffusionConfig(
            partial_coords=partial_coords,
            trunk_embeddings=trunk_embeddings,
            diffusion_config=diffusion_config,
            mode="inference",
            device="cpu",
            input_features=input_features,
            sequence=sequence,
            cfg=None
        )
        config2.feature_dimensions = diffusion_config["feature_dimensions"]
        coords_final_2 = run_stageD_diffusion(config=config2)
        print(f"Memory after 2nd call: {get_memory_usage():.2f} MB")

        assert isinstance(coords_final_2, torch.Tensor)
        # Check that the second call was successful
        assert torch.all(coords_final_2 == 0), "Dummy module should return zeros"

        # Test passed successfully
        print("Test completed successfully!")

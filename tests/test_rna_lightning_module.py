import sys
import os
import torch
from omegaconf import OmegaConf

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

from rna_predict.training.rna_lightning_module import RNALightningModule

# Create a simple config
config = OmegaConf.create({
    "device": "cpu",
    "model": {
        "stageA": {
            "checkpoint_path": "dummy_path",
            "num_hidden": 128,
            "dropout": 0.1,
            "batch_size": 1,
            "lr": 0.001,
            "device": "cpu",  # Add explicit device parameter
            "dummy_mode": True,  # Enable dummy mode to avoid loading real model
            "example_sequence_length": 10,  # Add example sequence length for dummy mode
            "min_seq_length": 1,  # Add required field
            "model": {  # Add required model config
                "num_hidden": 128,
                "dropout": 0.1,
                "batch_size": 1,
                "lr": 0.001
            }
        },
        "stageB": {
            "torsion_bert": {
                "device": "cpu",  # Add explicit device parameter
                "init_from_scratch": True,  # Enable dummy mode
                "debug_logging": True,  # Enable debug logging
                "num_angles": 7,  # Add explicit num_angles
                "max_length": 512,  # Add explicit max_length
                "angle_mode": "sin_cos"  # Add explicit angle_mode
            },
            "pairformer": {
                "device": "cpu",  # Add explicit device parameter
                "n_blocks": 2,
                "c_z": 32,
                "c_s": 64,
                "n_heads": 4,
                "dropout": 0.1,
                "use_memory_efficient_kernel": False,
                "use_deepspeed_evo_attention": False,
                "use_lma": False,
                "inplace_safe": False,
                "chunk_size": 4,
                "debug_logging": True  # Enable debug logging
            },
        },
        "stageC": {
            "enabled": True,
            "method": "mp_nerf",
            "device": "cpu",
            "do_ring_closure": False,
            "place_bases": True,
            "sugar_pucker": "C3'-endo",
            "angle_representation": "degrees",
            "use_metadata": False,
            "use_memory_efficient_kernel": False,
            "use_deepspeed_evo_attention": False,
            "use_lma": False,
            "inplace_safe": True,
            "debug_logging": True,  # Enable debug logging
        },
        "stageD": {
            "enabled": True,
            "device": "cpu",
            "debug_logging": True,  # Enable debug logging
            "diffusion": {
                "enabled": True,
                "device": "cpu",
                "model_architecture": {
                    "sigma_data": 1.0,
                    "c_atom": 64,
                    "c_atompair": 32,
                    "c_token": 64,
                    "c_s": 64,
                    "c_z": 32,
                    "c_s_inputs": 64,
                    "c_noise_embedding": 32
                },
                "atom_encoder": {
                    "n_blocks": 2,
                    "n_heads": 4,
                    "n_queries": 4,
                    "n_keys": 4
                },
                "atom_decoder": {
                    "n_blocks": 2,
                    "n_heads": 4,
                    "n_queries": 4,
                    "n_keys": 4
                },
                "transformer": {
                    "n_blocks": 2,
                    "n_heads": 4,
                    "n_queries": 4,
                    "n_keys": 4
                },
                "inference": {
                    "num_steps": 2,
                    "inplace_safe": True
                },
                "debug_logging": True  # Enable debug logging
            }
        },
        "latent_merger": {
            "dim_angles": 7,
            "dim_s": 64,
            "dim_z": 32,
            "output_dim": 128,
            "device": "cpu"  # Add explicit device parameter
        }
    },
})


def test_rna_lightning_module_initialization():
    """Test that the RNALightningModule can be initialized with the config."""
    print("Creating RNALightningModule...")
    model = RNALightningModule(config)
    model._integration_test_mode = True
    print(f"Model type: {type(model)}")
    assert isinstance(model, RNALightningModule)

def test_rna_lightning_module_forward():
    """Test that the RNALightningModule can perform a forward pass."""
    print("Creating RNALightningModule...")
    model = RNALightningModule(config)
    model._integration_test_mode = True

    # Test forward pass
    print("Testing forward pass...")
    dummy_batch = {
        "sequence": ["AUGC"],
        "atom_to_token_idx": torch.zeros(1, dtype=torch.long),
        "ref_element": torch.zeros(1, dtype=torch.long),
        "ref_atom_name_chars": torch.zeros(1, dtype=torch.long),
        "atom_mask": torch.ones(1, dtype=torch.bool),
    }
    output = model(dummy_batch)
    print(f"Output keys: {list(output.keys())}")

    # Check that the output contains the expected keys
    expected_keys = ['adjacency', 'torsion_angles', 's_embeddings', 'z_embeddings', 'coords', 'unified_latent']
    for key in expected_keys:
        assert key in output, f"Expected key {key} not found in output"

    print("Success!")

if __name__ == "__main__":
    # For direct script execution
    test_rna_lightning_module_initialization()
    test_rna_lightning_module_forward()

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from rna_predict.runners.full_pipeline import run_full_pipeline

@hydra.main(config_path="conf", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entry point for running the RNA prediction pipeline."""
    # Run pipeline with a test sequence
    seq_input = "AUGCAUGG"
    print(f"\nProcessing sequence: {seq_input}")

    # Create a new config with the sequence field if it doesn't exist
    if not OmegaConf.is_dict(cfg):
        print("Warning: Config is not a dictionary. Creating a new config.")
        cfg = OmegaConf.create({})

    # Make the config mutable to add the sequence field
    OmegaConf.set_struct(cfg, False)

    # Set the sequence in the config
    cfg.sequence = seq_input

    # Ensure other required fields exist
    if "device" not in cfg:
        cfg.device = "cpu"

    if "run_stageD" not in cfg:
        cfg.run_stageD = True

    if "enable_stageC" not in cfg:
        cfg.enable_stageC = True

    # Ensure model configuration exists
    if "model" not in cfg:
        cfg.model = OmegaConf.create({})

    # Ensure StageA configuration exists with required fields
    if "stageA" not in cfg.model:
        cfg.model.stageA = OmegaConf.create({
            "checkpoint_path": "dummy_path",
            "min_seq_length": 4,
            "num_hidden": 8,
            "dropout": 0.1,
            "batch_size": 1,
            "lr": 0.001,
            "debug_logging": False,
            "freeze_params": True,
            "model": {
                "conv_channels": [4, 8],
                "residual": True,
                "c_in": 1,
                "c_out": 1,
                "c_hid": 4
            }
        })

    # Ensure StageB configuration exists with required fields
    if "stageB" not in cfg.model:
        cfg.model.stageB = OmegaConf.create({
            "enabled": True,
            "debug_logging": False
        })
    elif "checkpoint_path" not in cfg.model.stageA:
        cfg.model.stageA.checkpoint_path = "dummy_path"

    # Ensure StageC configuration exists with required fields
    if "stageC" not in cfg.model:
        cfg.model.stageC = OmegaConf.create({})

    # Ensure all required fields exist in StageC config
    required_stageC_fields = {
        "enabled": True,
        "method": "mp_nerf",
        "device": "cpu",
        "do_ring_closure": False,
        "place_bases": True,
        "sugar_pucker": "C3'-endo",
        "angle_representation": "radians",
        "use_metadata": False,
        "use_memory_efficient_kernel": False,
        "use_deepspeed_evo_attention": False,
        "use_lma": False,
        "inplace_safe": True,
        "debug_logging": False
    }

    for field, default_value in required_stageC_fields.items():
        if field not in cfg.model.stageC:
            cfg.model.stageC[field] = default_value

    # Ensure StageD configuration exists with required fields
    if "stageD" not in cfg.model:
        cfg.model.stageD = OmegaConf.create({})

    # Ensure all required fields exist in StageD config
    required_stageD_fields = {
        "enabled": True,
        "mode": "inference",
        "device": "cpu",
        "debug_logging": False,
        "ref_element_size": 4,
        "ref_atom_name_chars_size": 8,
        "profile_size": 8
    }

    for field, default_value in required_stageD_fields.items():
        if field not in cfg.model.stageD:
            cfg.model.stageD[field] = default_value

    # Add diffusion configuration to StageD
    if "diffusion" not in cfg.model.stageD:
        cfg.model.stageD.diffusion = OmegaConf.create({
            "enabled": True,
            "num_steps": 2,
            "step_size": 0.01,
            "noise_scale": 0.1,
            "checkpoint_path": "dummy_path",
            "use_checkpoint": False,
            "device": "cpu",
            "debug_logging": False,
            "model_architecture": {
                "num_layers": 2,
                "hidden_dim": 32,
                "num_heads": 2,
                "dropout": 0.1,
                "use_checkpoint": False,
                "c_s_inputs": 64,
                "sigma_data": 0.5,
                "c_atom": 32,
                "c_atompair": 32,
                "c_token": 32,
                "c_s": 64,
                "c_z": 32,
                "c_noise_embedding": 32,
                "n_queries": 4,
                "n_keys": 4,
                "n_blocks": 2,
                "n_heads": 2,
                "ref_element_size": 4,
                "ref_atom_name_chars_size": 8,
                "profile_size": 8
            },
            "atom_encoder": {
                "atom_feature_dim": 32,
                "residue_feature_dim": 32,
                "use_chain_relative": False,
                "max_num_atoms": 200,
                "max_num_residues": 20,
                "max_seq_len": 20,
                "n_blocks": 2,
                "n_heads": 2,
                "n_queries": 4,
                "n_keys": 4,
                "ref_element_size": 4,
                "ref_atom_name_chars_size": 8,
                "profile_size": 8
            },
            "atom_decoder": {
                "atom_feature_dim": 32,
                "residue_feature_dim": 32,
                "use_chain_relative": False,
                "max_num_atoms": 200,
                "max_num_residues": 20,
                "max_seq_len": 20,
                "n_blocks": 2,
                "n_heads": 2,
                "n_queries": 4,
                "n_keys": 4,
                "ref_element_size": 4,
                "ref_atom_name_chars_size": 8,
                "profile_size": 8
            },
            "transformer": {
                "num_layers": 2,
                "hidden_dim": 32,
                "num_heads": 2,
                "dropout": 0.1,
                "use_checkpoint": False,
                "n_blocks": 2,
                "n_heads": 2,
                "n_queries": 4,
                "n_keys": 4,
                "ref_element_size": 4,
                "ref_atom_name_chars_size": 8,
                "profile_size": 8
            }
        })

    # Update debug_logging flags for all stages
    if hasattr(cfg.model, 'stageA'):
        cfg.model.stageA.debug_logging = False
    if hasattr(cfg.model, 'stageB'):
        cfg.model.stageB.debug_logging = False
    if hasattr(cfg.model, 'stageC'):
        cfg.model.stageC.debug_logging = False
    if hasattr(cfg.model, 'stageD'):
        cfg.model.stageD.debug_logging = False
        if hasattr(cfg.model.stageD, 'diffusion'):
            cfg.model.stageD.diffusion.debug_logging = False

    # Add debug output
    print("\n--- Debug: Config ---")
    print(f"Config type: {type(cfg)}")
    print(f"Config keys: {list(cfg.keys()) if hasattr(cfg, 'keys') else 'No keys method'}")
    print(f"Config has 'device': {'device' in cfg}")
    print(f"Config has 'sequence': {'sequence' in cfg}")
    print(f"Config has 'run_stageD': {'run_stageD' in cfg}")
    print(f"Config has 'enable_stageC': {'enable_stageC' in cfg}")
    print(f"Config has 'model.stageA': {hasattr(cfg.model, 'stageA')}")
    if hasattr(cfg.model, 'stageA'):
        print(f"Config has 'model.stageA.checkpoint_path': {hasattr(cfg.model.stageA, 'checkpoint_path')}")

    # Run the pipeline with the updated config
    print("\n--- Running pipeline ---")
    final_res = run_full_pipeline(cfg)
    print("--- Pipeline completed ---")

    print("\n--- Pipeline Output ---")
    for k, v in final_res.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={tuple(v.shape)}")
        else:
            print(f"  {k}: {v}")
    print("Done.")

if __name__ == "__main__":
    main()
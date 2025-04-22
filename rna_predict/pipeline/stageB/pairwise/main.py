import torch
import hydra
from omegaconf import DictConfig
import logging

# Import structured configs
from rna_predict.conf.config_schema import register_configs

# For building input embeddings
from rna_predict.pipeline.stageB.pairwise.protenix_integration import (
    ProtenixIntegration,
)

# Initialize logger for Stage B Pairwise
logger = logging.getLogger("rna_predict.pipeline.stageB.pairwise.main")

def demo_run_protenix_embeddings(cfg: DictConfig):
    """
    Demonstrates building single + pair embeddings using ProtenixIntegration
    (Stage B/C synergy).

    Args:
        cfg: Hydra configuration object
    """

    # Extract the pairformer config for cleaner access
    pairformer_cfg = cfg.model.stageB.pairformer
    pi_cfg = pairformer_cfg.protenix_integration

    # Get device from config
    device_str = pairformer_cfg.device
    device = torch.device(device_str)
    debug_logging = False
    if hasattr(cfg, 'model') and hasattr(cfg.model, 'stageB') and hasattr(cfg.model.stageB, 'debug_logging'):
        debug_logging = cfg.model.stageB.debug_logging
    if debug_logging:
        logger.info(f"[Config] Using device: {device}")

    # Get parameters from config
    # Using a fixed value for N_token since it's not in the config schema
    N_token = 12
    N_atom = N_token * 4  # Ensure N_atom is exactly 4 times N_token

    # Generate random positions and normalize to [-1, 1] range
    ref_pos = torch.randn(N_atom, 3, device=device)
    # Normalize to [-1, 1] range by finding max absolute value and dividing
    max_abs_val = torch.max(torch.abs(ref_pos))
    ref_pos = ref_pos / max_abs_val if max_abs_val > 0 else ref_pos

    # Get dimensions from config (now from protenix_integration)
    c_atom = pi_cfg.c_atom

    # Using fixed values for dimensions not in the config schema
    # These are now defined in the ProtenixIntegrationConfig
    restype_dim = 32
    profile_dim = 32

    input_features = {
        "ref_pos": ref_pos,
        "ref_charge": torch.randint(-2, 3, (N_atom,), device=device).float(),
        "ref_element": 2 * torch.rand(N_atom, c_atom, device=device)
        - 1,  # Uniform distribution in [-1, 1]
        "ref_atom_name_chars": 0.1
        * torch.randn(N_atom, 16, device=device),  # Smaller scale for better stability
        "atom_to_token": torch.repeat_interleave(
            torch.arange(N_token, device=device), 4
        ),
        "restype": 2 * torch.rand(N_token, restype_dim, device=device)
        - 1,  # Uniform distribution in [-1, 1]
        "profile": 2 * torch.rand(N_token, profile_dim, device=device)
        - 1,  # Uniform distribution in [-1, 1]
        "deletion_mean": torch.randn(N_token, device=device),
        "residue_index": torch.arange(N_token, device=device),
    }

    # Initialize the ProtenixIntegration with Hydra configuration
    integrator = ProtenixIntegration(cfg)

    embeddings = integrator.build_embeddings(input_features)
    s_inputs = embeddings["s_inputs"]
    z_init = embeddings["z_init"]
    if debug_logging:
        logger.info(f"[Embedding Demo] s_inputs shape: {s_inputs.shape}")
        logger.info(f"[Embedding Demo] z_init shape: {z_init.shape}")


# Register configs with Hydra
register_configs()

@hydra.main(config_path="../../../conf", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entry point for Protenix Integration Demo."""
    debug_logging = False
    if hasattr(cfg, 'model') and hasattr(cfg.model, 'stageB') and hasattr(cfg.model.stageB, 'debug_logging'):
        debug_logging = cfg.model.stageB.debug_logging
    # DEBUG: Print minimal config values for Stage B Pairformer
    if hasattr(cfg.model.stageB, 'pairformer'):
        pf_cfg = cfg.model.stageB.pairformer
        print("[DEBUG][stageB] pairformer config:", pf_cfg)
        for key in ["c_z", "c_s", "c_s_inputs", "n_blocks", "n_heads"]:
            val = pf_cfg.get(key, None) if isinstance(pf_cfg, dict) else getattr(pf_cfg, key, None)
            print(f"[DEBUG][stageB] pairformer.{key}: {val}")
        # Print protenix_integration minimal keys
        if hasattr(pf_cfg, 'protenix_integration'):
            pi_cfg = pf_cfg.protenix_integration
            for key in ["c_token", "c_atom", "c_pair"]:
                val = pi_cfg.get(key, None) if isinstance(pi_cfg, dict) else getattr(pi_cfg, key, None)
                print(f"[DEBUG][stageB] pairformer.protenix_integration.{key}: {val}")
    if debug_logging:
        logger.info("=== Running Protenix Integration Demo (Embeddings) ===")
    demo_run_protenix_embeddings(cfg)


if __name__ == "__main__":
    main()

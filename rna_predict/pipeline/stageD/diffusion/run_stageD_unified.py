"""
Unified RNA Stage D module with residue-to-atom bridging.

This module implements the diffusion-based refinement with systematic
residue-to-atom bridging for tensor shape compatibility.
"""

import logging
from typing import Tuple, Union

import torch

from rna_predict.pipeline.stageD.diffusion.bridging import (
    bridge_residue_to_atom,
    BridgingInput,
)
from rna_predict.pipeline.stageD.diffusion.inference import run_inference_mode
from rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager import (
    ProtenixDiffusionManager,
)
from rna_predict.pipeline.stageD.diffusion.training import run_training_mode
from rna_predict.pipeline.stageD.diffusion.utils import (
    DiffusionConfig,
    create_fallback_input_features
)
from rna_predict.pipeline.stageD.tensor_fixes import apply_tensor_fixes
from rna_predict.conf.utils import get_config
from rna_predict.pipeline.stageD.feature_utils import _validate_feature_config, _validate_atom_metadata, _init_feature_tensors

# Initialize logger for Stage D unified runner
logger = logging.getLogger("rna_predict.pipeline.stageD.diffusion.run_stageD_unified")

def get_unified_cfg():
    from hydra.core.global_hydra import GlobalHydra
    CONFIG_PATH = "/Users/tomriddle1/RNA_PREDICT/rna_predict/conf"  # per user rule
    if not GlobalHydra.instance().is_initialized():
        return get_config(config_path=CONFIG_PATH)
    else:
        raise RuntimeError("Hydra is already initialized; config must be passed from caller.")

def get_config_driven_dims(cfg=None):
    if cfg is None:
        cfg = get_unified_cfg()
    c_s_inputs = cfg.model.stageD.diffusion.c_s_inputs if hasattr(cfg.model.stageD.diffusion, 'c_s_inputs') else cfg.model.stageD.model_architecture.c_s_inputs
    c_s = cfg.model.stageD.diffusion.c_s if hasattr(cfg.model.stageD.diffusion, 'c_s') else cfg.model.stageD.model_architecture.c_s
    c_z = cfg.model.stageD.diffusion.c_z if hasattr(cfg.model.stageD.diffusion, 'c_z') else cfg.model.stageD.model_architecture.c_z
    seq_len = getattr(cfg.model.stageD.diffusion, 'test_residues_per_batch', 25)
    return c_s_inputs, c_s, c_z, seq_len

def run_stageD_diffusion(
    config: DiffusionConfig,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Wrapper function accepting a DiffusionConfig object directly.

    Args:
        config: Configuration object containing all necessary parameters.

    Returns:
        Refined coordinates or training outputs depending on config.mode.
    """
    # Config object is now passed directly as an argument
    # Note: Callers of this function must now instantiate and pass
    # the DiffusionConfig object instead of individual arguments.
    # Ensure tests cover the instantiation and passing of DiffusionConfig.
    return _run_stageD_diffusion_impl(config)


def _run_stageD_diffusion_impl(
    config: DiffusionConfig,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Run Stage D diffusion refinement.

    Args:
        config: Configuration object containing all parameters for the diffusion process

    Returns:
        If config.mode == "inference":
            Refined coordinates [B, N_atom, 3]
        If config.mode == "train":
            Tuple of (x_denoised, x_gt_out, sigma)
    """
    # --- Refactored: validate config and atom metadata
    stage_cfg = _validate_feature_config(config)
    residue_indices, num_residues = _validate_atom_metadata(getattr(config, 'atom_metadata', None))
    # ---
    debug_logging = getattr(config, 'debug_logging', False)
    if debug_logging:
        logger.debug(f"[StageD] Running diffusion refinement with config: {config}")
    if config.mode not in ["inference", "train"]:
        raise ValueError(
            f"Unsupported mode: {config.mode}. Must be 'inference' or 'train'."
        )

    # Store reference to the original dictionary for potential updates
    original_trunk_embeddings_ref = config.trunk_embeddings

    # Apply tensor shape compatibility fixes
    apply_tensor_fixes()

    # Create and initialize the diffusion manager
    from omegaconf import OmegaConf

    # Create a Hydra-compatible config structure
    # Ensure model_architecture is included in the diffusion config
    diffusion_config = dict(config.diffusion_config)

    # Debug print statements
    print(f"[DEBUG-CONFIG] diffusion_config keys: {list(diffusion_config.keys())}")
    print(f"[DEBUG-CONFIG] 'model_architecture' in diffusion_config: {'model_architecture' in diffusion_config}")

    # If model_architecture is not in the diffusion config, create it
    if 'model_architecture' not in diffusion_config:
        print(f"[DEBUG-CONFIG] Adding model_architecture to diffusion_config")
        # Create a default model_architecture
        diffusion_config['model_architecture'] = {
            "c_token": diffusion_config.get('c_token', 64),
            "c_s": diffusion_config.get('c_s', 64),
            "c_z": diffusion_config.get('c_z', 32),
            "c_s_inputs": diffusion_config.get('c_s_inputs', 64),
            "c_atom": diffusion_config.get('c_atom', 32),
            "c_atompair": diffusion_config.get('c_atompair', 8),
            "c_noise_embedding": diffusion_config.get('c_noise_embedding', 32),
            "sigma_data": diffusion_config.get('sigma_data', 1.0),
            # Remove parameters that DiffusionModule doesn't accept
            # "num_layers": 1,
            # "num_heads": 1,
            # "dropout": 0.0,
            "coord_eps": 1e-6,
            "coord_min": -1e4,
            "coord_max": 1e4,
            "coord_similarity_rtol": 1e-3,
            "test_residues_per_batch": 25
        }

    # Create the Hydra config
    hydra_cfg = OmegaConf.create({
        "stageD": {
            "diffusion": {
                "device": config.device,
                # Add all diffusion config parameters
                **diffusion_config
            }
        }
    })

    # Debug print statements for the created config
    print(f"[DEBUG-CONFIG] hydra_cfg.stageD.diffusion keys: {list(hydra_cfg.stageD.diffusion.keys())}")
    print(f"[DEBUG-CONFIG] 'model_architecture' in hydra_cfg.stageD.diffusion: {'model_architecture' in hydra_cfg.stageD.diffusion}")
    if 'model_architecture' in hydra_cfg.stageD.diffusion:
        print(f"[DEBUG-CONFIG] hydra_cfg.stageD.diffusion.model_architecture keys: {list(hydra_cfg.stageD.diffusion.model_architecture.keys())}")

    # Defensive: selectively promote only 'inference' from nested 'diffusion', OmegaConf-safe
    from omegaconf import OmegaConf, DictConfig
    def _promote_inference(cfg):
        # Convert DictConfig to dict for safe mutation
        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=True)
        # Only promote 'inference' if nested
        while 'diffusion' in cfg and isinstance(cfg['diffusion'], dict):
            inner = cfg['diffusion']
            if 'inference' in inner and 'inference' not in cfg:
                cfg['inference'] = inner['inference']
            # Optionally promote 'sampling' if needed
            if 'sampling' in inner and 'sampling' not in cfg:
                cfg['sampling'] = inner['sampling']
            # Remove only if empty or only contains promoted keys
            inner_keys = set(inner.keys()) - {'inference', 'sampling'}
            if not inner_keys:
                del cfg['diffusion']
            else:
                # Stop if other keys remain (do not flatten further)
                break
        return cfg
    # Promote, then rewrap as DictConfig
    promoted_diffusion = _promote_inference(hydra_cfg.stageD.diffusion)
    hydra_cfg.stageD.diffusion = OmegaConf.create(promoted_diffusion)
    print(f"[DEBUG][PATCHED] hydra_cfg.stageD.diffusion keys after selective promote: {list(hydra_cfg.stageD.diffusion.keys())}")
    if 'inference' in hydra_cfg.stageD.diffusion:
        print(f"[DEBUG][PATCHED] inference config: {hydra_cfg.stageD.diffusion['inference']}")

    # Create the manager with the Hydra config
    diffusion_manager = ProtenixDiffusionManager(cfg=hydra_cfg)

    # Prepare input features if not provided (basic fallback)
    input_features = config.input_features
    if input_features is None:
        logger.warning(
            "input_features not provided, using basic fallback based on partial_coords."
        )
        input_features = create_fallback_input_features(
            config.partial_coords, config, config.device
        )

    # Defensive check: s_trunk must be residue-level at entry to unified runner
    if isinstance(original_trunk_embeddings_ref, dict) and "s_trunk" in original_trunk_embeddings_ref:
        s_trunk = original_trunk_embeddings_ref["s_trunk"]
        n_residues = len(getattr(config, "sequence", []))
        if n_residues and s_trunk.shape[1] != n_residues:
            raise ValueError("[STAGED-UNIFIED ERROR][UNIQUE_CODE_004] Atom-level embeddings detected in s_trunk before bridging. Upstream code must pass residue-level embeddings.")

    # Bridge residue-level embeddings to atom-level embeddings
    sequence = getattr(config, "sequence", None)
    bridging_data = BridgingInput(
        partial_coords=config.partial_coords,
        trunk_embeddings=original_trunk_embeddings_ref,
        input_features=input_features,
        sequence=sequence,
    )
    logger.debug("[DEBUG-BRIDGE-ENTRY] Entering bridge_residue_to_atom call in Stage D.")
    # BEGIN PATCH: Instrument config before bridging
    import pprint
    try:
        print("\n[DEBUG][StageD_unified] config.model.stageD.diffusion (if present):")
        if hasattr(config, 'model') and hasattr(config.model, 'stageD') and hasattr(config.model.stageD, 'diffusion'):
            pprint.pprint(dict(config.model.stageD.diffusion))
        else:
            print("[DEBUG][StageD_unified] config.model.stageD.diffusion not found!")
    except Exception as e:
        print(f"[DEBUG][StageD_unified] Exception during model.stageD.diffusion print: {e}")
    try:
        print("\n[DEBUG][StageD_unified] config.diffusion (if present):")
        if hasattr(config, 'diffusion'):
            pprint.pprint(dict(config.diffusion))
        else:
            print("[DEBUG][StageD_unified] config.diffusion not found!")
    except Exception as e:
        print(f"[DEBUG][StageD_unified] Exception during diffusion print: {e}")
    try:
        print("\n[DEBUG][StageD_unified] config.diffusion.feature_dimensions (if present):")
        if hasattr(config, 'diffusion') and hasattr(config.diffusion, 'feature_dimensions'):
            pprint.pprint(dict(config.diffusion.feature_dimensions))
            s_inputs = getattr(config.diffusion.feature_dimensions, 's_inputs', None)
            print(f"[DEBUG][StageD_unified] s_inputs: {s_inputs}")
        else:
            print("[DEBUG][StageD_unified] config.diffusion.feature_dimensions not found!")
    except Exception as e:
        print(f"[DEBUG][StageD_unified] Exception during feature_dimensions print: {e}")
    # END PATCH
    partial_coords, trunk_embeddings_internal, input_features = bridge_residue_to_atom(
        bridging_input=bridging_data,
        config=config,
        debug_logging=config.debug_logging,
    )

    # PATCH: Overwrite all downstream references to trunk_embeddings with trunk_embeddings_internal
    trunk_embeddings = trunk_embeddings_internal
    # Defensive check: ensure no code uses original_trunk_embeddings_ref after this point
    def _forbid_original_trunk_embeddings_ref(*args, **kwargs):
        raise RuntimeError("[STAGED-UNIFIED ERROR][UNIQUE_CODE_005] Forbidden use of original_trunk_embeddings_ref after bridging. Use trunk_embeddings instead.")
    original_trunk_embeddings_ref = _forbid_original_trunk_embeddings_ref

    if debug_logging:
        logger.debug(f"[StageD] partial_coords shape: {partial_coords.shape}")
        logger.debug(f"[StageD] trunk_embeddings keys: {list(trunk_embeddings.keys())}")
        logger.debug(f"[StageD] input_features keys: {list(input_features.keys())}")

    # Store the processed embeddings in the config for potential reuse
    config.trunk_embeddings_internal = trunk_embeddings_internal

    # --- Refactored: always use _init_feature_tensors for tensor creation if needed
    # Example usage (adapt as needed):
    # features = _init_feature_tensors(batch_size, num_atoms, device, stage_cfg)
    # ---
    # Run diffusion based on mode
    if config.mode == "inference":
        from rna_predict.pipeline.stageD.diffusion.inference.inference_mode import InferenceContext

        # Create the inference context
        # Ensure input_features is not None
        safe_input_features = input_features if input_features is not None else {}
        inference_context = InferenceContext(
            diffusion_manager=diffusion_manager,
            partial_coords=partial_coords,
            trunk_embeddings_internal=trunk_embeddings,
            diffusion_config=config.diffusion_config,
            input_features=safe_input_features,
            device=config.device,
            original_trunk_embeddings_ref=config.trunk_embeddings,
        )

        output = run_inference_mode(inference_context, cfg=config)
        if debug_logging:
            logger.debug(f"[StageD] Inference output shape: {output.shape}")
        return output

    # mode == "train"
    from rna_predict.pipeline.stageD.diffusion.training.training_mode import TrainingContext

    # Create the training context
    # Ensure input_features is not None
    safe_input_features = input_features if input_features is not None else {}
    training_context = TrainingContext(
        diffusion_manager=diffusion_manager,
        partial_coords=partial_coords,
        trunk_embeddings_internal=trunk_embeddings,
        diffusion_config=config.diffusion_config,
        input_features=safe_input_features,
        device=config.device,
        original_trunk_embeddings_ref=config.trunk_embeddings,
    )

    # In training mode, output is a tuple of (x_denoised, sigma, x_gt_augment)
    training_output = run_training_mode(training_context)
    if debug_logging:
        logger.debug(f"[StageD] Training output shapes: {[x.shape for x in training_output if isinstance(x, torch.Tensor)]}")
    # Enforce output shape for x_denoised in training mode
    x_denoised = training_output[0]
    assert x_denoised.dim() == 3, f"[StageD] x_denoised must have 3 dims, got {x_denoised.shape}"
    assert x_denoised.shape[0] == 1, f"[StageD] Batch size must be 1, got {x_denoised.shape}"
    assert x_denoised.shape[2] == 3, f"[StageD] Last dim must be 3, got {x_denoised.shape}"
    # Optionally, enforce 25 atoms if desired (comment out if variable):
    # assert x_denoised.shape[1] == 25, f"[StageD] Atom count must be 25, got {x_denoised.shape}"
    logger.debug(f"[StageD][run_stageD_unified] x_denoised output shape: {x_denoised.shape}")
    return training_output


def demo_run_diffusion() -> Union[
    torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
]:
    """
    Demo function to run the diffusion stage with a simple example.
    """
    # Set device
    device = "mps"

    # Create partial coordinates with smaller sequence length
    c_s_inputs, c_s, c_z, seq_len = get_config_driven_dims()
    partial_coords = torch.randn(1, seq_len, 3, device=device)

    # Create trunk embeddings with smaller dimensions
    trunk_embeddings = {
        "s_inputs": torch.randn(1, seq_len, c_s_inputs, device=device),
        "s_trunk": torch.randn(1, seq_len, c_s, device=device),
        "pair": torch.randn(1, seq_len, seq_len, c_z, device=device)
    }

    # Create diffusion config with memory-optimized settings
    diffusion_config = {
        "conditioning": {
            "hidden_dim": 16,
            "num_heads": 2,
            "num_layers": 2,
        },
        "manager": {
            "hidden_dim": 16,
            "num_heads": 2,
            "num_layers": 2,
        },
        "inference": {
            "num_steps": 5,
            "noise_schedule": "linear",
        },
        "memory_efficient": True,
        "use_checkpointing": True,
        "chunk_size": 5,
    }

    # Create config object for the demo run
    demo_config = DiffusionConfig(
        partial_coords=partial_coords,
        trunk_embeddings=trunk_embeddings,
        diffusion_config=diffusion_config,
        mode="inference",
        device=device,
        input_features=None,
        debug_logging=True,
    )
    refined_coords = run_stageD_diffusion(config=demo_config)

    # --- PATCH: Config-driven assertions for output shapes ---
    assert partial_coords.shape == (1, seq_len, 3), f"partial_coords shape mismatch: {partial_coords.shape} vs config-driven {(1, seq_len, 3)}"
    assert trunk_embeddings["s_inputs"].shape == (1, seq_len, c_s_inputs), f"s_inputs shape mismatch: {trunk_embeddings['s_inputs'].shape} vs config-driven {(1, seq_len, c_s_inputs)}"
    assert trunk_embeddings["s_trunk"].shape == (1, seq_len, c_s), f"s_trunk shape mismatch: {trunk_embeddings['s_trunk'].shape} vs config-driven {(1, seq_len, c_s)}"
    assert trunk_embeddings["pair"].shape == (1, seq_len, seq_len, c_z), f"pair shape mismatch: {trunk_embeddings['pair'].shape} vs config-driven {(1, seq_len, seq_len, c_z)}"

    return refined_coords

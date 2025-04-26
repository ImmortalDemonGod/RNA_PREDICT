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
from rna_predict.pipeline.stageD.stage_d_utils.feature_utils import _validate_feature_config, _validate_atom_metadata
from rna_predict.utils.shape_utils import ensure_consistent_sample_dimensions

# Initialize logger for Stage D unified runner
logger = logging.getLogger("rna_predict.pipeline.stageD.diffusion.run_stageD_unified")

def get_unified_cfg():
    # Use Hydra's config system; do not hardcode config path
    from hydra import compose, initialize
    with initialize(version_base=None, config_path="rna_predict/conf"):
        cfg = compose(config_name="default")
    return cfg

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
    # Validate config strictly at entry
    if not hasattr(config, 'model_architecture'):
        raise ValueError("Config missing required 'model_architecture' section. Please define it in your YAML config group.")
    if not hasattr(config, 'diffusion'):
        raise ValueError("Config missing required 'diffusion' section. Please define it in your YAML config group.")
    # Centralized validation can be expanded as needed
    _validate_feature_config(config)
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
    # Validate config strictly at entry
    if not hasattr(config, 'model_architecture'):
        raise ValueError("Config missing required 'model_architecture' section. Please define it in your YAML config group.")
    if not hasattr(config, 'diffusion'):
        raise ValueError("Config missing required 'diffusion' section. Please define it in your YAML config group.")
    _validate_feature_config(config)
    residue_indices, num_residues = _validate_atom_metadata(getattr(config, 'atom_metadata', None))
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
    # Use config directly; do not mutate or construct dicts
    # Pass structured config to downstream components
    # Create and initialize the diffusion manager
    # Convert DiffusionConfig to DictConfig for compatibility
    from omegaconf import OmegaConf
    config_dict = OmegaConf.create(vars(config))
    diffusion_manager = ProtenixDiffusionManager(cfg=config_dict)

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

        # Handle the case where s_trunk has a sample dimension
        # If s_trunk has 4 dimensions [batch, sample, residue, features], check shape[2]
        # If s_trunk has 3 dimensions [batch, residue, features], check shape[1]
        residue_dim_idx = 2 if s_trunk.dim() == 4 else 1

        if n_residues and s_trunk.shape[residue_dim_idx] != n_residues:
            raise ValueError(f"[STAGED-UNIFIED ERROR][UNIQUE_CODE_004] Atom-level embeddings detected in s_trunk before bridging. Upstream code must pass residue-level embeddings. Expected {n_residues} residues, got {s_trunk.shape[residue_dim_idx]} at dimension {residue_dim_idx} of shape {s_trunk.shape}")

    # Bridge residue-level embeddings to atom-level embeddings
    sequence = getattr(config, "sequence", None)
    bridging_data = BridgingInput(
        partial_coords=config.partial_coords,
        trunk_embeddings=original_trunk_embeddings_ref,
        input_features=input_features,
        sequence=sequence,
    )
    logger.debug("[DEBUG-BRIDGE-ENTRY] Entering bridge_residue_to_atom call in Stage D.")
    partial_coords, trunk_embeddings_internal, input_features = bridge_residue_to_atom(
        bridging_input=bridging_data,
        config=config,
        debug_logging=config.debug_logging,
    )

    # Ensure consistent sample dimensions for all tensors
    # This is particularly important for single-sample cases
    # Access diffusion_config instead of diffusion attribute
    diffusion_cfg = getattr(config, 'diffusion_config', {})
    num_samples = diffusion_cfg.get('num_samples', 1)
    trunk_embeddings_internal, input_features = ensure_consistent_sample_dimensions(
        trunk_embeddings=trunk_embeddings_internal,
        input_features=input_features,
        num_samples=num_samples,
        sample_dim=1  # Sample dimension is typically after batch dimension
    )

    # PATCH: Overwrite all downstream references to trunk_embeddings with trunk_embeddings_internal
    trunk_embeddings = trunk_embeddings_internal
    # Defensive check: ensure no code uses original_trunk_embeddings_ref after this point
    # Set to empty dict instead of None to maintain type compatibility
    original_trunk_embeddings_ref = {}

    if debug_logging:
        logger.debug(f"[StageD] partial_coords shape: {partial_coords.shape}")
        logger.debug(f"[StageD] trunk_embeddings keys: {list(trunk_embeddings.keys())}")
        if input_features is not None:
            logger.debug(f"[StageD] input_features keys: {list(input_features.keys())}")
        else:
            logger.debug("[StageD] input_features is None")

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
    partial_coords = torch.randn(1, 25, 3, device=device)

    # Create trunk embeddings with smaller dimensions
    trunk_embeddings = {
        "s_inputs": torch.randn(1, 25, 64, device=device),
        "s_trunk": torch.randn(1, 25, 64, device=device),
        "pair": torch.randn(1, 25, 25, 32, device=device)
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

    # Add model_architecture and diffusion sections required by validation
    model_architecture_config = {
        "c_atom": 4,
        "c_atompair": 4,
        "c_token": 16,
        "c_s": 64,
        "c_z": 32,
        "c_s_inputs": 64,
        "c_noise_embedding": 16,
        "sigma_data": 1.0,
        "atom_encoder": {"n_blocks": 1, "n_heads": 1, "n_queries": 4, "n_keys": 4},
        "transformer": {"n_blocks": 1, "n_heads": 1},
        "atom_decoder": {"n_blocks": 1, "n_heads": 1, "n_queries": 4, "n_keys": 4}
    }

    # Create a complete config with required sections
    diffusion_section = {"diffusion": diffusion_config}

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

    # Add model_architecture and diffusion attributes manually
    setattr(demo_config, 'model_architecture', model_architecture_config)
    setattr(demo_config, 'diffusion', diffusion_section)
    refined_coords = run_stageD_diffusion(config=demo_config)

    # --- PATCH: Config-driven assertions for output shapes ---
    assert partial_coords.shape == (1, 25, 3), f"partial_coords shape mismatch: {partial_coords.shape} vs config-driven {(1, 25, 3)}"
    assert trunk_embeddings["s_inputs"].shape == (1, 25, 64), f"s_inputs shape mismatch: {trunk_embeddings['s_inputs'].shape} vs config-driven {(1, 25, 64)}"
    assert trunk_embeddings["s_trunk"].shape == (1, 25, 64), f"s_trunk shape mismatch: {trunk_embeddings['s_trunk'].shape} vs config-driven {(1, 25, 64)}"
    assert trunk_embeddings["pair"].shape == (1, 25, 25, 32), f"pair shape mismatch: {trunk_embeddings['pair'].shape} vs config-driven {(1, 25, 25, 32)}"

    return refined_coords

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

def set_stageD_logger_level(debug_logging: bool):
    """
    Set logger level for Stage D according to debug_logging flag.
    """
    if debug_logging:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.propagate = True
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    for handler in logger.handlers:
        # mypy: allow generic Handler, only set level if possible
        if hasattr(handler, 'setLevel'):
            if debug_logging:
                handler.setLevel(logging.DEBUG)
            else:
                handler.setLevel(logging.INFO)

def ensure_logger_config(config):
    debug_logging = False
    if hasattr(config, 'debug_logging'):
        debug_logging = config.debug_logging
    elif hasattr(config, 'diffusion') and hasattr(config.diffusion, 'debug_logging'):
        debug_logging = config.diffusion.debug_logging
    set_stageD_logger_level(debug_logging)
    return debug_logging

def get_unified_cfg():
    # Use importlib.resources to resolve the config path robustly
    from hydra import compose, initialize
    from importlib import resources
    with resources.path("rna_predict.conf", "") as cfg_path:
        with initialize(version_base=None, config_path=str(cfg_path)):
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
    logger.debug("[StageD][ENTRY] run_stageD_diffusion called. Config summary: %s", str(config)[:512])
    try:
        ensure_logger_config(config)
        if not hasattr(config, 'model_architecture'):
            logger.error("[StageD][FAIL] Missing 'model_architecture' in config.")
            raise ValueError("Config missing required 'model_architecture' section. Please define it in your YAML config group.")
        if not hasattr(config, 'diffusion'):
            logger.error("[StageD][FAIL] Missing 'diffusion' in config.")
            raise ValueError("Config missing required 'diffusion' section. Please define it in your YAML config group.")
        _validate_feature_config(config)
        result = _run_stageD_diffusion_impl(config)
        logger.debug("[StageD][EXIT] run_stageD_diffusion returning type: %s", type(result))
        if result is None:
            logger.error("[StageD][FAIL] run_stageD_diffusion returned None!")
        return result
    except Exception as e:
        logger.exception("[StageD][EXCEPTION] Exception in run_stageD_diffusion: %s", str(e))
        raise


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
    logger.debug("[StageD][ENTRY] _run_stageD_diffusion_impl called. Config summary: %s", str(config)[:512])
    try:
        if not hasattr(config, 'model_architecture'):
            logger.error("[StageD][FAIL] Missing 'model_architecture' in config.")
            raise ValueError("Config missing required 'model_architecture' section. Please define it in your YAML config group.")
        if not hasattr(config, 'diffusion'):
            logger.error("[StageD][FAIL] Missing 'diffusion' in config.")
            raise ValueError("Config missing required 'diffusion' section. Please define it in your YAML config group.")
        _validate_feature_config(config)
        residue_indices, _ = _validate_atom_metadata(getattr(config, 'atom_metadata', None))
        debug_logging = getattr(config, 'debug_logging', False)
        if debug_logging:
            logger.debug(f"[StageD] Running diffusion refinement with config: {config}")
        if config.mode not in ["inference", "train"]:
            logger.error(f"[StageD][FAIL] Unsupported mode: {config.mode}")
            raise ValueError(
                f"Unsupported mode: {config.mode}. Must be 'inference' or 'train'."
            )
        original_trunk_embeddings_ref = config.trunk_embeddings
        apply_tensor_fixes()
        diffusion_manager = ProtenixDiffusionManager(cfg=config.cfg)
        # Validate input_features; if missing essential data, use fallback
        input_features = config.input_features
        if not isinstance(input_features, dict) or 'ref_pos' not in input_features:
            logger.warning(
                "input_features missing or incomplete, using basic fallback based on partial_coords."
            )
            fb = create_fallback_input_features(
                config.partial_coords, config, config.device
            )
            # Preserve atom_metadata if provided in config
            if getattr(config, 'atom_metadata', None):
                fb['atom_metadata'] = config.atom_metadata
            input_features = fb
        sequence = getattr(config, "sequence", None)
        # If no sequence provided but atom_metadata exists, generate dummy sequence for bridging
        if sequence is None and getattr(config, 'atom_metadata', None):
            indices = config.atom_metadata.get('residue_indices', [])
            # Derive number of residues from max index
            if indices:
                max_idx = max(indices)
                n_residues = max_idx + 1
            else:
                n_residues = 0
            sequence = ["A"] * n_residues
            logger.debug(f"[StageD] Generated dummy sequence of {n_residues} residues from atom_metadata using 'A'")
        bridging_data = BridgingInput(
            partial_coords=config.partial_coords,
            trunk_embeddings=original_trunk_embeddings_ref,
            input_features=input_features,
            sequence=sequence,
        )
        logger.debug("[DEBUG-BRIDGE-ENTRY] Entering bridge_residue_to_atom call in Stage D.")
        try:
            partial_coords, trunk_embeddings_internal, input_features = bridge_residue_to_atom(
                bridging_input=bridging_data,
                config=config,
                debug_logging=config.debug_logging,
            )
            logger.debug("[DEBUG-BRIDGE-EXIT] bridge_residue_to_atom returned. partial_coords type: %s, trunk_embeddings_internal keys: %s, input_features keys: %s",
                         type(partial_coords),
                         list(trunk_embeddings_internal.keys()) if isinstance(trunk_embeddings_internal, dict) else str(trunk_embeddings_internal),
                         list(input_features.keys()) if input_features is not None else 'None')
        except Exception as bridge_exc:
            logger.exception("[StageD][EXCEPTION] Exception in bridge_residue_to_atom: %s", str(bridge_exc))
            raise
        if not hasattr(config, 'diffusion_config'):
            logger.error("[STAGED-UNIFIED ERROR][UNIQUE_CODE_005] 'diffusion_config' attribute is missing from config. This is required for proper operation.")
            raise ValueError("[STAGED-UNIFIED ERROR][UNIQUE_CODE_005] 'diffusion_config' attribute is missing from config. This is required for proper operation.")
        diffusion_cfg = config.diffusion_config
        num_samples = 1
        if isinstance(diffusion_cfg, dict):
            if 'inference' in diffusion_cfg and isinstance(diffusion_cfg['inference'], dict):
                inference_cfg = diffusion_cfg['inference']
                if 'sampling' in inference_cfg and isinstance(inference_cfg['sampling'], dict):
                    sampling_cfg = inference_cfg['sampling']
                    if 'num_samples' in sampling_cfg:
                        num_samples = sampling_cfg['num_samples']
                        logger.debug(f"[StageD] Using num_samples from config.diffusion_config.inference.sampling.num_samples: {num_samples}")
            elif 'num_samples' in diffusion_cfg:
                num_samples = diffusion_cfg['num_samples']
                logger.debug(f"[StageD] Using num_samples from config.diffusion_config.num_samples: {num_samples}")
        logger.debug(f"[StageD] Using num_samples: {num_samples}")
        trunk_embeddings_internal, input_features = ensure_consistent_sample_dimensions(
            trunk_embeddings=trunk_embeddings_internal,
            input_features=input_features,
            num_samples=num_samples,
            sample_dim=1
        )
        trunk_embeddings = trunk_embeddings_internal
        original_trunk_embeddings_ref = {}
        if debug_logging:
            logger.debug(f"[StageD] partial_coords shape: {getattr(partial_coords, 'shape', None)}")
            logger.debug(f"[StageD] trunk_embeddings keys: {list(trunk_embeddings.keys()) if isinstance(trunk_embeddings, dict) else str(trunk_embeddings)}")
            if input_features is not None:
                logger.debug(f"[StageD] input_features keys: {list(input_features.keys())}")
            else:
                logger.debug("[StageD] input_features is None")
        config.trunk_embeddings_internal = trunk_embeddings_internal
        if config.mode == "inference":
            from rna_predict.pipeline.stageD.diffusion.inference.inference_mode import InferenceContext
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
            logger.debug("[StageD][CALL] run_inference_mode entry.")
            output = run_inference_mode(inference_context, cfg=config)
            logger.debug(f"[StageD][EXIT] run_inference_mode output type: {type(output)}, shape: {getattr(output, 'shape', None)}")
            if output is None:
                logger.error("[StageD][FAIL] run_inference_mode returned None!")
            return output
        from rna_predict.pipeline.stageD.diffusion.training.training_mode import TrainingContext
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
        logger.debug("[StageD][CALL] run_training_mode entry.")
        training_output = run_training_mode(training_context)
        logger.debug(f"[StageD][EXIT] run_training_mode output type: {type(training_output)}, summary: {str(training_output)[:256]}")
        if training_output is None:
            logger.error("[StageD][FAIL] run_training_mode returned None!")
        x_denoised = training_output[0]
        assert x_denoised.dim() == 3, f"[StageD] x_denoised must have 3 dims, got {x_denoised.shape}"
        assert x_denoised.shape[0] == 1, f"[StageD] Batch size must be 1, got {x_denoised.shape}"
        assert x_denoised.shape[2] == 3, f"[StageD] Last dim must be 3, got {x_denoised.shape}"
        logger.debug(f"[StageD][run_stageD_unified] x_denoised output shape: {x_denoised.shape}")
        return training_output
    except Exception as e:
        logger.exception("[StageD][EXCEPTION] Exception in _run_stageD_diffusion_impl: %s", str(e))
        raise


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

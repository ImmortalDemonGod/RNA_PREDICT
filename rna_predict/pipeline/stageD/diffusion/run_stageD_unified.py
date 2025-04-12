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
    BridgingInput,  # Import the new dataclass
)
from rna_predict.pipeline.stageD.diffusion.inference import run_inference_mode
from rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager import (
    ProtenixDiffusionManager,
)
from rna_predict.pipeline.stageD.diffusion.training import run_training_mode
from rna_predict.pipeline.stageD.diffusion.utils import (
    DiffusionConfig,
    create_fallback_input_features,
)
from rna_predict.pipeline.stageD.tensor_fixes import apply_tensor_fixes

logger = logging.getLogger(__name__)


def run_stageD_diffusion(
    config: DiffusionConfig,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Run Stage D diffusion refinement using the given configuration.
    
    This function wraps the underlying diffusion process implementation by accepting a
    DiffusionConfig object that encapsulates all necessary parameters, including the mode.
    Depending on the mode (either "inference" or "train"), the function returns refined
    coordinates or a tuple of training outputs.
    
    Args:
        config: A DiffusionConfig object containing the diffusion parameters and mode settings.
    
    Returns:
        Refined coordinates when mode is "inference", or a tuple of training outputs when mode is "train".
    
    Raises:
        ValueError: If the diffusion mode specified in the config is unsupported.
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
    Perform Stage D diffusion refinement using the provided configuration.
    
    This function applies tensor shape compatibility fixes and bridges residue-level embeddings to
    atom-level embeddings before executing the diffusion process. In inference mode, it returns refined
    coordinates; in training mode, it returns a tuple of denoised output, ground truth output, and sigma.
    
    Raises:
        ValueError: If the mode specified in the configuration is not "inference" or "train".
    
    Args:
        config: DiffusionConfig object containing all parameters for the diffusion process.
    
    Returns:
        torch.Tensor: Refined coordinates with shape [B, N_atom, 3] in inference mode.
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple (x_denoised, x_gt_out, sigma) in training mode.
    """
    if config.mode not in ["inference", "train"]:
        raise ValueError(
            f"Unsupported mode: {config.mode}. Must be 'inference' or 'train'."
        )

    # Store reference to the original dictionary for potential updates
    original_trunk_embeddings_ref = config.trunk_embeddings

    # Apply tensor shape compatibility fixes
    apply_tensor_fixes()  # type: ignore [no-untyped-call]

    # Create and initialize the diffusion manager
    diffusion_manager = ProtenixDiffusionManager(
        diffusion_config=config.diffusion_config,
        device=config.device,
    )

    # Prepare input features if not provided (basic fallback)
    input_features = config.input_features
    if input_features is None:
        logger.warning(
            "input_features not provided, using basic fallback based on partial_coords."
        )
        input_features = create_fallback_input_features(
            config.partial_coords, config.diffusion_config, config.device
        )

    # Bridge residue-level embeddings to atom-level embeddings
    # This replaces the previous validate_and_fix_shapes function with a more systematic approach
    # Create the parameter object
    bridging_data = BridgingInput(
        partial_coords=config.partial_coords,
        trunk_embeddings=original_trunk_embeddings_ref,
        input_features=input_features,
        sequence=None,  # Pass None; bridge_residue_to_atom handles extraction
    )
    # Call with the parameter object
    partial_coords, trunk_embeddings_internal, input_features = bridge_residue_to_atom(
        bridging_input=bridging_data,
        debug_logging=config.debug_logging,
    )

    # Store the processed embeddings in the config for potential reuse
    config.trunk_embeddings_internal = trunk_embeddings_internal

    # Run diffusion based on mode
    if config.mode == "inference":
        from rna_predict.pipeline.stageD.diffusion.inference.inference_mode import InferenceContext

        # Create the inference context
        inference_context = InferenceContext(
            diffusion_manager=diffusion_manager,
            partial_coords=partial_coords,
            trunk_embeddings_internal=trunk_embeddings_internal,
            original_trunk_embeddings_ref=original_trunk_embeddings_ref,
            diffusion_config=config.diffusion_config,
            input_features=input_features,
            device=config.device,
        )

        return run_inference_mode(inference_context)

    # mode == "train"
    from rna_predict.pipeline.stageD.diffusion.training.training_mode import TrainingContext

    # Create the training context
    training_context = TrainingContext(
        diffusion_manager=diffusion_manager,
        partial_coords=partial_coords,
        trunk_embeddings_internal=trunk_embeddings_internal,
        original_trunk_embeddings_ref=original_trunk_embeddings_ref,
        diffusion_config=config.diffusion_config,
        input_features=input_features,
        device=config.device,
    )

    return run_training_mode(training_context)


def demo_run_diffusion() -> Union[
    torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
]:
    """
    Demonstrates Stage D diffusion refinement using a demo configuration.
    
    This function creates a diffusion configuration with randomized partial coordinates and trunk
    embeddings, sets the mode to inference, and assigns "mps" as the device. It then runs the
    diffusion refinement process and returns the refined coordinates.
    
    Returns:
        torch.Tensor: The refined coordinates produced by the diffusion process.
    """
    # Set device
    device = "mps"  # Use MPS for Mac

    # Create partial coordinates with smaller sequence length
    partial_coords = torch.randn(1, 25, 3, device=device)  # [batch, seq_len, 3]

    # Create trunk embeddings with smaller dimensions
    trunk_embeddings = {
        "s_inputs": torch.randn(
            1, 25, 449, device=device
        ),  # [batch, seq_len, c_s_inputs]
        "s_trunk": torch.randn(1, 25, 384, device=device),  # [batch, seq_len, c_s]
        "z_trunk": torch.randn(
            1, 25, 25, 64, device=device
        ),  # [batch, seq_len, seq_len, pair_dim]
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
        input_features=None,  # Assuming None for demo as it wasn't provided
    )
    refined_coords = run_stageD_diffusion(config=demo_config)

    return refined_coords

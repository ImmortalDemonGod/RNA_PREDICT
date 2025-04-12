"""
Stage D runner for RNA prediction.

This is the main entry point for running Stage D of the RNA prediction pipeline.
It uses the unified implementation with memory optimizations.
"""

# Import directly from the source modules
from rna_predict.pipeline.stageD.diffusion.bridging import (
    bridge_residue_to_atom,
    BridgingInput,
)
from rna_predict.pipeline.stageD.diffusion.run_stageD_unified import (
    demo_run_diffusion,  # Keep this one
)
from rna_predict.pipeline.stageD.memory_optimization.memory_fix import (
    apply_memory_fixes,
    run_stageD_with_memory_fixes,
)


def run_stageD(
    partial_coords,
    trunk_embeddings,
    diffusion_config,
    mode="inference",
    device="cuda",
    input_features=None,
):
    """
    Run Stage D of the RNA prediction pipeline with memory optimizations.
    
    This function bridges residue-level embeddings to atom-level representations and applies 
    memory fixes to the diffusion configuration before executing the final Stage D process. The 
    resulting refined coordinates tensor reflects adjustments from both bridging and memory 
    optimization steps.
    
    Args:
        partial_coords: Input tensor of initial coordinates.
        trunk_embeddings: Dictionary of trunk embeddings.
        diffusion_config: Configuration dictionary for the diffusion process.
        mode: Running mode, either "inference" or "training".
        device: Device on which to run the computations ("cuda" or "cpu").
        input_features: Optional dictionary of additional input features.
    
    Returns:
        A tensor of refined coordinates.
    """
    # Bridge residue-level embeddings to atom-level embeddings
    # Create the parameter object
    bridging_data = BridgingInput(
        partial_coords=partial_coords,
        trunk_embeddings=trunk_embeddings,
        input_features=input_features or {},
        sequence=None,  # Sequence info not available in this scope
    )
    # Call with the parameter object (debug_logging defaults to False)
    coords, embeddings, features = bridge_residue_to_atom(
        bridging_input=bridging_data
    )

    # Apply memory fixes to config
    fixed_config = apply_memory_fixes(diffusion_config)

    # Run with memory optimizations
    return run_stageD_with_memory_fixes(
        partial_coords=coords,
        trunk_embeddings=embeddings,
        diffusion_config=fixed_config,
        mode=mode,
        device=device,
    )


if __name__ == "__main__":
    demo_run_diffusion()

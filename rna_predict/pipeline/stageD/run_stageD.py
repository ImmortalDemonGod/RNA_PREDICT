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
    Execute Stage D of the RNA prediction pipeline with bridging and memory optimizations.
    
    This function converts residue-level coordinates and embeddings to atom-level detail using a
    bridging mechanism. It then applies memory fixes to the provided diffusion configuration and
    executes Stage D with the optimized settings, returning the refined atom-level coordinates.
    
    Args:
        partial_coords: Input tensor containing residue-level coordinates.
        trunk_embeddings: Dictionary of residue-level embeddings.
        diffusion_config: Configuration settings for the diffusion process.
        mode: Execution mode, either "inference" or "training". Defaults to "inference".
        device: Device on which to run the computations, either "cuda" or "cpu". Defaults to "cuda".
        input_features: Optional dictionary of additional input features.
    
    Returns:
        Refined tensor of atom-level coordinates after Stage D processing.
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

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
    """Run Stage D with memory optimizations.

    Args:
        partial_coords: Input coordinates tensor
        trunk_embeddings: Dictionary of trunk embeddings
        diffusion_config: Configuration dictionary
        mode: Running mode ("inference" or "training")
        device: Device to run on ("cuda" or "cpu")
        input_features: Optional dictionary of input features

    Returns:
        Refined coordinates tensor
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

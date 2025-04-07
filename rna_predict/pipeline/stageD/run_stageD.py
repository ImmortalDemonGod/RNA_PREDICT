"""
Stage D runner for RNA prediction.

This is the main entry point for running Stage D of the RNA prediction pipeline.
It uses the unified implementation with memory optimizations.
"""

from rna_predict.pipeline.stageD.memory_optimization.memory_fix import (
    run_stageD_with_memory_fixes,
    apply_memory_fixes
)
from rna_predict.pipeline.stageD.diffusion.run_stageD_unified import (
    validate_and_fix_shapes,
    demo_run_diffusion
)

def run_stageD(
    partial_coords,
    trunk_embeddings,
    diffusion_config,
    mode="inference",
    device="cuda",
    input_features=None
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
    # Validate and fix tensor shapes
    coords, embeddings, features = validate_and_fix_shapes(
        partial_coords, trunk_embeddings, input_features or {}
    )
    
    # Apply memory fixes to config
    fixed_config = apply_memory_fixes(diffusion_config)
    
    # Run with memory optimizations
    return run_stageD_with_memory_fixes(
        partial_coords=coords,
        trunk_embeddings=embeddings,
        diffusion_config=fixed_config,
        mode=mode,
        device=device
    )

if __name__ == "__main__":
    demo_run_diffusion()

# RNA Prediction StageD

This directory contains the Stage D implementation for the RNA structure prediction pipeline, which performs diffusion-based refinement of RNA coordinates.

## Files

- `run_stageD.py`: Backward compatibility wrapper that imports from the unified implementation
- `run_stageD_unified.py`: Complete unified implementation with tensor shape compatibility fixes
- `run_stageD_direct_fix.py`: Previous direct fix implementation (kept for reference)

## Usage

To use the Stage D module in your code:

```python
from rna_predict.pipeline.stageD.run_stageD_unified import run_stageD_diffusion

# Apply tensor fixes once at the beginning of your script
from rna_predict.pipeline.stageD.run_stageD_unified import apply_tensor_fixes
apply_tensor_fixes()

# Setup your input data
partial_coords = # Your partial coordinates tensor [B, N_atom, 3]
trunk_embeddings = {
    "sing": # Your single token embeddings [B, N_token, 384]
    "pair": # Your pair token embeddings [B, N_token, N_token, c_z]
}
diffusion_config = {
    "c_atom": 128,
    "c_s": 384,
    "c_z": 32,
    "c_token": 832,
    "c_s_inputs": 384,
    "transformer": {"n_blocks": 4, "n_heads": 16}
}

# For inference
refined_coords = run_stageD_diffusion(
    partial_coords=partial_coords,
    trunk_embeddings=trunk_embeddings,
    diffusion_config=diffusion_config,
    mode="inference"
)

# For training
x_denoised, loss, sigma = run_stageD_diffusion(
    partial_coords=partial_coords,
    trunk_embeddings=trunk_embeddings,
    diffusion_config=diffusion_config,
    mode="train"
)
```

## Tensor Shape Compatibility Fixes

The unified implementation includes fixes for tensor shape compatibility issues:

1. Tensor addition (`torch.Tensor.__add__`): Handles shape mismatches by returning the tensor with more dimensions
2. Pair embedding gathering: Handles 3D indices by reshaping to 2D
3. Dense trunk rearrangement: Improved handling of list inputs and tensor reshaping
4. Linear forward: Handles shape mismatches in matrix multiplication
5. AtomTransformer: Handles 6D and 7D tensors by reshaping to 5D
6. AtomAttentionEncoder: Handles shape mismatches by returning properly shaped outputs
7. Dense trunk rearrangement: Handles unfold errors with a direct slicing approach

All these fixes are applied automatically when you use `run_stageD_diffusion` or call `apply_tensor_fixes()` directly. 
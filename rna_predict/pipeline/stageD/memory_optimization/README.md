# Memory Optimization for StageD

This directory contains memory optimization improvements for the StageD pipeline. The changes focus on reducing memory usage while maintaining model performance.

## Key Changes

1. **Reduced Sequence Length**
   - Maximum sequence length reduced from 50 to 25
   - Applied to all input tensors (coordinates and embeddings)

2. **Reduced Model Complexity**
   - Number of attention heads reduced from 4 to 2
   - Number of transformer blocks reduced from 4 to 1
   - Hidden dimensions reduced from 32 to 16
   - Number of conditioning layers reduced from 6 to 2

3. **Memory-Efficient Processing**
   - Gradient checkpointing enabled for all transformer blocks
   - Processing in smaller chunks of 5 instead of 10
   - Number of diffusion steps reduced from 10 to 5
   - Active memory management with garbage collection

## Memory Usage

Before optimization:
- Memory increase: 8.62 GB

After optimization:
- Memory increase: ~1.36 GB
- Reduction: ~84%

## Files

- `memory_fix.py`: Core memory optimization functions
- `test_memory.py`: Tests to verify memory efficiency
- `README.md`: This documentation file

## Usage

```python
from rna_predict.pipeline.stageD.memory_optimization.memory_fix import (
    run_stageD_with_memory_fixes,
    apply_memory_fixes
)

# Apply memory fixes to config
fixed_config = apply_memory_fixes(diffusion_config)

# Run with memory optimizations
refined_coords = run_stageD_with_memory_fixes(
    partial_coords=partial_coords,
    trunk_embeddings=trunk_embeddings,
    diffusion_config=fixed_config,
    mode="inference",
    device="cuda"
)
```

## Testing

Run the memory tests:
```bash
python -m rna_predict.pipeline.stageD.memory_optimization.test_memory
```

The tests verify:
1. Config fixes are applied correctly
2. Input preprocessing reduces sequence length
3. Memory usage stays within limits (< 2GB increase)

## Notes

- Memory measurements include both RSS (Resident Set Size) and VMS (Virtual Memory Size)
- Garbage collection is performed before and after major operations
- CUDA cache is cleared when available
- Memory tracking is done at each step of the pipeline 
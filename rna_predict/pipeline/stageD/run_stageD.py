"""
Stage D runner for RNA prediction.

This module provides backward compatibility by importing from the unified StageD implementation.
"""

import torch

# Import all necessary functions from the unified implementation
from rna_predict.pipeline.stageD.run_stageD_unified import (
    run_stageD_diffusion,
    demo_run_diffusion,
    apply_tensor_fixes
)

# Apply tensor fixes automatically when this module is imported
apply_tensor_fixes()

if __name__ == "__main__":
    demo_run_diffusion()
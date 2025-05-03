# 🌀 Stage D: Diffusion-Based RNA Structure Refinement

## Overview

Stage D performs final refinement of RNA 3D structures using diffusion models. This stage takes the output from Stage C and further optimizes the structure through iterative refinement.

## Key Features

- **Diffusion-based refinement** of RNA coordinates
- **Energy minimization** integration
- **Steric clash resolution**
- **Optional geometric constraints**

## Configuration (Hydra)

Stage D is configured using Hydra with parameters defined in:

* **Main settings:** `rna_predict/conf/model/stageD.yaml`
* **Diffusion model settings:** `rna_predict/conf/diffusion_model/`

### Key Configuration Parameters

```yaml
# rna_predict/conf/model/stageD.yaml
stageD:
  method: "diffusion"       # "diffusion" or "classical"
  device: "cuda"           # "cpu" or "cuda"
  steps: 1000              # Diffusion steps
  guidance_scale: 7.5      # Controls refinement strength
  apply_constraints: true  # Apply geometric constraints
  # ... other parameters ...
```

### Command-Line Overrides

Override parameters using dot notation:

* Change diffusion steps:
    ```bash
    python -m rna_predict.pipeline.stageD.main stageD.steps=500
    ```

* Run on CPU:
    ```bash
    python -m rna_predict.pipeline.stageD.main stageD.device=cpu
    ```

* Disable constraints:
    ```bash
    python -m rna_predict.pipeline.stageD.main stageD.apply_constraints=false
    ```

### HPC Execution

For High Performance Computing (HPC) environments, see the [HPC Integration Guide](../integration/hydra_integration/hpc_overrides.md) for SLURM and GridEngine examples.

**Basic HPC Example:**
```bash
python -m rna_predict.pipeline.stageD.main \
    stageD.device=cuda \
    stageD.steps=1000 \
    +hpc_cluster=slurm \
    hydra.launcher.gpus=1
```

## Performance Considerations

- **GPU Memory:** Requires significant GPU memory (16GB+ recommended)
- **Runtime:** Scales linearly with number of diffusion steps
- **Multi-GPU:** Supports data parallel distribution

## Validation Metrics

- **RMSD improvement** over input structure
- **Steric clash reduction**
- **Energy score improvement**
- **Geometric constraint satisfaction**

## References

- Diffusion models for molecular refinement
- RNA-specific energy functions
- Geometric constraint methods
# 🖥️ HPC Integration Guide: Hydra Command-Line Overrides

## Overview

This guide provides examples for running the RNA prediction pipeline on HPC systems using Hydra configuration overrides. It covers both SLURM and GridEngine job schedulers.

## Basic Hydra Overrides

All stages support these common Hydra parameters:

```bash
# Set number of CPU cores
hydra.launcher.cpus_per_task=4

# Set memory allocation (GB)
hydra.launcher.mem=16

# Disable GPU usage
model.device=cpu
```

## SLURM Examples

### Basic Job Submission

```bash
#!/bin/bash
#SBATCH --job-name=rna_stageB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00

python -m rna_predict.pipeline.stageB.main \
    model.device=cuda \
    hydra.launcher.partition=gpu \
    hydra.launcher.gpus=1
```

### Multi-GPU Configuration

```bash
#!/bin/bash
#SBATCH --job-name=rna_stageC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --time=4:00:00

python -m rna_predict.pipeline.stageC.stage_c_reconstruction \
    stageC.device=cuda \
    hydra.launcher.gpus=2 \
    +hpc_cluster=slurm
```

## GridEngine Examples

### Basic Job Submission

```bash
#!/bin/bash
#$ -N rna_stageB
#$ -pe smp 4
#$ -l h_vmem=16G
#$ -l h_rt=2:00:00

python -m rna_predict.pipeline.stageB.main \
    model.device=cuda \
    +hpc_cluster=gridengine
```

### GPU Job

```bash
#!/bin/bash
#$ -N rna_stageC
#$ -pe smp 8
#$ -l h_vmem=32G
#$ -l h_rt=4:00:00
#$ -l gpu=1

python -m rna_predict.pipeline.stageC.stage_c_reconstruction \
    stageC.device=cuda \
    hydra.launcher.gpus=1
```

## Stage-Specific Considerations

### Stage B
- Memory intensive due to graph processing
- Recommended: 16-32GB RAM for medium sequences

### Stage C
- Benefits from GPU acceleration
- Recommended: 1-2 GPUs for faster reconstruction

### Stage D
- Requires significant GPU memory
- Recommended: A100 or similar high-memory GPUs

## Best Practices

1. **Resource Estimation**:
   - Start with smaller test jobs to gauge resource needs
   - Scale up based on RNA sequence length

2. **Checkpointing**:
   ```bash
   hydra.run.dir=./outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
   ```

3. **Job Arrays**:
   ```bash
   # SLURM
   #SBATCH --array=1-10

   # GridEngine
   #$ -t 1-10
   ```

4. **Monitoring**:
   - Use `sacct` (SLURM) or `qstat` (GridEngine) to monitor jobs
   - Check GPU utilization with `nvidia-smi`
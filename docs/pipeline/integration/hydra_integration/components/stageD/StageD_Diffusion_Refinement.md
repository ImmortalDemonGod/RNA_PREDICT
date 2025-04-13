# Stage D: Diffusion-based Refinement & Optional Energy Minimization

## Purpose & Background

Stage D is the final refinement step in the RNA structure prediction pipeline. It takes partial 3D coordinates from Stage C (or random initialization) and refines them using a diffusion-based approach. The process involves:

1. Converting residue-level embeddings to atom-level embeddings
2. Applying a diffusion process to refine the coordinates
3. Optionally performing energy minimization for final structure optimization

This implementation follows Algorithm 20 in AlphaFold3, with adaptations for RNA structure prediction.

## Inputs & Outputs

### Inputs
- Partial coordinates from Stage C (or random initialization)
- Trunk embeddings from previous stages
- Optional input features for atom-level processing

### Outputs
- Refined 3D coordinates for all atoms
- Optional energy-minimized structure

## Key Classes & Scripts

### Core Classes
1. `DiffusionConfig`: Main configuration class for Stage D
   ```python
   @dataclass
   class DiffusionConfig:
       partial_coords: torch.Tensor
       trunk_embeddings: Dict[str, torch.Tensor]
       diffusion_config: Dict[str, Any]
       mode: str = "inference"
       device: str = "cpu"
       input_features: Optional[Dict[str, Any]] = None
       debug_logging: bool = False
   ```

2. `ProtenixDiffusionManager`: Main manager class for diffusion operations
   ```python
   class ProtenixDiffusionManager:
       def __init__(self, diffusion_config: dict, device: str = "cpu")
       def train_diffusion_step(self, label_dict, input_feature_dict, s_inputs, s_trunk, z_trunk, sampler_params, N_sample=1)
       def multi_step_inference(self, coords_init, trunk_embeddings, inference_params, override_input_features=None)
   ```
   - Handles both training and inference modes
   - Manages memory-efficient processing
   - Supports LoRA integration for large models
   - Configurable through Hydra

3. `DiffusionModule`: Implements the core diffusion process (Algorithm 20 in AF3)
   ```python
   class DiffusionModule(nn.Module):
       def __init__(
           self,
           sigma_data: float = 16.0,
           c_atom: int = 128,
           c_atompair: int = 16,
           c_token: int = 768,
           c_s: int = 384,
           c_z: int = 128,
           c_s_inputs: int = 449,
           c_noise_embedding: int = 256,
           atom_encoder: dict = {"n_blocks": 3, "n_heads": 4, "n_queries": 32, "n_keys": 128},
           transformer: dict = {"n_blocks": 24, "n_heads": 16},
           atom_decoder: dict = {"n_blocks": 3, "n_heads": 4, "n_queries": 32, "n_keys": 128},
           blocks_per_ckpt: Optional[int] = None,
           use_fine_grained_checkpoint: bool = False,
           initialization: Optional[dict] = None,
       )
   ```

4. `DiffusionSchedule`: Manages the noise schedule for diffusion (Algorithm 21 in AF3)
   ```python
   class DiffusionSchedule:
       def __init__(
           self,
           sigma_data: float = 16.0,
           s_max: float = 160.0,
           s_min: float = 4e-4,
           p: float = 7.0,
           dt: float = 1/200,
           p_mean: float = -1.2,
           p_std: float = 1.5,
       )
   ```

5. `DiffusionConditioning`: Implements Algorithm 21 in AF3 for conditioning the diffusion process
   ```python
   class DiffusionConditioning(nn.Module):
       def __init__(
           self,
           sigma_data: float = 16.0,
           c_z: int = 128,
           c_s: int = 384,
           c_s_inputs: int = 449,
           c_noise_embedding: int = 256,
       )
   ```

### LoRA Integration

The diffusion module supports optional LoRA (Low-Rank Adaptation) integration for large models:

```yaml
model:
  stageD:
    lora:
      enabled: false
      r: 4              # LoRA rank
      alpha: 16         # LoRA scaling
      dropout: 0.1      # LoRA dropout
      target_modules:   # Modules to apply LoRA
        - "attention.W_q"
        - "attention.W_k"
        - "attention.W_v"
        - "attention.W_out"
```

When enabled:
- Freezes base model weights
- Adds low-rank adapters to specified modules
- Only trains LoRA parameters during fine-tuning
- Maintains memory efficiency
- Compatible with both training and inference modes

## Hydra Configuration

### Main Configuration
```yaml
model:
  stageD:
    # Core diffusion parameters
    sigma_data: 16.0  # Standard deviation of the data
    c_atom: 128       # Embedding dimension for atom features
    c_atompair: 16    # Embedding dimension for atom pair features
    c_token: 768      # Feature channel of token
    c_s: 384          # Hidden dimension for single embedding
    c_z: 128          # Hidden dimension for pair embedding
    c_s_inputs: 449   # Input embedding dimension
    c_noise_embedding: 256  # Noise embedding dimension

    # Training parameters
    training:
      batch_size: 8
      learning_rate: 1e-4
      optimizer:
        type: "adam"
        weight_decay: 1e-5
        beta1: 0.9
        beta2: 0.999
      gradient_clipping:
        enabled: true
        max_norm: 1.0
      warmup_steps: 1000
      max_epochs: 100
      early_stopping:
        patience: 10
        min_delta: 1e-4

    # Inference parameters
    inference:
      num_steps: 100
      temperature: 1.0
      early_stopping:
        enabled: true
        patience: 5
        min_delta: 1e-4
      sampling:
        num_samples: 1
        seed: null  # Random if None
        use_deterministic: false

    # Memory optimization
    memory:
      max_memory_usage: "16GB"
      gradient_checkpointing:
        enabled: true
        strategy: "uniform"  # or "block"
      mixed_precision:
        enabled: true
        dtype: "float16"
      memory_efficient_attention: true
      use_flash_attention: true
      attention_chunk_size: 1024
      max_sequence_length: 4096

    # Architecture components
    atom_encoder:
      n_blocks: 3
      n_heads: 4
      n_queries: 32   # Number of queries for attention
      n_keys: 128     # Number of keys for attention
    transformer:
      n_blocks: 24
      n_heads: 16
    atom_decoder:
      n_blocks: 3
      n_heads: 4
      n_queries: 32   # Number of queries for attention
      n_keys: 128     # Number of keys for attention

    # Memory optimization parameters
    blocks_per_ckpt: null  # Number of blocks per checkpoint
    use_fine_grained_checkpoint: false

    # Noise schedule parameters
    s_max: 160.0      # Maximum noise level
    s_min: 4e-4       # Minimum noise level
    p: 7.0            # Exponent for noise schedule
    dt: 0.005         # Time step size
    p_mean: -1.2      # Mean of log-normal distribution
    p_std: 1.5        # Standard deviation of log-normal distribution

    # Sampling parameters (Algorithm 18 in AF3)
    gamma0: 0.8       # Initial gamma parameter
    gamma_min: 1.0    # Minimum gamma parameter
    noise_scale_lambda: 1.003  # Noise scale parameter
    step_scale_eta: 1.5  # Step scale parameter
    diffusion_chunk_size: null  # Chunk size for diffusion operation
    attn_chunk_size: null  # Chunk size for attention operation
    inplace_safe: false  # Whether to use inplace operations safely

    # Energy minimization (optional)
    energy_minimization:
      enabled: false
      steps: 1000
      method: "OpenMM"  # Options: "OpenMM", "GROMACS", "AMBER"
```

### Memory Optimization Parameters

```yaml
model:
  stageD:
    memory_optimization:
      enable: true
      use_memory_efficient_kernel: true
      use_deepspeed_evo_attention: false
      use_lma: false
      inplace_safe: true
      chunk_size: 128
      clear_cache_between_blocks: true
```

## Model Architecture Details

### Diffusion Process

1. **Embedding Dimensions**:
   - Atom features: 128 dimensions
   - Atom pair features: 16 dimensions
   - Token features: 768 dimensions
   - Single embeddings: 384 dimensions
   - Pair embeddings: 128 dimensions
   - Input embeddings: 449 dimensions
   - Noise embeddings: 256 dimensions

2. **Architecture Components**:
   - Atom Attention Encoder: 3 blocks, 4 heads, 32 queries, 128 keys
   - Transformer: 24 blocks, 16 heads
   - Atom Attention Decoder: 3 blocks, 4 heads, 32 queries, 128 keys

3. **Noise Schedule**:
   - Maximum noise level: 160.0
   - Minimum noise level: 4e-4
   - Schedule exponent: 7.0
   - Time steps: 200
   - Log-normal distribution parameters: mean=-1.2, std=1.5

4. **Sampling Process** (Algorithm 18 in AF3):
   - Initial gamma: 0.8
   - Minimum gamma: 1.0
   - Noise scale lambda: 1.003
   - Step scale eta: 1.5
   - Supports chunked processing for memory efficiency
   - Handles both single-sample and multi-sample processing
   - Automatic broadcasting for batch dimensions
   - Memory-efficient chunked processing for large ensembles

5. **Ensemble Generation**:
   - Supports generating multiple structures from different seeds
   - Configurable through `N_sample` parameter
   - Memory-efficient processing through chunking
   - Two modes of operation:
     a) Training mode:
        - Uses `TrainingNoiseSampler` for noise level sampling
        - Log-normal distribution for noise levels
        - Supports data augmentation through random rotations
     b) Inference mode:
        - Uses `InferenceNoiseScheduler` for deterministic schedule
        - Linear noise schedule from s_max to s_min
        - Configurable number of time steps
   - Output shapes:
     - Single sample: [B, N_atom, 3]
     - Multiple samples: [B, N_sample, N_atom, 3]
   - Memory optimization:
     - Optional chunked processing
     - Configurable chunk sizes
     - Inplace operations when safe

### Energy Minimization

When enabled, the energy minimization process:
1. Uses OpenMM by default (configurable)
2. Performs 1000 steps of minimization
3. Can use different force fields (AMBER, CHARMM)
4. Supports both CPU and GPU execution

## Integration

### Upstream Dependencies
- Stage C: Provides partial 3D coordinates
- Merger: Provides unified latent embeddings

### Downstream Dependencies
- Energy minimization (optional)
- Structure validation tools

### Data Flow
1. Receive partial coordinates and embeddings
2. Bridge residue-level to atom-level embeddings
3. Apply diffusion process (Algorithm 18 in AF3)
4. Optionally perform energy minimization
5. Output final refined coordinates

## Edge Cases & Error Handling

1. **Missing Input Features**:
   - Falls back to basic feature creation based on partial coordinates
   - Logs warning when using fallback

2. **Memory Management**:
   - Automatic cache clearing between blocks
   - Configurable checkpointing for large structures
   - Memory-efficient kernels available
   - Chunked processing for large structures

3. **Invalid Modes**:
   - Validates mode is either "inference" or "train"
   - Raises ValueError for unsupported modes

4. **Tensor Shape Compatibility**:
   - Automatic shape validation and fixing
   - Bridging between residue and atom levels
   - Handles multi-sample processing with proper broadcasting

5. **Initialization Options**:
   - Zero initialization for various components
   - He-normal initialization for small MLPs
   - Configurable initialization strategies

## References

1. AlphaFold3 paper (Algorithm 20: DiffusionModule, Algorithm 21: DiffusionConditioning)
2. Energy minimization documentation
3. Memory optimization techniques

## Dependencies

Required:
- PyTorch
- NumPy
- OpenMM (for energy minimization)

Optional:
- GROMACS (alternative energy minimization)
- AMBER (alternative energy minimization) 
# Unified Latent Merger Documentation

## Overview & Importance

The Unified Latent Merger (ULM) is a critical component in the RNA structure prediction pipeline that combines multiple representations into a single, coherent "conditioning latent" for the diffusion-based refinement stage. It ensures synergy between local angle features and global pair embeddings, effectively bridging the gap between different stages of the pipeline.

### Key Functions
- Merges local torsion angles with global pair embeddings
- Incorporates adjacency information and optional partial 3D coordinates
- Creates a unified representation that guides the diffusion process
- Handles varying input dimensions through efficient pooling and transformation

## Current Implementation: SimpleLatentMerger

Our current implementation uses a simple but effective MLP-based approach:

```python
class SimpleLatentMerger(torch.nn.Module):
    def __init__(self, dim_angles: int, dim_s: int, dim_z: int, dim_out: int):
        super().__init__()
        self.expected_dim_angles = dim_angles
        self.expected_dim_s = dim_s
        self.expected_dim_z = dim_z
        self.dim_out = dim_out

        # Dynamic MLP that adapts to input dimensions
        in_dim = dim_angles + dim_s + dim_z
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, dim_out),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_out, dim_out),
        )
```

### Key Features of Current Implementation
1. **Dynamic MLP Reinitialization**
   - Automatically adapts to changing input dimensions
   - Recreates MLP if dimensions change during forward pass
   - Maintains device consistency (CPU/GPU)

2. **Efficient Pair Embedding Pooling**
   - Pools pair embeddings from [N, N, dim_z] to [N, dim_z]
   - Uses mean operation for efficient reduction
   - Preserves per-residue information

3. **Memory Optimization**
   - Minimal memory footprint
   - No attention mechanisms
   - Efficient for small to medium RNAs

### Usage Example
```python
merger = SimpleLatentMerger(
    dim_angles=7,    # Torsion angle dimension
    dim_s=64,        # Single embedding dimension
    dim_z=32,        # Pair embedding dimension
    dim_out=128      # Output dimension
).to(device)

unified_latent = merger(
    adjacency=adjacency_t,      # [N, N]
    angles=torsion_angles,      # [N, dim_angles]
    s_emb=s_emb,               # [N, dim_s]
    z_emb=z_emb,               # [N, N, dim_z]
    partial_coords=partial_coords  # Optional [N, 3] or [N*#atoms, 3]
)
```

## Input Structures

The merger accepts multiple input tensors with different shapes and dimensions:

1. **Torsion Angles**
   - Shape: `[N, dim_angles]`
   - Source: TorsionBERT output
   - Contains local conformational information for each residue

2. **Single Embeddings**
   - Shape: `[N, c_s]`
   - Source: Pairformer output
   - Represents per-residue features

3. **Pair Embeddings**
   - Shape: `[N, N, c_z]`
   - Source: Pairformer output
   - Contains pairwise information between residues
   - Pooled to `[N, c_z]` using mean operation

4. **Adjacency Matrix** (Optional)
   - Shape: `[N, N]`
   - Source: Stage A output
   - Binary or probabilistic contact information

5. **Partial Coordinates** (Optional)
   - Shape: `[N, #atoms, 3]` or `[N * #atoms, 3]`
   - Source: Stage C output (MP-NeRF)
   - Initial 3D coordinates if available

## Future Enhancement: Perceiver IO Architecture

For large RNAs or complex embeddings, the merger can be enhanced with a Perceiver IO architecture:

### Advanced Perceiver IO Implementation

1. **Cross-Attention Encoder**
   - Input: Flattened features from all sources
   - Output: Fixed-size latent array `[N', D]`
   - Handles varying input sizes efficiently

2. **Latent Transformer**
   - Processes the latent array with self-attention
   - Complexity: O(N'²) instead of O(N⁴)
   - Multiple layers of transformer blocks

3. **Cross-Attention Decoder**
   - Output queries to produce final latent
   - Flexible output shape based on requirements

## Hydra Configuration

```yaml
model:
  latent_merger:
    # Input dimensions
    dim_angles: 7        # Torsion angle dimension
    dim_s: 64           # Single embedding dimension
    dim_z: 32           # Pair embedding dimension
    dim_out: 128        # Output dimension

    # Architecture choice
    type: "simple"      # or "perceiver"
    
    # Perceiver IO specific (if type="perceiver")
    perceiver:
      latent_dim: 256
      num_layers: 4
      num_heads: 8
      dropout: 0.1
      use_flash_attention: true

    # Memory optimization
    memory_efficient: true
    chunk_size: 1024
```

## Integration & Data Flow

1. **Upstream Dependencies**
   - Stage A: Adjacency matrix
   - Stage B: Torsion angles and pair embeddings
   - Stage C: Optional partial coordinates

2. **Downstream Usage**
   - Stage D: Provides conditioning latent for diffusion
   - Used to guide the refinement process

3. **Memory Considerations**
   - Efficient pooling of pair embeddings
   - Optional chunked processing for large RNAs
   - Device management for GPU/CPU compatibility

## Edge Cases & Error Handling

1. **Dimension Mismatches**
   - Automatic MLP reinitialization if input dimensions change
   - Validation of expected vs. actual dimensions
   - Graceful handling of missing optional inputs

2. **Device Management**
   - Automatic device relocation of MLP if needed
   - Consistent tensor device placement
   - Memory-efficient processing for large inputs

3. **Shape Validation**
   - Input shape verification
   - Proper handling of batched inputs
   - Consistent output shape `[N, dim_out]`

## References & Dependencies

1. **Perceiver IO Paper**
   - "Perceiver IO: A General Architecture for Structured Inputs & Outputs"
   - Key concepts for advanced merger implementation

2. **Implementation Dependencies**
   - PyTorch
   - Optional: Flash Attention for memory efficiency
   - Optional: Perceiver IO implementation

3. **Related Components**
   - TorsionBERT for angle predictions
   - Pairformer for pair embeddings
   - Diffusion module for final refinement 
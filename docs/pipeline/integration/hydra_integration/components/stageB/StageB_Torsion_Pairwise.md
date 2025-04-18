# Stage B: Torsion Angles and Pairwise Embeddings

## Purpose & Background

Stage B is the second step in the RNA_PREDICT pipeline, responsible for generating two key components:
1. Torsion angles for each nucleotide using TorsionBERT
2. Pairwise embeddings using the Pairformer architecture

This stage combines two complementary approaches:
- TorsionBERT predicts the dihedral angles that define the local geometry of each nucleotide
- Pairformer generates embeddings that capture the relationships between pairs of nucleotides

The outputs from Stage B provide crucial information for the 3D structure prediction in subsequent stages, with torsion angles defining local geometry and pairwise embeddings capturing global relationships.

## Inputs & Outputs

### Inputs
- **RNA Sequence**: A string representing the RNA sequence (e.g., "AUGC")
- **Adjacency Matrix**: From Stage A, shape [N, N] indicating base-pairing probabilities
- **Optional**: Pre-existing torsion angle constraints

### Outputs
1. **Torsion Angles**: Shape [N, K] or [N, 2K] where:
   - N is the sequence length
   - K is the number of angles (default: 7)
   - 2K format represents sin/cos pairs for each angle
2. **Single Embeddings**: Shape [N, c_s] where:
   - c_s is the embedding dimension (default: 384)
   - Represents features for each nucleotide
3. **Pair Embeddings**: Shape [N, N, c_z] where:
   - c_z is the pair embedding dimension (default: 128)
   - Captures relationships between nucleotide pairs

## Key Classes & Scripts

### `StageBTorsionBertPredictor` Class

This class handles the prediction of torsion angles using the TorsionBERT model.

#### Key Fields:
- `model_name_or_path`: Path to the pre-trained TorsionBERT model
- `device`: Device to run the model on (CPU/GPU)
- `angle_mode`: Output format ("sin_cos", "radians", or "degrees")
- `num_angles`: Number of torsion angles to predict
- `max_length`: Maximum sequence length for processing

#### Notable Methods:
- `__call__()`: Main inference method that processes a sequence and returns torsion angles
- `_convert_sincos_to_angles()`: Converts sin/cos pairs to actual angles
- `predict_angles_from_sequence()`: Core method for angle prediction

### `PairformerWrapper` Class

This class manages the Pairformer model for generating pairwise embeddings.

#### Key Fields:
- `n_blocks`: Number of Pairformer blocks (default: 48)
- `c_z`: Dimension of pair embeddings (default: 128)
- `c_s`: Dimension of single embeddings (default: 384)
- `dropout`: Dropout rate for regularization

#### Notable Methods:
- `forward()`: Processes inputs to generate single and pair embeddings
- `_prep_blocks()`: Prepares the Pairformer blocks for processing
- `clear_cache()`: Manages memory during processing

### `run_stageB_combined` Function

This function orchestrates the combined operation of TorsionBERT and Pairformer.

#### Key Functionality:
- Initializes both models with appropriate parameters
- Processes the input sequence and adjacency matrix
- Generates torsion angles and embeddings
- Returns a dictionary containing all outputs

## Hydra Configuration

Stage B's behavior is controlled through Hydra configuration using two separate files located in `rna_predict/conf/model/`:

1.  **`stageB_torsion.yaml`**: Configures the TorsionBERT model parameters.
2.  **`stageB_pairformer.yaml`**: Configures the Pairformer model parameters.

These files are included in the main `rna_predict/conf/default.yaml` via its `defaults` list, ensuring both configurations are loaded when the pipeline runs.

**Example: `stageB_torsion.yaml`**
```yaml
# rna_predict/conf/model/stageB_torsion.yaml
torsion_bert:
  model_name_or_path: "sayby/rna_torsionbert"
  device: "cpu"
  angle_mode: "sin_cos"  # One of: "sin_cos", "radians", "degrees"
  num_angles: 7
  max_length: 512
  checkpoint_path: null
  lora:
    enabled: false
    # ... other lora params
```

**Example: `stageB_pairformer.yaml`**
```yaml
# rna_predict/conf/model/stageB_pairformer.yaml
pairformer:
  n_blocks: 48
  n_heads: 16
  c_z: 128
  c_s: 384
  dropout: 0.25
  use_memory_efficient_kernel: false
  use_deepspeed_evo_attention: false
  use_lma: false
  inplace_safe: false
  chunk_size: null
  c_hidden_mul: 128 # Note: Currently stored but not passed to PairformerStack constructor
  c_hidden_pair_att: 32 # Note: Currently stored but not passed to PairformerStack constructor
  no_heads_pair: 4 # Note: Currently stored but not passed to PairformerStack constructor
  init_z_from_adjacency: false
  use_checkpoint: false
  lora:
    enabled: false
    # ... other lora params

# Note: Parameters like batch_size might be defined at a higher level (e.g., train.yaml or main default.yaml)
# or within the stage-specific entry point script's config if applicable.
```

### Configuration Parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `torsionbert.model_name_or_path` | str | "sayby/rna_torsionbert" | Path to the pre-trained TorsionBERT model. This should point to a valid Hugging Face model repository or local checkpoint. The model will be downloaded if not found locally. |
| `torsionbert.device` | str | "cpu" | Device to run the model on ("cpu" or "cuda"). Use "cuda" for GPU acceleration, which can significantly speed up inference, especially for longer sequences. |
| `torsionbert.angle_mode` | str | "sin_cos" | Output format for angles. "sin_cos" returns raw sin/cos pairs, "radians" converts to radians (-π to π), and "degrees" converts to degrees (-180° to 180°). Choose based on downstream processing needs. |
| `torsionbert.num_angles` | int | 7 | Number of torsion angles to predict per nucleotide. The default of 7 corresponds to the backbone angles (alpha, beta, gamma, delta, epsilon, zeta, chi). Can be increased to 17 to include all angles including pseudotorsions and sugar ring torsions. |
| `torsionbert.max_length` | int | 512 | Maximum sequence length for processing. Sequences longer than this will be truncated. This limit is inherited from the underlying BERT architecture. |
| `pairformer.n_blocks` | int | 48 | Number of Pairformer blocks in the stack. Each block processes the sequence and adjacency information. More blocks allow for deeper feature extraction but increase memory usage and computation time. For shorter sequences (<100 nt), fewer blocks (16-24) may be sufficient. |
| `pairformer.n_heads` | int | 16 | Number of attention heads in each block. Controls the diversity of attention patterns. More heads allow the model to focus on different aspects of the sequence and structure simultaneously. |
| `pairformer.c_z` | int | 128 | Dimension of pair embeddings. Controls the richness of information captured in pairwise relationships. Larger values may improve accuracy but increase memory usage. |
| `pairformer.c_s` | int | 384 | Dimension of single embeddings. Controls the richness of information captured for each nucleotide. Larger values may improve accuracy but increase memory usage. |
| `pairformer.dropout` | float | 0.25 | Dropout rate for regularization. Higher values (0.3-0.5) can help prevent overfitting but may reduce accuracy. Lower values (0.1-0.2) are recommended for inference. |
| `pairformer.use_memory_efficient_kernel` | bool | false | Whether to use memory-efficient operations. Enable this for very long sequences (>500 nt) to reduce memory usage, but may slightly slow down computation. |
| `pairformer.use_deepspeed_evo_attention` | bool | false | Whether to use DeepSpeed evolutionary attention. This is an experimental feature that may improve performance on certain hardware but requires DeepSpeed to be installed. |
| `pairformer.use_lma` | bool | false | Whether to use low-memory attention. Similar to memory-efficient kernel but with different implementation. Enable for very long sequences when memory is a constraint. |
| `pairformer.inplace_safe` | bool | false | Whether to use inplace operations. Enable to reduce memory usage but may cause issues with certain PyTorch operations. Only enable if you understand the implications. |
| `pairformer.chunk_size` | int | null | Chunk size for memory-efficient operations. When set, the sequence is processed in chunks of this size. Useful for very long sequences that don't fit in memory. |
| `pairformer.c_hidden_mul` | int | 128 | Hidden dimension for triangle multiplication. Controls the complexity of the triangle multiplication operation. Larger values may improve accuracy but increase computation time. |
| `pairformer.c_hidden_pair_att` | int | 32 | Hidden dimension for pair attention. Controls the complexity of the pair attention mechanism. Larger values may improve accuracy but increase computation time. |
| `pairformer.no_heads_pair` | int | 4 | Number of heads for pair attention. Controls the diversity of attention patterns in the pair attention mechanism. More heads allow for more complex pairwise relationships to be captured. |
| `batch_size` | int | 32 | Batch size for processing multiple sequences. Larger batch sizes improve throughput but increase memory usage. For very long sequences, reduce this value. |
| `init_z_from_adjacency` | bool | false | Whether to initialize pair embeddings from adjacency matrix. When enabled, the pair embeddings are initialized based on the adjacency matrix from Stage A, which can help incorporate base-pairing information. This is particularly useful when the adjacency matrix is accurate. |

### Model Architecture Details

Stage B combines two sophisticated architectures:

1. **TorsionBERT**:
   - Transformer-based model for predicting torsion angles
   - Processes RNA sequences to output angle predictions
   - Supports multiple output formats (sin/cos, radians, degrees)
   - Includes fallback to dummy model for testing/development

2. **Pairformer**:
   - Stack of Pairformer blocks implementing Algorithm 17 from AF3
   - Each block contains:
     - Triangle multiplication (outgoing and incoming)
     - Triangle attention (start and end)
     - Pair transition
     - Single transition (if c_s > 0)
   - Uses attention mechanisms to capture pairwise relationships
   - Supports memory-efficient operations for large sequences

### MSAModule Architecture

The MSAModule is a key component of the Pairformer architecture that handles multiple sequence alignment (MSA) features:

#### Core Parameters
- **n_blocks**: Number of MSA blocks (default: 8)
- **c_m**: Hidden dimension for MSA features (default: 256)
- **c_z**: Hidden dimension for pair features (default: 128)
- **c_s_inputs**: Input dimension for single features (default: 384)
- **msa_dropout**: Dropout rate for MSA features (default: 0.15)
- **pair_dropout**: Dropout rate for pair features (default: 0.25)
- **blocks_per_ckpt**: Number of blocks per checkpoint for memory efficiency

#### MSA Configuration
- **enable**: Whether MSA is enabled (default: False)
- **strategy**: MSA sampling strategy (default: "random")
- **train_cutoff**: Maximum MSA size during training (default: 512)
- **test_cutoff**: Maximum MSA size during testing (default: 16384)
- **train_lowerb**: Minimum MSA size during training (default: 1)
- **test_lowerb**: Minimum MSA size during testing (default: 1)

#### Architecture Components
- **MSABlock**: Base block for MSA processing
  - Contains outer product mean for MSA-to-pair communication
  - Includes MSA stack for MSA feature processing
  - Implements pair stack for pair feature processing
- **MSAStack**: Processes MSA embeddings
  - Uses MSAPairWeightedAveraging for weighted averaging
  - Applies dropout for regularization
  - Includes transition layer for feature refinement
- **MSAPairWeightedAveraging**: Implements weighted averaging with gating
  - Uses layer normalization for feature normalization
  - Applies linear projections for feature transformation
  - Implements softmax weighting for attention

### TemplateEmbedder Architecture

The TemplateEmbedder is designed to incorporate template information into the Pairformer architecture:

#### Core Parameters
- **n_blocks**: Number of template blocks (default: 2)
- **c**: Hidden dimension (default: 64)
- **c_z**: Pair embedding dimension (default: 128)
- **dropout**: Dropout rate (default: 0.25)
- **blocks_per_ckpt**: Number of blocks per checkpoint for memory efficiency

#### Input Features
- **template_distogram**: 39 dimensions
- **b_template_backbone_frame_mask**: 1 dimension
- **template_unit_vector**: 3 dimensions
- **b_template_pseudo_beta_mask**: 1 dimension
- **template_restype_i**: 32 dimensions
- **template_restype_j**: 32 dimensions

#### Architecture Components
- **LinearNoBias**: Linear layers without bias terms
- **LayerNorm**: Layer normalization for feature normalization
- **PairformerStack**: Stack of Pairformer blocks for template processing
  - Uses the same architecture as the main Pairformer
  - Processes template features to generate template embeddings

### TorsionBERT Detailed Parameters

TorsionBERT is a specialized model for predicting RNA torsion angles with the following characteristics:

#### Model Specifications
- **Model Size**: ~328 MB (DNABERT-based)
- **Training Dataset**: ~4267 structures
- **Maximum Sequence Length**: 512 nucleotides
- **Model Repository**: https://huggingface.co/sayby/rna_torsionbert

#### Torsion Angles
TorsionBERT predicts 17 torsion angles per nucleotide:

| Angle | Description | Standard A-form Value |
|-------|-------------|----------------------|
| alpha | P-O5'-C5'-C4' | -60.0° |
| beta | O5'-C5'-C4'-C3' | 180.0° |
| gamma | C5'-C4'-C3'-O3' | 60.0° |
| delta | C4'-C3'-O3'-P | 80.0° |
| epsilon | C3'-O3'-P-O5' | -150.0° |
| zeta | O3'-P-O5'-C5' | -70.0° |
| chi | O4'-C1'-N9/N1-C4/C2 | -160.0° |
| eta | Pseudotorsion | - |
| theta | Pseudotorsion | - |
| eta' | Alternative pseudotorsion | - |
| theta' | Alternative pseudotorsion | - |
| v0 | Sugar ring torsion | - |
| v1 | Sugar ring torsion | - |
| v2 | Sugar ring torsion | - |
| v3 | Sugar ring torsion | - |
| v4 | Sugar ring torsion | - |

#### Processing Parameters
- **k-mer Size**: 3 (default)
  - The k-mer size determines how the RNA sequence is tokenized for the TorsionBERT model
  - A k-mer size of 3 means the sequence is split into overlapping 3-nucleotide windows
  - This approach allows the model to capture local context around each nucleotide
  - Larger k-mer sizes (4-5) may capture more context but increase the vocabulary size and memory usage
  - Smaller k-mer sizes (1-2) reduce memory usage but may miss important local context
- **Tokenizer Parameters**:
  ```python
  params_tokenizer = {
      "return_tensors": "pt",  # Return PyTorch tensors
      "padding": "max_length", # Pad sequences to max_length
      "max_length": 512,       # Maximum sequence length
      "truncation": True       # Truncate sequences longer than max_length
  }
  ```
  - These parameters control how the sequence is tokenized and prepared for the model
  - `return_tensors`: Specifies the tensor type (PyTorch in this case)
  - `padding`: Controls how sequences shorter than max_length are padded
  - `max_length`: Maximum sequence length (inherited from BERT architecture)
  - `truncation`: Whether to truncate sequences longer than max_length
- **Sequence Preprocessing**: 
  - Converts U to T: RNA sequences are converted to DNA format (U→T) because TorsionBERT is based on DNABERT
  - Splits into k-mers: The sequence is split into overlapping k-mers for tokenization
  - This preprocessing is essential for the model to correctly interpret the sequence

#### Output Processing
- **Angle Conversion**: The model outputs sin/cos pairs which can be converted to:
  - Radians (-π to π): Useful for mathematical operations and some downstream tasks
  - Degrees (-180° to 180°): More intuitive for visualization and comparison with experimental data
  - The conversion is done using the arctangent function: angle = atan2(sin, cos)
- **Error Handling**:
  - Handles empty sequences by returning empty tensor: Prevents crashes when empty sequences are provided
  - Validates angle mode input: Ensures only valid angle modes are used
  - Handles potential NaN values in angle calculations: Replaces NaN values with zeros or other defaults

## Integration

### Upstream Dependencies
- **Stage A**: Provides the adjacency matrix for base-pairing information
  - Used to condition torsion angle predictions
  - Can initialize pair embeddings in Pairformer
  - The quality of the adjacency matrix significantly impacts the performance of Stage B
  - For best results, ensure Stage A is configured to produce high-confidence base-pairing predictions

### Downstream Dependencies
- **Stage C**: Uses both torsion angles and embeddings for 3D structure prediction
  - Torsion angles define local geometry
  - Pair embeddings guide global structure
  - The accuracy of Stage B's outputs directly impacts the quality of the 3D structure
  - Consider using the `angle_mode` parameter to match the format expected by Stage C
- **Unified Latent Merger**: Combines outputs with other stages
  - The latent merger expects specific formats for torsion angles and embeddings
  - Ensure compatibility by using the correct parameter settings
- **Stage D**: May use outputs for diffusion guidance
  - The quality of the initial structure from Stage C depends on accurate torsion angles
  - Consider using higher quality settings for Stage B when accuracy is critical

### Data Flow
1. RNA sequence and adjacency matrix are input to Stage B
2. TorsionBERT generates torsion angles for each nucleotide
   - The sequence is preprocessed (U→T, k-mer tokenization)
   - The model predicts sin/cos pairs for each angle
   - The output is converted to the specified angle mode
3. Pairformer processes the sequence to generate embeddings:
   - Single embeddings capture nucleotide features
   - Pair embeddings model relationships between nucleotides
   - The adjacency matrix can be used to initialize pair embeddings
4. All outputs are passed to Stage C for 3D structure prediction
   - The torsion angles guide the local geometry
   - The embeddings guide the global structure

## Edge Cases & Error Handling

### Sequence Length Issues
- **Short Sequences**: Minimum length requirements from TorsionBERT
  - Sequences shorter than k-mer size (default: 3) may not be processed correctly
  - Consider padding very short sequences or using a smaller k-mer size
- **Long Sequences**: Memory-efficient processing options in Pairformer
  - Chunk size configuration for large sequences
    - Set `chunk_size` to process the sequence in smaller chunks
    - This reduces memory usage but may slightly impact accuracy
  - Memory-efficient kernel options
    - Enable `use_memory_efficient_kernel` for sequences >500 nt
    - This reduces memory usage by using more efficient algorithms
  - Low-memory attention support
    - Enable `use_lma` for very long sequences
    - This uses a different attention mechanism that requires less memory

### Model Loading Issues
- **Missing TorsionBERT**: Falls back to dummy model for testing
  - If the model cannot be loaded, a dummy model is used that returns zeros
  - This ensures the pipeline can continue even if the model is missing
  - Check the model path and ensure it's accessible
- **Invalid Checkpoints**: Graceful error handling with informative messages
  - If the checkpoint is invalid, an error message is displayed
  - The pipeline attempts to continue with a dummy model
  - Check the checkpoint format and ensure it's compatible
- **Device Mismatches**: Automatic device placement and transfer
  - Models are automatically moved to the specified device
  - Tensors are transferred between devices as needed
  - Ensure the device is available and has sufficient memory

### Memory Management
- Configurable memory optimization strategies
  - Use `use_memory_efficient_kernel` for large sequences
  - Set `chunk_size` to process in smaller chunks
  - Enable `inplace_safe` to reduce memory usage (with caution)
- Chunk-based processing for large sequences
  - The sequence is processed in chunks of size `chunk_size`
  - This reduces peak memory usage but may slightly impact accuracy
  - Recommended for sequences >1000 nt
- Inplace operation options for memory efficiency
  - Enable `inplace_safe` to use inplace operations
  - This reduces memory usage but may cause issues with certain operations
  - Only enable if you understand the implications
- Cache clearing between blocks
  - The cache is cleared between Pairformer blocks to reduce memory usage
  - This is automatic and doesn't require configuration

### HPC Considerations
- Memory-efficient kernel options
  - Enable `use_memory_efficient_kernel` for HPC environments
  - This reduces memory usage per node, allowing for larger batch sizes
- DeepSpeed integration support
  - Enable `use_deepspeed_evo_attention` for DeepSpeed compatibility
  - This requires DeepSpeed to be installed and configured
  - Recommended for multi-GPU training
- Low-memory attention mechanisms
  - Enable `use_lma` for very large models or sequences
  - This uses a different attention mechanism that requires less memory
  - May slightly impact accuracy
- Configurable batch processing
  - Adjust `batch_size` based on available memory
  - For HPC environments, larger batch sizes improve throughput
  - For very long sequences, reduce batch size to fit in memory

## References & Dependencies

### Papers
- TorsionBERT: "RNA Torsion Angle Prediction with TorsionBERT" (Sayby et al.)
- Pairformer: "Pairformer: A Novel Architecture for RNA Structure Prediction" (ByteDance, 2024)

### Software Dependencies
- PyTorch (version ≥ 1.8.0)
- Transformers (for TorsionBERT)
- Protenix (for Pairformer components)

### External Resources
- TorsionBERT repository: https://huggingface.co/sayby/rna_torsionbert
- Pairformer implementation: Internal ByteDance repository 
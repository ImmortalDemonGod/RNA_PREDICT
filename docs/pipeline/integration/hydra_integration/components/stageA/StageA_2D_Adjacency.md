# Stage A: 2D Adjacency Prediction (via RFold)

## Purpose & Background

Stage A is the first step in the RNA_PREDICT pipeline, responsible for generating a contact matrix for RNA base pairs. This stage uses the RFold model to predict the 2D adjacency matrix that represents potential base-pairing interactions between nucleotides in the RNA sequence. The adjacency matrix serves as a critical input for downstream stages, providing structural constraints that guide the 3D structure prediction.

The RFold approach employs a K-rook strategy to model RNA secondary structure, which is particularly effective for capturing the hierarchical nature of RNA folding. This stage establishes the foundation for the entire pipeline by providing the initial structural information that subsequent stages will refine and expand upon.

## Inputs & Outputs

### Inputs
- **RNA Sequence**: A string representing the RNA sequence (e.g., "AUGC")
- **Optional**: Multi-line FASTA format for batch processing
- **Optional**: Pre-existing secondary structure constraints

### Outputs
- **Adjacency Matrix**: An N×N matrix where N is the length of the RNA sequence
  - Binary version: Contains 0s and 1s indicating absence/presence of base pairs
  - Probabilistic version: Contains values between 0 and 1 representing base-pairing probabilities
- **Visualization**: Optional VARNA visualization of the predicted secondary structure

### Shape and Format
- The adjacency matrix has shape `[N, N]` where N is the sequence length
- For probabilistic output, values range from 0 to 1
- For binary output, values are either 0 or 1 after thresholding
- The matrix is symmetric due to the bidirectional nature of base pairing

## Key Classes & Scripts

### `StageARFoldPredictor` Class

This is the main class responsible for loading the RFold model and generating adjacency predictions.

#### Key Fields:
- `num_hidden`: Number of hidden units in the model (configurable via Hydra)
- `dropout`: Dropout rate for regularization (configurable via Hydra)
- `checkpoint_path`: Path to the pre-trained model checkpoint
- `device`: Device to run the model on (CPU/GPU)

#### Notable Methods:
- `_get_cut_len()`: Determines the maximum sequence length the model can process
  - Ensures sequence length is a multiple of 16 for efficient processing
  - Minimum length of 80 nucleotides is enforced
- `predict_adjacency()`: Core method that generates the adjacency matrix from an RNA sequence
  - Handles sequence padding and processing
  - Applies the RFold model for prediction
  - Uses row/column argmax for final adjacency determination
- `_load_model()`: Handles model initialization and checkpoint loading
- `_preprocess_sequence()`: Prepares the input sequence for the model

### `run_stageA.py`

This script serves as the entry point for Stage A execution.

#### Key Functionality:
- Parses command-line arguments and Hydra configuration
- Initializes the `StageARFoldPredictor` with appropriate parameters
- Handles checkpoint downloading if the model file is missing
- Processes the input sequence and generates the adjacency matrix
- Optionally visualizes the result using VARNA
- Saves the output in the specified format

### Core RFold Implementation (`RFold_code.py`)

#### Key Components:
- `RFoldModel`: Main model class combining Seq2Map and U-Net architecture
  - Encoder: Processes input through a series of convolutional layers
  - Decoder: Reconstructs the adjacency matrix through upsampling
  - Seq2Map: Handles sequence to map conversion with attention mechanisms
- `constraint_matrix`: Enforces base-pairing rules (A-U, C-G, G-U)
- `row_col_argmax`: Implements the K-rook strategy for structure prediction

## Hydra Configuration

Stage A's behavior is controlled through Hydra configuration in `conf/model/stageA.yaml`:

```yaml
stageA:
  # Model Architecture
  num_hidden: 128  # Number of hidden units in Seq2Map
  dropout: 0.3     # Dropout rate for regularization
  
  # Model Loading
  checkpoint_path: "checkpoints/RNAStralign_trainset_pretrained.pth"
  checkpoint_url: "https://www.dropbox.com/s/l04l9bf3v6z2tfd/checkpoints.zip?dl=1"  # URL to download checkpoint if not found
  use_gpu: true    # Whether to use GPU for inference
  
  # Processing Parameters
  min_seq_length: 80  # Minimum sequence length for processing
  batch_size: 32      # Batch size for processing multiple sequences
  lr: 0.001           # Learning rate (used during training/fine-tuning)
  
  # Output Configuration
  threshold: 0.5      # Threshold for converting probabilistic to binary adjacency
  
  # Visualization
  visualization:
    enabled: true
    varna_jar_path: "tools/varna-3-93.jar"
    resolution: 8.0   # VARNA visualization resolution
    
  # Model Architecture Details
  model:
    # U-Net Architecture Parameters
    conv_channels: [64, 128, 256, 512]  # Channel dimensions for U-Net
    residual: true                      # Whether to use residual connections
    c_in: 1                            # Input channels for U-Net
    c_out: 1                           # Output channels for U-Net
    c_hid: 32                          # Hidden channels for U-Net
    
    # Seq2Map Parameters
    seq2map:
      input_dim: 4                     # Input dimension (number of RNA bases)
      max_length: 3000                 # Maximum sequence length for positional encoding
      attention_heads: 8               # Number of attention heads
      attention_dropout: 0.1           # Dropout rate for attention layers
      positional_encoding: true        # Whether to use positional encoding
      query_key_dim: 128               # Query/key dimension for attention mechanism
      expansion_factor: 2.0            # Expansion factor for attention mechanism
      heads: 1                         # Number of heads for OffsetScale
      
    # Decoder Parameters
    decoder:
      up_conv_channels: [256, 128, 64]  # Channel dimensions for upsampling
      skip_connections: true            # Whether to use skip connections
```

### Configuration Parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_hidden` | int | 128 | Number of hidden units in the Seq2Map component. Controls the dimensionality of the token embeddings and positional encodings. Larger values increase model capacity but require more memory. |
| `dropout` | float | 0.3 | Dropout rate for regularization. Applied to the Seq2Map component to prevent overfitting. Higher values increase regularization but may reduce model performance. |
| `checkpoint_path` | str | "checkpoints/RNAStralign_trainset_pretrained.pth" | Path to the pre-trained model checkpoint. The model will load weights from this file during initialization. |
| `checkpoint_url` | str | "https://www.dropbox.com/s/l04l9bf3v6z2tfd/checkpoints.zip?dl=1" | URL to download checkpoint if not found locally. Used by the automatic checkpoint download mechanism. |
| `use_gpu` | bool | true | Whether to use GPU for inference. If true and a GPU is available, the model will run on the GPU for faster processing. |
| `min_seq_length` | int | 80 | Minimum sequence length for processing. Sequences shorter than this will be padded to this length. This ensures efficient processing by making sequence lengths multiples of 16. |
| `batch_size` | int | 32 | Batch size for processing multiple sequences. Larger batch sizes improve throughput but require more memory. |
| `lr` | float | 0.001 | Learning rate used during training or fine-tuning. Controls how quickly the model adapts to new data. |
| `threshold` | float | 0.5 | Threshold for converting probabilistic to binary adjacency. Values above this threshold are considered base pairs. |
| `visualization.enabled` | bool | true | Whether to generate VARNA visualization of the predicted secondary structure. |
| `visualization.varna_jar_path` | str | "tools/varna-3-93.jar" | Path to the VARNA JAR file used for visualization. VARNA is a tool for visualizing RNA secondary structures. |
| `visualization.resolution` | float | 8.0 | Resolution for VARNA visualization. Higher values produce higher quality images but require more processing time. |
| `model.conv_channels` | list | [64, 128, 256, 512] | Channel dimensions for U-Net encoder. Defines the number of filters at each level of the U-Net architecture. |
| `model.residual` | bool | true | Whether to use residual connections in conv blocks. Residual connections help with gradient flow and can improve training stability. |
| `model.c_in` | int | 1 | Input channels for U-Net. Set to 1 as the input is a single-channel attention map. |
| `model.c_out` | int | 1 | Output channels for U-Net. Set to 1 as the output is a single-channel adjacency matrix. |
| `model.c_hid` | int | 32 | Hidden channels for U-Net. Controls the number of features in the bottleneck layer. |
| `model.seq2map.input_dim` | int | 4 | Input dimension for Seq2Map, representing the number of RNA bases (A, U, C, G). |
| `model.seq2map.max_length` | int | 3000 | Maximum sequence length for positional encoding. Sequences longer than this may not be processed correctly. |
| `model.seq2map.attention_heads` | int | 8 | Number of attention heads in Seq2Map. Multi-head attention allows the model to focus on different aspects of the sequence simultaneously. |
| `model.seq2map.attention_dropout` | float | 0.1 | Dropout rate for attention layers. Applied to the attention weights to prevent overfitting. |
| `model.seq2map.positional_encoding` | bool | true | Whether to use positional encoding. Positional encodings help the model understand the relative positions of nucleotides in the sequence. |
| `model.seq2map.query_key_dim` | int | 128 | Query/key dimension for attention mechanism. Controls the dimensionality of the query and key vectors in the attention computation. |
| `model.seq2map.expansion_factor` | float | 2.0 | Expansion factor for attention mechanism. Controls how much the attention scores are scaled before normalization. |
| `model.seq2map.heads` | int | 1 | Number of heads for OffsetScale. Controls the number of parallel scaling operations in the attention mechanism. |
| `model.decoder.up_conv_channels` | list | [256, 128, 64] | Channel dimensions for upsampling in the decoder. Defines the number of filters at each upsampling level. |
| `model.decoder.skip_connections` | bool | true | Whether to use skip connections in decoder. Skip connections help preserve fine-grained details from the encoder. |

### Model Architecture Details

The RFold model combines a Seq2Map attention mechanism with a U-Net architecture to predict RNA secondary structure:

1. **Seq2Map Component**:
   - Converts the RNA sequence into a contact map using attention mechanisms
   - Uses token embeddings to represent each nucleotide
   - Applies positional encodings to capture sequence position information
   - Employs multi-head attention to model interactions between nucleotides

2. **U-Net Architecture**:
   - Encoder: Processes the attention map through a series of convolutional layers with increasing channel dimensions
   - Decoder: Reconstructs the adjacency matrix through upsampling and convolutional layers
   - Skip connections: Preserve fine-grained details from the encoder to the decoder
   - Final readout: Converts the decoder output into the final adjacency matrix

3. **K-Rook Strategy**:
   - Uses row/column argmax to ensure each nucleotide pairs with at most one other nucleotide
   - Applies constraint matrix to enforce valid base-pairing rules (A-U, C-G, G-U)
   - Produces a binary adjacency matrix representing the predicted secondary structure

### Code Integration:

The configuration is loaded and used in the following way:

```python
def __init__(self, config, checkpoint_path=None, device=None):
    # Load config from file if string path is provided
    if isinstance(config, str):
        with open(config, "r") as f:
            config = json.load(f)
    
    # Set device based on config
    use_gpu = config.get("use_gpu", True)
    self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    
    # Initialize model with config parameters
    self.model = RFoldModel(args_namespace(config))
    self.model.to(self.device)
    self.model.eval()
    
    # Load checkpoint if provided
    if checkpoint_path:
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
```

## Integration

### Upstream Dependencies
- None. Stage A is the starting point of the pipeline for structure prediction.

### Downstream Dependencies
- **Stage B**: Uses the adjacency matrix as input for TorsionBERT and Pairformer
  - TorsionBERT may use adjacency information to condition angle predictions
  - Pairformer uses the adjacency matrix to guide pair embedding generation
- **Unified Latent Merger**: Incorporates adjacency information into the conditioning latent
- **Stage D**: May use adjacency information for diffusion guidance

### Data Flow
1. RNA sequence is input to Stage A
2. Stage A generates the adjacency matrix through the following steps:
   - Sequence preprocessing and padding
   - RFold model inference
   - Row/column argmax processing
   - Constraint matrix application
3. The adjacency matrix is passed to Stage B for further processing
4. The adjacency information flows through the pipeline, influencing subsequent stages

## Edge Cases & Error Handling

### Sequence Length Issues
- **Short Sequences**: If the sequence is less than 4 nucleotides, the model returns a zero adjacency matrix
- **Long Sequences**: For sequences exceeding the model's maximum length, they are processed in chunks
  - The `_get_cut_len()` method ensures sequences are padded to multiples of 16
  - Minimum sequence length of 80 nucleotides is enforced

### Checkpoint Issues
- **Missing Checkpoint**: The code attempts to download the checkpoint if it's not found locally
  - Uses a predefined URL for checkpoint download
  - Verifies downloaded file integrity
- **Corrupted Checkpoint**: Error handling for corrupted model files with informative error messages

### Dimension Mismatches
- Validation to ensure the adjacency matrix dimensions match the input sequence length
- Shape assertions to catch potential issues early in the pipeline
- Proper handling of padded sequences in the output

### HPC Considerations
- Memory usage optimization for large sequences
- GPU memory management for batch processing
- Efficient sequence padding and processing

## References & Dependencies

### Papers
- RFold: "RNA Secondary Structure Prediction by Learning Unrolled Algorithms" (Chen et al., ICLR 2019)
- K-rook approach for RNA secondary structure: "Deciphering RNA Secondary Structure Prediction: A Probabilistic K-Rook Matching Perspective" (Tan et al., arXiv 2024)

### Software Dependencies
- PyTorch (version ≥ 1.8.0)
- VARNA (for visualization)
- NumPy (for matrix operations)

### External Resources
- RFold GitHub repository: https://github.com/keio-bioinformatics/neuralfold/
- VARNA documentation: http://varna.lri.fr/ 
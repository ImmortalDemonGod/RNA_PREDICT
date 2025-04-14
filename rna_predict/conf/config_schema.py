# rna_predict/conf/config_schema.py
from dataclasses import dataclass, field
from typing import Optional, List

# --- Nested Config Structures for Stage A based on StageA_2D_Adjacency.md ---

@dataclass
class VisualizationConfig:
    """Configuration for VARNA visualization."""
    enabled: bool = True
    varna_jar_path: str = "tools/varna-3-93.jar" # Path to the VARNA JAR file
    resolution: float = 8.0 # Resolution for VARNA visualization

@dataclass
class Seq2MapConfig:
    """Configuration for the Seq2Map component of the RFold model."""
    input_dim: int = 4  # Input dimension (A, U, C, G)
    max_length: int = 3000 # Max sequence length for positional encoding
    attention_heads: int = 8 # Number of attention heads
    attention_dropout: float = 0.1 # Dropout rate for attention layers
    positional_encoding: bool = True # Whether to use positional encoding
    query_key_dim: int = 128 # Query/key dimension for attention
    expansion_factor: float = 2.0 # Expansion factor for attention mechanism
    heads: int = 1 # Number of heads for OffsetScale

@dataclass
class DecoderConfig:
    """Configuration for the Decoder component (U-Net based) of the RFold model."""
    up_conv_channels: List[int] = field(default_factory=lambda: [256, 128, 64]) # Channels for upsampling layers
    skip_connections: bool = True # Whether to use skip connections

@dataclass
class ModelArchConfig:
    """Configuration for the overall RFold model architecture."""
    # U-Net Architecture Parameters
    conv_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512]) # Channels for U-Net encoder layers
    residual: bool = True # Whether to use residual connections
    c_in: int = 1 # Input channels for U-Net
    c_out: int = 1 # Output channels for U-Net
    c_hid: int = 32 # Hidden channels for U-Net bottleneck
    # Nested Seq2Map and Decoder Configs
    seq2map: Seq2MapConfig = field(default_factory=Seq2MapConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

# --- Main Stage A Config ---

@dataclass
class StageAConfig:
    """Configuration specific to Stage A (RFold Adjacency Prediction)."""
    # Top-level architecture and processing params (may overlap/inform nested model params)
    num_hidden: int = 128 # Often relates to attention dims, e.g., query_key_dim
    dropout: float = 0.3 # General dropout, potentially distinct from attention_dropout

    # Processing Parameters
    min_seq_length: int = 80 # Minimum sequence length for padding
    batch_size: int = 32 # Batch size for processing (if applicable)
    lr: float = 0.001 # Learning rate (for training/fine-tuning)

    # Model Loading & Device
    device: str = "cuda"  # Specify device as string ("cuda" or "cpu")
    checkpoint_path: str = "RFold/checkpoints/RNAStralign_trainset_pretrained.pth"
    checkpoint_url: str = "[https://www.dropbox.com/s/l04l9bf3v6z2tfd/checkpoints.zip?dl=1](https://www.dropbox.com/s/l04l9bf3v6z2tfd/checkpoints.zip?dl=1)" # URL for download if needed

    # Output Configuration
    threshold: float = 0.5 # Threshold for binary adjacency map

    # Nested Configurations
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    model: ModelArchConfig = field(default_factory=ModelArchConfig) # Contains U-Net, Seq2Map, Decoder params

# --- Overall RNA Pipeline Config ---

@dataclass
class RNAConfig:
    """Root configuration for the entire RNA_PREDICT pipeline."""
    # Default to StageAConfig. Fields for other stages can be added later.
    stageA: StageAConfig = field(default_factory=StageAConfig)
    # TODO: Add dataclasses for Stage B, C, D etc. when integrating them
    # stageB_torsion: Optional[TorsionBertConfig] = None
    # stageB_pairformer: Optional[PairformerConfig] = None
    # stageC: Optional[StageCConfig] = None
    # stageD: Optional[StageDConfig] = None
    # latent_merger: Optional[LatentMergerConfig] = None
    # memory_optimization: Optional[MemoryOptimizationConfig] = None
    # energy_minimization: Optional[EnergyMinimizationConfig] = None
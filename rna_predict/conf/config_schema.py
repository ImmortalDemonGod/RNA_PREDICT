# rna_predict/conf/config_schema.py
from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict, Any
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from rna_predict.pipeline.stageD.config import NoiseScheduleConfig, SamplingConfig

@dataclass
class DimensionsConfig:
    """Centralized configuration for model dimensions across all stages."""
    # Single representation dimensions
    c_s: int = field(
        default=384,
        metadata={"help": "Single representation dimension used across all stages"}
    )

    # Pair representation dimensions
    c_z: int = field(
        default=128,
        metadata={"help": "Pair representation dimension used across all stages"}
    )

    # Input feature dimensions
    c_s_inputs: int = field(
        default=449,
        metadata={"help": "Input representation dimension"}
    )
    c_token: int = field(
        default=449,
        metadata={"help": "Token dimension for embeddings"}
    )

    # Atom dimensions
    c_atom: int = field(
        default=128,
        metadata={"help": "Atom dimension for embeddings"}
    )
    c_atom_coords: int = field(
        default=3,
        metadata={"help": "Coordinate dimension (x, y, z)"}
    )

    # Other common dimensions
    c_noise_embedding: int = field(
        default=32,
        metadata={"help": "Noise embedding dimension"}
    )
    restype_dim: int = field(
        default=32,
        metadata={"help": "Dimension for residue type embeddings"}
    )
    profile_dim: int = field(
        default=32,
        metadata={"help": "Dimension for profile embeddings"}
    )
    c_pair: int = field(
        default=32,
        metadata={"help": "Pair dimension for embeddings (different from c_z)"}
    )

    # Reference dimensions
    ref_element_size: int = field(
        default=128,
        metadata={"help": "Size of reference element embeddings"}
    )
    ref_atom_name_chars_size: int = field(
        default=256,
        metadata={"help": "Size of atom name character embeddings"}
    )

# Register configs with Hydra's config store
cs = ConfigStore.instance()

def register_configs() -> None:
    """Register all configurations with Hydra's config store for validation."""
    cs.store(name="rna_config_schema", node=RNAConfig)
    cs.store(group="model", name="dimensions", node=DimensionsConfig)
    cs.store(group="model", name="stageA", node=StageAConfig)
    cs.store(group="model", name="stageB_torsion", node=TorsionBertConfig)
    cs.store(group="model", name="stageB_pairformer", node=PairformerConfig)
    cs.store(group="model", name="stageC", node=StageCConfig)
    cs.store(group="model", name="stageD", node=StageDConfig)
    cs.store(group="model", name="stageD_model_arch", node=StageDModelArchConfig)
    cs.store(group="model", name="stageD_transformer", node=StageDTransformerConfig)
    cs.store(group="model", name="stageD_atom_encoder", node=StageDAtomEncoderConfig)
    cs.store(group="model", name="stageD_atom_decoder", node=StageDAtomDecoderConfig)
    cs.store(group="model", name="latent_merger", node=LatentMergerConfig)
    cs.store(group="optimization", name="memory", node=MemoryOptimizationConfig)
    cs.store(group="optimization", name="energy", node=EnergyMinimizationConfig)
    cs.store(group="pipeline", name="default", node=PipelineConfig)
    cs.store(group="test", name="data", node=TestDataConfig)
    cs.store(group="model", name="protenix_integration", node=ProtenixIntegrationConfig)

def validate_config(cfg: Union[dict, "RNAConfig"]) -> None:
    """Validate configuration using OmegaConf structured validation.

    Args:
        cfg: Either a dict-like config or an instantiated RNAConfig
    Raises:
        ValidationError: If config is invalid
    """
    try:
        # Convert to OmegaConf if needed
        if not OmegaConf.is_config(cfg):
            try:
                cfg = OmegaConf.structured(cfg)
            except Exception as e:
                print(f"[ERROR] Failed to convert config to OmegaConf: {e}")
                raise ValueError(f"Failed to convert config to OmegaConf: {e}") from e

        # Merge with schema for validation
        try:
            schema = OmegaConf.structured(RNAConfig)
        except Exception as e:
            print(f"[ERROR] Failed to create schema from RNAConfig: {e}")
            raise ValueError(f"Failed to create schema from RNAConfig: {e}") from e

        try:
            OmegaConf.merge(schema, cfg)
        except Exception as e:
            print(f"[ERROR] Configuration validation failed during merge: {e}")
            raise ValueError(f"Configuration validation failed: {str(e)}") from e
    except Exception as e:
        print(f"[ERROR] Unexpected error in validate_config: {e}")
        raise

# --- Nested Config Structures for Stage A based on StageA_2D_Adjacency.md ---

@dataclass
class VisualizationConfig:
    """Configuration for VARNA visualization."""
    enabled: bool = True
    varna_jar_path: str = "tools/varna-3-93.jar" # Path to the VARNA JAR file
    resolution: float = 8.0 # Resolution for VARNA visualization
    output_path: str = "test_seq.png" # Output path for visualization

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
    # Top-level architecture and processing params
    num_hidden: int = field(
        default=128,
        metadata={"help": "Hidden dimension size, often relates to attention dims"}
    )
    dropout: float = field(
        default=0.3,
        metadata={
            "help": "General dropout rate",
            "validate": lambda x: 0.0 <= x <= 1.0
        }
    )
    debug_logging: bool = field(
        default=True,
        metadata={"help": "Enable debug logging for Stage A"}
    )
    # NEW: freeze all parameters for testing (default True for test parity)
    freeze_params: bool = field(
        default=True,
        metadata={"help": "If True, freeze all model parameters (requires_grad=False) for eval/test."}
    )

    # Processing Parameters
    min_seq_length: int = field(
        default=80,
        metadata={
            "help": "Minimum sequence length for padding",
            "validate": lambda x: x > 0
        }
    )
    batch_size: int = field(
        default=32,
        metadata={
            "help": "Batch size for processing",
            "validate": lambda x: x > 0
        }
    )
    lr: float = field(
        default=0.001,
        metadata={
            "help": "Learning rate",
            "validate": lambda x: x > 0.0
        }
    )

    # Model Loading & Device
    device: str = field(
        default="cuda",
        metadata={"help": "Device to run the model on: 'cuda', 'cpu', or 'mps' (Apple Silicon)"}
    )
    checkpoint_path: str = field(
        default="RFold/checkpoints/RNAStralign_trainset_pretrained.pth",
        metadata={"help": "Path to model checkpoint"}
    )
    checkpoint_url: Optional[str] = field(
        default="https://www.dropbox.com/s/l04l9bf3v6z2tfd/checkpoints.zip?dl=1",
        metadata={"help": "URL to download checkpoint if not found locally"}
    )
    checkpoint_zip_path: str = field(
        default="RFold/checkpoints.zip",
        metadata={"help": "Path to save downloaded checkpoint zip file"}
    )

    # Output Configuration
    threshold: float = field(
        default=0.5,
        metadata={
            "help": "Threshold for binary adjacency map",
            "validate": lambda x: 0.0 <= x <= 1.0
        }
    )

    # Example/Testing
    run_example: bool = field(
        default=True,
        metadata={"help": "Whether to run an example inference"}
    )
    example_sequence: str = field(
        default="AAGUCUGGUGGACAUUGGCGUCCUGAGGUGUUAAAACCUCUUAUUGCUGACGCCAGAAAGAGAAGAACUUCGGUUCUACUAGUCGACUAUACUACAAGCUUUGGGUGUAUAGCGGCAAGACAACCUGGAUCGGGGGAGGCUAAGGGCGCAAGCCUAUGCUAACCCCGAGCCGAGCUACUGGAGGGCAACCCCCAGAUAGCCGGUGUAGAGCGCGGAAAGGUGUCGGUCAUCCUAUCUGAUAGGUGGCUUGAGGGACGUGCCGUCUCACCCGAAAGGGUGUUUCUAAGGAGGAGCUCCCAAAGGGCAAAUCUUAGAAAAGGGUGUAUACCCUAUAAUUUAACGGCCAGCAGCC",
        metadata={"help": "Example RNA sequence to use for testing"}
    )

    # Nested Configurations
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    model: ModelArchConfig = field(default_factory=ModelArchConfig)

    def _validate_numeric_ranges(self):
        """Validate dropout and threshold ranges."""
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError(f"dropout must be between 0 and 1, got {self.dropout}")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be between 0 and 1, got {self.threshold}")

    def _validate_positive_integers(self):
        """Validate positive integer constraints."""
        if self.min_seq_length <= 0:
            raise ValueError(f"min_seq_length must be positive, got {self.min_seq_length}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

    def _validate_learning_rate(self):
        """Validate learning rate."""
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")

    def _validate_device(self):
        """Validate device selection."""
        if self.device not in ["cuda", "cpu", "mps"]:
            raise ValueError(f"device must be 'cuda', 'cpu', or 'mps', got {self.device}")

    def __post_init__(self):
        """Validate configuration after initialization by calling helper methods."""
        self._validate_numeric_ranges()
        self._validate_positive_integers()
        self._validate_learning_rate()
        self._validate_device()

# --- Overall RNA Pipeline Config ---

@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) settings."""
    enabled: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA adaptation"}
    )
    r: int = field(
        default=8,
        metadata={
            "help": "LoRA rank",
            "validate": lambda x: x > 0
        }
    )
    alpha: int = field(
        default=16,
        metadata={
            "help": "LoRA alpha scaling factor",
            "validate": lambda x: x > 0
        }
    )
    dropout: float = field(
        default=0.1,
        metadata={
            "help": "LoRA dropout rate",
            "validate": lambda x: 0.0 <= x <= 1.0
        }
    )
    target_modules: List[str] = field(
        default_factory=list,
        metadata={"help": "List of module names to apply LoRA to"}
    )

@dataclass
class TorsionBertConfig:
    """Configuration for TorsionBERT model."""
    model_name_or_path: str = field(
        default="sayby/rna_torsionbert",
        metadata={"help": "HuggingFace model name or local path"}
    )
    device: str = field(
        default="cuda",
        metadata={"help": "Device to run the model on"}
    )
    angle_mode: str = field(
        default="sin_cos",
        metadata={"help": "Angle representation mode: sin_cos, degrees, or radians"}
    )
    num_angles: int = field(
        default=7,
        metadata={"help": "Number of torsion angles to predict"}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length"}
    )
    checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to model checkpoint"}
    )
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    debug_logging: bool = field(
        default=True,
        metadata={"help": "Enable debug logging for Stage B (TorsionBert)"}
    )
    init_from_scratch: bool = field(
        default=False,
        metadata={"help": "If true, initialize TorsionBert from scratch instead of loading weights"}
    )

@dataclass
class PairformerBlockConfig:
    """Configuration for PairformerBlock."""
    n_heads: int = field(
        default=16,
        metadata={
            "help": "Number of attention heads for AttentionPairBias",
            "validate": lambda x: x > 0
        }
    )
    c_z: int = field(default=128, metadata={"help": "Hidden dimension for pair embedding"})
    c_s: int = field(default=384, metadata={"help": "Hidden dimension for single embedding"})
    c_hidden_mul: int = field(default=128, metadata={"help": "Hidden dimension for TriangleMultiplicationOutgoing"})
    c_hidden_pair_att: int = field(default=32, metadata={"help": "Hidden dimension for TriangleAttention"})
    no_heads_pair: int = field(default=4, metadata={"help": "Number of heads for TriangleAttention"})
    dropout: float = field(
        default=0.25,
        metadata={
            "help": "Dropout ratio for TriangleUpdate",
            "validate": lambda x: 0.0 <= x <= 1.0
        }
    )

@dataclass
class PairformerStackConfig:
    """Configuration for PairformerStack."""
    n_blocks: int = field(
        default=48,
        metadata={
            "help": "Number of transformer blocks",
            "validate": lambda x: x > 0
        }
    )
    n_heads: int = field(
        default=16,
        metadata={
            "help": "Number of attention heads",
            "validate": lambda x: x > 0
        }
    )
    c_z: int = field(default=128, metadata={"help": "Hidden dimension for pair embedding"})
    c_s: int = field(default=384, metadata={"help": "Hidden dimension for single embedding"})
    dropout: float = field(
        default=0.25,
        metadata={
            "help": "Dropout rate",
            "validate": lambda x: 0.0 <= x <= 1.0
        }
    )
    blocks_per_ckpt: Optional[int] = field(
        default=None,
        metadata={"help": "Number of blocks per checkpoint, None for no checkpointing"}
    )

@dataclass
class MSAConfig:
    """Configuration for MSA-related components."""
    c_m: int = field(default=64, metadata={"help": "Hidden dimension for MSA embedding"})
    c: int = field(default=32, metadata={"help": "Hidden dimension for MSA components"})
    c_z: int = field(default=128, metadata={"help": "Hidden dimension for pair embedding"})
    dropout: float = field(
        default=0.15,
        metadata={
            "help": "Dropout ratio for MSA components",
            "validate": lambda x: 0.0 <= x <= 1.0
        }
    )
    n_blocks: int = field(
        default=4,
        metadata={
            "help": "Number of MSA blocks",
            "validate": lambda x: x >= 0
        }
    )
    enable: bool = field(default=False, metadata={"help": "Whether to use MSA embedding"})
    strategy: str = field(default="random", metadata={"help": "Strategy for MSA sampling"})
    train_cutoff: int = field(default=512, metadata={"help": "MSA sample cutoff during training"})
    test_cutoff: int = field(default=16384, metadata={"help": "MSA sample cutoff during testing"})
    train_lowerb: int = field(default=1, metadata={"help": "Minimum MSA sample size during training"})
    test_lowerb: int = field(default=1, metadata={"help": "Minimum MSA sample size during testing"})

    # Additional parameters needed by MSAPairWeightedAveraging
    n_heads: int = field(default=8, metadata={"help": "Number of attention heads for MSA pair weighted averaging"})

    # Additional parameters needed by MSABlock
    pair_dropout: float = field(
        default=0.25,
        metadata={
            "help": "Dropout ratio for pair stack in MSA block",
            "validate": lambda x: 0.0 <= x <= 1.0
        }
    )

    # Input feature dimensions
    input_feature_dims: Dict[str, int] = field(
        default_factory=lambda: {
            "msa": 32,
            "has_deletion": 1,
            "deletion_value": 1,
        },
        metadata={"help": "Dimensions for input features"}
    )

    # Parameters for MSAModule
    c_s_inputs: int = field(default=449, metadata={"help": "Hidden dimension for single embedding from InputFeatureEmbedder"})
    blocks_per_ckpt: Optional[int] = field(
        default=1,
        metadata={"help": "Number of MSA blocks in each activation checkpoint"}
    )

@dataclass
class TemplateEmbedderConfig:
    """Configuration for TemplateEmbedder."""
    n_blocks: int = field(
        default=2,
        metadata={
            "help": "Number of blocks for TemplateEmbedder",
            "validate": lambda x: x >= 0
        }
    )
    c: int = field(default=64, metadata={"help": "Hidden dimension of TemplateEmbedder"})
    c_z: int = field(default=128, metadata={"help": "Hidden dimension for pair embedding"})
    dropout: float = field(
        default=0.25,
        metadata={
            "help": "Dropout ratio for PairformerStack",
            "validate": lambda x: 0.0 <= x <= 1.0
        }
    )
    blocks_per_ckpt: Optional[int] = field(
        default=None,
        metadata={"help": "Number of blocks per checkpoint, None for no checkpointing"}
    )

    # Input feature dimensions
    input_feature_dims: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {
            "feature1": {
                "template_distogram": 39,
                "b_template_backbone_frame_mask": 1,
                "template_unit_vector": 3,
                "b_template_pseudo_beta_mask": 1,
            },
            "feature2": {
                "template_restype_i": 32,
                "template_restype_j": 32,
            }
        },
        metadata={"help": "Dimensions for input features"}
    )

    # Distogram parameters
    distogram: Dict[str, float] = field(
        default_factory=lambda: {
            "max_bin": 50.75,
            "min_bin": 3.25,
            "no_bins": 39,
        },
        metadata={"help": "Distogram parameters"}
    )

@dataclass
class ProtenixIntegrationConfig:
    """Configuration for ProtenixIntegration class."""
    # Device setting
    device: str = field(
        default="cpu",
        metadata={"help": "Device to run the model on: 'cpu', 'cuda', or 'mps'"}
    )

    # Embedding dimensions
    c_token: int = field(default=449, metadata={"help": "Token dimension for embeddings"})
    restype_dim: int = field(default=32, metadata={"help": "Dimension for residue type embeddings"})
    profile_dim: int = field(default=32, metadata={"help": "Dimension for profile embeddings"})
    c_atom: int = field(default=128, metadata={"help": "Atom dimension for embeddings"})
    c_pair: int = field(default=32, metadata={"help": "Pair dimension for embeddings"})

    # Attention parameters
    num_heads: int = field(default=4, metadata={"help": "Number of attention heads"})
    num_layers: int = field(default=3, metadata={"help": "Number of attention layers"})

    # Relative position encoding parameters
    r_max: int = field(default=32, metadata={"help": "Maximum relative position"})
    s_max: int = field(default=2, metadata={"help": "Maximum sequence separation"})

    # Optimization flags
    use_optimized: bool = field(default=False, metadata={"help": "Whether to use optimized implementation"})

@dataclass
class PairformerConfig:
    """Configuration for Pairformer model."""
    # Device setting
    device: str = field(
        default="cpu",
        metadata={"help": "Device to run the model on: 'cpu', 'cuda', or 'mps'"}
    )

    # Core model parameters
    n_blocks: int = field(
        default=1,
        metadata={
            "help": "Number of transformer blocks (minimal for test/memory)"
        }
    )
    n_heads: int = field(
        default=1,
        metadata={
            "help": "Number of attention heads (minimal for test/memory)"
        }
    )
    c_z: int = field(
        default=2,
        metadata={
            "help": "Pair embedding dimension (minimal for test/memory)"
        }
    )
    c_s: int = field(
        default=2,
        metadata={
            "help": "Single embedding dimension (minimal for test/memory)"
        }
    )
    dropout: float = field(
        default=0.1,
        metadata={
            "help": "Dropout rate for attention",
            "validate": lambda x: 0.0 <= x <= 1.0
        }
    )
    # NEW: freeze all parameters for test parity
    freeze_params: bool = field(
        default=False,
        metadata={"help": "If True, freeze all model parameters (requires_grad=False) for eval/test."}
    )

    # ProtenixIntegration configuration
    protenix_integration: ProtenixIntegrationConfig = field(default_factory=ProtenixIntegrationConfig)

    # Triangle Attention parameters
    c_hidden_mul: int = field(default=2, metadata={"help": "Hidden dimension multiplier"})
    c_hidden_pair_att: int = field(default=128, metadata={"help": "Hidden dimension for pair attention"})
    no_heads_pair: int = field(default=8, metadata={"help": "Number of heads for pair attention"})
    init_z_from_adjacency: bool = field(default=False, metadata={"help": "Whether to initialize Z from adjacency matrix"})

    # Memory optimization flags
    use_checkpoint: bool = field(
        default=True,
        metadata={"help": "Whether to use gradient checkpointing for memory efficiency"}
    )
    use_memory_efficient_kernel: bool = field(
        default=False,
        metadata={"help": "Whether to use memory efficient attention"}
    )
    use_deepspeed_evo_attention: bool = field(
        default=False,
        metadata={"help": "Whether to use DeepSpeed evolution attention"}
    )
    use_lma: bool = field(
        default=False,
        metadata={"help": "Whether to use linear multi-head attention"}
    )
    inplace_safe: bool = field(
        default=False,
        metadata={"help": "Whether to use inplace operations safely"}
    )
    chunk_size: Optional[int] = field(
        default=None,
        metadata={"help": "Chunk size for attention computation"}
    )

    # Component-specific configurations
    block: PairformerBlockConfig = field(default_factory=PairformerBlockConfig)
    stack: PairformerStackConfig = field(default_factory=PairformerStackConfig)
    msa: MSAConfig = field(default_factory=MSAConfig)
    template: TemplateEmbedderConfig = field(default_factory=TemplateEmbedderConfig)

    # LoRA configuration
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    debug_logging: bool = field(
        default=True,
        metadata={"help": "Enable debug logging for Stage B (Pairformer)"}
    )

@dataclass
class StageCConfig:
    """Configuration for Stage C (MP-NeRF)."""
    enabled: bool = True
    method: str = field(
        default="mp_nerf",
        metadata={"help": "Method to use: mp_nerf or legacy"}
    )
    do_ring_closure: bool = False
    place_bases: bool = True
    sugar_pucker: str = field(
        default="C3'-endo",
        metadata={"help": "Sugar pucker conformation: C3'-endo or C2'-endo"}
    )
    device: str = field(
        default="cuda",
        metadata={"help": "Device to run the model on"}
    )
    angle_representation: str = field(
        default="cartesian",
        metadata={"help": "Angle representation: cartesian or internal"}
    )
    use_metadata: bool = False
    use_memory_efficient_kernel: bool = False
    use_deepspeed_evo_attention: bool = False
    use_lma: bool = False
    inplace_safe: bool = False
    chunk_size: Optional[int] = None
    debug_logging: bool = True

@dataclass
class StageDInferenceConfig:
    """Configuration for Stage D inference."""
    num_steps: int = field(
        default=2,
        metadata={"help": "Number of diffusion steps"}
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for sampling"}
    )
    use_ddim: bool = field(
        default=True,
        metadata={"help": "Whether to use DDIM sampling"}
    )
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

@dataclass
class StageDModelArchConfig:
    """Configuration for Stage D model architecture."""
    c_token: int = field(
        default=384,
        metadata={"help": "Token dimension"}
    )
    c_s: int = field(
        default=384,
        metadata={"help": "Single representation dimension"}
    )
    c_z: int = field(
        default=128,
        metadata={"help": "Pair representation dimension"}
    )
    c_s_inputs: int = field(
        default=449,
        metadata={"help": "Input representation dimension"}
    )
    c_atom: int = field(
        default=128,
        metadata={"help": "Atom embedding dimension"}
    )
    c_noise_embedding: int = field(
        default=32,
        metadata={"help": "Noise embedding dimension"}
    )
    num_layers: int = field(
        default=6,
        metadata={"help": "Number of transformer layers"}
    )
    num_heads: int = field(
        default=8,
        metadata={"help": "Number of attention heads"}
    )
    dropout: float = field(
        default=0.1,
        metadata={
            "help": "Dropout rate",
            "validate": lambda x: 0.0 <= x <= 1.0
        }
    )
    coord_eps: float = field(
        default=1e-6,
        metadata={"help": "Epsilon for coordinate calculations"}
    )
    coord_min: float = field(
        default=-1e4,
        metadata={"help": "Minimum coordinate value"}
    )
    coord_max: float = field(
        default=1e4,
        metadata={"help": "Maximum coordinate value"}
    )
    coord_similarity_rtol: float = field(
        default=1e-3,
        metadata={"help": "Relative tolerance for coordinate similarity"}
    )
    test_residues_per_batch: int = field(
        default=25,
        metadata={"help": "Number of residues per batch during testing"}
    )
    sigma_data: float = field(
        default=16.0,
        metadata={"help": "Sigma data parameter for diffusion (should be under model_architecture)"}
    )

@dataclass
class StageDTransformerConfig:
    """Configuration for Stage D transformer."""
    n_blocks: int = field(
        default=6,
        metadata={"help": "Number of transformer blocks"}
    )
    n_heads: int = field(
        default=8,
        metadata={"help": "Number of attention heads"}
    )
    blocks_per_ckpt: Optional[int] = field(
        default=None,
        metadata={"help": "Number of blocks per checkpoint, None for no checkpointing"}
    )

@dataclass
class StageDAtomEncoderConfig:
    """Configuration for Stage D atom encoder."""
    c_in: int = field(
        default=3,
        metadata={"help": "Input dimension for atom encoder"}
    )
    c_hidden: List[int] = field(
        default_factory=lambda: [32, 64, 128],
        metadata={"help": "Hidden dimensions for atom encoder"}
    )
    c_out: int = field(
        default=64,
        metadata={"help": "Output dimension for atom encoder"}
    )
    dropout: float = field(
        default=0.1,
        metadata={
            "help": "Dropout rate",
            "validate": lambda x: 0.0 <= x <= 1.0
        }
    )

@dataclass
class StageDAtomDecoderConfig:
    """Configuration for Stage D atom decoder."""
    c_in: int = field(
        default=64,
        metadata={"help": "Input dimension for atom decoder"}
    )
    c_hidden: List[int] = field(
        default_factory=lambda: [128, 64, 32],
        metadata={"help": "Hidden dimensions for atom decoder"}
    )
    c_out: int = field(
        default=3,
        metadata={"help": "Output dimension for atom decoder"}
    )
    dropout: float = field(
        default=0.1,
        metadata={
            "help": "Dropout rate",
            "validate": lambda x: 0.0 <= x <= 1.0
        }
    )

@dataclass
class StageDDiffusionConfig:
    """Configuration for Stage D diffusion."""
    init_from_scratch: bool = field(default=False, metadata={"help": "If true, initialize diffusion model from scratch"})
    enabled: bool = field(default=True, metadata={"help": "Enable diffusion for Stage D"})
    mode: str = field(default="inference", metadata={"help": "Mode: inference or training"})
    device: str = field(default="cpu", metadata={"help": "Device to run the diffusion model on"})
    debug_logging: bool = field(default=True, metadata={"help": "Enable debug logging for diffusion"})
    sigma_data: float = field(default=16.0, metadata={"help": "Sigma data parameter for diffusion"})
    c_atom: int = field(default=128, metadata={"help": "Atom embedding dimension"})
    c_s: int = field(default=384, metadata={"help": "Single representation dimension"})
    c_z: int = field(default=128, metadata={"help": "Pair representation dimension"})
    c_s_inputs: int = field(default=449, metadata={"help": "Input representation dimension"})
    c_noise_embedding: int = field(default=32, metadata={"help": "Noise embedding dimension"})
    ref_element_size: int = field(default=128, metadata={"help": "Reference element embedding size"})
    ref_atom_name_chars_size: int = field(default=256, metadata={"help": "Atom name char embedding size"})
    atom_metadata: Optional[dict] = None
    # Feature dimensions required for bridging
    feature_dimensions: Dict[str, int] = field(
        default_factory=lambda: {
            "c_s": 384,  # Single representation dimension
            "c_s_inputs": 449,  # Input representation dimension
            "c_sing": 384,  # Set to same as c_s by default
        },
        metadata={"help": "Dimensions for various features"}
    )
    test_residues_per_batch: int = field(
        default=25,
        metadata={"help": "Number of residues per batch during testing"}
    )
    # Nested configs (add as Any for now, can be typed later)
    model_architecture: Any = None
    transformer: Any = None
    atom_encoder: Any = None
    atom_decoder: Any = None
    noise_schedule: NoiseScheduleConfig = field(default_factory=NoiseScheduleConfig)
    inference: StageDInferenceConfig = field(default_factory=StageDInferenceConfig)
    use_memory_efficient_kernel: bool = field(default=False, metadata={"help": "Whether to use memory efficient attention kernel"})
    use_deepspeed_evo_attention: bool = field(default=False, metadata={"help": "Whether to use DeepSpeed evolution attention"})
    use_lma: bool = field(default=False, metadata={"help": "Whether to use linear multi-head attention"})
    inplace_safe: bool = field(default=False, metadata={"help": "Whether to use inplace operations safely"})
    chunk_size: Optional[int] = field(default=None, metadata={"help": "Chunk size for attention computation"})

@dataclass
class StageDConfig:
    """Configuration for Stage D (Diffusion)."""
    enabled: bool = True
    mode: str = field(
        default="inference",
        metadata={"help": "Mode: inference or training"}
    )
    device: str = field(
        default="cpu",
        metadata={"help": "Device to run the model on: cpu, cuda, or mps"}
    )
    debug_logging: bool = field(
        default=True,
        metadata={"help": "Enable debug logging"}
    )
    sigma_data: float = field(
        default=16.0,
        metadata={"help": "Sigma data parameter for diffusion"}
    )
    c_atom: int = field(
        default=128,
        metadata={"help": "Atom embedding dimension"}
    )
    c_s: int = field(
        default=384,
        metadata={"help": "Single representation dimension"}
    )
    c_z: int = field(
        default=128,
        metadata={"help": "Pair representation dimension"}
    )
    c_s_inputs: int = field(
        default=449,
        metadata={"help": "Input representation dimension"}
    )
    c_noise_embedding: int = field(
        default=32,
        metadata={"help": "Noise embedding dimension"}
    )
    ref_element_size: int = field(
        default=128,
        metadata={"help": "Reference element embedding size"}
    )
    ref_atom_name_chars_size: int = field(
        default=256,
        metadata={"help": "Atom name char embedding size"}
    )
    chunk_size: Optional[int] = field(
        default=None,
        metadata={"help": "Chunk size for attention computation"}
    )
    # Add nested configurations for model architecture and components
    model_architecture: StageDModelArchConfig = field(default_factory=StageDModelArchConfig)
    transformer: StageDTransformerConfig = field(default_factory=StageDTransformerConfig)
    atom_encoder: StageDAtomEncoderConfig = field(default_factory=StageDAtomEncoderConfig)
    atom_decoder: StageDAtomDecoderConfig = field(default_factory=StageDAtomDecoderConfig)
    diffusion: StageDDiffusionConfig = field(default_factory=StageDDiffusionConfig)

@dataclass
class LatentMergerConfig:
    """Configuration for merging latent representations."""
    merge_method: str = field(
        default="concat",
        metadata={"help": "Method to merge latents: concat, add, or attention"}
    )
    attention_heads: int = field(
        default=8,
        metadata={"help": "Number of attention heads if using attention merge"}
    )
    dropout: float = field(
        default=0.1,
        metadata={
            "help": "Dropout rate for attention",
            "validate": lambda x: 0.0 <= x <= 1.0
        }
    )
    output_dim: int = field(
        default=384,
        metadata={
            "help": "Output dimension after merging",
            "validate": lambda x: x > 0
        }
    )
    use_residual: bool = field(
        default=True,
        metadata={"help": "Whether to use residual connections"}
    )

@dataclass
class MemoryOptimizationConfig:
    """Configuration for memory optimization settings."""
    use_checkpointing: bool = field(
        default=True,
        metadata={"help": "Whether to use gradient checkpointing"}
    )
    checkpoint_every_n_layers: int = field(
        default=2,
        metadata={
            "help": "Number of layers between checkpoints",
            "validate": lambda x: x > 0
        }
    )
    optimize_memory_layout: bool = field(
        default=True,
        metadata={"help": "Whether to optimize memory layout"}
    )
    mixed_precision: bool = field(
        default=True,
        metadata={"help": "Whether to use mixed precision training"}
    )

@dataclass
class EnergyMinimizationConfig:
    """Configuration for energy minimization."""
    enabled: bool = True
    method: str = field(
        default="steepest_descent",
        metadata={"help": "Minimization method: steepest_descent or conjugate_gradient"}
    )
    max_iterations: int = field(
        default=1000,
        metadata={"help": "Maximum number of minimization iterations"}
    )
    tolerance: float = field(
        default=1e-6,
        metadata={"help": "Convergence tolerance"}
    )
    learning_rate: float = field(
        default=0.01,
        metadata={"help": "Learning rate for minimization"}
    )

@dataclass
class LanceDBConfig:
    """Configuration for LanceDB logging integration."""
    enabled: bool = field(
        default=False,
        metadata={"help": "Enable LanceDB logging (M3 only, stub in M2)"}
    )

@dataclass
class PipelineConfig:
    """Configuration for the overall pipeline execution."""
    verbose: bool = field(
        default=True,
        metadata={"help": "Whether to enable verbose logging"}
    )
    save_intermediates: bool = field(
        default=True,
        metadata={"help": "Whether to save intermediate results"}
    )
    output_dir: str = field(
        default="outputs",
        metadata={"help": "Directory to save outputs"}
    )
    ignore_nan_values: bool = field(
        default=False,
        metadata={"help": "Whether to ignore NaN values and continue pipeline execution"}
    )
    nan_replacement_value: float = field(
        default=0.0,
        metadata={"help": "Value to replace NaNs with when ignore_nan_values is True"}
    )
    lance_db: LanceDBConfig = field(
        default_factory=LanceDBConfig,
        metadata={"help": "LanceDB logging config (M3: enable for real logging)"}
    )

@dataclass
class TestDataConfig:
    """Configuration for test data used in demos and tests."""
    # Test sequence
    sequence: str = field(
        default="ACGUACGU",
        metadata={"help": "RNA sequence to use for testing"}
    )
    # Sequence properties
    sequence_length: int = field(
        default=8,
        metadata={"help": "Length of the sequence"}
    )
    atoms_per_residue: int = field(
        default=44,
        metadata={"help": "Standard RNA residue has ~44 atoms"}
    )
    # Adjacency matrix parameters
    adjacency_fill_value: float = field(
        default=1.0,
        metadata={"help": "Value to fill the adjacency matrix with"}
    )
    # Target parameters for gradient flow test
    target_dim: int = field(
        default=3,
        metadata={"help": "Dimension of target tensor for gradient flow test"}
    )
    # Torsion angle parameters
    torsion_angle_dim: int = field(
        default=7,
        metadata={"help": "Number of torsion angles per residue"}
    )

    # Embedding dimensions
    embedding_dims: Dict[str, int] = field(
        default_factory=lambda: {
            "s_trunk": 384,  # Single representation dimension
            "z_trunk": 128,  # Pair representation dimension
            "s_inputs": 32,  # Input representation dimension
        },
        metadata={"help": "Dimensions for various embeddings"}
    )
    # Allow arbitrary model group for test config composition
    model: Optional[Any] = None

@dataclass
class RNAConfig:
    """Root configuration for the entire RNA_PREDICT pipeline."""
    sequence: str = field(
        default="ACGUACGU",
        metadata={"help": "RNA sequence to predict structure for"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility"}
    )
    device: str = field(
        default="cpu",
        metadata={"help": "Global device setting, can be overridden per stage"}
    )
    dimensions: DimensionsConfig = field(default_factory=DimensionsConfig)
    model: Dict[str, Any] = field(default_factory=lambda: {
        "stageA": StageAConfig(),
        "stageB_torsion": TorsionBertConfig(),
        "stageB_pairformer": PairformerConfig(),
        "stageB": {
            "torsion_bert": TorsionBertConfig(),
            "pairformer": PairformerConfig()
        },
        "stageC": StageCConfig(),
        "stageD": StageDConfig()  # Add stageD under model
    })
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    latent_merger: LatentMergerConfig = field(default_factory=LatentMergerConfig)
    memory_optimization: MemoryOptimizationConfig = field(default_factory=MemoryOptimizationConfig)
    energy_minimization: EnergyMinimizationConfig = field(default_factory=EnergyMinimizationConfig)
    test_data: TestDataConfig = field(default_factory=TestDataConfig)
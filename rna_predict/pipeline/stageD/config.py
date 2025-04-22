"""
Configuration schema for Stage D using dataclasses.

This module defines the configuration structure for Stage D (Diffusion Refinement)
using Python dataclasses for type safety and validation.
"""
from dataclasses import dataclass, field
from typing import Optional, List
from hydra.core.config_store import ConfigStore

@dataclass
class NoiseScheduleConfig:
    """Configuration for the noise schedule."""
    schedule_type: str = "linear"
    s_max: float = 160.0
    s_min: float = 4e-4
    p: float = 7.0
    sigma_data: float = 16.0
    p_mean: float = -1.2
    p_std: float = 1.5

@dataclass
class SamplingConfig:
    """Configuration for the sampling process."""
    num_samples: int = 1
    gamma0: float = 0.8
    gamma_min: float = 1.0
    noise_scale_lambda: float = 1.003
    step_scale_eta: float = 1.5

@dataclass
class InferenceConfig:
    """Configuration for inference."""
    num_steps: int = 100
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

@dataclass
class AtomEncoderConfig:
    """Configuration for the atom encoder."""
    c_in: int = 3
    c_hidden: List[int] = field(default_factory=lambda: [32, 64, 128])
    c_out: int = 64
    dropout: float = 0.1

@dataclass
class AtomDecoderConfig:
    """Configuration for the atom decoder."""
    c_in: int = 64
    c_hidden: List[int] = field(default_factory=lambda: [128, 64, 32])
    c_out: int = 3
    dropout: float = 0.1

@dataclass
class TransformerConfig:
    """Configuration for the transformer."""
    n_blocks: int = 6
    n_heads: int = 8
    blocks_per_ckpt: Optional[int] = None

@dataclass
class InputFeatureConfig:
    """Configuration for a single input feature."""
    size: List[int] = field(default_factory=lambda: [128])
    _target_: Optional[str] = None

@dataclass
class AtomToTokenConfig:
    """Configuration for atom to token mapping."""
    repeats: Optional[int] = None

@dataclass
class ModelConfig:
    """Configuration for the diffusion model architecture."""
    c_token: int = 768
    c_s: int = 384
    c_z: int = 128
    c_s_inputs: int = 32
    c_atom: int = 3
    c_noise_embedding: int = 32
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1

    # Coordinate handling parameters
    coord_eps: float = 1e-6
    coord_min: float = -1e4
    coord_max: float = 1e4
    coord_similarity_rtol: float = 1e-3

    # Testing parameters
    test_residues_per_batch: int = 25
    atoms_per_residue: Optional[int] = None

@dataclass
class InputFeaturesConfig:
    """Configuration for input features."""
    ref_element: InputFeatureConfig = field(default_factory=lambda: InputFeatureConfig(size=[128]))
    ref_atom_name_chars: InputFeatureConfig = field(default_factory=lambda: InputFeatureConfig(size=[256]))
    profile: InputFeatureConfig = field(default_factory=lambda: InputFeatureConfig(size=[32]))
    atom_to_token_idx: AtomToTokenConfig = field(default_factory=AtomToTokenConfig)

@dataclass
class FeatureDimensionsConfig:
    """Feature dimensions for Stage D diffusion."""
    s_inputs: int = 32

@dataclass
class DiffusionConfig:
    """Main configuration for Stage D diffusion refinement."""
    # Core settings
    enabled: bool = True
    mode: str = "inference"
    device: str = "cpu"
    debug_logging: bool = True

    # Core diffusion parameters
    sigma_data: float = 16.0
    c_atom: int = 128
    c_s: int = 384
    c_z: int = 128
    c_s_inputs: int = 32
    c_noise_embedding: int = 32

    # Feature dimensions group
    feature_dimensions: FeatureDimensionsConfig = field(default_factory=FeatureDimensionsConfig)

    # Component configurations
    noise_schedule: NoiseScheduleConfig = field(default_factory=NoiseScheduleConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    atom_encoder: AtomEncoderConfig = field(default_factory=AtomEncoderConfig)
    atom_decoder: AtomDecoderConfig = field(default_factory=AtomDecoderConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    input_features: InputFeaturesConfig = field(default_factory=InputFeaturesConfig)

    # Memory optimization flags
    use_memory_efficient_kernel: bool = False
    use_deepspeed_evo_attention: bool = False
    use_lma: bool = False
    inplace_safe: bool = False
    chunk_size: Optional[int] = None

@dataclass
class StageDConfig:
    """Configuration for Stage D."""
    stageD: DiffusionConfig = field(default_factory=DiffusionConfig)

# Register configs with Hydra
cs = ConfigStore.instance()
cs.store(name="stageD", node=StageDConfig)
cs.store(name="stageD/diffusion", node=DiffusionConfig)
cs.store(name="stageD/noise_schedule", node=NoiseScheduleConfig)
cs.store(name="stageD/sampling", node=SamplingConfig)
cs.store(name="stageD/inference", node=InferenceConfig)
cs.store(name="stageD/model", node=ModelConfig)
cs.store(name="stageD/atom_encoder", node=AtomEncoderConfig)
cs.store(name="stageD/atom_decoder", node=AtomDecoderConfig)
cs.store(name="stageD/transformer", node=TransformerConfig)
cs.store(name="stageD/input_features", node=InputFeaturesConfig)
cs.store(name="stageD/input_feature", node=InputFeatureConfig)
cs.store(name="stageD/atom_to_token", node=AtomToTokenConfig)
cs.store(name="stageD/feature_dimensions", node=FeatureDimensionsConfig)
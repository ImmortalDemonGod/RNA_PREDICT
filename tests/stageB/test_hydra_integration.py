"""
Tests to verify that Stage B is using Hydra configuration correctly.
"""
import pytest
import torch
from omegaconf import OmegaConf
from hypothesis import given, strategies as st, settings, HealthCheck

from rna_predict.pipeline.stageB.torsion.torsion_bert_predictor import StageBTorsionBertPredictor
from rna_predict.pipeline.stageB.pairwise.pairformer_wrapper import PairformerWrapper


def test_stageB_torsion_bert_hydra_config(monkeypatch):
    """Test that TorsionBERT correctly uses Hydra configuration."""
    # Mock the AutoTokenizer and AutoModel to avoid actual model loading
    import transformers
    import torch
    from unittest.mock import MagicMock

    # Create dummy tokenizer class (not a MagicMock)
    class DummyTokenizer:
        def __call__(self, *args, **kwargs):
            return {"input_ids": torch.zeros((1, 8), dtype=torch.long), "attention_mask": torch.ones((1, 8), dtype=torch.long)}

    mock_tokenizer = DummyTokenizer()
    mock_model = MagicMock()

    # Patch the from_pretrained methods
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: mock_tokenizer)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", lambda *args, **kwargs: mock_model)

    # Create a test configuration
    test_config = OmegaConf.create({
        "model": {
            "stageB": {
                "torsion_bert": {
                    "model_name_or_path": "test_model_path",
                    "device": "cpu",
                    "angle_mode": "degrees",
                    "num_angles": 5,
                    "max_length": 256,
                    "lora": {
                        "enabled": False,
                        "r": 4,
                        "alpha": 8,
                        "dropout": 0.2,
                        "target_modules": ["query", "key"]
                    }
                }
            }
        }
    })

    # Initialize the predictor with the test config
    predictor = StageBTorsionBertPredictor(test_config)

    # Verify that the configuration was correctly applied
    assert predictor.model_name_or_path == "test_model_path"
    assert predictor.device == torch.device("cpu")
    assert predictor.angle_mode == "degrees"
    assert predictor.num_angles == 5
    assert predictor.max_length == 256


@settings(
    deadline=None,  # Disable deadline checks since model loading can be slow
    max_examples=5,  # Limit number of examples to keep test runtime reasonable
    suppress_health_check=[HealthCheck.too_slow]
)
@given(
    n_blocks=st.integers(min_value=1, max_value=32),
    c_z=st.integers(min_value=16, max_value=256),
    c_s=st.integers(min_value=32, max_value=512),
    dropout=st.floats(min_value=0.0, max_value=0.5),
    use_memory_efficient_kernel=st.booleans(),
    use_deepspeed_evo_attention=st.booleans(),
    use_lma=st.booleans(),
    inplace_safe=st.booleans(),
    chunk_size=st.one_of(st.none(), st.integers(min_value=16, max_value=64)),
    use_checkpoint=st.booleans()
)
def test_stageB_pairformer_hydra_config(
    n_blocks, c_z, c_s, dropout, use_memory_efficient_kernel,
    use_deepspeed_evo_attention, use_lma, inplace_safe, chunk_size, use_checkpoint
):
    """Property-based test: Verify that Pairformer correctly uses Hydra configuration.

    Args:
        n_blocks: Number of transformer blocks
        c_z: Dimension of pair embeddings
        c_s: Dimension of single embeddings
        dropout: Dropout rate
        use_memory_efficient_kernel: Whether to use memory efficient attention
        use_deepspeed_evo_attention: Whether to use DeepSpeed Evo attention
        use_lma: Whether to use linear multi-head attention
        inplace_safe: Whether to use inplace operations
        chunk_size: Chunk size for attention (or None)
        use_checkpoint: Whether to use checkpointing
    """
    # Create a test configuration
    test_config = OmegaConf.create({
        "model": {
            "stageB": {
                "pairformer": {
                    "n_blocks": n_blocks,
                    "c_z": c_z,
                    "c_s": c_s,
                    "dropout": dropout,
                    "use_memory_efficient_kernel": use_memory_efficient_kernel,
                    "use_deepspeed_evo_attention": use_deepspeed_evo_attention,
                    "use_lma": use_lma,
                    "inplace_safe": inplace_safe,
                    "chunk_size": chunk_size,
                    "use_checkpoint": use_checkpoint,
                    "lora": {
                        "enabled": False,
                        "r": 4,
                        "alpha": 8,
                        "dropout": 0.2,
                        "target_modules": ["query", "key"]
                    }
                }
            }
        }
    })

    # Initialize the wrapper with the test config
    wrapper = PairformerWrapper(test_config)

    # Verify that the configuration was correctly applied
    assert wrapper.n_blocks == n_blocks
    assert wrapper.c_z == c_z
    assert wrapper.c_s == c_s
    assert wrapper.dropout == dropout
    assert wrapper.use_memory_efficient_kernel is use_memory_efficient_kernel
    assert wrapper.use_deepspeed_evo_attention is use_deepspeed_evo_attention
    assert wrapper.use_lma is use_lma
    assert wrapper.inplace_safe is inplace_safe
    assert wrapper.chunk_size == chunk_size
    assert wrapper.use_checkpoint is use_checkpoint


def test_stageB_torsion_bert_default_config(monkeypatch):
    """Test that TorsionBERT uses default configuration values when not specified."""
    # Mock the AutoTokenizer and AutoModel to avoid actual model loading
    import transformers
    import torch
    from unittest.mock import MagicMock

    # Create dummy tokenizer class (not a MagicMock)
    class DummyTokenizer:
        def __call__(self, *args, **kwargs):
            return {"input_ids": torch.zeros((1, 8), dtype=torch.long), "attention_mask": torch.ones((1, 8), dtype=torch.long)}

    mock_tokenizer = DummyTokenizer()
    mock_model = MagicMock()

    # Patch the from_pretrained methods
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: mock_tokenizer)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", lambda *args, **kwargs: mock_model)

    # Create a test configuration with minimal fields but including all required ones
    test_config = OmegaConf.create({
        "model": {
            "stageB": {
                "torsion_bert": {
                    "model_name_or_path": "sayby/rna_torsionbert",
                    "device": "cpu",
                    "angle_mode": "sin_cos",
                    "num_angles": 7,
                    "max_length": 512,
                    "checkpoint_path": None,
                    "lora": {
                        "enabled": False,
                        "r": 8,
                        "alpha": 16,
                        "dropout": 0.1,
                        "target_modules": []
                    }
                }
            }
        }
    })

    # Initialize the predictor with the config
    predictor = StageBTorsionBertPredictor(test_config)

    # Verify that values were applied
    assert predictor.model_name_or_path == "sayby/rna_torsionbert"
    assert predictor.device == torch.device("cpu")
    assert predictor.angle_mode == "sin_cos"
    assert predictor.num_angles == 7
    assert predictor.max_length == 512


def test_stageB_missing_config_section():
    """Test that appropriate error is raised when config section is missing."""
    # Create a config without the required section
    test_config = OmegaConf.create({
        "model": {
            "stageA": {}  # Missing stageB section
        }
    })

    # Verify that initialization raises ValueError with expected message
    with pytest.raises(ValueError, match="Configuration must contain either stageB_torsion or model.stageB.torsion_bert section"):
        StageBTorsionBertPredictor(test_config)

    with pytest.raises(ValueError, match="Pairformer config not found in Hydra config"):
        PairformerWrapper(test_config)


def test_stageB_config_override(monkeypatch):
    """Test that command-line overrides work correctly with Hydra."""
    # Mock the AutoTokenizer and AutoModel to avoid actual model loading
    from unittest.mock import MagicMock
    import transformers

    # Create mock objects
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    # Patch the from_pretrained methods
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: mock_tokenizer)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", lambda *args, **kwargs: mock_model)

    # Create a test configuration with overrides directly
    test_config = OmegaConf.create({
        "model": {
            "stageB": {
                "torsion_bert": {
                    "model_name_or_path": "test_model_path",
                    "device": "cuda" if torch.cuda.is_available() else "cpu",
                    "angle_mode": "radians",
                    "num_angles": 5,
                    "max_length": 256,
                    "lora": {
                        "enabled": False,
                        "r": 4,
                        "alpha": 8,
                        "dropout": 0.2,
                        "target_modules": ["query", "key"]
                    }
                }
            }
        }
    })

    # Skip test if CUDA is not available when device=cuda is specified
    if test_config.model.stageB.torsion_bert.device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping test")

    # Initialize the predictor with the overridden config
    predictor = StageBTorsionBertPredictor(test_config)

    # Verify that overrides were applied
    assert predictor.device == torch.device(test_config.model.stageB.torsion_bert.device)
    assert predictor.angle_mode == "radians"

"""
Tests to verify that Stage B is using Hydra configuration correctly.
"""
import pytest
import torch
from omegaconf import OmegaConf
from hypothesis import given, strategies as st, settings, HealthCheck
import os

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
                    "device": "cpu",
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
    assert wrapper.device == torch.device("cpu")


def test_stageB_torsion_bert_default_config(monkeypatch):
    """
    Tests that StageBTorsionBertPredictor correctly applies default configuration values,
    including when `num_angles` is set to 7 and the relevant environment variable is enabled.
    
    Temporarily sets the `ALLOW_NUM_ANGLES_7_FOR_TESTS` environment variable to allow
    testing with `num_angles=7`, mocks model and tokenizer loading, and verifies that
    the predictor's attributes match the provided minimal configuration.
    """
    original_env_var = os.environ.get("ALLOW_NUM_ANGLES_7_FOR_TESTS")
    os.environ["ALLOW_NUM_ANGLES_7_FOR_TESTS"] = "1"
    try:
        # Mock the AutoTokenizer and AutoModel to avoid actual model loading
        import transformers
        import torch
        from unittest.mock import MagicMock

        # Create dummy tokenizer class (not a MagicMock)
        class DummyTokenizer:
            def __call__(self, *args, **kwargs):
                """
                Returns a dummy input dictionary with zeroed input IDs and an attention mask of ones.
                
                Returns:
                    dict: A dictionary containing 'input_ids' and 'attention_mask' tensors.
                """
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
    finally:
        if original_env_var is None:
            del os.environ["ALLOW_NUM_ANGLES_7_FOR_TESTS"]
        else:
            os.environ["ALLOW_NUM_ANGLES_7_FOR_TESTS"] = original_env_var


def test_stageB_missing_config_section():
    """
    Tests that initializing StageBTorsionBertPredictor or PairformerWrapper with a config missing required sections raises a ValueError with the expected error message.
    """
    # Create a config without the required section
    test_config = OmegaConf.create({
        "model": {
            "stageB": {
                # Missing torsion_bert and pairformer sections
            }
        }
    })

    # Verify that attempting to initialize without required sections raises an error
    with pytest.raises(ValueError, match=".*UNIQUE-ERR-TORSIONBERT-NOCONFIG.*"):
        StageBTorsionBertPredictor(test_config)

    with pytest.raises(ValueError, match=".*UNIQUE-ERR-PAIRFORMER-NOCONFIG.*"):
        PairformerWrapper(test_config)


def test_stageB_config_override(monkeypatch):
    """Test that configuration overrides work correctly."""
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

    # Create a test configuration with overrides
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
                        "enabled": True,
                        "r": 16,
                        "alpha": 32,
                        "dropout": 0.3,
                        "target_modules": ["query", "key", "value"]
                    }
                },
                "pairformer": {
                    "n_blocks": 4,
                    "c_z": 32,
                    "c_s": 64,
                    "dropout": 0.2,
                    "device": "cpu",
                    "use_memory_efficient_kernel": True,
                    "use_deepspeed_evo_attention": False,
                    "use_lma": True,
                    "inplace_safe": True,
                    "chunk_size": 32,
                    "use_checkpoint": True,
                    "lora": {
                        "enabled": True,
                        "r": 16,
                        "alpha": 32,
                        "dropout": 0.3,
                        "target_modules": ["query", "key", "value"]
                    }
                }
            }
        }
    })

    # Initialize both predictors with the test config
    torsion_predictor = StageBTorsionBertPredictor(test_config)
    pairformer_wrapper = PairformerWrapper(test_config)

    # Verify that the configuration overrides were correctly applied
    assert torsion_predictor.model_name_or_path == "test_model_path"
    assert torsion_predictor.device == torch.device("cpu")
    assert torsion_predictor.angle_mode == "degrees"
    assert torsion_predictor.num_angles == 5
    assert torsion_predictor.max_length == 256

    assert pairformer_wrapper.n_blocks == 4
    assert pairformer_wrapper.c_z == 32
    assert pairformer_wrapper.c_s == 64
    assert pairformer_wrapper.dropout == 0.2
    assert pairformer_wrapper.device == torch.device("cpu")
    assert pairformer_wrapper.use_memory_efficient_kernel is True
    assert pairformer_wrapper.use_deepspeed_evo_attention is False
    assert pairformer_wrapper.use_lma is True
    assert pairformer_wrapper.inplace_safe is True
    assert pairformer_wrapper.chunk_size == 32
    assert pairformer_wrapper.use_checkpoint is True

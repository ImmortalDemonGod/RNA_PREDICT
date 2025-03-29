"""
test_torsionbert.py

Comprehensive pytest test suite for TorsionBertModel and StageBTorsionBertPredictor.

This file merges best practices to achieve:
  - Thorough testing of TorsionBertModel initialization, forward pass, and
    predict_angles_from_sequence
  - Edge-case checks (empty seq, short seq, fallback to last_hidden_state, invalid device, invalid path)
  - Use of Hypothesis-based property tests for random RNA sequences
  - Integration tests for StageBTorsionBertPredictor usage
  - Mock-based coverage for Hugging Face components
  - Clear docstrings and robust assertions
"""

import pytest
import torch
from torch import Tensor
from unittest.mock import patch, MagicMock
from hypothesis import given, settings, strategies as st, HealthCheck
from typing import Any, Dict

# Importing classes under test:
# Adjust paths to reflect your project structure if they differ.
from rna_predict.pipeline.stageB.torsion.torsionbert_inference import TorsionBertModel
from rna_predict.pipeline.stageB.torsion.torsion_bert_predictor import StageBTorsionBertPredictor


# ----------------------------------------------------------------------
#                           Mock Helpers
# ----------------------------------------------------------------------
class DummyLastHiddenStateOutput:
    """
    Simple container that mimics a huggingface model output with .last_hidden_state
    instead of the 'logits' key. This helps us test fallback logic.
    """
    def __init__(self, tensor: torch.Tensor):
        self.last_hidden_state = tensor


def mock_forward_logits(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """
    Mock forward function that returns a dictionary with 'logits'.
    The shape is (batch=1, tokens=5, features=2*self.num_angles) by default.
    We can dynamically detect number of tokens from inputs['input_ids'] if present.
    """
    batch_size = 1
    if "input_ids" in inputs:
        tokens = inputs["input_ids"].shape[1]
    else:
        tokens = 5  # fallback if not found

    feats = 2 * getattr(self, "num_angles", 7)
    logits = torch.randn(batch_size, tokens, feats)
    return {"logits": logits}


def mock_forward_last_hidden(self, inputs: Dict[str, Tensor]) -> Any:
    """
    Mock forward function that returns an object with .last_hidden_state.
    Always uses shape (1, tokens=5, 2*self.num_angles).
    """
    batch_size = 1
    tokens = 5
    feats = 2 * getattr(self, "num_angles", 7)
    tensor = torch.ones(batch_size, tokens, feats)
    return DummyLastHiddenStateOutput(tensor)


# ----------------------------------------------------------------------
#                        Pytest Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def model_with_logits() -> TorsionBertModel:
    """
    Fixture that patches AutoTokenizer/AutoModel so that the forward pass
    returns a dictionary with 'logits'.
    """
    with patch("rna_predict.pipeline.stageB.torsion.torsionbert_inference.AutoTokenizer") as mock_tok_cls, \
         patch("rna_predict.pipeline.stageB.torsion.torsionbert_inference.AutoModel") as mock_model_cls:

        # Mock tokenizer
        mock_tokenizer = MagicMock()

        def _fake_tokenizer(*args, **kwargs):
            max_len = kwargs.get("max_length", 512)
            return {
                "input_ids": torch.zeros((1, max_len), dtype=torch.long),
                "attention_mask": torch.ones((1, max_len), dtype=torch.long),
            }

        mock_tokenizer.side_effect = _fake_tokenizer
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer

        # Mock model
        mock_model = MagicMock()
        mock_model.num_angles = 7
        mock_model.__call__.side_effect = lambda inp: mock_forward_logits(mock_model, inp)
        mock_model_cls.from_pretrained.return_value = mock_model

        # Instantiate the TorsionBertModel with our mock
        model = TorsionBertModel(
            model_name_or_path="dummy_path_logits",
            device=torch.device("cpu"),
            num_angles=7,
            max_length=20
        )
    return model


@pytest.fixture
def model_with_last_hidden() -> TorsionBertModel:
    """
    Fixture that patches AutoTokenizer/AutoModel so that the forward pass
    returns an object with .last_hidden_state.
    """
    with patch("rna_predict.pipeline.stageB.torsion.torsionbert_inference.AutoTokenizer") as mock_tok_cls, \
         patch("rna_predict.pipeline.stageB.torsion.torsionbert_inference.AutoModel") as mock_model_cls:

        mock_tokenizer = MagicMock()

        def _fake_tokenizer(*args, **kwargs):
            max_len = kwargs.get("max_length", 512)
            return {
                "input_ids": torch.zeros((1, max_len), dtype=torch.long),
                "attention_mask": torch.ones((1, max_len), dtype=torch.long),
            }

        mock_tokenizer.side_effect = _fake_tokenizer
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer

        # Mock model
        mock_model = MagicMock()
        mock_model.num_angles = 7
        mock_model.__call__.side_effect = lambda inp: mock_forward_last_hidden(mock_model, inp)
        mock_model_cls.from_pretrained.return_value = mock_model

        model = TorsionBertModel(
            model_name_or_path="dummy_path_last_hidden",
            device=torch.device("cpu"),
            num_angles=7,
            max_length=20
        )
    return model


@pytest.fixture
def predictor_fixture(model_with_logits: TorsionBertModel) -> StageBTorsionBertPredictor:
    """
    Fixture that sets up a StageBTorsionBertPredictor using the model_with_logits
    to test integrated behavior. We override predictor.model with our mock-based TorsionBertModel.
    """
    # Instantiate the predictor
    predictor = StageBTorsionBertPredictor(
        model_name_or_path="dummy_path_logits",
        device="cpu",
        angle_mode="degrees",
        num_angles=16
    )
    # Override the predictor's internal model with our TorsionBertModel mock
    predictor.model = model_with_logits
    # Keep them in sync
    predictor.model.num_angles = 16
    return predictor


# ----------------------------------------------------------------------
#               Tests for TorsionBertModel
# ----------------------------------------------------------------------
class TestTorsionBertModel:
    """
    Tests for the TorsionBertModel class, ensuring:
    - Initialization
    - Forward pass returning logits or last_hidden_state
    - predict_angles_from_sequence behavior
    - Edge cases and error handling
    """

    def test_init_success_logits(self, model_with_logits: TorsionBertModel) -> None:
        """
        Ensures the model is initialized with correct attributes
        and that tokenizer/model are not None.
        """
        assert model_with_logits.num_angles == 7
        assert model_with_logits.max_length == 20
        assert model_with_logits.device.type == "cpu"
        assert model_with_logits.tokenizer is not None
        assert model_with_logits.model is not None

    def test_init_success_last_hidden(self, model_with_last_hidden: TorsionBertModel) -> None:
        """
        Ensures the model is initialized when the forward pass
        uses last_hidden_state instead of logits.
        """
        assert model_with_last_hidden.num_angles == 7
        assert model_with_last_hidden.max_length == 20
        assert model_with_last_hidden.device.type == "cpu"
        assert model_with_last_hidden.tokenizer is not None
        assert model_with_last_hidden.model is not None

    def test_forward_logits(self, model_with_logits: TorsionBertModel) -> None:
        """
        Confirm forward returns a dict containing 'logits'
        and that shape is as expected.
        """
        fake_inputs = {
            "input_ids": torch.randint(0, 50, (1, 10)),
            "attention_mask": torch.ones((1, 10), dtype=torch.long),
        }
        outputs = model_with_logits.forward(fake_inputs)
        assert isinstance(outputs, dict)
        assert "logits" in outputs
        # 2 * num_angles => 2*7=14
        assert outputs["logits"].shape == (1, 10, 14)

    def test_forward_last_hidden(self, model_with_last_hidden: TorsionBertModel) -> None:
        """
        Confirm forward returns an object with last_hidden_state
        if 'logits' isn't present.
        """
        fake_inputs = {
            "input_ids": torch.randint(0, 50, (1, 8)),
            "attention_mask": torch.ones((1, 8), dtype=torch.long),
        }
        outputs = model_with_last_hidden.forward(fake_inputs)
        # Should not be a dict
        assert not isinstance(outputs, dict)
        assert hasattr(outputs, "last_hidden_state")
        # 2 * num_angles => 2*7=14
        assert outputs.last_hidden_state.shape == (1, 8, 14)

    @pytest.mark.parametrize("rna_seq,expected_len", [
        ("", 0),
        ("A", 1),
        ("AC", 2),
        ("ACG", 3),
        ("ACGU", 4),
    ])
    def test_predict_angles_basic(
        self,
        model_with_logits: TorsionBertModel,
        rna_seq: str,
        expected_len: int,
    ) -> None:
        """
        Parametrized test for predict_angles_from_sequence, verifying shape
        matches the sequence length.
        """
        out = model_with_logits.predict_angles_from_sequence(rna_seq)
        assert out.shape == (expected_len, 14)  # 2*7=14

    def test_predict_angles_empty(self, model_with_logits: TorsionBertModel) -> None:
        """
        Specifically test empty RNA sequence returns [0,14].
        """
        out = model_with_logits.predict_angles_from_sequence("")
        assert out.shape == (0, 14)
        assert out.numel() == 0  # zero elements

    def test_predict_angles_short_seq(self, model_with_logits: TorsionBertModel) -> None:
        """
        A single char sequence => [1,14] but no 3-mer tokens => row is zeros.
        """
        out = model_with_logits.predict_angles_from_sequence("A")
        assert out.shape == (1, 14)
        assert torch.allclose(out[0], torch.zeros(14))

    def test_predict_angles_partial_fill(self, model_with_logits: TorsionBertModel) -> None:
        """
        For a normal sequence of length 4 => up to 2 tokens => partial fill.
        The code uses mid_idx to place data, leaving last row zeros if mid_idx >= seq_len.
        """
        seq = "ACGU"
        out = model_with_logits.predict_angles_from_sequence(seq)
        assert out.shape == (4, 14)
        # We won't do super-detailed row checks but at least ensure no crash, shape correct.

    def test_predict_angles_fallback_last_hidden(self, model_with_logits: TorsionBertModel) -> None:
        """
        If no 'logits' in the output, fallback to last_hidden_state. We manually override
        to produce last_hidden_state only, verifying correct shape and no crash.
        """
        mock_model = model_with_logits.model
        mock_model.__call__.side_effect = lambda i: DummyLastHiddenStateOutput(
            torch.ones((1, 5, 2*mock_model.num_angles))
        )
        seq = "ACG"
        result = model_with_logits.predict_angles_from_sequence(seq)
        assert result.shape == (3, 14)  # 3 residues => shape[0] = 3

    @given(st.text(alphabet=["A", "C", "G", "U", "T"], min_size=0, max_size=25))
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=30)
    def test_predict_angles_hypothesis(
        self,
        model_with_logits: TorsionBertModel,
        rna_sequence: str
    ) -> None:
        """
        Hypothesis-based test generating random RNA sequences from ACGUT,
        verifying shape correctness for each.
        """
        out = model_with_logits.predict_angles_from_sequence(rna_sequence)
        assert out.shape == (len(rna_sequence), 14)

    def test_invalid_device(self) -> None:
        """
        Creating TorsionBertModel with an invalid device string => RuntimeError.
        """
        with pytest.raises(RuntimeError, match="Invalid device string"):
            TorsionBertModel(
                model_name_or_path="any_path",
                device=torch.device("invalid_device"),
                num_angles=7,
                max_length=512
            )

    @pytest.mark.parametrize("invalid_model_path", [None, 123, ""])
    def test_invalid_model_path(self, invalid_model_path) -> None:
        """
        Passing invalid model paths => OSError/ValueError/TypeError
        when huggingface tries to load. We'll patch to simulate that.
        """
        with patch("rna_predict.pipeline.stageB.torsion.torsionbert_inference.AutoTokenizer.from_pretrained",
                   side_effect=OSError("Mock HF load fail")):
            with patch("rna_predict.pipeline.stageB.torsion.torsionbert_inference.AutoModel.from_pretrained",
                       side_effect=OSError("Mock HF load fail")):
                with pytest.raises((OSError, ValueError, TypeError)):
                    TorsionBertModel(
                        model_name_or_path=invalid_model_path,
                        device=torch.device("cpu"),
                        num_angles=7,
                        max_length=512
                    )


# ----------------------------------------------------------------------
#        Tests for StageBTorsionBertPredictor (Integration)
# ----------------------------------------------------------------------
class TestStageBTorsionBertPredictor:
    """
    Simple integration tests to confirm that StageBTorsionBertPredictor
    calls TorsionBertModel as expected and returns correct shapes.
    """

    def test_short_seq(self, predictor_fixture: StageBTorsionBertPredictor) -> None:
        """
        A short 2-letter sequence => expect output shape [2, 16],
        because predictor is set to num_angles=16 in degrees mode.
        """
        seq = "AC"
        result = predictor_fixture(seq)
        angles = result["torsion_angles"]
        assert angles.shape == (2, 16)

    def test_normal_seq(self, predictor_fixture: StageBTorsionBertPredictor) -> None:
        """
        A normal 4-letter sequence => shape [4,16].
        """
        seq = "ACGU"
        result = predictor_fixture(seq)
        angles = result["torsion_angles"]
        assert angles.shape == (4, 16)

    @pytest.mark.parametrize("seq", ["A", "ACG", "ACGUAC"])
    def test_various_lengths(self, predictor_fixture: StageBTorsionBertPredictor, seq: str) -> None:
        """
        Check the predictor's shape for various input lengths.
        """
        out = predictor_fixture(seq)
        angles = out["torsion_angles"]
        assert angles.shape == (len(seq), 16)

    def test_predictor_consistency(self, predictor_fixture: StageBTorsionBertPredictor) -> None:
        """
        If the same sequence is passed multiple times, ensure shape is consistent
        and no crash occurs. This is a minimal check for consistency.
        """
        seq = "ACGUACGU"
        out1 = predictor_fixture(seq)
        out2 = predictor_fixture(seq)
        angles1 = out1["torsion_angles"]
        angles2 = out2["torsion_angles"]
        assert angles1.shape == angles2.shape == (8, 16)
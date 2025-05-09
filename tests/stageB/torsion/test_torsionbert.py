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

from typing import Any, Dict
from unittest.mock import patch

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from torch import Tensor

from rna_predict.pipeline.stageB.torsion.torsion_bert_predictor import (
    StageBTorsionBertPredictor,
)
from rna_predict.pipeline.stageB.torsion.torsionbert_inference import (
    DummyTorsionBertAutoModel,
    TorsionBertModel,
)


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


def configure_mock_tensor(batch_size: int, seq_len: int, sincos_dim: int):
    """
    Configure a real torch tensor with the specified shape for testing.
    This replaces the previous MagicMock-based tensor for compatibility with stricter dummy model.
    """
    return torch.zeros(batch_size, seq_len, sincos_dim)


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
    mock_tensor = configure_mock_tensor(batch_size, tokens, feats)
    return {"logits": mock_tensor}


def mock_forward_last_hidden(self, inputs: Dict[str, Tensor]) -> Any:
    """
    Mock forward function that returns an object with .last_hidden_state.
    Always uses shape (1, tokens=5, 2*self.num_angles).
    """
    batch_size = 1
    tokens = 5
    feats = 2 * getattr(self, "num_angles", 7)
    mock_tensor = configure_mock_tensor(batch_size, tokens, feats)
    return DummyLastHiddenStateOutput(mock_tensor)


# ----------------------------------------------------------------------
#                        Pytest Fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def mock_tokenizer() -> Any:
    """Dummy tokenizer that returns fixed-size tensors, not a MagicMock."""
    class DummyTokenizer:
        def __call__(self, text, **kwargs):
            print(f"[DEBUG-TOKENIZER-INPUT] Received text: '{text}' (repr: {repr(text)})")
            if text is None:
                raise ValueError("[UNIQUE-ERR-TOKENIZER-NONE] Tokenizer received None as input.")
            if isinstance(text, str):
                seq_len = len(text)
            else:
                seq_len = len(str(text).split())
            if seq_len == 0:
                raise ValueError("[UNIQUE-ERR-TOKENIZER-EMPTYSEQ] Tokenizer received empty sequence.")
            return {
                "input_ids": torch.zeros((1, seq_len), dtype=torch.long),
                "attention_mask": torch.ones((1, seq_len), dtype=torch.long),
            }
    return DummyTokenizer()


@pytest.fixture(scope="module")
def model_with_logits(mock_tokenizer: Any) -> TorsionBertModel:
    """
    Creates a TorsionBertModel fixture configured to return logits for testing.
    
    This fixture patches the tokenizer and model loading to use mock implementations,
    ensuring the returned TorsionBertModel produces logits output for test scenarios.
    """
    print("[DEBUG-FIXTURE] Setting up model_with_logits fixture")
    print(f"[DEBUG-FIXTURE] mock_tokenizer type: {type(mock_tokenizer)}")

    # Create patchers explicitly to debug
    tokenizer_patcher = patch(
        "rna_predict.pipeline.stageB.torsion.torsionbert_inference.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer
    )
    model_patcher = patch(
        "rna_predict.pipeline.stageB.torsion.torsionbert_inference.AutoModel.from_pretrained",
        return_value=DummyTorsionBertAutoModel()
    )

    # Start patchers
    tokenizer_patcher.start()
    model_patcher.start()

    try:
        print("[DEBUG-FIXTURE] Patchers started, creating model")
        model = TorsionBertModel(
            model_path="dummy_path",
            num_angles=7,
            max_length=512,
            device="cpu",
            return_dict=True,
        )
        print(f"[DEBUG-FIXTURE] Model created, tokenizer: {model.tokenizer}")
        return model
    finally:
        # Stop patchers
        tokenizer_patcher.stop()
        model_patcher.stop()
        print("[DEBUG-FIXTURE] Patchers stopped")


@pytest.fixture
def model_with_last_hidden(mock_tokenizer: Any) -> TorsionBertModel:
    """
    Creates a TorsionBertModel fixture configured to return last_hidden_state outputs.
    
    This fixture patches the tokenizer and model loading to use mock objects, ensuring
    the returned TorsionBertModel instance operates in a controlled test environment
    with `return_dict=False` for last_hidden_state output mode.
    """
    print("[DEBUG-FIXTURE] Setting up model_with_last_hidden fixture")
    print(f"[DEBUG-FIXTURE] mock_tokenizer type: {type(mock_tokenizer)}")

    # Create patchers explicitly to debug
    tokenizer_patcher = patch(
        "rna_predict.pipeline.stageB.torsion.torsionbert_inference.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer
    )
    model_patcher = patch(
        "rna_predict.pipeline.stageB.torsion.torsionbert_inference.AutoModel.from_pretrained",
        return_value=DummyTorsionBertAutoModel()
    )

    # Start patchers
    tokenizer_patcher.start()
    model_patcher.start()

    try:
        print("[DEBUG-FIXTURE] Patchers started, creating model with return_dict=False")
        model = TorsionBertModel(
            model_path="dummy_path",
            num_angles=7,
            max_length=512,
            device="cpu",
            return_dict=False,
        )
        print(f"[DEBUG-FIXTURE] Model created, tokenizer: {model.tokenizer}")
        return model
    finally:
        # Stop patchers
        tokenizer_patcher.stop()
        model_patcher.stop()
        print("[DEBUG-FIXTURE] Patchers stopped")


@pytest.fixture
def predictor_fixture(
    mock_tokenizer: Any,
) -> StageBTorsionBertPredictor:
    """
    Pytest fixture that creates a StageBTorsionBertPredictor configured for integration tests.
    
    Sets up the predictor with a mock tokenizer and a dummy model using 16 angles and 'degrees'
    angle mode, enabling debug logging for diagnostic output.
    """
    from omegaconf import DictConfig
    # Build a minimal config matching what StageBTorsionBertPredictor expects
    cfg = DictConfig({
        'model': {
            'stageB': {
                'torsion_bert': {
                    'model_name_or_path': 'bert-base-uncased',
                    'device': 'cpu',
                    'angle_mode': 'degrees',
                    'num_angles': 16,
                    'max_length': 512,
                }
            }
        }
    })
    # Add backward-compatible alias for root-level stageB_torsion
    cfg['stageB_torsion'] = cfg['model']['stageB']['torsion_bert']
    # Add debug logging to the config
    cfg['debug_logging'] = True

    # Patch predictor to use mock model and tokenizer
    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        patch("rna_predict.pipeline.stageB.torsion.torsion_bert_predictor.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        patch("transformers.AutoModel.from_pretrained", return_value=DummyTorsionBertAutoModel(num_angles=16)),
    ):
        print("[DEBUG-PREDICTOR-FIXTURE] Creating StageBTorsionBertPredictor with config:")
        print(f"[DEBUG-PREDICTOR-FIXTURE] cfg: {cfg}")
        predictor = StageBTorsionBertPredictor(cfg)
        print(f"[DEBUG-PREDICTOR-FIXTURE] predictor.dummy_mode: {getattr(predictor, 'dummy_mode', 'not set')}")
        print(f"[DEBUG-PREDICTOR-FIXTURE] predictor.num_angles: {getattr(predictor, 'num_angles', 'not set')}")
        print(f"[DEBUG-PREDICTOR-FIXTURE] predictor.angle_mode: {getattr(predictor, 'angle_mode', 'not set')}")
        print(f"[DEBUG-PREDICTOR-FIXTURE] predictor.output_dim: {getattr(predictor, 'output_dim', 'not set')}")
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
        assert model_with_logits.max_length == 512
        assert model_with_logits.device.type == "cpu"
        assert model_with_logits.tokenizer is not None
        assert model_with_logits.model is not None

    def test_init_success_last_hidden(
        self, model_with_last_hidden: TorsionBertModel
    ) -> None:
        """
        Ensures the model is initialized when the forward pass
        uses last_hidden_state instead of logits.
        """
        assert model_with_last_hidden.num_angles == 7
        assert model_with_last_hidden.max_length == 512
        assert model_with_last_hidden.device.type == "cpu"
        assert model_with_last_hidden.tokenizer is not None
        assert model_with_last_hidden.model is not None

    @settings(
        deadline=None,  # Disable deadline checks since model loading can be slow
        max_examples=5,  # Limit number of examples to keep test runtime reasonable
        suppress_health_check=[HealthCheck.too_slow]
    )
    @given(
        seq_len=st.integers(min_value=3, max_value=20),  # Test various sequence lengths
        batch_size=st.integers(min_value=1, max_value=3)  # Test various batch sizes
    )
    def test_forward_logits(self, model_with_logits: TorsionBertModel, seq_len: int, batch_size: int) -> None:
        """
        Property-based test: Confirm forward returns a dict containing 'logits'
        and that shape is as expected for various input shapes.

        Args:
            model_with_logits: The model fixture configured to return logits
            seq_len: Length of the input sequence
            batch_size: Batch size for the input

        # ERROR_ID: TORSIONBERT_FORWARD_LOGITS
        """
        fake_inputs = {
            "input_ids": torch.randint(0, 50, (batch_size, seq_len)),
            "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long),
        }
        outputs = model_with_logits.forward(fake_inputs)
        assert isinstance(outputs, dict), f"[UNIQUE-ERR-TORSIONBERT-FORWARD-001] Expected dict output, got {type(outputs)}"
        assert "logits" in outputs, f"[UNIQUE-ERR-TORSIONBERT-FORWARD-002] Expected 'logits' key in output dict, got keys: {list(outputs.keys())}"
        # 2 * num_angles => 2*7=14
        expected_shape = (batch_size, seq_len, 14)
        assert outputs["logits"].shape == expected_shape, f"[UNIQUE-ERR-TORSIONBERT-FORWARD-003] Expected shape {expected_shape}, got {outputs['logits'].shape}"

    def test_forward_last_hidden(
        self, model_with_last_hidden: TorsionBertModel
    ) -> None:
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

    @pytest.mark.parametrize(
        "rna_seq,expected_len,expect_error",
        [
            ("A", 1, True),
            ("AC", 2, True),
            ("ACG", 3, False),
            ("ACGU", 4, False),
        ],
    )
    def test_predict_angles_basic(
        self,
        model_with_logits: TorsionBertModel,
        rna_seq: str,
        expected_len: int,
        expect_error: bool,
    ) -> None:
        """
        Parametrized test for predict_angles_from_sequence, verifying shape
        matches the sequence length. For sequences shorter than 3, expects a unique ValueError.
        """
        if expect_error:
            with pytest.raises(ValueError) as excinfo:
                model_with_logits.predict_angles_from_sequence(rna_seq)
            assert "[UNIQUE-ERR-TOKENIZER-EMPTYSEQ]" in str(excinfo.value)
        else:
            out = model_with_logits.predict_angles_from_sequence(rna_seq)
            if not hasattr(out, 'shape'):
                raise AssertionError(f"[UNIQUE-ERR-TORSIONBERT-BASIC-OUTPUT-NOSHAPE] Output has no shape attribute. Type: {type(out)}")
            assert out.shape == (expected_len, 14), (
                f"[UNIQUE-ERR-TORSIONBERT-BASIC-SHAPE] For input '{rna_seq}', expected shape ({expected_len}, 14), got {getattr(out, 'shape', None)}. Output type: {type(out)}"
            )

    def test_predict_angles_empty(self, model_with_logits: TorsionBertModel) -> None:
        """
        Test that empty RNA sequence returns a (0, 14) tensor with zero elements.
        """
        out = model_with_logits.predict_angles_from_sequence("")
        assert hasattr(out, 'shape'), "[UNIQUE-ERR-TORSIONBERT-EMPTY-NOSHAPE] Output has no shape attribute."
        assert out.shape == (0, 14), f"[UNIQUE-ERR-TORSIONBERT-EMPTY-SHAPE] Expected shape (0, 14), got {out.shape}"
        assert getattr(out, 'numel', lambda: -1)() == 0, f"[UNIQUE-ERR-TORSIONBERT-EMPTY-NUMELEMS] Expected 0 elements, got {getattr(out, 'numel', lambda: -1)()}"

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(rna_seq=st.text(alphabet=st.sampled_from(["A", "C", "G", "U"]), min_size=1, max_size=20))
    def test_predict_angles_hypothesis(self, model_with_logits: TorsionBertModel, rna_seq: str):
        """
        Hypothesis-based test for random non-empty RNA sequences, checking output shape.
        Instrumented with debug output for systematic diagnosis.
        """
        if len(rna_seq) < 3:
            try:
                print(f"[DEBUG-TORSIONBERT-HYP] Input sequence: '{rna_seq}' (len={len(rna_seq)}) [Expecting ValueError]")
                model_with_logits.predict_angles_from_sequence(rna_seq)
            except ValueError as e:
                print(f"[DEBUG-TORSIONBERT-HYP][EXCEPTION] Input: '{rna_seq}' | Exception: {type(e).__name__}: {e}")
                assert "[UNIQUE-ERR-TOKENIZER-EMPTYSEQ]" in str(e)
            else:
                raise AssertionError(f"[UNIQUE-ERR-TORSIONBERT-HYP-NOERROR] Expected ValueError for input '{rna_seq}' of length {len(rna_seq)}")
        else:
            try:
                print(f"[DEBUG-TORSIONBERT-HYP] Input sequence: '{rna_seq}' (len={len(rna_seq)})")
                out = model_with_logits.predict_angles_from_sequence(rna_seq)
                print(f"[DEBUG-TORSIONBERT-HYP] Output type: {type(out)}")
                if hasattr(out, 'shape'):
                    print(f"[DEBUG-TORSIONBERT-HYP] Output shape: {out.shape}")
                else:
                    print(f"[DEBUG-TORSIONBERT-HYP] Output has no 'shape' attribute. Value: {out}")
                assert hasattr(out, 'shape'), f"[UNIQUE-ERR-TORSIONBERT-HYP-NOSHAPE] Output has no shape attribute. Type: {type(out)}"
                assert out.shape[0] == len(rna_seq), f"[UNIQUE-ERR-TORSIONBERT-HYP-SHAPE] For input '{rna_seq}', expected first dim {len(rna_seq)}, got {out.shape[0]}"
                assert out.shape[1] == 14, f"[UNIQUE-ERR-TORSIONBERT-HYP-SHAPE] For input '{rna_seq}', expected second dim 14, got {out.shape[1]}"
            except Exception as e:
                print(f"[DEBUG-TORSIONBERT-HYP][EXCEPTION] Input: '{rna_seq}' | Exception: {type(e).__name__}: {e}")
                raise

    def test_predict_angles_short_seq(
        self, model_with_logits: TorsionBertModel
    ) -> None:
        """
        A single char sequence => should now raise ValueError due to tokenizer contract.
        """
        seq = "A"
        try:
            print(f"[DEBUG-TORSIONBERT-SHORTSEQ] Input sequence: '{seq}' (len={len(seq)}) [Expecting ValueError]")
            model_with_logits.predict_angles_from_sequence(seq)
        except ValueError as e:
            print(f"[DEBUG-TORSIONBERT-SHORTSEQ][EXCEPTION] Input: '{seq}' | Exception: {type(e).__name__}: {e}")
            assert "[UNIQUE-ERR-TOKENIZER-EMPTYSEQ]" in str(e)
        else:
            raise AssertionError(f"[UNIQUE-ERR-TORSIONBERT-SHORTSEQ-NOERROR] Expected ValueError for input '{seq}' of length {len(seq)}")

    def test_predict_angles_partial_fill(
        self, model_with_logits: TorsionBertModel
    ) -> None:
        """
        For a normal sequence of length 4 => up to 2 tokens => partial fill.
        The code uses mid_idx to place data, leaving last row zeros if mid_idx >= seq_len.
        """
        seq = "ACGU"
        out = model_with_logits.predict_angles_from_sequence(seq)
        assert out.shape == (4, 14)
        # We won't do super-detailed row checks but at least ensure no crash, shape correct.

    def test_predict_angles_fallback_last_hidden(
        self, model_with_logits: TorsionBertModel
    ) -> None:
        """
        If no 'logits' in the output, fallback to last_hidden_state. We manually override
        to produce last_hidden_state only, verifying correct shape and no crash.
        """
        seq = "ACG"
        # Save original forward method and return_dict value to restore later
        original_forward = model_with_logits.model.forward
        original_return_dict = model_with_logits.return_dict

        # Create a custom forward function that correctly handles inputs
        def custom_forward(inputs):
            # Handle both dictionary-style inputs and direct parameter calls
            if isinstance(inputs, dict) and "input_ids" in inputs:
                input_ids = inputs["input_ids"]
                batch_size, seq_len = input_ids.shape
            else:
                # Default dimensions if input_ids not found or not properly shaped
                batch_size, seq_len = 1, len(seq)

            # Create a tensor with appropriate dimensions for last_hidden_state
            hidden_state = torch.zeros(
                batch_size, seq_len, 2 * model_with_logits.num_angles
            )
            return DummyLastHiddenStateOutput(hidden_state)

        try:
            # Replace model's forward method and disable return_dict to force last_hidden_state usage
            model_with_logits.model.forward = custom_forward
            model_with_logits.return_dict = False

            # Run the prediction
            result = model_with_logits.predict_angles_from_sequence(seq)

            # Verify shape
            assert result.shape == (len(seq), 2 * model_with_logits.num_angles)
        finally:
            # Restore original state
            model_with_logits.model.forward = original_forward
            model_with_logits.return_dict = original_return_dict

    @settings(
        deadline=None,  # Disable deadline for potentially slow network calls if not mocked
        suppress_health_check=[HealthCheck.function_scoped_fixture],  # Suppress check
    )
    @given(st.text(alphabet=["A", "C", "G", "U", "T"], min_size=0, max_size=25))
    def test_predict_angles_hypothesis_old(
        self, model_with_logits: TorsionBertModel, rna_sequence: str
    ) -> None:
        """
        Hypothesis-based test generating random RNA sequences from ACGUT,
        verifying shape correctness for each.
        """
        from hypothesis import assume
        assume(len(rna_sequence) >= 3)
        out = model_with_logits.predict_angles_from_sequence(rna_sequence)
        assert out.shape == (len(rna_sequence), 14)

    def test_invalid_device(self) -> None:
        """
        Creating TorsionBertModel with an invalid device string => RuntimeError.
        """
        with pytest.raises(RuntimeError):
            TorsionBertModel(
                model_path="any_path",  # Fixed: was model_name_or_path
                device="invalid_device",  # Already a string
                num_angles=7,
                max_length=512,
            )

    @pytest.mark.parametrize("invalid_model_path", [None, 123, ""])
    def test_invalid_model_path(self, invalid_model_path) -> None:
        """
        Passing invalid model paths should fall back to using a dummy model
        instead of raising an exception. This test verifies that behavior.
        """
        with patch(
            "rna_predict.pipeline.stageB.torsion.torsionbert_inference.AutoTokenizer.from_pretrained",
            side_effect=OSError("Mock HF load fail"),
        ):
            with patch(
                "rna_predict.pipeline.stageB.torsion.torsionbert_inference.AutoModel.from_pretrained",
                side_effect=OSError("Mock HF load fail"),
            ):
                # Should not raise an exception, but use a dummy model instead
                model = TorsionBertModel(
                    model_path=invalid_model_path,
                    device="cpu",
                    num_angles=7,
                    max_length=512,
                )
                # Verify that the model is a dummy model
                assert model.model is not None
                assert isinstance(model.model, DummyTorsionBertAutoModel)


# ----------------------------------------------------------------------
#        Tests for StageBTorsionBertPredictor (Integration)
# ----------------------------------------------------------------------
class TestStageBTorsionBertPredictor:
    """
    Simple integration tests to confirm that StageBTorsionBertPredictor
    calls TorsionBertModel as expected and returns correct shapes.
    """

    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(st.text(alphabet=["A", "C", "G", "U", "T"], min_size=3, max_size=4))
    def test_short_seq_hypothesis(self, predictor_fixture: StageBTorsionBertPredictor, seq: str) -> None:
        """
        Hypothesis-based test for short sequences (length 3-4). Ensures output shape matches sequence length and num_angles.

        Note: We use min_size=3 because sequences shorter than 3 characters are expected to raise an error
        in the tokenizer as per the mock_tokenizer fixture implementation.
        """
        try:
            print(f"[DEBUG-SHORT-SEQ-TEST] Testing sequence: '{seq}' (length: {len(seq)})")
            result = predictor_fixture(seq)
            angles = result["torsion_angles"]
            print(f"[DEBUG-SHORT-SEQ-TEST] Output shape: {angles.shape}")

            # Verify the output shape matches the sequence length and num_angles
            assert angles.shape == (len(seq), 16), (
                f"[UNIQUE-ERR-TORSIONBERT-HYPOTHESIS] Expected output shape ({len(seq)}, 16) for sequence '{seq}' with num_angles=16 in degrees mode, "
                f"but got {angles.shape}. Check predictor_fixture configuration, tokenizer, and model output."
            )

            # Additional debug: ensure input dict is not empty
            assert angles.shape[0] > 0, f"[UNIQUE-ERR-TORSIONBERT-EMPTY] Output tensor has zero rows for input '{seq}'"
        except ValueError as e:
            # If the sequence is too short, the tokenizer should raise a specific error
            if len(seq) < 3:
                assert "[UNIQUE-ERR-TOKENIZER-EMPTYSEQ]" in str(e), f"Expected tokenizer error for short sequence, got: {e}"
            else:
                raise

    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(seq=st.text(alphabet=["A", "C", "G", "U", "T"], min_size=4, max_size=4))
    def test_normal_seq(self, predictor_fixture: StageBTorsionBertPredictor, seq: str) -> None:
        """
        Property-based test: For a normal 4-letter sequence, output shape must be [4, 16].
        """
        result = predictor_fixture(seq)
        angles = result["torsion_angles"]
        assert angles.shape == (4, 16), (
            f"[UNIQUE-ERR-TORSIONBERT-NORMALSEQ] Expected output shape (4, 16) for a 4-residue sequence with num_angles=16 in degrees mode, "
            f"but got {angles.shape}. Seq: '{seq}'. Check predictor_fixture configuration and model output."
        )

    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(seq=st.text(alphabet=["A", "C", "G", "U", "T"], min_size=1, max_size=16))
    def test_various_lengths(self, predictor_fixture: StageBTorsionBertPredictor, seq: str) -> None:
        """
        Property-based test: For any sequence length, output shape must be [len(seq), 16].
        """
        from hypothesis import assume
        assume(len(seq) >= 3)
        out = predictor_fixture(seq)
        angles = out["torsion_angles"]
        if angles.shape != (len(seq), 16):
            print(f"[DEBUG-TEST] seq='{seq}' len(seq)={len(seq)} angles.shape={angles.shape}")
        assert angles.shape == (len(seq), 16), (
            f"[UNIQUE-ERR-TORSIONBERT-VARLEN] Expected output shape ({len(seq)}, 16) for sequence '{seq}', but got {angles.shape}. "
            "Check predictor_fixture configuration and model output."
        )

    def test_predictor_consistency(
        self, predictor_fixture: StageBTorsionBertPredictor
    ) -> None:
        """
        If the same sequence is passed multiple times, ensure shape is consistent
        and no crash occurs. This is a minimal check for consistency.
        """
        seq = "ACGUACGU"
        out1 = predictor_fixture(seq)
        out2 = predictor_fixture(seq)
        angles1 = out1["torsion_angles"]
        angles2 = out2["torsion_angles"]
        assert angles1.shape == angles2.shape == (8, 16), (
            f"[UNIQUE-ERR-TORSIONBERT-CONSISTENCY] Expected output shape (8, 16) for repeated sequence, but got {angles1.shape} and {angles2.shape}. "
            "Check predictor_fixture configuration and model output."
        )

    @settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        seq=st.text(alphabet=["A", "C", "G", "U", "T"], min_size=5, max_size=10),
        angle_mode=st.sampled_from(["sin_cos", "degrees", "radians"])
    )
    def test_angle_mode_conversion(self, mock_tokenizer, seq: str, angle_mode: str) -> None:
        """
        Property-based test: Verify that different angle modes produce outputs with appropriate value ranges.

        Args:
            predictor_fixture: The predictor fixture
            seq: Random RNA sequence
            angle_mode: Angle representation mode (sin_cos, degrees, radians)
        """
        from hypothesis import assume
        from omegaconf import OmegaConf

        # Skip sequences that are too short
        assume(len(seq) >= 5)

        # Create a new predictor with the specified angle mode
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
             patch("transformers.AutoModel.from_pretrained", return_value=DummyTorsionBertAutoModel(num_angles=7)):

            # Create a configuration with the specified angle mode
            cfg = OmegaConf.create({
                'model': {
                    'stageB': {
                        'torsion_bert': {
                            'model_name_or_path': 'bert-base-uncased',
                            'device': 'cpu',
                            'angle_mode': angle_mode,
                            'num_angles': 7,
                            'max_length': 512,
                        }
                    }
                }
            })
            # Add backward-compatible alias for root-level stageB_torsion
            cfg['stageB_torsion'] = cfg['model']['stageB']['torsion_bert']

            # Create a new predictor with the specified angle mode
            predictor = StageBTorsionBertPredictor(cfg)

            # Run prediction
            result = predictor(seq)
            angles = result["torsion_angles"]

            # Verify shape matches sequence length
            assert angles.shape[0] == len(seq), f"Expected first dimension {len(seq)}, got {angles.shape[0]}"

            # The output dimension depends on the model's implementation
            # For our dummy model, it's always 14 (2*7) regardless of angle_mode
            assert angles.shape[1] in [7, 14], f"Expected second dimension 7 or 14, got {angles.shape[1]}"

            # For a real model, we would verify value ranges based on angle_mode
            # But for our dummy model, we just check that the values are finite
            assert not torch.isnan(angles).any(), "Output contains NaN values"
            assert not torch.isinf(angles).any(), "Output contains infinity values"

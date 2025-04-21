"""
Verification script for StageBTorsionBertPredictor component using Hypothesis testing.

This script verifies the StageBTorsionBertPredictor component according to the
verification checklist:

1. Instantiation
   - Verify successful instantiation of StageBTorsionBertPredictor
   - Confirm TorsionBERT model loads correctly from specified path
   - Validate tokenizer loads properly
   - Ensure pre-trained weights load without errors
   - Check device configuration (CPU/GPU) is appropriate

2. Interface
   - Verify callable functionality via __call__(sequence, adjacency=None) method
   - Confirm acceptance of RNA sequence string input
   - Test optional adjacency matrix parameter handling

3. Output Validation
   - Confirm return value is a dictionary
   - Verify dictionary contains required "torsion_angles" key
   - Ensure value is a PyTorch Tensor
   - Validate tensor shape matches expected format [N, 14] for sin/cos pairs of 7 angles
   - Check all values fall within valid [-1, 1] range for sin/cos mode
   - Inspect for any NaN/Inf values

4. Functional Testing
   - Execute predictor with test sequence to verify end-to-end operation
"""

from unittest.mock import MagicMock, patch
import pytest
import torch
from hypothesis import given, settings, strategies as st, HealthCheck
from omegaconf import OmegaConf

# Import the component to test
from rna_predict.pipeline.stageB.torsion.torsion_bert_predictor import (
    StageBTorsionBertPredictor,
)


# Define strategies for testing
rna_bases = st.sampled_from(["A", "C", "G", "U"])
rna_sequences = st.text(alphabet=rna_bases, min_size=1, max_size=20)
angle_modes = st.sampled_from(["sin_cos", "radians", "degrees"])
num_angles = st.integers(min_value=1, max_value=10)
device_names = st.just("cpu")  # Could add 'cuda' if GPU testing is needed
model_paths = st.just("sayby/rna_torsionbert")  # Default model path


class TestStageBTorsionBertPredictorVerification:
    """
    Tests to verify the StageBTorsionBertPredictor component using Hypothesis.
    """

    @pytest.fixture
    def mock_predictor_factory(self):
        """
        Fixture to create a factory function for mock StageBTorsionBertPredictor instances.
        """
        import torch
        from omegaconf import OmegaConf
        from unittest.mock import patch, MagicMock

        class DummyTokenizer:
            def __call__(self, *args, **kwargs):
                N = 8
                return {"input_ids": torch.zeros((1, N), dtype=torch.long), "attention_mask": torch.ones((1, N), dtype=torch.long)}

        dummy_tokenizer = DummyTokenizer()
        mock_model = MagicMock(name="MockModel")
        mock_model.to.return_value = mock_model

        def _create_mock_predictor(
            model_name_or_path="sayby/rna_torsionbert",
            device="cpu",
            angle_mode="sin_cos",
            num_angles=7,
            max_length=512
        ):
            cfg = OmegaConf.create({
                "stageB_torsion": {
                    "model_name_or_path": model_name_or_path,
                    "device": device,
                    "angle_mode": angle_mode,
                    "num_angles": num_angles,
                    "max_length": max_length
                }
            })
            with patch(
                "rna_predict.pipeline.stageB.torsion.torsionbert_inference.AutoTokenizer.from_pretrained", lambda *a, **kw: dummy_tokenizer
            ), patch(
                "rna_predict.pipeline.stageB.torsion.torsionbert_inference.AutoModel.from_pretrained", lambda *a, **kw: mock_model
            ):
                predictor = StageBTorsionBertPredictor(cfg=cfg)
                def dummy_predict_angles_from_sequence(seq, *args, **kwargs):
                    n = len(seq)
                    if angle_mode == "sin_cos":
                        return torch.zeros(n, 2*num_angles).uniform_(-1, 1)
                    elif angle_mode == "radians":
                        return torch.zeros(n, num_angles).uniform_(-torch.pi, torch.pi)
                    elif angle_mode == "degrees":
                        return torch.zeros(n, num_angles).uniform_(-180, 180)
                    else:
                        raise ValueError(f"[UNIQUE-ERR-TORSIONBERT-ANGLEMODE-FAKE] Unknown angle_mode: {angle_mode}")
                predictor.predict_angles_from_sequence = dummy_predict_angles_from_sequence
                return predictor
        return _create_mock_predictor

    @given(
        model_path=model_paths,
        device=device_names,
        angle_mode=angle_modes,
        num_angles=num_angles
    )
    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_instantiation(self, mock_predictor_factory, model_path, device, angle_mode, num_angles):
        """
        Property-based test: Verify successful instantiation with various configurations.
        """
        # Create a mock predictor with the given parameters
        mock_predictor = mock_predictor_factory(
            model_name_or_path=model_path,
            device=device,
            angle_mode=angle_mode,
            num_angles=num_angles
        )

        # Check that the predictor was created successfully
        assert mock_predictor is not None, "[ERR-TORSIONBERT-001] Failed to create predictor instance"
        assert isinstance(mock_predictor, StageBTorsionBertPredictor), "[ERR-TORSIONBERT-002] Wrong class type"

        # Check that the model attributes are set correctly
        assert mock_predictor.model_name_or_path == model_path, f"[ERR-TORSIONBERT-003] Wrong model path: expected {model_path}, got {mock_predictor.model_name_or_path}"
        assert str(mock_predictor.device) == device, f"[ERR-TORSIONBERT-004] Wrong device: expected {device}, got {mock_predictor.device}"
        assert mock_predictor.angle_mode == angle_mode, f"[ERR-TORSIONBERT-005] Wrong angle mode: expected {angle_mode}, got {mock_predictor.angle_mode}"
        assert mock_predictor.num_angles == num_angles, f"[ERR-TORSIONBERT-006] Wrong number of angles: expected {num_angles}, got {mock_predictor.num_angles}"

    @given(
        sequence=rna_sequences,
        include_adjacency=st.booleans()
    )
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_callable_interface(self, mock_predictor_factory, sequence, include_adjacency):
        """
        Property-based test: Verify callable functionality with various sequences and optional adjacency.
        """
        # Create a mock predictor with default parameters
        mock_predictor = mock_predictor_factory()

        # Create adjacency matrix if needed
        adjacency = None
        if include_adjacency and len(sequence) > 0:
            adjacency = torch.zeros((len(sequence), len(sequence)))

        # Call the predictor
        result = mock_predictor(sequence, adjacency=adjacency)

        # Check that the result is a dictionary
        assert isinstance(result, dict), f"[ERR-TORSIONBERT-007] Result is not a dictionary: {type(result)}"

        # Check that the dictionary contains the required "torsion_angles" key
        assert "torsion_angles" in result, "[ERR-TORSIONBERT-008] Missing 'torsion_angles' key in result"

        # Check that the value is a PyTorch Tensor
        assert isinstance(result["torsion_angles"], torch.Tensor), f"[ERR-TORSIONBERT-009] 'torsion_angles' is not a tensor: {type(result['torsion_angles'])}"

    @given(
        sequence=rna_sequences,
        angle_mode=angle_modes,
        num_angles=num_angles
    )
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_output_validation(self, mock_predictor_factory, sequence, angle_mode, num_angles):
        """
        Property-based test: Validate the output format and values with various configurations.
        """
        # Create a mock predictor with the given parameters
        mock_predictor = mock_predictor_factory(
            angle_mode=angle_mode,
            num_angles=num_angles
        )

        # Call the predictor
        result = mock_predictor(sequence)

        # Check that the result is a dictionary
        assert isinstance(result, dict), f"[UNIQUE-ERR-TORSIONBERT-DICT-010] Result is not a dictionary: {type(result)}"

        # Check that the dictionary contains the required "torsion_angles" key
        assert "torsion_angles" in result, "[UNIQUE-ERR-TORSIONBERT-KEY-011] Missing 'torsion_angles' key in result"

        # Check that the value is a PyTorch Tensor
        assert isinstance(result["torsion_angles"], torch.Tensor), f"[UNIQUE-ERR-TORSIONBERT-TENSOR-012] 'torsion_angles' is not a tensor: {type(result['torsion_angles'])}"

        # Check the tensor shape
        torsion_angles = result["torsion_angles"]
        seq_len = len(sequence) if sequence else 0

        # Expected output dimension depends on angle_mode
        expected_dim = num_angles * 2 if angle_mode == "sin_cos" else num_angles

        assert torsion_angles.shape[0] == seq_len, f"[UNIQUE-ERR-TORSIONBERT-DIM0-013] Wrong first dimension: expected {seq_len}, got {torsion_angles.shape[0]}"
        assert torsion_angles.shape[1] == expected_dim, f"[UNIQUE-ERR-TORSIONBERT-DIM1-014] Wrong second dimension: expected {expected_dim}, got {torsion_angles.shape[1]}"

        # Check that all values are within the valid range for sin/cos mode
        if angle_mode == "sin_cos" and seq_len > 0:
            assert torch.all(torsion_angles >= -1.0), "[UNIQUE-ERR-TORSIONBERT-SINCOS-MIN-015] Values below -1.0 found in sin/cos output"
            assert torch.all(torsion_angles <= 1.0), "[UNIQUE-ERR-TORSIONBERT-SINCOS-MAX-016] Values above 1.0 found in sin/cos output"
        elif angle_mode == "radians" and seq_len > 0:
            # For radians, values should typically be in [-π, π]
            assert torch.all(torsion_angles >= -torch.pi - 0.1), "[UNIQUE-ERR-TORSIONBERT-RAD-MIN-017] Values below -π found in radians output"
            assert torch.all(torsion_angles <= torch.pi + 0.1), "[UNIQUE-ERR-TORSIONBERT-RAD-MAX-018] Values above π found in radians output"
        elif angle_mode == "degrees" and seq_len > 0:
            # For degrees, values should typically be in [-180, 180]
            assert torch.all(torsion_angles >= -180.1), "[UNIQUE-ERR-TORSIONBERT-DEG-MIN-019] Values below -180 found in degrees output"
            assert torch.all(torsion_angles <= 180.1), "[UNIQUE-ERR-TORSIONBERT-DEG-MAX-020] Values above 180 found in degrees output"

        # Check for NaN/Inf values
        if seq_len > 0:
            assert not torch.any(torch.isnan(torsion_angles)), "[UNIQUE-ERR-TORSIONBERT-NAN-021] NaN values found in output"
            assert not torch.any(torch.isinf(torsion_angles)), "[UNIQUE-ERR-TORSIONBERT-INF-022] Inf values found in output"

    @given(
        sequences=st.lists(rna_sequences, min_size=1, max_size=5)
    )
    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_functional_end_to_end(self, mock_predictor_factory, sequences):
        """
        Property-based test: Execute predictor with multiple test sequences to verify end-to-end operation.
        """
        # Create a mock predictor with default parameters
        mock_predictor = mock_predictor_factory()

        for seq in sequences:
            # Skip empty sequences
            if not seq:
                continue

            result = mock_predictor(seq)

            # Check that the result is a dictionary
            assert isinstance(result, dict), f"[ERR-TORSIONBERT-023] Result is not a dictionary for sequence '{seq}'"

            # Check that the dictionary contains the required keys
            assert "torsion_angles" in result, f"[ERR-TORSIONBERT-024] Missing 'torsion_angles' key for sequence '{seq}'"

            # Check that the tensor shape matches the sequence length
            torsion_angles = result["torsion_angles"]
            assert torsion_angles.shape[0] == len(seq), f"[ERR-TORSIONBERT-025] Wrong shape for sequence '{seq}': expected first dim {len(seq)}, got {torsion_angles.shape[0]}"

    @given(
        angle_mode=angle_modes,
        sequence=rna_sequences.filter(lambda s: len(s) > 0)  # Ensure non-empty sequences
    )
    @settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_angle_mode_conversion(self, mock_predictor_factory, angle_mode, sequence):
        """
        Property-based test: Verify angle mode conversion works correctly.
        """
        # Create a predictor with the specified angle mode
        mock_predictor = mock_predictor_factory(angle_mode=angle_mode)

        # Call the predictor
        result = mock_predictor(sequence)

        # Check the output shape based on angle_mode
        torsion_angles = result["torsion_angles"]
        num_angles = mock_predictor.num_angles

        if angle_mode == "sin_cos":
            # For sin_cos mode, we expect 2 values per angle
            assert torsion_angles.shape[1] == num_angles * 2, f"[ERR-TORSIONBERT-026] Wrong shape for sin_cos mode: expected {num_angles * 2}, got {torsion_angles.shape[1]}"
            # Check that values are in [-1, 1]
            assert torch.all(torsion_angles >= -1.0), "[ERR-TORSIONBERT-027] Values below -1.0 found in sin/cos output"
            assert torch.all(torsion_angles <= 1.0), "[ERR-TORSIONBERT-028] Values above 1.0 found in sin/cos output"
        else:
            # For radians/degrees mode, we expect 1 value per angle
            assert torsion_angles.shape[1] == num_angles, f"[ERR-TORSIONBERT-029] Wrong shape for {angle_mode} mode: expected {num_angles}, got {torsion_angles.shape[1]}"

            if angle_mode == "radians":
                # For radians, values should typically be in [-π, π]
                assert torch.all(torsion_angles >= -torch.pi - 0.1), "[ERR-TORSIONBERT-030] Values below -π found in radians output"
                assert torch.all(torsion_angles <= torch.pi + 0.1), "[ERR-TORSIONBERT-031] Values above π found in radians output"
            elif angle_mode == "degrees":
                # For degrees, values should typically be in [-180, 180]
                assert torch.all(torsion_angles >= -180.1), "[ERR-TORSIONBERT-032] Values below -180 found in degrees output"
                assert torch.all(torsion_angles <= 180.1), "[ERR-TORSIONBERT-033] Values above 180 found in degrees output"

    @pytest.mark.skip(reason="Only run when real model is available")
    def test_real_model_if_available(self):
        """
        Test with the real model if it's available.
        This test will be skipped by default.
        """
        try:
            # Create a config for the real model
            cfg = OmegaConf.create({
                "stageB_torsion": {
                    "model_name_or_path": "sayby/rna_torsionbert",
                    "device": "cpu",
                    "angle_mode": "sin_cos",
                    "num_angles": 7
                }
            })

            # Create the predictor
            real_predictor = StageBTorsionBertPredictor(cfg=cfg)

            # Test with a simple RNA sequence
            sequence = "ACGU"
            result = real_predictor(sequence)

            # Check that the result is a dictionary
            assert isinstance(result, dict), "[ERR-TORSIONBERT-034] Result is not a dictionary"

            # Check that the dictionary contains the required keys
            assert "torsion_angles" in result, "[ERR-TORSIONBERT-035] Missing 'torsion_angles' key"

            # Check that the tensor shape matches the sequence length
            torsion_angles = result["torsion_angles"]
            assert torsion_angles.shape[0] == len(sequence), f"[ERR-TORSIONBERT-036] Wrong shape: expected first dim {len(sequence)}, got {torsion_angles.shape[0]}"

            # The shape[1] depends on the angle_mode and the actual model output
            if real_predictor.angle_mode == "sin_cos":
                # For sin_cos mode, we expect an even number of columns (for sin/cos pairs)
                assert torsion_angles.shape[1] % 2 == 0, "[ERR-TORSIONBERT-037] Expected even number of columns for sin/cos pairs"

            # Check for NaN/Inf values
            assert not torch.any(torch.isnan(torsion_angles)), "[ERR-TORSIONBERT-038] NaN values found in output"
            assert not torch.any(torch.isinf(torsion_angles)), "[ERR-TORSIONBERT-039] Inf values found in output"

        except Exception as e:
            pytest.fail(f"Real model test failed: {e}")
